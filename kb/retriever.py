"""检索器：BM25 关键词召回 + 元数据硬过滤。

为什么不用向量库？
  政策问答的查询词高度术语化（"申请""审批""适飞空域""真高120米"），
  BM25 在这类精确术语上比稠密向量更稳，而且**完全离线、零 API、可复现**。
  需要语义泛化时再叠加向量检索（见 EmbeddingRetriever 预留接口）。

真正影响答对率的不是打分函数，而是过滤：
  「已废止的文件」字面上往往和问题最像，向量和 BM25 都会把它排第一。
  所以先把不该出现的文档 mask 掉，再排序 —— 这一步在 Dify 里要靠手工配元数据过滤，
  在这里是代码，可测试、可解释。
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .schema import freshness, is_usable, region_matches, to_date, tokenize

RECENCY_WINDOW_DAYS = 1095  # 3 年

# 问题里出现这些词，说明问的是"要不要审批"，命中同类词的条款应加权
INTENT_WORDS = ("申请", "审批", "批准", "备案", "许可", "工作日", "无需", "不需要", "手续", "资质")
APPROVAL_HINTS = ("申请", "审批", "批准", "备案", "许可", "工作日", "无需", "不需要", "合格证", "执照")

# 罚则条款：除非问题本身在问处罚，否则不该排第一 ——
# 问"4 千克要不要执照"，最像的却是"未取得执照罚 5000 元"，答非所问。
PENALTY_HINTS = ("处罚", "罚款", "责令", "违反本规定", "法律责任", "追究")
PENALTY_ASK = ("处罚", "罚款", "违规", "违反", "罚", "后果", "责任")

# 问题里有具体数字（1500米 / 25千克 / 3个工作日）时，含数字的条款更可能有答案
_DIGIT_RE = re.compile(r"\d+")


class PolicyRetriever:
    def __init__(self, docs: list[Document], tokens: Optional[list[list[str]]] = None,
                 today: Optional[date] = None, enforce_freshness: bool = True,
                 newer_bias: float = 0.35, intent_boost: float = 0.20):
        if not docs:
            raise ValueError("索引为空，先跑 kb_ingest.py")
        self.docs = docs
        self.tokens = tokens or [tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(self.tokens)
        self.today = today
        self.enforce_freshness = enforce_freshness
        # 「新法优于旧法」：同一问题命中多篇时，施行日期更新的应排前面
        self.newer_bias = newer_bias
        # 意图重排：问"要不要审批"时，含审批动作词的条款更相关
        self.intent_boost = intent_boost

    # ------------------------------------------------------------------
    def score(self, query: str) -> list[float]:
        return list(self.bm25.get_scores(tokenize(query)))

    def retrieve(
        self,
        query: str,
        k: int = 4,
        region: str = "",
        enforce_freshness: Optional[bool] = None,
        dedupe_by_policy: bool = True,
    ) -> list[Document]:
        """返回 top-k chunk（metadata 里带 _score / _usable / _reason）。

        不可用的文档不是"排后面"，而是直接剔除 —— 否则 top-1 很容易是已废止文件。
        """
        use_filter = self.enforce_freshness if enforce_freshness is None else enforce_freshness
        raw = self.score(query)
        query_wants_approval = any(w in query for w in INTENT_WORDS)
        query_asks_penalty = any(w in query for w in PENALTY_ASK)
        query_has_digit = bool(_DIGIT_RE.search(query))

        # 先把"能用的"筛出来，再在这个子集里做归一化，否则会被失效文档的高分污染
        pool: list[tuple[float, int]] = []
        for i, s in enumerate(raw):
            if s <= 0:
                continue
            md = self.docs[i].metadata
            if use_filter and not is_usable(md, self.today, region):
                continue
            text = self.docs[i].page_content
            score = float(s)

            if query_wants_approval and any(w in text for w in APPROVAL_HINTS):
                score *= 1.0 + self.intent_boost
            # 罚则条款：问的不是"罚多少"就降权（不剔除——问违规后果时还要用）
            if not query_asks_penalty and any(w in text for w in PENALTY_HINTS):
                score *= 0.55
            # 数字对齐：问「1500 米」时，「在境内适用本规定」这种空条款不该赢过写了具体米数的条款
            if query_has_digit and _DIGIT_RE.search(text):
                score *= 1.15

            pool.append((score, i))

        if not pool:
            return []

        # 新法优先：用**绝对**时间窗算新鲜度，而不是在候选池里做 min-max。
        # 池内归一会把"相差两个月"放大成 0→1 的差距，反而压过相关性；
        # 固定 3 年窗口下，两个月只差 0.05，真正的新旧法（跨年）才有明显权重。
        today = self.today or date.today()
        window_start = today - timedelta(days=RECENCY_WINDOW_DAYS)

        def recency_of(md: dict) -> float:
            d = to_date(md.get("effective_date", ""))
            if not d:
                return 0.0
            return max(0.0, min(1.0, (d - window_start).days / RECENCY_WINDOW_DAYS))

        max_score = max(s for s, _ in pool)

        def rank_key(score: float, idx: int) -> float:
            relevance = score / max_score
            recency = recency_of(self.docs[idx].metadata)
            return relevance * (1 - self.newer_bias) + recency * self.newer_bias

        ranked = sorted(((rank_key(s, i), s, i) for s, i in pool), reverse=True)

        picked: list[Document] = []
        seen_policy: set[str] = set()
        for rk, s, i in ranked:
            doc = self.docs[i]
            pid = doc.metadata.get("policy_id") or doc.metadata.get("policy_name", "")
            if dedupe_by_policy and pid in seen_policy:
                continue
            md = dict(doc.metadata)
            ok, reason = freshness(md, self.today)
            md.update({"_score": round(float(s), 4), "_rank": round(float(rk), 4),
                       "_usable": ok, "_reason": reason,
                       "_region_ok": region_matches(md.get("apply_region", ""), region)})
            picked.append(Document(page_content=doc.page_content, metadata=md))
            seen_policy.add(pid)
            if len(picked) >= k:
                break
        return picked

    # ------------------------------------------------------------------
    def stale_probe(self, query: str, k: int = 4, region: str = "") -> list[Document]:
        """不过滤地召回 top-k，用于时效门禁判断"是不是全过期了"。"""
        return self.retrieve(query, k=k, region=region, enforce_freshness=False, dedupe_by_policy=True)

    def stats(self, today: Optional[date] = None) -> dict:
        regions: dict[str, int] = {}
        stale = 0
        for d in self.docs:
            md = d.metadata
            regions[md.get("apply_region") or "未知"] = regions.get(md.get("apply_region") or "未知", 0) + 1
            ok, _ = freshness(md, today or self.today)
            if not ok:
                stale += 1
        return {"chunks": len(self.docs), "stale_chunks": stale, "by_region": regions}
