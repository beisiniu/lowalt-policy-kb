"""政策文档切分：按"第X条"切，保留条款边界。

为什么不直接 RecursiveCharacterTextSplitter？
政策文本的价值单位是**条**。"第九条 使用不超过25千克的…无需申请" 被从中间切断，
检索出来的 chunk 就只剩半句话，模型要么答错要么编。
所以先在条文边界切，再对超长条款做二次句号切分。
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

# 条款标题：第九条 / 第十二条 / 第二十三条，或"一、"这类项
ARTICLE_RE = re.compile(r"^(第[一二三四五六七八九十百零〇\d]+条)\s*")
SECTION_RE = re.compile(r"^(第[一二三四五六七八九十]+章)\s*")
# 无条号文件（通知/实施方案）常用「一、」「（一）」分节。
# 如果不按小节先分组，greedy 装包会把“物流补贴”和上一条“信息基础设施”粘进同一块，
# 向量语义被前一条稀释，检索就排不上来。
SUBSECTION_RE = re.compile(r"^(?:（[一二三四五六七八九十\d]+）|[一二三四五六七八九十]+、)")
_SENT_SPLIT = re.compile(r"(?<=[。；！？])")


def _pack_lines(lines: list[str], max_chars: int) -> list[str]:
    """把若干行按顺序装进不超过 max_chars 的块（行本身不切断）。"""
    out: list[str] = []
    buf = ""
    for ln in lines:
        if buf and len(buf) + len(ln) > max_chars:
            out.append(buf)
            buf = ""
        buf += ln
    if buf.strip():
        out.append(buf)
    return out


class PolicyChunkSplitter(TextSplitter):
    """把一篇政策切成若干「条文级」chunk。

    - 每块开头补上《政策名》，让 BM25 能命中"深圳市""条例"这类检索词
    - 单条超过 max_chars 时按句号再切，overlap 用一整句，避免半句截断
    - 没有条号的文本（通知、函）退化为段落装包切分
    """

    def __init__(self, max_chars: int = 600, overlap: int = 60, **kwargs):
        super().__init__(chunk_size=max_chars, chunk_overlap=overlap, **kwargs)
        self.max_chars = max_chars
        self.overlap_chars = overlap

    # TextSplitter 要求实现 split_text
    def split_text(self, text: str) -> list[str]:
        return [c for c in self._split(text) if c.strip()]

    # ------------------------------------------------------------------
    def _split(self, text: str) -> list[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        blocks: list[tuple[str, str]] = []  # (条号, 内容)
        cur_head, cur_body = "", []
        found_article = False

        for ln in lines:
            m = ARTICLE_RE.match(ln)
            if m:
                found_article = True
                if cur_body:
                    blocks.append((cur_head, "".join(cur_body)))
                cur_head = m.group(1)
                # 条号已单独抽成 head，正文里要剥掉，否则拼出来是"第九条 第九条 …"
                cur_body = [ARTICLE_RE.sub("", ln, count=1)]
            elif SECTION_RE.match(ln):
                if cur_body:
                    blocks.append((cur_head, "".join(cur_body)))
                cur_head, cur_body = "", []
            else:
                cur_body.append(ln)
        if cur_body:
            blocks.append((cur_head, "".join(cur_body)))

        if not found_article:
            # 通篇没有「第X条」（通知、函、公告）：先按小节分组，再在组内装包。
            # 这样不会把不同主题的两节粘成一个 chunk。
            groups: list[list[str]] = []
            cur: list[str] = []
            for ln in lines:
                if cur and len(ln) <= 80 and SUBSECTION_RE.match(ln):
                    groups.append(cur)
                    cur = [ln]
                else:
                    cur.append(ln)
            if cur:
                groups.append(cur)
            blocks = []
            for group in groups or [lines]:
                blocks.extend(("", c) for c in _pack_lines(group, self.max_chars))

        out: list[str] = []
        for head, body in blocks:
            if len(body) <= self.max_chars:
                out.append(f"{head} {body}".strip() if head else body)
            else:
                out.extend(self._split_long(head, body))
        return out

    def _split_long(self, head: str, body: str) -> list[str]:
        """超长条款按句子切，相邻块之间带 overlap。"""
        sents = [s for s in _SENT_SPLIT.split(body) if s.strip()]
        chunks, buf = [], ""
        for s in sents:
            if len(buf) + len(s) > self.max_chars and buf:
                chunks.append(buf)
                buf = buf[-self.overlap_chars:] if self.overlap_chars else ""
            buf += s
        if buf.strip():
            chunks.append(buf)
        return [f"{head} {c}".strip() if head else c for c in chunks]


def build_documents(text: str, meta: dict, splitter: PolicyChunkSplitter | None = None) -> list[Document]:
    """正文 + 元数据 → LangChain Document 列表（每条政策若干个）。

    每个 chunk 都**冗余带上全部时效元数据**，这是后面做过滤/时效门禁的前提：
    检索时不需要回表查政策表，chunk 自己就能回答"我还作不作数"。
    """
    splitter = splitter or PolicyChunkSplitter()
    name = meta.get("policy_name", "")
    parts = splitter.split_text(text)

    docs: list[Document] = []
    for i, part in enumerate(parts):
        content = f"《{name}》{part}" if name and not part.startswith("《") else part
        docs.append(
            Document(
                page_content=content,
                metadata={
                    **{k: meta.get(k, "") for k in
                       ("policy_name", "policy_id", "issue_date", "effective_date",
                        "expiry_date", "apply_region", "status", "source_url")},
                    "chunk_index": i,
                    "chunk_total": len(parts),
                    "article": ARTICLE_RE.match(part).group(1) if ARTICLE_RE.match(part) else "",
                },
            )
        )
    return docs
