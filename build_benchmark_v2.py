"""按 TQ 流程重做出题：一次多题、多题型、金标证据可机械验证。

相对第一版（build_benchmark.py）的四处改动：
  1. **一对条款出 1-3 题**，题型在提示词里给定枚举，由模型按材料实际情况选择，
     不再由数据（数值差异）单方面决定题型——缓解"八成是补贴/门槛题"的偏科。
  2. **每题强制输出金标证据**：B 条款的切片编号 + 逐字摘录的支撑句，
     本地计算字符区间。摘录必须能在 B 原文中逐字命中，否则整题丢弃（规则校验，不靠裁判）。
  3. **数值题额外校验**：标准答案里的每个数字都必须出现在金标证据里，杜绝"答案数字来自别处"。
  4. **新增前提错误题**（false_premise）：编造一个该地区现行政策中确实不存在的补贴/资质，
     问用户"能不能申请"，正确行为是拒绝编造。校验方式是机械的——编造名不得出现在该地区任何政策文本中。

用法：
  python build_benchmark_v2.py --limit-pairs 3 --out data/national/benchmark_tq2_sample.json
  python build_benchmark_v2.py --max-pairs 120 --out data/national/benchmark_tq2.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_benchmark as bm  # noqa: E402  复用配对/分桶/检索
from policycrawler.llm import LLMError, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402
from rag_eval.backends import build_backend  # noqa: E402
from rag_eval.temporal import clause_status, iso_or_none  # noqa: E402

CONFLICT_TYPES = (
    "subsidy_changed",       # 补贴金额/比例/奖励标准变化
    "threshold_changed",     # 门槛、数量、时限、规模等数值标准变化
    "scope_changed",         # 适用范围（地域/机型/场景/主体）变化
    "eligibility_changed",   # 资格、资质、备案前置条件变化
    "procedure_changed",     # 办理流程、申报材料、审批环节变化
    "repeal_recognition",    # 现行侧只是废止/停止执行，问"现在还能否享受"
)
TYPE_CN = {
    "subsidy_changed": "补贴标准",
    "threshold_changed": "门槛数值",
    "scope_changed": "适用范围",
    "eligibility_changed": "资格资质",
    "procedure_changed": "办理流程",
    "repeal_recognition": "失效识别",
    "false_premise": "前提错误",
}

GEN_SYSTEM = """你是时序 RAG 测试集的设计专家，熟悉中国低空经济与无人机政策。
给你两条同主题、同地域的条款：A 是已失效的旧条款，B 是评估时点仍有效的现行条款。

任务：先判断 A 与 B 是否构成"实质性规则变化"（同一个现实问题会得到不同答案）。
若构成，设计 1 到 3 道用户会真实问出口的问题，并为每道题给出**金标证据**。

只能输出一个 JSON 对象：
{
  "has_substantive_difference": true/false,
  "reason": "不超过 30 字",
  "questions": [
    {
      "conflict_type": "从下列六类中选一个",
      "question": "用户问题",
      "ground_truth": "只依据 B 的标准答案",
      "answer_type": "numeric|boolean|enum|text",
      "gold_evidence": {
        "quote": "从 B 中逐字摘录、能直接推出该答案的原句",
        "note": "该句如何支撑答案，不超过 20 字"
      }
    }
  ]
}

题型枚举（一题只属于一类）：
- subsidy_changed：补贴金额、比例、奖励上限变了
- threshold_changed：门槛、数量、规模、时限等数值标准变了
- scope_changed：适用范围（区域、机型、场景、主体）变了
- eligibility_changed：资格、资质、备案等前置条件变了
- procedure_changed：办理流程、申报材料、审批环节变了
- repeal_recognition：B 只是废止/停止执行旧规定，问题是"现在还能否享受"

硬约束（违反任一条，该题作废）：
1. gold_evidence.quote 必须**逐字摘自 B**，不得改写、不得跨段拼接、不得包含省略号；
   长度控制在 15~300 字。B 里找不到能直接支撑答案的原句，就不要出这道题。
2. question 不得出现：政策名称、文号、日期年份、"旧/新/废止/失效/修订/原来/此前"等泄漏词。
   要用地域、机型、场景等自然限定把问题问具体。
3. 不得使用"B 没写 = 政策取消"的缺失推断。若 B 没有直接规定所问事项，直接不出该题。
4. 一题只问一件事。数值题的答案数字必须能在 quote 中找到。
5. 优先让多道题的题型互不相同；若 B 的材料只支撑一类，就只出一类，**不要为凑数硬造**。
6. 不得编造 B 中不存在的数字、豁免条件或适用对象。
"""

VERIFY_SYSTEM = """你是测试集审核员。逐题审核给定题目是否可用，只输出 JSON：
{"items":[{"index":0,"valid":true/false,"reason":"不超过30字"}]}

审核标准（任一不满足即 valid=false）：
1. 标准答案能否由 gold_evidence.quote 直接推出；需要额外信息才能推出 → false。
2. quote 是否真的在给到的 B 原文里逐字出现（含数字、单位）。
3. 问题是否泄漏政策名、文号、日期年份或"旧/新/废止/修订"等词。
4. 是否把 B 没写的内容当成"已取消"。
5. 问题是否一题问了两件以上的事。
"""

FALSE_PREMISE_SYSTEM = """你在为政策问答系统构造"前提错误"测试题：用户问的东西在该地区现行政策中**根本不存在**，
系统的正确行为是明确说明没有该政策依据，而不是编造条件或金额。

给你：某地区的现行政策清单（标题）、其中一条真实扶持条款的原文。
请只输出一个 JSON 对象（字段如下，不要 Markdown、不要代码块）：

{
  "fake_name": "编造的补贴/资质/许可名称（8~20 字）",
  "question": "用户问题（含地域与场景，20~60 字）",
  "ground_truth": "标准答案（说明现行政策中没有该补贴/资质，不能申请，并提示以官方清单为准，40~120 字）",
  "basis_quote": "从给定的真实条款中逐字摘录的一句，用来说明该条款实际覆盖哪些情形",
  "reasoning": "不超过 25 字"
}

要求：
1. fake_name 必须**听起来合理但确实不存在**：把真实条款覆盖的场景替换成"本次指定场景"，
   拼成一个该地区政策从未出现过的名称。例如真实条款只覆盖物流配送，而指定场景是农业植保，
   则名称应围绕农业植保构造。**不得使用指定场景之外的场景**。
2. fake_name 不得是政策清单里任何标题的原文片段，也不得是真实条款里的连续原文。
3. basis_quote 必须逐字摘录自给定条款的**正文要求句**（含补贴对象、条件或标准），不得改写，也不得只摘政策名称/标题，长度 20~200 字。
4. ground_truth 必须明确"不存在"，不能给出任何金额或条件（给了就等于配合编造）。
"""

# 问题里出现这些词说明题面泄漏了时点或版本信息
LEAK_WORDS = ("旧版", "新版", "已废止", "已失效", "废止前", "修订前", "原来", "此前", "以前",
              "旧政策", "新政策", "旧规定", "新规定", "原政策", "老政策")
YEAR_RE = re.compile(r"20\d{2}\s*年")
# 缺失推断：标准答案声称"没写=取消/不要求"，而金标证据里并无对应的否定表述
INFER_RE = re.compile(r"未(将|对|规定|明确|提及|作|纳入)|不再要求|没有(规定|明确|提及)|"
                      r"不(再)?将.{0,12}作为|无相关(规定|要求)|未设置")
NEG_RE = re.compile(r"不(要求|需要|得|再)|取消|废止|停止|删除|不再执行|禁止")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def norm(s: str) -> str:
    return "".join(str(s or "").split())


def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """在原文中定位逐字摘录（容忍空白差异），返回正文字符区间。"""
    q = norm(quote)
    if not q:
        return None
    direct = text.find(quote)
    if direct >= 0:
        return direct, direct + len(quote)
    # 逐字未命中时按去空白后的位置反推
    flat, idx = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            flat.append(ch)
            idx.append(i)
    pos = "".join(flat).find(q)
    if pos < 0:
        return None
    return idx[pos], idx[pos + len(q) - 1] + 1


@dataclass
class Item:
    id: str
    source: str
    region: str
    topic: str
    conflict_type: str
    question: str
    ground_truth: str
    answer_type: str
    gold_evidence: dict[str, Any]
    old_clause_id: str = ""
    new_clause_id: str = ""
    old_clause_text: str = ""
    new_clause_text: str = ""
    old_policy: str = ""
    new_policy: str = ""
    old_status: str = ""
    new_status: str = ""
    old_valid_to: str = ""
    new_valid_from: str = ""
    as_of: str = ""
    llm_reasoning: str = ""
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def rule_check(question: str, ground_truth: str, quote: str, answer_type: str,
               new_text: str, new_policy: str) -> tuple[bool, list[str]]:
    """机械校验：这些条件不满足就不需要裁判介入，直接弃用。"""
    bad: list[str] = []
    q = question or ""
    gt = ground_truth or ""
    if len(q) < 10:
        bad.append("问题过短")
    if len(gt) < 5:
        bad.append("答案过短")
    for w in LEAK_WORDS:
        if w in q:
            bad.append(f"问题含泄漏词「{w}」")
            break
    if YEAR_RE.search(q):
        bad.append("问题含年份")
    # 政策名片段出现在问题里也算泄漏
    core = re.sub(r"[（(].*?[）)]", "", new_policy or "")
    core = core.replace("关于印发", "").replace("的通知", "").replace("市人民政府", "").replace("人民政府", "")
    for frag in (core[:12], core[-12:]):
        if len(frag) >= 8 and frag in q:
            bad.append("问题含政策名片段")
            break
    if not norm(quote):
        bad.append("缺金标证据")
    elif norm(quote) not in norm(new_text):
        bad.append("金标证据不在新条款原文中")
    if INFER_RE.search(gt) and not NEG_RE.search(quote):
        bad.append("疑似缺失推断（证据里没有否定表述，却回答\"不要求/未规定\"）")
    if answer_type == "numeric":
        nums = NUM_RE.findall(gt)
        if not nums:
            bad.append("数值题答案无数字")
        else:
            qn = norm(quote)
            for n in nums:
                if n not in qn:
                    bad.append(f"答案数字 {n} 不在金标证据中")
                    break
    return (not bad), bad


def gen_pair(client, pair, as_of: date) -> tuple[list[dict], str]:
    ovt = iso_or_none(bm.clause_valid_to(pair.old)) or "无"
    nvt = iso_or_none(bm.clause_valid_to(pair.new)) or "无（现行有效）"
    user = (
        f"评估时点：{as_of.isoformat()}\n主题：{pair.topic}\n地域：{pair.region}\n\n"
        f"【条款 A · 已失效】《{pair.old.policy_name}》{pair.old.article}\n"
        f"失效日={ovt}\n{pair.old.text}\n\n"
        f"【条款 B · 现行】《{pair.new.policy_name}》{pair.new.article}\n"
        f"到期日={nvt}\n{pair.new.text}"
    )
    raw = client.chat_json(GEN_SYSTEM, user, temperature=0.0, max_tokens=2600, thinking=False)
    if not bm._as_bool(raw.get("has_substantive_difference")):
        return [], str(raw.get("reason") or "")
    return list(raw.get("questions") or []), str(raw.get("reason") or "")


def verify_questions(client, pair, cands: list[dict]) -> list[bool]:
    """一次调用审完该对的所有候选题。"""
    payload = []
    for c in cands:
        payload.append({
            "question": c.get("question"),
            "ground_truth": c.get("ground_truth"),
            "gold_quote": (c.get("gold_evidence") or {}).get("quote"),
            "answer_type": c.get("answer_type"),
        })
    user = (
        f"【B 条款原文】《{pair.new.policy_name}》{pair.new.article}\n{pair.new.text}\n\n"
        f"【待审题目】\n{json.dumps(payload, ensure_ascii=False, indent=1)}"
    )
    raw = client.chat_json(VERIFY_SYSTEM, user, temperature=0.0, max_tokens=1500, thinking=False)
    flags = [True] * len(cands)
    for it in raw.get("items") or []:
        try:
            i = int(it.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(flags):
            flags[i] = bm._as_bool(it.get("valid"))
    return flags


SCENES = ("婚庆航拍", "农业植保", "电力巡检", "地形测绘", "夜景灯光表演", "外卖送餐",
          "医疗急救送样", "影视拍摄", "林业防火巡查", "水上救援", "建筑外墙清洗",
          "环境监测", "快递末端配送", "景区观光", "港口理货", "渔业巡查",
          "桥梁检测", "通信中继", "气象探测", "交通疏导")


def gen_false_premise(client, region: str, titles: list[str], basis_clause, as_of: date,
                      scene: str = "") -> dict | None:
    user = (
        f"评估时点：{as_of.isoformat()}\n地域：{region}\n"
        f"本次指定场景：{scene or '由你选择一个与给定条款覆盖范围不同的场景'}\n\n"
        f"【该地区现行政策清单】\n" + "\n".join(f"- {t}" for t in titles[:40]) +
        f"\n\n【一条真实扶持条款】《{basis_clause.policy_name}》{basis_clause.article}\n{basis_clause.text}"
    )
    try:
        raw = client.chat_json(FALSE_PREMISE_SYSTEM, user, temperature=0.2, max_tokens=1200, thinking=False)
    except LLMError as e:
        log(f"  [false_premise 失败] {region}: {e}")
        return None
    return raw


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description="TQ-v2 出题：多题多类型 + 金标证据 + 前提错误题")
    ap.add_argument("--as-of", default="2026-10-27")
    ap.add_argument("--out", default=str(ROOT / "data" / "national" / "benchmark_tq2.json"))
    ap.add_argument("--chroma-dir", default=str(ROOT / "chroma_lowalt"))
    ap.add_argument("--collection", default="lowalt_policy")
    ap.add_argument("--embedding-function", default="bge-zh")
    ap.add_argument("--min-similarity", type=float, default=0.12)
    ap.add_argument("--pairs-per-clause", type=int, default=2, help="每条旧条款最多配几个新条款")
    ap.add_argument("--max-pairs", type=int, default=120, help="最多处理多少候选对（成本上限）")
    ap.add_argument("--limit-pairs", type=int, default=0, help="只跑前 N 对（调试）")
    ap.add_argument("--max-items", type=int, default=0, help="出够多少题就停（0=不限）")
    ap.add_argument("--region-cap", type=float, default=0.20, help="单一地区题量占比上限")
    ap.add_argument("--false-premise-per-region", type=int, default=1)
    ap.add_argument("--no-verify", action="store_true", help="跳过 LLM 复核（只做机械校验）")
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "qwen", "glm"])
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--report", default=str(ROOT / "output" / "benchmark_tq2_table.md"))
    args = ap.parse_args()
    as_of = date.fromisoformat(args.as_of)

    backend = build_backend(argparse.Namespace(
        backend="chroma", chroma_dir=args.chroma_dir, chroma_host="", chroma_port="8000",
        collection=args.collection, embedding_function=args.embedding_function,
        index_dir=str(ROOT / "kb" / "index"),
    ))
    clauses = backend.all_clauses()
    log(f"条款：{len(clauses)} 条（{backend.describe()}）")

    pairs, stats = bm.build_candidate_pairs(
        clauses, as_of, min_similarity=args.min_similarity, max_pairs_per_topic=args.pairs_per_clause,
    )
    pairs = bm.diversify_pairs(pairs)
    log(f"候选对 {len(pairs)}；{json.dumps(stats, ensure_ascii=False)}")
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]
    else:
        pairs = pairs[: args.max_pairs]

    # 该地区现行政策清单（供前提错误题确认"不存在"）
    title_by_region: dict[str, list[str]] = {}
    valid_by_region: dict[str, list] = {}
    inv_names = {c.policy_name for c in clauses if (bm.clause_valid_to(c) or date.max) < as_of}
    for c in clauses:
        if c.policy_name in inv_names:
            continue
        ok, _ = bm.usable(c, as_of)
        if not ok:
            continue
        title_by_region.setdefault(c.region or "", [])
        if c.policy_name not in title_by_region[c.region or ""]:
            title_by_region[c.region or ""].append(c.policy_name)
        valid_by_region.setdefault(c.region or "", []).append(c)

    client = provider_client(args.provider)
    items: list[Item] = []
    rejected: list[dict] = []
    region_count: Counter[str] = Counter()
    type_count: Counter[str] = Counter()
    idx = 0

    for pi, pair in enumerate(pairs, 1):
        log(f"[{pi}/{len(pairs)}] {pair.region}·{pair.topic} sim={pair.similarity:.3f}")
        cap = int(args.max_pairs * args.region_cap) if args.max_pairs else 10 ** 6
        if region_count[pair.region] >= max(2, cap):
            log(f"  - 跳过：{pair.region} 已达地区上限 {cap}")
            continue
        try:
            cands, reason = gen_pair(client, pair, as_of)
        except LLMError as e:
            log(f"  [生成失败] {e}")
            continue
        if not cands:
            log(f"  - 无实质差异（{reason[:30]}）")
            continue

        # 机械校验
        passed: list[dict] = []
        for c in cands:
            ct = str(c.get("conflict_type") or "").strip()
            ge = c.get("gold_evidence") or {}
            quote = str(ge.get("quote") or "")
            ok, why = rule_check(
                str(c.get("question") or ""), str(c.get("ground_truth") or ""), quote,
                str(c.get("answer_type") or "text"), pair.new.text, pair.new.policy_name,
            )
            if ct not in CONFLICT_TYPES:
                ok, why = False, why + [f"题型非法 {ct}"]
            if ok:
                passed.append(c)
            else:
                rejected.append({"region": pair.region, "question": str(c.get("question"))[:60], "reasons": why})
                log(f"  × 弃用（{'; '.join(why)}）")
        if not passed:
            continue

        # LLM 复核
        if not args.no_verify:
            try:
                flags = verify_questions(client, pair, passed)
            except LLMError as e:
                log(f"  [复核失败，保留机械校验结果] {e}")
                flags = [True] * len(passed)
            kept = []
            for c, f in zip(passed, flags):
                if f:
                    kept.append(c)
                else:
                    rejected.append({"region": pair.region, "question": str(c.get("question"))[:60],
                                     "reasons": ["LLM 复核未通过"]})
            passed = kept

        for c in passed:
            if args.max_items and len(items) >= args.max_items:
                break
            if region_count[pair.region] >= max(2, cap):
                break
            idx += 1
            quote = str((c.get("gold_evidence") or {}).get("quote") or "")
            span = find_span(pair.new.text, quote)
            items.append(Item(
                id=f"TQ2-{idx:03d}",
                source="TQ2",
                region=pair.region,
                topic=pair.topic,
                conflict_type=str(c.get("conflict_type")),
                question=str(c.get("question") or "").strip(),
                ground_truth=str(c.get("ground_truth") or "").strip(),
                answer_type=str(c.get("answer_type") or "text"),
                gold_evidence={
                    "chunk_id": pair.new.id,
                    "policy_name": pair.new.policy_name,
                    "article": pair.new.article,
                    "quote": quote,
                    "char_span": list(span) if span else None,
                    "source_url": "",
                    "negative": False,
                    "note": str((c.get("gold_evidence") or {}).get("note") or ""),
                },
                old_clause_id=pair.old.id,
                new_clause_id=pair.new.id,
                old_clause_text=pair.old.text,
                new_clause_text=pair.new.text,
                old_policy=pair.old.policy_name,
                new_policy=pair.new.policy_name,
                old_status=clause_status(pair.old),
                new_status=clause_status(pair.new),
                old_valid_to=iso_or_none(bm.clause_valid_to(pair.old)) or "",
                new_valid_from=iso_or_none(bm.clause_valid_from(pair.new)) or "",
                as_of=as_of.isoformat(),
                llm_reasoning=str(c.get("reasoning") or ""),
                checks={"rule": "pass", "verify": "skip" if args.no_verify else "pass",
                        "quote_span_found": bool(span)},
            ))
            region_count[pair.region] += 1
            type_count[str(c.get("conflict_type"))] += 1
            log(f"  + {items[-1].id} [{items[-1].conflict_type}] {items[-1].question[:34]}")
        time.sleep(args.sleep)

    # 前提错误题
    fp_n = 0
    for region in sorted(title_by_region):
        if args.false_premise_per_region <= 0:
            break
        pool = valid_by_region.get(region) or []
        if len(pool) < 2:
            continue
        basis = max(pool, key=lambda c: len(c.text))
        raw = gen_false_premise(client, region, title_by_region[region], basis, as_of,
                                scene=SCENES[fp_n % len(SCENES)])
        if not raw:
            continue
        fake = str(raw.get("fake_name") or "").strip()
        q = str(raw.get("question") or "").strip()
        gt = str(raw.get("ground_truth") or "").strip()
        bq = str(raw.get("basis_quote") or "").strip()
        # 机械校验：编造名不得出现在该地区任何政策标题或条款里
        all_titles = " ".join(title_by_region[region])
        all_text = " ".join(c.text for c in pool)
        if not fake or len(fake) < 4:
            rejected.append({"region": region, "question": q[:60], "reasons": ["fake_name 缺失"]})
            continue
        if fake in all_titles or norm(fake) in norm(all_text):
            rejected.append({"region": region, "question": q[:60], "reasons": [f"编造名「{fake}」在该地区政策中真实存在"]})
            log(f"  × 前提错误题弃用：{fake} 真实存在")
            continue
        if norm(bq) not in norm(basis.text):
            rejected.append({"region": region, "question": q[:60], "reasons": ["basis_quote 不在原文"]})
            continue
        # 证据必须是条款正文里的句子，不能是政策标题本身
        if len(bq) < 15 or norm(bq) in norm(basis.policy_name) or norm(basis.policy_name) in norm(bq):
            rejected.append({"region": region, "question": q[:60], "reasons": ["basis_quote 是标题而非条款正文"]})
            log(f"  × 前提错误题弃用：basis_quote 取成了标题")
            continue
        fp_n += 1
        idx += 1
        span = find_span(basis.text, bq)
        items.append(Item(
            id=f"TQ2-FP{fp_n:02d}",
            source="TQ2",
            region=region,
            topic="前提错误",
            conflict_type="false_premise",
            question=q,
            ground_truth=gt,
            answer_type="refuse",
            gold_evidence={
                "chunk_id": basis.id,
                "policy_name": basis.policy_name,
                "article": basis.article,
                "quote": bq,
                "char_span": list(span) if span else None,
                "source_url": "",
                "negative": True,
                "fake_name": fake,
                "checked_policies": title_by_region[region][:20],
                "note": "该地区现行政策中不存在此名称；正确行为是明确指出没有依据",
            },
            new_clause_id=basis.id,
            new_clause_text=basis.text,
            new_policy=basis.policy_name,
            new_status=clause_status(basis),
            new_valid_from=iso_or_none(bm.clause_valid_from(basis)) or "",
            as_of=as_of.isoformat(),
            llm_reasoning=str(raw.get("reasoning") or ""),
            checks={"rule": "pass", "fake_absent": True, "quote_span_found": bool(span)},
        ))
        type_count["false_premise"] += 1
        log(f"  + {items[-1].id} [false_premise] {q[:34]}")

    # 落盘
    out = Path(args.out)
    bm.atomic_write_json(out, [it.to_dict() for it in items])
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "as_of": as_of.isoformat(),
        "pairs_used": len(pairs),
        "items": len(items),
        "by_region": dict(region_count.most_common()),
        "by_type": dict(type_count.most_common()),
        "rejected": len(rejected),
        "pair_stats": stats,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 报表
    write_table(items, Path(args.report), meta)

    print(json.dumps(meta, ensure_ascii=False, indent=1))
    backend.close()
    return 0


def write_table(items: list[Item], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TQ-v2 题库：问题、标准答案与金标证据",
        "",
        f"生成时间：{meta['built_at']}　|　基准日：{meta['as_of']}　|　题量：**{meta['items']} 道**",
        f"地区 {len(meta['by_region'])} 个　|　题型 {len(meta['by_type'])} 类　|　弃用 {meta['rejected']} 题",
        "",
        "## 分布",
        "",
        "| 题型 | 题数 | 占比 |",
        "| --- | --- | --- |",
    ]
    total = max(1, meta["items"])
    for k, v in meta["by_type"].items():
        lines.append(f"| {TYPE_CN.get(k, k)}（{k}） | {v} | {v/total:.0%} |")
    lines += ["", "| 地区 | 题数 |", "| --- | --- |"]
    for k, v in meta["by_region"].items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## 逐题明细", ""]
    for it in items:
        ge = it.gold_evidence
        lines += [
            f"### {it.id}　[{TYPE_CN.get(it.conflict_type, it.conflict_type)}]　{it.region}",
            "",
            f"**问题**：{it.question}",
            "",
            f"**标准答案**：{it.ground_truth}",
            "",
            f"**金标证据**：",
            f"- 切片编号：`{ge.get('chunk_id', '')}`",
            f"- 出处：《{ge.get('policy_name', '')}》{ge.get('article', '')}",
            f"- 原文摘录：「{ge.get('quote', '')}」",
            f"- 字符区间：{ge.get('char_span')}",
        ]
        if ge.get("negative"):
            lines.append(f"- 类型：**否定证据**（编造名「{ge.get('fake_name', '')}」在该地区政策中不存在）")
        lines += [
            "",
            f"<details><summary>旧条款（已失效）与现行条款对照</summary>",
            "",
            f"**旧**：《{it.old_policy}》（失效日 {it.old_valid_to or '—'}）",
            f"> {it.old_clause_text[:400]}",
            "",
            f"**现行**：《{it.new_policy}》（生效日 {it.new_valid_from or '—'}）",
            f"> {it.new_clause_text[:400]}",
            "",
            "</details>",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报表 → {path}")


if __name__ == "__main__":
    raise SystemExit(main())
