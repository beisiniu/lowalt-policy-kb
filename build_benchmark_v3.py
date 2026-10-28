#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""出题契约 v3：先分级 → 再列冲突点 → 按点出题，不相关就不出。

与 v2 的根本差别（不是加校验，是改产出关系）：

  v2：给定一对条款，要求"设计 1 到 3 道题"。
      → LLM 即使只找到半个差异点也会凑满 1 道，出现大量「题目问的点旧条款压根没讲」的题。

  v3：给定一对条款，先判断两段是否在讨论同一个现实问题（High/Medium/Low），
      只有 High 才进入"列冲突点"，且**每个冲突点必须写成 a_answer / b_answer 成对形式**，
      两条答案实质相同则该点作废；题目数量 = 有效冲突点数量，不设上下限。

  一对条款能有几个冲突点，就出几道题；一个都没有，就一道不出——这是正常结果。

冲突点自带可机械验证的证据：
  a_quote 必须逐字出现在旧条款 A 中，b_quote 必须逐字出现在现行条款 B 中。
  编造不出来（原文里没有那句话就找不到），因此不需要再做"毒饵有效性"的事后检验。

用法：
  python build_benchmark_v3.py --limit-pairs 3 --out data/national/benchmark_tq3_sample.json
  python build_benchmark_v3.py --max-pairs 120 --out data/national/benchmark_tq3.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_benchmark as bm  # noqa: E402
from build_benchmark_v2 import (  # noqa: E402
    CONFLICT_TYPES,
    Item,
    LEAK_WORDS,
    NUM_RE,
    YEAR_RE,
    find_span,
    log,
    norm,
)
from policycrawler.llm import LLMError, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402
from rag_eval.backends import build_backend  # noqa: E402
from rag_eval.temporal import bigram_jaccard, clause_status, iso_or_none  # noqa: E402

GEN3_SYSTEM = """你是中国低空经济政策的评测题目设计专家。给你同地域的两条条款：
A 是已失效的旧条款，B 是评估时点仍有效的现行条款。

请严格按下面三步执行。**宁可不出题，也不要为凑数出题。**

第一步 · 契合度分级
判断 A 与 B 是否在讨论"同一个现实问题"——同一类主体、同一类事项（补贴、门槛、资格、流程、范围）。
- "High"：同一事项，且对同一问题 A 与 B 的答案会不同
- "Medium"：同一事项，但答案相同，只是表述、体例或条款序号变化
- "Low"：不是同一件事，或无法就同一问题作答

第二步 · 冲突点清单
仅当 level 为 "High" 时执行。逐条列出 A 与 B 对同一问题给出的不同答案。每条写成：
{"issue": "现实问题（不含政策名称与日期）",
 "a_answer": "按 A 会得到的答案",
 "b_answer": "按 B 会得到的答案",
 "a_quote": "支撑 a_answer 的 A 原文逐字摘录，15~200 字",
 "b_quote": "支撑 b_answer 的 B 原文逐字摘录，15~200 字"}

冲突点的硬性要求（违反则该条作废）：
1. a_quote 必须逐字出现在 A 中，b_quote 必须逐字出现在 B 中；不得改写、不得跨段拼接、不得用省略号。
2. a_answer 与 b_answer 必须**实质不同**：是/否相反，或数值、比例、范围、主体、条件不同。
   若两条答案意思相同（只是措辞不同），说明此处没有冲突，删掉这一条。
3. 一个冲突点只对应一件具体的事，不要把整条条款概括成一个点。
4. 找不到符合上述要求的冲突点时，conflict_points 返回空数组。

第三步 · 按点出题
仅当 conflict_points 非空时执行。**每个冲突点出 1 道题**，题目必须精确问在这个点上：
{"point_index": 对应冲突点下标（从 0 开始）,
 "conflict_type": "subsidy_changed|threshold_changed|scope_changed|eligibility_changed|procedure_changed|repeal_recognition 之一",
 "question": "用户会真实问出口的问题",
 "ground_truth": "只依据 B 的答案",
 "answer_type": "numeric|boolean|enum",
 "gold_quote": "从该冲突点的 b_quote 中摘取、能直接推出答案的句子"}

题目的硬性要求：
1. question 不得出现：政策名称、文号、年份、"旧/新/废止/失效/修订/原来/此前"等泄漏词。
   用地域、机型、场景把问题问具体。
2. **question 控制在 50 字以内**，只保留区分答案所必需的最少条件，不要把条款里的限定条件逐条抄进问题。
   正确示范：「在黄埔区建无人机小型起降点，是否只要建成就能申请补贴？」
   错误示范（照抄条款）：「在黄埔区建无人机小型起降点，投入 50% 补贴、每个不超过 10 万元、每家企业每年不超过 600 万元，是否只要建成就能申请？」
3. ground_truth 只能依据 B，不得混入 A 的内容。
4. answer_type 只用 numeric / boolean / enum 三者之一，不要用 text。
5. 数值题的答案数字必须出现在 gold_quote 中。
6. 不得使用"B 没规定 = 政策取消"的缺失推断；若冲突确实来自某一侧的沉默，
   答案要如实写成"B 未规定该要求"这类表述，不要断言"政策取消了该要求"。
7. 题目数量不得超过有效冲突点数量，不要为凑题型而重复问同一个点。

数量预期：1~2 个冲突点是常见情况，3 个及以上较少见。
若两条条款没有冲突（level 为 Medium 或 Low），questions 返回空数组——这是正常结果，不扣分。

只输出一个 JSON 对象：
{"level": "High|Medium|Low",
 "level_reason": "不超过 30 字",
 "conflict_points": [],
 "questions": []}"""

ANSWER_TYPES = ("numeric", "boolean", "enum")
MIN_QUOTE = 15
MAX_QUOTE = 300


def check_point(p: dict, old_text: str, new_text: str) -> tuple[bool, list[str]]:
    """冲突点可机械验证：两段引文各自逐字命中原文，两个答案实质不同。"""
    bad: list[str] = []
    aq = str(p.get("a_quote") or "").strip()
    bq = str(p.get("b_quote") or "").strip()
    aa = str(p.get("a_answer") or "").strip()
    bb = str(p.get("b_answer") or "").strip()

    if not aq or not bq:
        bad.append("冲突点缺引文")
    else:
        if not (MIN_QUOTE <= len(aq) <= MAX_QUOTE):
            bad.append(f"a_quote 长度 {len(aq)} 超范围")
        elif norm(aq) not in norm(old_text):
            bad.append("a_quote 不在旧条款原文中")
        if not (MIN_QUOTE <= len(bq) <= MAX_QUOTE):
            bad.append(f"b_quote 长度 {len(bq)} 超范围")
        elif norm(bq) not in norm(new_text):
            bad.append("b_quote 不在现行条款原文中")

    if not aa or not bb:
        bad.append("冲突点缺答案")
    else:
        na, nb = norm(aa), norm(bb)
        if na == nb:
            bad.append("两答案完全相同")
        elif na in nb or nb in na:
            bad.append("两答案互相包含")
        elif bigram_jaccard(aa, bb) >= 0.85:
            bad.append(f"两答案高度相似（{bigram_jaccard(aa, bb):.2f}）")
    return (not bad), bad


def check_question(q: dict, points: list[dict], new_text: str,
                   new_policy: str) -> tuple[bool, list[str]]:
    """题目校验：指向有效冲突点、无泄漏、金标证据在 B 中、数值题数字在引文中。

    答案长度门槛按 answer_type 分档：boolean/enum 的正确答法本就极短
    （「不能」「不需要」「应当」），不能用统一的长文本门槛把它挡掉。
    """
    bad: list[str] = []
    pi = q.get("point_index")
    if not isinstance(pi, int) or not (0 <= pi < len(points)):
        bad.append("point_index 非法")

    at = str(q.get("answer_type") or "").strip()
    if at not in ANSWER_TYPES:
        bad.append(f"answer_type 非法「{at}」（只允许 numeric/boolean/enum）")
        at = "text"

    ct = str(q.get("conflict_type") or "").strip()
    if ct not in CONFLICT_TYPES:
        bad.append(f"conflict_type 非法「{ct}」")

    qs = str(q.get("question") or "").strip()
    gt = str(q.get("ground_truth") or "").strip()
    quote = str(q.get("gold_quote") or "").strip()

    if len(qs) < 8:
        bad.append("问题过短")
    if len(qs) > 60:
        bad.append(f"问题过长（{len(qs)} 字，上限 60）")
    if len(gt) < 2:
        bad.append("答案为空")
    if at == "text" and len(gt) < 5:
        bad.append("文本答案过短")

    for w in LEAK_WORDS:
        if w in qs:
            bad.append(f"问题含泄漏词「{w}」")
            break
    if YEAR_RE.search(qs):
        bad.append("问题含年份")

    core = re.sub(r"[（(].*?[）)]", "", new_policy or "")
    core = (core.replace("关于印发", "").replace("的通知", "")
                .replace("市人民政府", "").replace("人民政府", ""))
    for frag in (core[:12], core[-12:]):
        if len(frag) >= 8 and frag in qs:
            bad.append("问题含政策名片段")
            break

    if not quote:
        bad.append("缺金标证据")
    elif norm(quote) not in norm(new_text):
        bad.append("金标证据不在现行条款原文中")

    if at == "numeric":
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


def guess_mechanism(point: dict) -> str:
    """毒化机理三分：

    contradiction  A 明确规定了相反结论（A 说"必须 5 人"，B 说不要求）
    omission_a    冲突来自 A 的沉默（A 未设上限，B 设了上限）
    omission_b    冲突来自 B 的沉默（A 要求 5 人，B 未提人数）

    三者对模型的压力不同：contradiction 是"看到相反规则后选错版本"，
    omission 是"把某侧的沉默读成规定"。分析时应分开报告。
    """
    aa = str(point.get("a_answer") or "")
    bb = str(point.get("b_answer") or "")
    silence = r"未(设|规定|涉及|明确|提及|要求|列入)|没有(规定|明确|限制|上限)|不设|无(相关|上限)"
    if re.search(silence, aa):
        return "omission_a"
    if re.search(silence, bb):
        return "omission_b"
    return "contradiction"


def gen_pair_v3(client, pair, as_of: date) -> dict:
    ovt = iso_or_none(bm.clause_valid_to(pair.old)) or "无"
    nvt = iso_or_none(bm.clause_valid_to(pair.new)) or "无（现行有效）"
    user = (
        f"评估时点：{as_of.isoformat()}\n主题：{pair.topic}\n地域：{pair.region}\n\n"
        f"【条款 A · 已失效】《{pair.old.policy_name}》{pair.old.article}\n"
        f"失效日={ovt}\n{pair.old.text}\n\n"
        f"【条款 B · 现行】《{pair.new.policy_name}》{pair.new.article}\n"
        f"到期日={nvt}\n{pair.new.text}"
    )
    raw = client.chat_json(GEN3_SYSTEM, user, temperature=0.0, max_tokens=3000, thinking=False)
    return raw or {}


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description="TQ-v3 出题：分级→冲突点→按点出题")
    ap.add_argument("--as-of", default="2026-10-27")
    ap.add_argument("--out", default=str(ROOT / "data" / "national" / "benchmark_tq3.json"))
    ap.add_argument("--chroma-dir", default=str(ROOT / "chroma_lowalt"))
    ap.add_argument("--collection", default="lowalt_policy")
    ap.add_argument("--embedding-function", default="bge-zh")
    ap.add_argument("--min-similarity", type=float, default=0.12)
    ap.add_argument("--pairs-per-clause", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--report-dir", default=str(ROOT / "output"))
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
        clauses, as_of, min_similarity=args.min_similarity,
        max_pairs_per_topic=args.pairs_per_clause,
    )
    pairs = bm.diversify_pairs(pairs)
    log(f"候选对 {len(pairs)}；{json.dumps(stats, ensure_ascii=False)}")
    pairs = pairs[: args.limit_pairs] if args.limit_pairs else pairs[: args.max_pairs]

    client = provider_client(args.provider)
    log(f"模型：{client.model}")

    items: list[dict] = []
    rejected: list[dict] = []
    util = {
        "pairs_attempted": 0,
        "level": Counter(),
        "high_no_valid_point": 0,
        "points_raw": 0,
        "points_valid": 0,
        "questions_raw": 0,
        "questions_valid": 0,
        "points_per_high_pair": [],
    }
    idx = 0

    for pi, pair in enumerate(pairs, 1):
        log(f"[{pi}/{len(pairs)}] {pair.region}·{pair.topic} sim={pair.similarity:.3f}")
        try:
            raw = gen_pair_v3(client, pair, as_of)
        except LLMError as e:
            log(f"  [生成失败] {str(e)[:90]}")
            continue

        level = str(raw.get("level") or "").strip()
        level = level if level in ("High", "Medium", "Low") else "Low"
        reason = str(raw.get("level_reason") or "")[:40]
        points = raw.get("conflict_points") or []
        questions = raw.get("questions") or []
        util["pairs_attempted"] += 1
        util["level"][level] += 1
        util["points_raw"] += len(points)
        util["questions_raw"] += len(questions)

        if level != "High":
            log(f"  - 不出题：{level}（{reason}）")
            rejected.append({"region": pair.region, "level": level, "reason": reason})
            if args.sleep:
                time.sleep(args.sleep)
            continue

        # 校验冲突点
        point_ok = []
        for k, p in enumerate(points):
            ok, why = check_point(p, pair.old.text, pair.new.text)
            point_ok.append(ok)
            if ok:
                util["points_valid"] += 1
            else:
                log(f"  × 冲突点 {k} 作废（{'; '.join(why)}）")
                rejected.append({"region": pair.region, "level": level,
                                 "issue": str(p.get("issue"))[:50], "reasons": why})

        if not any(point_ok):
            util["high_no_valid_point"] += 1
            log(f"  - 判为 High 但无有效冲突点（{reason}）")
            if args.sleep:
                time.sleep(args.sleep)
            continue

        util["points_per_high_pair"].append(sum(point_ok))

        # 校验题目
        for q in questions:
            qi = q.get("point_index")
            if not isinstance(qi, int) or not (0 <= qi < len(points)) or not point_ok[qi]:
                rejected.append({"region": pair.region, "level": level,
                                 "question": str(q.get("question"))[:50],
                                 "reasons": ["指向的冲突点无效"]})
                continue
            ok, why = check_question(q, points, pair.new.text, pair.new.policy_name)
            if not ok:
                log(f"  × 题目弃用（{'; '.join(why)}）")
                rejected.append({"region": pair.region, "level": level,
                                 "question": str(q.get("question"))[:60], "reasons": why})
                continue
            if args.max_items and len(items) >= args.max_items:
                break
            point = points[qi]
            quote = str(q.get("gold_quote") or point.get("b_quote") or "")
            span = find_span(pair.new.text, quote)
            idx += 1
            util["questions_valid"] += 1
            items.append({
                "id": f"TQ3-{idx:03d}",
                "source": "TQ3",
                "region": pair.region,
                "topic": pair.topic,
                "conflict_type": str(q.get("conflict_type")),
                "question": str(q.get("question") or "").strip(),
                "ground_truth": str(q.get("ground_truth") or "").strip(),
                "answer_type": str(q.get("answer_type")),
                "gold_evidence": {
                    "chunk_id": pair.new.id,
                    "policy_name": pair.new.policy_name,
                    "article": pair.new.article,
                    "quote": quote,
                    "char_span": list(span) if span else None,
                    "source_url": "",
                    "negative": False,
                    "note": str(q.get("note") or "")[:60],
                },
                "bait_evidence": {
                    "issue": str(point.get("issue") or ""),
                    "a_answer": str(point.get("a_answer") or ""),
                    "b_answer": str(point.get("b_answer") or ""),
                    "a_quote": str(point.get("a_quote") or ""),
                    "mechanism": guess_mechanism(point),
                },
                "old_clause_id": pair.old.id,
                "new_clause_id": pair.new.id,
                "old_clause_text": pair.old.text,
                "new_clause_text": pair.new.text,
                "old_policy": pair.old.policy_name,
                "new_policy": pair.new.policy_name,
                "old_status": clause_status(pair.old),
                "new_status": clause_status(pair.new),
                "old_valid_to": iso_or_none(bm.clause_valid_to(pair.old)) or "",
                "new_valid_from": iso_or_none(bm.clause_valid_from(pair.new)) or "",
                "as_of": as_of.isoformat(),
                "level": level,
                "level_reason": reason,
            })
            log(f"  + {items[-1]['id']} [{items[-1]['conflict_type']}] "
                f"{items[-1]['question'][:30]}")
        if args.sleep:
            time.sleep(args.sleep)

    out = Path(args.out)
    bm.atomic_write_json(out, items)

    pp = util["points_per_high_pair"]
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "as_of": as_of.isoformat(),
        "version": "v3",
        "items": len(items),
        "pairs_stage": stats,
        "utilization": {
            "pairs_attempted": util["pairs_attempted"],
            "level_counts": dict(util["level"]),
            "level_high_rate": round(util["level"]["High"] / util["pairs_attempted"], 4)
            if util["pairs_attempted"] else 0.0,
            "high_pairs_without_valid_point": util["high_no_valid_point"],
            "points_raw": util["points_raw"],
            "points_valid": util["points_valid"],
            "questions_raw": util["questions_raw"],
            "questions_valid": util["questions_valid"],
            "mean_points_per_high_pair": round(sum(pp) / len(pp), 2) if pp else 0.0,
            "max_points_in_a_pair": max(pp) if pp else 0,
            "by_region": dict(Counter(i["region"] for i in items)),
            "by_type": dict(Counter(i["conflict_type"] for i in items)),
            "by_mechanism": dict(Counter(i["bait_evidence"]["mechanism"] for i in items)),
            "by_answer_type": dict(Counter(i["answer_type"] for i in items)),
        },
        "rejected": rejected[:400],
        "rejected_total": len(rejected),
    }
    bm.atomic_write_json(out.with_suffix(".meta.json"), meta)

    log("")
    log("=" * 60)
    log(f"题数 {len(items)}；尝试配对 {util['pairs_attempted']}")
    lc = dict(util["level"])
    log(f"分级：High {lc.get('High', 0)} / Medium {lc.get('Medium', 0)} / Low {lc.get('Low', 0)}")
    log(f"High 但无有效冲突点：{util['high_no_valid_point']}")
    log(f"冲突点：原始 {util['points_raw']} → 有效 {util['points_valid']}"
        f"；平均每个 High 对 {meta['utilization']['mean_points_per_high_pair']} 个")
    log(f"题目：原始 {util['questions_raw']} → 有效 {util['questions_valid']}")
    log(f"产物：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
