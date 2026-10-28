#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层 2：为"2026-12-31 到期"的政策生成答案翻转题。

这些政策在 as_of=2026-12-30 仍现行、在 2027-01-01 已失效。
同一道题在两个时点各跑一次，正确答案从"能"翻转为"不能"，
直接观测 Naive 的时间断崖与 Temporal 的时点切换。

题的结构（与 benchmark_tq3 不同——没有"新版条款"，翻转的是时间）：
  question      不含任何日期（模型不能从题面知道今天是几号）
  clause_id/text 到期条款本身（T1 时现行、T2 时失效）
  t1_answer     2026-12-30 的标准答案（肯定）
  t2_answer     2027-01-01 的标准答案（否定：有效期届满已失效）

用法：
    python tools/gen_temporal_flip.py --per-policy 1 --out data/national/temporal_flip.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from policycrawler.llm import LLMError, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402
from rag_eval.backends import build_backend  # noqa: E402
from rag_eval.temporal import clause_valid_to  # noqa: E402

GEN_SYSTEM = """你在构造"政策到期翻转"测试题。给你一条政策条款原文，它在 2026-12-31 到期失效。

请构造一个用户问题，问"该事项现在能否办理/申请"，并给出两个时点的标准答案：
- 2026-12-30（到期前一天）：该条款现行有效，答案应为肯定
- 2027-01-01（到期后）：该条款已失效，答案应为否定

只输出一个 JSON 对象：
{
  "question": "用户问题（20~50 字，含地域，不含任何日期）",
  "t1_answer": "2026-12-30 的标准答案（肯定，依据条款给出标准/条件，30~90 字）",
  "t2_answer": "2027-01-01 的标准答案（否定，说明该政策有效期至 2026-12-31 已届满失效，30~90 字）",
  "key_quote": "条款中支撑该事项的原文逐字摘录（15~150 字）"
}

硬性要求：
1. 问题所问的事项必须在条款里有明确规定（补贴/奖励/补助/资格/备案/流程）。若该条款只是宣导性表述、没有可问的具体事项，输出 {"skip": true}。
2. question 不得包含任何年份日期，不得包含政策名称、文号；用地域、主体、场景把问题问具体。
3. t1_answer 只依据条款原文，不得编造数字；数字必须出现在 key_quote 中。
4. t2_answer 的否定理由必须是"有效期届满已失效"，不得编造"被新政策替代"（你没有新政策的信息）。
5. key_quote 必须逐字摘自条款原文，不得改写拼接。"""

FLUSH_WORDS = ("补贴", "奖励", "补助", "扶持", "支持", "资助", "申请", "备案", "奖励标准", "优化", "服务")
LEAK = ("2026", "2027", "明年", "去年", "今年", "到期", "失效", "废止")


def norm(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]+", "", s or "")


def pick_clauses(clauses, per_policy: int):
    """按政策分组，每组挑最"可问"的条款：含扶持词、长度适中。"""
    groups: dict[str, list] = {}
    for c in clauses:
        v = clause_valid_to(c)
        v = str(v) if v else None
        if not v or not ("2026-10-27" < v <= "2027-01-01"):
            continue
        groups.setdefault(c.policy_name, []).append(c)
    picked = []
    for name, arr in groups.items():
        cand = [c for c in arr
                if any(w in c.text for w in FLUSH_WORDS) and 80 <= len(c.text) <= 600]
        cand.sort(key=lambda c: -len(c.text))
        picked += cand[:per_policy]
    return picked


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description="生成到期翻转题")
    ap.add_argument("--out", default=str(ROOT / "data" / "national" / "temporal_flip.json"))
    ap.add_argument("--per-policy", type=int, default=1)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--max-policies", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    backend = build_backend(argparse.Namespace(
        backend="chroma", chroma_dir=str(ROOT / "chroma_lowalt"), chroma_host="",
        chroma_port="8000", collection="lowalt_policy", embedding_function="bge-zh",
        index_dir=str(ROOT / "kb" / "index"),
    ))
    clauses = backend.all_clauses()
    picked = pick_clauses(clauses, args.per_policy)
    if args.max_policies:
        picked = picked[: args.max_policies]
    print(f"到期候选条款 {len(picked)} 条（去重后政策 {len({c.policy_name for c in picked})} 部）")

    client = provider_client(args.provider)
    items, skipped = [], []
    for i, c in enumerate(picked, 1):
        try:
            raw = client.chat_json(
                GEN_SYSTEM,
                f"【政策条款】《{c.policy_name}》{c.article}\n{c.text}",
                temperature=0.0, max_tokens=900, thinking=False,
            )
        except LLMError as e:
            print(f"  [{i}/{len(picked)}] {c.policy_name[:26]} 失败：{str(e)[:70]}")
            continue
        if raw.get("skip"):
            skipped.append(c.policy_name)
            continue
        q = str(raw.get("question") or "").strip()
        q1 = str(raw.get("t1_answer") or "").strip()
        q2 = str(raw.get("t2_answer") or "").strip()
        quote = str(raw.get("key_quote") or "").strip()
        bad = []
        if len(q) < 10 or len(q) > 60:
            bad.append("问题长度异常")
        for w in LEAK:
            if w in q:
                bad.append(f"问题含泄漏词「{w}」")
                break
        if len(quote) < 12 or norm(quote) not in norm(c.text):
            bad.append("key_quote 不在条款原文")
        for w in ("能", "可以", "按"):
            if w in q1:
                break
        else:
            bad.append("t1_answer 不是肯定表述")
        if not any(w in q2 for w in ("失效", "届满", "到期", "不再")):
            bad.append("t2_answer 未说明失效")
        for n in re.findall(r"\d+(?:\.\d+)?", q1):
            if n not in norm(quote) and len(n) >= 2:
                bad.append(f"t1 数字 {n} 不在引文中")
                break
        if bad:
            print(f"  [{i}/{len(picked)}] 弃用（{'; '.join(bad)}）")
            skipped.append(c.policy_name)
            continue
        items.append({
            "id": f"TF-{len(items) + 1:03d}",
            "region": c.region,
            "policy_name": c.policy_name,
            "policy_valid_to": str(clause_valid_to(c)),
            "question": q,
            "clause_id": c.id,
            "clause_text": c.text,
            "key_quote": quote,
            "t1_answer": q1,
            "t2_answer": q2,
        })
        print(f"  [{i}/{len(picked)}] + {items[-1]['id']} {q[:30]}")
        if args.sleep:
            import time
            time.sleep(args.sleep)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n题 {len(items)} 道 → {out}（跳过 {len(skipped)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
