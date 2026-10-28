"""把候选结果按可信度分层，输出可核对的候选表。

分层口径（严格版）：
  A 可直接采纳：证据逐字可核 + 置信 high + 日期与文件名/证据中的年份区间末年和 12-31 对齐，
                或正文明确写"有效期/至YYYY年"。
  B 需抽检：给了日期但未通过上述全部条件（含"到2030年"这类目标年被误当规划期末的情况）。
  C 不给日期：open_ended / unclear / repealed 无日期。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROWS = ROOT / "output" / "validity_candidates_report.json"
OUT_MD = ROOT / "output" / "validity_candidates_table.md"

RANGE = re.compile(r"(20\d{2})\s*[—\-－~至到]\s*(20\d{2})")
EXPL = re.compile(r"有效期|至\s*20\d{2}\s*年")
TARGET_ONLY = re.compile(r"纲要|指南|公告|意见|标准|通知")


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    A, B, C = [], [], []
    for r in rows:
        d = r.get("inferred_expiry")
        if not d:
            C.append(r)
            continue
        txt = r["policy_name"] + " " + (r.get("evidence") or "")
        ok = False
        if r["kind"] == "plan_period":
            m = RANGE.search(txt)
            ok = bool(m) and str(d)[:4] == m.group(2) and str(d)[5:] == "12-31"
        elif r["kind"] == "explicit_expiry":
            ok = bool(EXPL.search(txt))
        if ok and r["evidence_verified"] and r["confidence"] == "high":
            A.append(r)
        else:
            B.append(r)

    dup = Counter(r["policy_name"] for r in A)
    dup_names = {k: v for k, v in dup.items() if v > 1}
    target_flag = [r for r in A + B if TARGET_ONLY.search(r["policy_name"]) and r["kind"] == "plan_period"]

    print(f"合计 {len(rows)} 条")
    print(f"  A 可直接采纳 {len(A)} 条（其中重复政策 {len(dup_names)} 组，去重后 {len(A)-sum(v-1 for v in dup_names.values())} 份文件）")
    print(f"  B 需抽检     {len(B)} 条")
    print(f"  C 不给日期   {len(C)} 条  {dict(Counter(r['kind'] for r in C))}")
    print(f"  提示：A/B 中含 {len(target_flag)} 条属纲要/指南/意见/公告，其年份是目标年而非规划期末，建议剔除")

    lines = ["# 失效日候选表（大模型判断，2026-10-27）", "",
             f"来源：`tools/infer_validity.py` 对 83 份无失效日文件的逐份判断；本表由 `tools/tier_validity_candidates.py` 分层。", "",
             f"- A 级（可直接采纳）：{len(A)} 条",
             f"- B 级（需抽检）：{len(B)} 条",
             f"- C 级（不给日期）：{len(C)} 条", "",
             "## A 级", "",
             "| 政策 | 地区 | 候选失效日 | 判定 | 证据 |",
             "| --- | --- | --- | --- | --- |"]
    for r in sorted(A, key=lambda x: (x["inferred_expiry"], x["policy_name"])):
        lines.append(f"| {r['policy_name'][:32]} | {r['region']} | {r['inferred_expiry']} | {r['kind']} | {(r.get('evidence') or '')[:34]} |")
    lines += ["", "## B 级（需抽检）", "", "| 政策 | 地区 | 候选失效日 | 判定 | 置信 | 未通过原因 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in sorted(B, key=lambda x: (x["policy_name"])):
        why = []
        if not r["evidence_verified"]:
            why.append("证据不可逐字核")
        if r["confidence"] != "high":
            why.append("置信非 high")
        if r["kind"] == "plan_period" and not RANGE.search(r["policy_name"] + " " + (r.get("evidence") or "")):
            why.append("未见区间表述")
        if TARGET_ONLY.search(r["policy_name"]) and r["kind"] == "plan_period":
            why.append("疑为目标年")
        lines.append(f"| {r['policy_name'][:32]} | {r['region']} | {r.get('inferred_expiry')} | {r['kind']} | {r['confidence']} | {'、'.join(why) or '—'} |")
    lines += ["", "## C 级（不给日期，保持长期有效或无依据）", "", "| 政策 | 地区 | 判定 | 置信 | 依据 |", "| --- | --- | --- | --- | --- |"]
    for r in sorted(C, key=lambda x: (x["kind"], x["policy_name"])):
        lines.append(f"| {r['policy_name'][:32]} | {r['region']} | {r['kind']} | {r['confidence']} | {(r.get('reasoning') or '')[:38]} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("候选表写入:", OUT_MD)


if __name__ == "__main__":
    main()
