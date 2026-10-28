"""规则判分器：用可抽取的数值/日期集合，替代 LLM 裁判判「是否照抄失效条款」。

核心思想：
  旧条款独有值 O_only = values(old) - values(new)
  新条款值     N      = values(new)
  答案中的值   A
    A 命中 O_only 且不命中 N  → POISONED（照抄了旧值）
    A 命中 N 且不命中 O_only  → CORRECT（用了现行值）
    其余（无值/无差异）        → ABSTAIN（交给 LLM 裁判）

用法：python tools/auto_score_rules.py --benchmark ... --results ... [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

UNIT = {
    "亿元": 1e8, "万元": 1e4, "元": 1.0, "%": 1.0, "％": 1.0,
    "年": 1.0, "个月": 1.0, "月": 1.0, "个": 1.0, "架": 1.0,
    "平方米": 1.0, "米": 1.0, "天": 1.0, "小时": 1.0, "倍": 1.0, "人": 1.0,
}
PAT = re.compile(r"(\d+(?:\.\d+)?)\s*(亿元|万元|元|%|％|个月|平方米|年|月|个|架|米|天|小时|倍|人)")


def values(text: str) -> set[tuple[float, str]]:
    """抽取 (归一化数值, 单位类别) 集合；金额统一到元，比例统一到 %。"""
    out = set()
    for num, unit in PAT.findall(text or ""):
        u = "％" if unit == "%" else unit
        try:
            v = float(num)
        except ValueError:
            continue
        if u in ("亿元", "万元", "元"):
            out.add((round(v * UNIT[u], 2), "money"))
        elif u in ("%", "％"):
            out.add((round(v, 4), "pct"))
        elif u in ("年",):
            out.add((round(v, 4), "year"))
        else:
            out.add((round(v, 4), "count"))
    return out


def score_answer(answer: str, old_text: str, new_text: str) -> dict:
    O, N = values(old_text), values(new_text)
    A = values(answer)
    old_only = O - N
    if not A:
        return {"label": "ABSTAIN", "reason": "答案无可抽取数值"}
    if not old_only and not N:
        return {"label": "ABSTAIN", "reason": "新旧条款均无可抽取数值/无差异"}
    hit_old = A & old_only
    hit_new = A & N
    if hit_old and not hit_new:
        return {"label": "POISONED", "reason": f"命中旧独有值 {sorted(hit_old)[:3]}"}
    if hit_new and not hit_old:
        return {"label": "CORRECT", "reason": f"命中现行值 {sorted(hit_new)[:3]}"}
    if hit_old and hit_new:
        return {"label": "ABSTAIN", "reason": "同时命中新旧值，无法裁定"}
    return {"label": "ABSTAIN", "reason": "未命中任何对照值"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/national/benchmark_v9.json")
    ap.add_argument("--results", default="results/temporal_eval_v9.json")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    bm = {x["id"]: x for x in json.loads(Path(args.benchmark).read_text(encoding="utf-8"))}
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))["items"]

    rows = []
    for r in res:
        b = bm.get(r["id"], {})
        old_t, new_t = b.get("old_clause_text") or "", b.get("new_clause_text") or ""
        if not values(old_t):
            continue  # 该题无自动判分基础
        for side in ("naive", "agent"):
            ans = r.get(f"{side}_answer") or ""
            pred = score_answer(ans, old_t, new_t)
            gold = (r.get(f"{side}_judge") or {}).get("status", "")
            rows.append({"id": r["id"], "side": side, "rule": pred["label"], "llm": gold,
                         "reason": pred["reason"]})

    judged = [x for x in rows if x["rule"] != "ABSTAIN"]
    agree = [x for x in judged if x["rule"] == x["llm"]]
    print(f"可进入规则判分的题数: {len({x['id'] for x in rows})} / {len(res)}")
    print(f"规则给出结论的判定数: {len(judged)}（{len(judged)/max(1,len(rows)):.1%}）")
    print(f"规则 vs LLM 一致: {len(agree)} / {len(judged)} = {len(agree)/max(1,len(judged)):.1%}")
    print("规则标签分布:", dict(Counter(x["rule"] for x in rows)))
    print("不一致样本:")
    for x in [y for y in judged if y["rule"] != y["llm"]][:8]:
        print(f"  {x['id']} [{x['side']}] rule={x['rule']} llm={x['llm']} | {x['reason'][:50]}")

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"rows": rows, "summary": {"judged": len(judged), "agree": len(agree)}},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print("报告已写入:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
