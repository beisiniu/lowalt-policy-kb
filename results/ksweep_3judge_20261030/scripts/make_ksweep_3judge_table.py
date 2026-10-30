"""从 K 扫描三裁判复核结果生成论文表与图表数据。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "rejudge/ksweep/merged_summary.json").read_text(encoding="utf-8"))
OUT_MD = ROOT / "evidence" / "ksweep_3judge_table.md"
OUT_JSON = ROOT / "data" / "ksweep_3judge.json"
KEY_KS = (1, 5, 9, 10, 12, 17, 20)


def by_k():
    d = {}
    for g in S["groups"]:
        d.setdefault(g["k"], {})[g["branch"]] = g
    return d


def row(k, g):
    n = g["naive"]["majority"]
    t = g["temporal"]["majority"]
    gain = round(t["weighted_score"] - n["weighted_score"], 3)
    return {
        "k": k,
        "recall": round(g["temporal"]["new_recall"] / 31, 4),
        "penetration": round(g["temporal"]["old_penetration"] / 31, 4),
        "naive_poison": n["POISONED"],
        "naive_correct": n["CORRECT"],
        "naive_refused": n["REFUSED"],
        "naive_other": n["OTHER_ERROR"],
        "temporal_poison": t["POISONED"],
        "temporal_correct": t["CORRECT"],
        "temporal_refused": t["REFUSED"],
        "temporal_other": t["OTHER_ERROR"],
        "strict_rescue": g["temporal"].get("strict_rescue_majority"),
        "weighted_gain": gain,
        "temporal_weighted": t["weighted_score"],
        "naive_weighted": n["weighted_score"],
    }


def main():
    d = by_k()
    rows = [row(k, d[k]) for k in range(1, 21)]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"meta": {k: S.get(k) for k in
                              ("merged", "unresolved", "no_majority", "human_reviewed", "human_overrides",
                               "ds_qwen_agreement", "ds_qwen_kappa", "glm_used",
                               "glm_pair_agreement", "glm_pair_kappa")}, "rows": rows},
                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# K 扫描三裁判复核（Qwen 全量 + GLM 分歧）", "",
             f"合并 {S['merged']} 条，未决 {S['unresolved']} 条；DS/Qwen 一致率 {S['ds_qwen_agreement']}，"
             f"κ={S['ds_qwen_kappa']}；GLM 用于 {S['glm_used']} 条分歧（成对一致率 {S['glm_pair_agreement']}、κ={S['glm_pair_kappa']}）；无多数 {S['no_majority']} 条，人工复核 {S.get('human_reviewed',0)} 条、改判 {S.get('human_overrides',0)} 条。",
             "", "## 关键 K 点", "",
             "| K | 现行召回 | 旧穿透 | Naive 毒化 | Temporal 毒化 | Temporal 正确 | Temporal 拒答 | 严格拯救 | 加权增益 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["k"] in KEY_KS:
            rescue = "—" if r["strict_rescue"] is None else f"{r['strict_rescue']:.1%}"
            lines.append(f"| {r['k']} | {r['recall']:.1%} | {r['penetration']:.1%} | {r['naive_poison']}/31 | "
                         f"{r['temporal_poison']}/31 | {r['temporal_correct']}/31 | {r['temporal_refused']}/31 | "
                         f"{rescue} | {r['weighted_gain']:+.3f} |")
    lines += ["", "## 全表", "",
              "| K | 现行召回 | 旧穿透 | Naive 毒化 | Temporal 毒化 | Temporal 正确 | Temporal 拒答 | 严格拯救 | 加权增益 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        rescue = "—" if r["strict_rescue"] is None else f"{r['strict_rescue']:.1%}"
        lines.append(f"| {r['k']} | {r['recall']:.1%} | {r['penetration']:.1%} | {r['naive_poison']}/31 | "
                     f"{r['temporal_poison']}/31 | {r['temporal_correct']}/31 | {r['temporal_refused']}/31 | "
                     f"{rescue} | {r['weighted_gain']:+.3f} |")
    lines += ["", "> K 的选择仍只用检索层指标（现行召回/旧穿透），毒化/正确/拒答列为三裁判复核后的趋势记录。",
              "> K 扫描的 11 条无多数票已由作者人工复核（7 条改判、4 条维持）；它仍只用于趋势与部署折中，不是最终主表。"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
