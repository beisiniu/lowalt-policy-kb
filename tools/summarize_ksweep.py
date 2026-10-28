#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 K 敏感性实验：表格 + 曲线图。

读取 results/ksweep/k*.json（evaluate.py 的产物），输出：
  output/k_sweep.md                   Markdown 对照表（含指标定义与读法）
  output/figures/k_sensitivity.png    左右两图：毒化率 / 拯救率与召回率

用法：
    python tools/summarize_ksweep.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIELDS = [
    ("new_recall", "现行条款召回率", "Top-K 里召回到现行条款的题占比"),
    ("old_recall", "旧条款穿透率", "Top-K 里召回到已失效条款的题占比"),
    ("naive_poison", "Naive 毒化率", "Naive 判 POISONED 的题占比"),
    ("agent_poison", "Temporal 毒化率", "Temporal 判 POISONED 的题占比"),
    ("rescue", "严格拯救率", "Naive 毒化且 Temporal 正确的题 / Naive 毒化题数"),
    ("depoison", "毒化消除率", "Naive 毒化且 Temporal 不再毒化的题 / Naive 毒化题数"),
    ("agent_refuse", "Temporal 拒答率", "Temporal 判 REFUSED 的题占比"),
    ("gain", "加权增益", "Temporal 加权得分 − Naive 加权得分"),
]


def load_runs(folder: Path) -> list[dict]:
    runs = []
    for p in sorted(folder.glob("k*.json")):
        m = re.fullmatch(r"k(\d+)", p.stem)
        if not m:
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        mt = payload.get("metrics") or {}
        if not mt.get("n"):
            continue
        runs.append({
            "k": int(m.group(1)),
            "path": p,
            "n": mt["n"],
            "new_recall": mt.get("new_clause_penetration_rate", 0.0),
            "old_recall": mt.get("penetration_rate", 0.0),
            "naive_poison": mt.get("naive_poison_rate", 0.0),
            "agent_poison": mt.get("agent_poison_rate", 0.0),
            "naive_correct": mt.get("naive_correct_rate", 0.0),
            "agent_correct": mt.get("agent_correct_rate", 0.0),
            "rescue": mt.get("rescue_rate_strict", 0.0),
            "depoison": mt.get("de_poison_rate", 0.0),
            "agent_refuse": mt.get("agent_refusal_rate", 0.0),
            "naive_refuse": mt.get("naive_refusal_rate", 0.0),
            "gain": mt.get("weighted_gain", 0.0),
            "naive_weighted": mt.get("naive_weighted_score", 0.0),
            "agent_weighted": mt.get("agent_weighted_score", 0.0),
            "cond_poison_naive": mt.get("naive_conditionally_poisoned_rate", 0.0),
            "cond_poison_agent": mt.get("agent_conditionally_poisoned_rate", 0.0),
        })
    runs.sort(key=lambda r: r["k"])
    return runs


def write_table(runs: list[dict], out: Path) -> None:
    k_list = [r["k"] for r in runs]
    lines = [
        "# K 敏感性实验（Temporal 过滤的 top-k 影响）",
        "",
        f"题库：`data/national/benchmark_tq3_final.json` ｜ 样本数 n={runs[0]['n'] if runs else 0}"
        f" ｜ 检索：Chroma + BGE-zh（Naive 与 Temporal 共用同一次检索）",
        f" ｜ 判分：单裁判 DeepSeek（主表 k=9 为三裁判，趋势判定用单裁判以控成本）",
        f" ｜ as_of=2026-10-27",
        "",
        "## 指标对照表",
        "",
        "| k | " + " | ".join(f[1] for f in FIELDS) + " |",
        "|---|" + "---|" * len(FIELDS),
    ]
    for r in runs:
        cells = []
        for key, _, _ in FIELDS:
            v = r[key]
            cells.append(f"{v:+.3f}" if key == "gain" else f"{v * 100:.1f}%")
        lines.append(f"| **{r['k']}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 指标定义",
        "",
        "| 指标 | 定义 |",
        "|---|---|",
    ]
    for _, name, desc in FIELDS:
        lines.append(f"| {name} | {desc} |")

    # k 的选择：判据只用检索层指标（确定性、可复现），不用准确率（受判分噪声影响）
    rec = [r for r in runs if r["new_recall"] >= 0.90 and r["old_recall"] >= 0.90]
    lines += ["", "## k 的选择", ""]
    if rec:
        best = min(rec, key=lambda r: r["k"])
        lines += [
            f"**判据**：现行条款召回率 ≥ 90% 且旧条款穿透率 ≥ 90%，取满足条件的**最小 k**。",
            "",
            f"**结果：k = {best['k']}**（召回 {best['new_recall']:.1%} / 穿透 {best['old_recall']:.1%}）。"
            f"最小 k 优先，因为上下文更短、推理成本更低，且更接近真实部署配置。",
            "",
            "为什么不用准确率挑 k：准确率受裁判噪声影响，在同一测试集上按准确率挑 k 等于在测试集上调超参数，"
            "会高估效果。检索层指标（召回率/穿透率）只由向量检索决定，同一 k 下完全可复现，适合作为选择依据；"
            "准确率只用于**验证**选定 k 的表现。",
            "",
        ]
        if len(rec) > 1:
            tail = "、".join(f"k={r['k']}" for r in rec[:8])
            lines.append(f"满足条件的 k 共 {len(rec)} 个：{tail}{'…' if len(rec) > 8 else ''}。")
    else:
        lines.append("没有 k 同时满足「现行召回 ≥90% 且旧穿透 ≥90%」，需要放宽阈值或改进检索。")

    lines += [
        "",
        "## 读法",
        "",
        "- **旧条款穿透率**随 k 单调上升是必然的（候选越多越容易包含旧条款）；关键不是它，"
        "而是它上升的同时 **Naive 毒化率**是否也跟着上升。",
        "- **现行条款召回率**是 Temporal 分支的弹药量：k 太小时过滤完手里没牌，只能拒答；"
        "它上来之后，严格拯救率才有上升空间。",
        "- **严格拯救率**（毒化→正确）与 **毒化消除率**（毒化→不再毒化）的差额，"
        "就是「过滤了但答不出」的部分，反映检索层的天花板。",
        "- 若 **Temporal 毒化率**在某个 k 之后重新抬头，说明候选过多引入了新的过期条款，"
        "该 k 即为上限。",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"表 → {out}")


def draw(runs: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ks = [r["k"] for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300)

    ax = axes[0]
    ax.plot(ks, [r["naive_poison"] * 100 for r in runs], "o-", color="#C0504D",
            linewidth=2, markersize=6, label="Naive 毒化率")
    ax.plot(ks, [r["agent_poison"] * 100 for r in runs], "s-", color="#2F6FB2",
            linewidth=2, markersize=6, label="Temporal 毒化率（全为 0）")
    ax.plot(ks, [r["old_recall"] * 100 for r in runs], "^--", color="#9E9E9E",
            linewidth=1.2, markersize=4, label="旧条款穿透率")
    ax.set_xlabel("top-k", fontsize=11)
    ax.set_ylabel("比例（%）", fontsize=11)
    ax.set_title("毒化率随 k 的变化", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 21))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, loc="center right")

    ax = axes[1]
    ax.plot(ks, [r["new_recall"] * 100 for r in runs], "o-", color="#4E7A3A",
            linewidth=2, markersize=5, label="现行条款召回率")
    ax.plot(ks, [r["rescue"] * 100 for r in runs], "s-", color="#B8860B",
            linewidth=2, markersize=5, label="严格拯救率")
    ax.plot(ks, [r["depoison"] * 100 for r in runs], "^--", color="#6A5ACD",
            linewidth=1.4, markersize=4, label="毒化消除率")
    ax.plot(ks, [r["agent_refuse"] * 100 for r in runs], "v:", color="#808080",
            linewidth=1.2, markersize=4, label="Temporal 拒答率")
    ax.set_xlabel("top-k", fontsize=11)
    ax.set_ylabel("比例（%）", fontsize=11)
    ax.set_title("Temporal 分支：现行条款召回率与拯救效果", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 21))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, loc="center right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"图 → {out}")


def main() -> int:
    folder = ROOT / "results" / "ksweep"
    runs = load_runs(folder)
    if not runs:
        print("没有找到 results/ksweep/k*.json")
        return 1
    print(f"读到 {len(runs)} 个 k：{[r['k'] for r in runs]}")
    write_table(runs, ROOT / "output" / "k_sweep.md")
    draw(runs, ROOT / "output" / "figures" / "k_sensitivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
