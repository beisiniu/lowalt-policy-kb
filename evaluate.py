#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真实向量检索 + Naive RAG / Temporal LangGraph 双分支评估。

实验设计（只改变一个变量）：

    question
       │
       └─ similarity_search(k=5)   ← 只检索一次
             │
             ├─ Naive RAG：Top-5 原样拼接 → LLM
             └─ Temporal LangGraph：同一 Top-5 作为初始 State
                    → temporal_check（按 valid_from/valid_to 过滤）
                    → generate / refuse

Naive 和 Temporal 使用**同一个模型、同一个 temperature、同一个回答 Prompt**。
唯一差异是 Temporal_Check 是否在生成前剔除过期条款，因此可以把差异归因到架构。

指标：
  过期政策穿透率 = Top-5 含旧条款的题数 / 总题数
  Naive 毒化率   = Naive 答案为 POISONED / 总题数
  Agent 毒化率   = Temporal 答案为 POISONED / 总题数
  严格拯救率     = Naive POISONED 且 Temporal CORRECT / Naive POISONED
  毒化消除率     = Naive POISONED 且 Temporal 不再 POISONED / Naive POISONED

为什么额外有 REFUSED / OTHER_ERROR：
  裁判若被迫二分类，Temporal 分支把全部旧条款过滤后拒答，会被判成 CORRECT 而虚高；
  普通答错也会被塞进 CORRECT。因此保留四类，毒化率只统计 POISONED：
    POISONED      错误且由旧政策/旧标准导致
    CORRECT       核心事实与标准答案一致
    REFUSED       拒答，或明确说无法确定
    OTHER_ERROR   答错但不是旧政策导致

用法：
    python evaluate.py \
        --chroma-dir ./chroma_db --collection lowalt_policy \
        --benchmark benchmark.json --top-k 5 \
        --out results/temporal_poisoning_eval.json \
        --figure output/figures/temporal_poisoning_compare.png

    # 当前仓库没有 Chroma 时，用本地索引跑通同一套逻辑
    python evaluate.py --backend local --index-dir kb/index --benchmark benchmark.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.llm import ChatClient, LLMError, provider_client
from policycrawler.utils import load_env_file
from rag_eval.backends import add_backend_args, build_backend
from rag_eval.models import Clause
from rag_eval.temporal import (
    clause_matches,
    clause_valid_from,
    clause_valid_to,
    iso_or_none,
)
from rag_eval.temporal_graph import TemporalRAGGraph, generate_answer

ROOT = Path(__file__).resolve().parent
STATUSES = ("POISONED", "CORRECT", "REFUSED", "OTHER_ERROR")
_STATUS_ALIASES = {
    "毒化": "POISONED", "中毒": "POISONED", "被毒化": "POISONED",
    "正确": "CORRECT", "答对": "CORRECT",
    "拒答": "REFUSED", "无法确定": "REFUSED",
    "其他错误": "OTHER_ERROR", "其他": "OTHER_ERROR", "其他错误（非旧政策）": "OTHER_ERROR",
}


def _normalize_status(v: str) -> str:
    """裁判输出可能是英文枚举、中文判定或带引号/空白，统一成 STATUSES。"""
    t = str(v or "").strip().strip("\'\u201c\u201d").upper()
    if t in STATUSES:
        return t
    raw = str(v or "").strip()
    for zh, en in _STATUS_ALIASES.items():
        if zh in raw:
            return en
    return t
STATUS_CN = {
    "POISONED": "旧政策毒化",
    "CORRECT": "正确",
    "REFUSED": "拒答",
    "OTHER_ERROR": "其他错误",
}

JUDGE_SYSTEM = """你是时序 RAG 的评估专家。请判断【模型回答】是否被过期政策毒化。

判定必须优先看“核心结论/核心数值”：
- POISONED：核心结论或关键数值错误，而且错误来自使用了旧条款、旧版标准或已失效的适用范围。
  即使资料里同时存在正确的新条款，只要模型最终采纳了旧条款，也判 POISONED。
- CORRECT：核心结论和关键事实与【标准答案】一致。
  措辞更简略、额外引用旧政策但没有影响最终结论，也算 CORRECT。
- REFUSED：明确拒答、说资料不足/无法确定，没有给出可判错的结论。
- OTHER_ERROR：给出了错误结论，但错误原因不是旧政策（如计算错误、答非所问、混用其他地区数值、凭空编造）。

注意：
1. “引用旧政策”本身不等于 POISONED；必须导致最终答案错误。
2. 拒答不能算 CORRECT，否则时序过滤后的安全拒答会虚高正确率。
3. 只输出 JSON：
{"status": "POISONED|CORRECT|REFUSED|OTHER_ERROR", "uses_old_clause": true/false, "reason": "不超过 60 字"}
"""


def log(msg: str) -> None:
    print(msg, file=sys.stderr if msg.startswith("  ") else sys.stdout, flush=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} 中没有 benchmark 题目")
    required = {"topic", "region", "question", "ground_truth", "old_clause_text", "new_clause_text"}
    for i, item in enumerate(items, 1):
        missing = required - set(item)
        if missing:
            raise ValueError(f"第 {i} 题缺少字段：{sorted(missing)}")
        item.setdefault("id", f"TQ-{i:04d}")
    return items


def benchmark_fingerprint(items: list[dict[str, Any]]) -> str:
    canonical = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _majority(votes: list[dict]) -> dict:
    if not votes:
        return {"status": "OTHER_ERROR", "reason": "无裁判结果", "agreement": 0.0}
    counts = Counter(v["status"] for v in votes)
    # 平票时优先更严格/更保守的判定，避免平均后虚高
    priority = {"POISONED": 3, "OTHER_ERROR": 2, "REFUSED": 1, "CORRECT": 0}
    best = max(counts, key=lambda s: (counts[s], priority.get(s, 0)))
    return {
        "status": best,
        "reason": next(v["reason"] for v in votes if v["status"] == best),
        "uses_old_clause": any(v.get("uses_old_clause") for v in votes if v["status"] == best),
        "agreement": round(counts[best] / len(votes), 3),
    }


def judge_answer(
    client: ChatClient,
    item: dict[str, Any],
    answer: str,
    *,
    repeat: int = 1,
) -> dict[str, Any]:
    user = (
        f"【问题】\n{item['question']}\n\n"
        f"【标准答案】\n{item['ground_truth']}\n\n"
        f"【旧条款（已失效）】\n{item['old_clause_text'][:1600]}\n\n"
        f"【现行条款】\n{item['new_clause_text'][:1600]}\n\n"
        f"【模型回答】\n{answer[:2200]}"
    )
    votes: list[dict] = []
    for _ in range(max(1, repeat)):
        # 偶发空 status（如 GLM 思考模式吞掉正文）→ 重试而非中止；
        # 重试仍失败才上抛，保留"不把 API 错误伪装成答错"的原则
        raw: dict | None = None
        last_err = ""
        for attempt in range(3):
            try:
                raw = client.chat_json(
                    JUDGE_SYSTEM, user, temperature=0.0, max_tokens=400, thinking=False
                )
                status = _normalize_status((raw or {}).get("status"))
                if status in STATUSES:
                    break
                last_err = f"非法 status：{status!r}"
                raw = None
            except LLMError as e:
                last_err = str(e)[:120]
                raw = None
        if raw is None:
            raise LLMError(f"裁判连续 3 次失败：{last_err}")
        votes.append({
            "status": _normalize_status(raw["status"]),
            "reason": " ".join(str(raw.get("reason") or "").split())[:200],
            "uses_old_clause": bool(raw.get("uses_old_clause")),
        })
    return _majority(votes)


def naive_answer(client: ChatClient, item: dict[str, Any], top5: list[Clause],
                 *, thinking: bool) -> str:
    """对照组：Top-5 原样拼接，不做任何时效过滤。"""
    return generate_answer(client, item["question"], top5, thinking=thinking)


def _clause_dump(c: Clause) -> dict[str, Any]:
    return {
        "id": c.id,
        "policy_name": c.policy_name,
        "article": c.article,
        "region": c.region,
        "score": c.score,
        "valid_from": iso_or_none(clause_valid_from(c)),
        "valid_to": iso_or_none(clause_valid_to(c)),
        "text": c.text,
    }


def majority_vote(js: list[dict]) -> dict:
    """三裁判多数票：>=半数即胜出；无多数取主裁判并打 no_majority 标记。"""
    js = [j for j in js if j and j.get("status") in STATUSES]
    if not js:
        return {"status": "OTHER_ERROR", "reason": "全部裁判失败"}
    from collections import Counter
    c = Counter(j["status"] for j in js)
    top, cnt = c.most_common(1)[0]
    win = next(j for j in js if j["status"] == top)
    out = dict(win)
    out["vote"] = dict(c)
    out["judges"] = [j["status"] for j in js]
    if cnt * 2 <= len(js):
        out["no_majority"] = True
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """多分类 Cohen's κ：衡量两个裁判的一致性是否超过随机。"""
    n = len(pairs)
    if not n:
        return None
    labels = {a for a, _ in pairs} | {b for _, b in pairs}
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for a, _ in pairs if a == lab) / n
        pb = sum(1 for _, b in pairs if b == lab) / n
        pe += pa * pb
    if pe >= 1:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"n": 0}

    penetration = sum(1 for r in records if r["old_clause_penetrated"])
    new_penetration = sum(1 for r in records if r.get("new_clause_penetrated"))
    old_filtered = sum(1 for r in records if r.get("old_clause_filtered"))
    naive_poisoned = sum(1 for r in records if r["naive_judge"]["status"] == "POISONED")
    agent_poisoned = sum(1 for r in records if r["agent_judge"]["status"] == "POISONED")
    naive_correct = sum(1 for r in records if r["naive_judge"]["status"] == "CORRECT")
    agent_correct = sum(1 for r in records if r["agent_judge"]["status"] == "CORRECT")
    naive_refused = sum(1 for r in records if r["naive_judge"]["status"] == "REFUSED")
    agent_refused = sum(1 for r in records if r["agent_judge"]["status"] == "REFUSED")

    jpairs: list[tuple[str, str]] = []
    for r in records:
        for a, b in ((r.get("naive_judge"), r.get("naive_judge2")),
                     (r.get("agent_judge"), r.get("agent_judge2"))):
            if a and b and a.get("status") in STATUSES and b.get("status") in STATUSES:
                jpairs.append((a["status"], b["status"]))
    judge_agree = round(sum(1 for x, y in jpairs if x == y) / len(jpairs), 4) if jpairs else None
    judge_kappa = cohen_kappa(jpairs) if jpairs else None

    # 条件毒化率的正确分子：必须同时"穿透"且"毒化"，与分母同一集合
    naive_pen_poisoned = sum(
        1 for r in records
        if r["old_clause_penetrated"] and r["naive_judge"]["status"] == "POISONED")
    agent_pen_poisoned = sum(
        1 for r in records
        if r["old_clause_penetrated"] and r["agent_judge"]["status"] == "POISONED")

    # 毒化加权得分（CRAG 式分档惩罚）：正确 +1，拒答 +0.2，其他错误 0，毒化 -1
    # 理由：毒化给出的是"已废止的错误依据"，对用户的危害超过答对带来的收益，
    # 因此权重取与正确答案相同的绝对值、方向相反；拒答安全但无帮助，给部分分。
    W = {"CORRECT": 1.0, "REFUSED": 0.2, "OTHER_ERROR": 0.0, "POISONED": -1.0}
    def weighted(side: str) -> float:
        tot = sum(W.get(r[f"{side}_judge"]["status"], 0.0) for r in records)
        return round(tot / n, 4)

    rescued_correct = sum(
        1 for r in records
        if r["naive_judge"]["status"] == "POISONED" and r["agent_judge"]["status"] == "CORRECT"
    )
    de_poisoned = sum(
        1 for r in records
        if r["naive_judge"]["status"] == "POISONED" and r["agent_judge"]["status"] != "POISONED"
    )

    return {
        "n": n,
        "penetration_count": penetration,
        "penetration_rate": round(penetration / n, 4),
        "new_clause_penetration_count": new_penetration,
        "new_clause_penetration_rate": round(new_penetration / n, 4),
        "old_clause_filtered_count": old_filtered,
        "old_clause_filter_rate": round(old_filtered / penetration, 4) if penetration else 0.0,
        "naive_poisoned": naive_poisoned,
        "agent_poisoned": agent_poisoned,
        "naive_poison_rate": round(naive_poisoned / n, 4),
        "agent_poison_rate": round(agent_poisoned / n, 4),
        "naive_correct": naive_correct,
        "agent_correct": agent_correct,
        "naive_correct_rate": round(naive_correct / n, 4),
        "agent_correct_rate": round(agent_correct / n, 4),
        "naive_refused": naive_refused,
        "agent_refused": agent_refused,
        "naive_refusal_rate": round(naive_refused / n, 4),
        "agent_refusal_rate": round(agent_refused / n, 4),
        # 条件毒化率：在旧条款确实被召回的前提下，有多少被模型采纳（分子分母同集合）
        "naive_conditionally_poisoned_count": naive_pen_poisoned,
        "agent_conditionally_poisoned_count": agent_pen_poisoned,
        "naive_conditionally_poisoned_rate": round(naive_pen_poisoned / penetration, 4) if penetration else 0.0,
        "agent_conditionally_poisoned_rate": round(agent_pen_poisoned / penetration, 4) if penetration else 0.0,
        # 四分类计数 + 毒化加权得分（毒化扣分等于正确加分，拒答给 0.2）
        "naive_other_error": sum(1 for r in records if r["naive_judge"]["status"] == "OTHER_ERROR"),
        "agent_other_error": sum(1 for r in records if r["agent_judge"]["status"] == "OTHER_ERROR"),
        "naive_weighted_score": weighted("naive"),
        "agent_weighted_score": weighted("agent"),
        "weighted_gain": round(weighted("agent") - weighted("naive"), 4),
        "rescued_to_correct": rescued_correct,
        "de_poisoned": de_poisoned,
        "rescue_rate_strict": round(rescued_correct / naive_poisoned, 4) if naive_poisoned else 0.0,
        "de_poison_rate": round(de_poisoned / naive_poisoned, 4) if naive_poisoned else 0.0,
        "absolute_poison_reduction": round((naive_poisoned - agent_poisoned) / n, 4),
        "judge_pairs": len(jpairs),
        "judge_agreement": judge_agree,
        "cohen_kappa": judge_kappa,
    }


def setup_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["font.size"] = 11
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "PingFang SC", "Heiti SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    else:
        print("[WARN] 没有找到中文字体，图中中文可能显示为方块")
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_metrics(metrics: dict[str, Any], out: Path, *, n: int) -> Path:
    import matplotlib.pyplot as plt

    setup_font()
    labels = ["Naive RAG", "Temporal LangGraph"]
    values = [metrics["naive_poison_rate"] * 100, metrics["agent_poison_rate"] * 100]
    colors = ["#C0504D", "#2F6FB2"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    for rect, v, color in zip(bars, values, colors):
        ax.annotate(
            f"{v:.1f}%", (rect.get_x() + rect.get_width() / 2, v),
            ha="center", va="bottom", fontsize=12, fontweight="bold", xytext=(0, 4),
            textcoords="offset points",
        )
        if v > 0:
            ax.annotate(
                f"{int(round(v * n / 100))}/{n}",
                (rect.get_x() + rect.get_width() / 2, v / 2),
                ha="center", va="center", fontsize=9, color="white",
            )
        else:
            # 零值柱在白色背景上看不见，画一条基线粗线表示“确实测了，结果为 0”
            cx = rect.get_x() + rect.get_width() / 2
            ax.plot([cx - 0.2, cx + 0.2], [0.8, 0.8], color=color, linewidth=4,
                    solid_capstyle="butt")
    ax.set_ylim(0, 100)
    ax.set_ylabel("毒化率（%）")
    ax.set_title(f"时序过滤对过期政策毒化的防御效果（N={n}）", fontsize=12.5, pad=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(
        0.01, 0.01,
        f"旧条款穿透率 {metrics['penetration_rate']:.1%}；"
        f"拦截率 {metrics['old_clause_filter_rate']:.1%}；"
        f"严格拯救率 {metrics['rescue_rate_strict']:.1%}；"
        f"毒化消除率 {metrics['de_poison_rate']:.1%}",
        fontsize=8, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="时序 RAG 双分支毒化评估")
    add_backend_args(ap)
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark.json"))
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "temporal_poisoning_eval.json"))
    ap.add_argument("--figure", default=str(ROOT / "output" / "figures" / "temporal_poisoning_compare.png"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试/控成本）")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true", help="复用同一 benchmark/配置下的已完成题目")
    ap.add_argument("--plot-only", action="store_true", help="不调用任何 API，只从 --out 重绘图表")
    ap.add_argument("--judge-repeat", type=int, default=1)
    ap.add_argument("--answer-provider", default="deepseek", choices=["deepseek", "qwen", "glm"],
                    help="答题模型供应商（换底座做稳健性检验）")
    ap.add_argument("--judge-provider", default="", choices=["", "deepseek", "qwen", "glm"],
                    help="主裁判供应商，默认与答题模型相同")
    ap.add_argument("--judge2-provider", default="", choices=["", "deepseek", "qwen", "glm"],
                    help="第二裁判供应商：填了就双裁判并计算一致性/Cohen κ")
    ap.add_argument("--judge3-provider", default="", choices=["", "deepseek", "qwen", "glm"],
                    help="第三裁判供应商：三裁判时按多数票合成最终判决")
    ap.add_argument("--agent-mode", default="filter", choices=["filter", "downweight"],
                    help="消融实验：downweight=旧条款不剔除只降权，验证毒化防御来源")
    ap.add_argument("--thinking", action="store_true", help="回答生成开启 DeepSeek thinking；两组必须一致")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-model", default=None)
    return ap.parse_args()


def main() -> int:
    load_env_file(ROOT / ".env")
    args = parse_args()
    as_of = date.fromisoformat(args.as_of)
    bench_path = Path(args.benchmark)
    out_path = Path(args.out)

    if args.plot_only:
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            metrics = old.get("metrics") or compute_metrics(old.get("items", []))
        except (OSError, json.JSONDecodeError) as e:
            log(f"[ERROR] 读取评估结果失败：{e}")
            return 2
        fig = plot_metrics(metrics, Path(args.figure), n=metrics["n"])
        print(f"已重绘 → {fig}")
        return 0

    try:
        items = load_benchmark(bench_path)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log(f"[ERROR] 读取 benchmark 失败：{e}")
        return 2
    if args.limit:
        items = items[: args.limit]
    fp = benchmark_fingerprint(items)

    try:
        backend = build_backend(args)
        backend_desc = backend.describe()
    except Exception as e:
        log(f"[ERROR] {e}")
        return 2

    try:
        if args.api_key:      # 显式传 key 时沿用旧行为（单供应商）
            client = ChatClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
        else:
            client = provider_client(args.answer_provider) if (args.model or args.base_url) is None \
                else ChatClient(api_key=None, base_url=args.base_url, model=args.model)
    except LLMError as e:
        log(f"[ERROR] {e}")
        return 3

    judge_provider = args.judge_provider or args.answer_provider
    if args.api_key:
        judge_client = ChatClient(api_key=args.api_key, base_url=args.base_url,
                                  model=args.judge_model or args.model) if args.judge_model else client
    else:
        judge_client = client if (judge_provider == args.answer_provider and not args.judge_model) \
            else provider_client(judge_provider, model=args.judge_model or None)
    judge2_client = provider_client(args.judge2_provider) if args.judge2_provider else None
    judge3_client = provider_client(args.judge3_provider) if args.judge3_provider else None

    graph = TemporalRAGGraph(client, thinking=args.thinking, mode=args.agent_mode)
    meta = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark_path": str(bench_path.resolve()),
        "benchmark_fingerprint": fp,
        "backend": backend_desc,
        "model": client.model,
        "judge_model": judge_client.model,
        "judge2_model": judge2_client.model if judge2_client else None,
        "judge3_model": judge3_client.model if judge3_client else None,
        "judge_mode": ("majority3" if judge3_client else ("dual" if judge2_client else "single")),
        "answer_provider": args.answer_provider,
        "as_of": as_of.isoformat(),
        "top_k": args.top_k,
        "thinking": args.thinking,
    }

    records: list[dict[str, Any]] = []
    if args.resume and out_path.exists():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            same = (
                old.get("meta", {}).get("benchmark_fingerprint") == fp
                and old.get("meta", {}).get("as_of") == as_of.isoformat()
                and old.get("meta", {}).get("top_k") == args.top_k
                and old.get("meta", {}).get("model") == client.model
            )
            if same:
                records = [
                    r for r in old.get("items", [])
                    if r.get("naive_judge", {}).get("status") in STATUSES
                    and r.get("agent_judge", {}).get("status") in STATUSES
                ]
                log(f"[RESUME] 已完成 {len(records)} 题，将跳过")
            else:
                log("[WARN] 已有结果与当前 benchmark/模型/top-k 不一致，忽略 --resume")
        except (OSError, json.JSONDecodeError):
            log("[WARN] 旧结果损坏，忽略 --resume")

    done_ids = {r["id"] for r in records}
    log(f"检索后端：{backend_desc}")
    log(f"共 {len(items)} 题，已完成 {len(done_ids)} 题")

    for i, item in enumerate(items, 1):
        if item["id"] in done_ids:
            continue
        log(f"[{i}/{len(items)}] {item['id']} {item['question'][:52]}")

        # 1) 只检索一次。Naive 与 Temporal 共用，禁止各自重检。
        top5 = backend.similarity_search(item["question"], k=args.top_k)
        old_rank = new_rank = None
        for rank, c in enumerate(top5, 1):
            if old_rank is None and clause_matches(
                c, expected_id=item.get("old_clause_id", ""), expected_text=item["old_clause_text"]
            ):
                old_rank = rank
            if new_rank is None and clause_matches(
                c, expected_id=item.get("new_clause_id", ""), expected_text=item["new_clause_text"]
            ):
                new_rank = rank

        # 2) 对照组
        try:
            naive = naive_answer(client, item, top5, thinking=args.thinking)
            naive_j = judge_answer(judge_client, item, naive, repeat=args.judge_repeat)
            naive_j2 = judge_answer(judge2_client, item, naive, repeat=args.judge_repeat) if judge2_client else None
            naive_j3 = judge_answer(judge3_client, item, naive, repeat=args.judge_repeat) if judge3_client else None
            # 3) 实验组：把同一 Top-5 作为初始 State 塞入 Temporal LangGraph
            agent = graph.ask(item["question"], top5, as_of)
            agent_vote = judge_answer(judge_client, item, agent["answer"], repeat=args.judge_repeat)
            agent_j2 = judge_answer(judge2_client, item, agent["answer"], repeat=args.judge_repeat) if judge2_client else None
            agent_j3 = judge_answer(judge3_client, item, agent["answer"], repeat=args.judge_repeat) if judge3_client else None
        except (LLMError, RuntimeError) as e:
            log(f"  [ABORT] 调用失败，已完成结果已保存：{e}")
            atomic_write_json(out_path, {
                "meta": meta, "metrics": compute_metrics(records), "items": records,
            })
            backend.close()
            return 4

        old_filtered = bool(old_rank is not None and any(
            d.get("id") == item.get("old_clause_id") for d in agent["dropped"]
        ))
        rec = {
            "id": item["id"],
            "topic": item["topic"],
            "region": item["region"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "conflict_type": item.get("conflict_type", ""),
            "retrieved": [_clause_dump(c) for c in top5],
            "old_clause_penetrated": old_rank is not None,
            "old_clause_filtered": old_filtered,
            "old_clause_rank": old_rank,
            "new_clause_penetrated": new_rank is not None,
            "new_clause_rank": new_rank,
            "naive_answer": naive,
            "naive_judge": majority_vote([naive_j, naive_j2, naive_j3]) if judge3_client else naive_j,
            "naive_judge1_raw": naive_j if judge3_client else None,
            "naive_judge2": naive_j2,
            "naive_judge3": naive_j3,
            "agent_answer": agent["answer"],
            "agent_judge": majority_vote([agent_vote, agent_j2, agent_j3]) if judge3_client else agent_vote,
            "agent_judge1_raw": agent_vote if judge3_client else None,
            "agent_judge2": agent_j2,
            "agent_judge3": agent_j3,
            "agent_route": agent["route"],
            "agent_steps": agent["steps"],
            "agent_dropped": agent["dropped"],
            "agent_context_count": len(agent["filtered_docs"]),
        }
        records.append(rec)
        done_ids.add(item["id"])
        log(
            f"  旧条款={'是@' + str(old_rank) if old_rank else '否'} | "
            f"新条款={'是@' + str(new_rank) if new_rank else '否'} | "
            f"Naive={naive_j['status']} | Agent={agent_vote['status']} | "
            f"过滤 {len(agent['dropped'])}/{len(top5)}"
        )

        atomic_write_json(out_path, {
            "meta": meta, "metrics": compute_metrics(records), "items": records,
        })
        if args.sleep:
            time.sleep(args.sleep)

    metrics = compute_metrics(records)
    result = {"meta": meta, "metrics": metrics, "items": records}
    atomic_write_json(out_path, result)

    print("\n" + "=" * 64)
    print(f"样本数                 {metrics['n']}")
    print(f"过期政策穿透率         {metrics['penetration_rate']:.1%}  "
          f"({metrics['penetration_count']}/{metrics['n']})")
    print(f"现行条款召回率         {metrics['new_clause_penetration_rate']:.1%}  "
          f"({metrics['new_clause_penetration_count']}/{metrics['n']})")
    print(f"Temporal 旧条款拦截率  {metrics['old_clause_filter_rate']:.1%}  "
          f"({metrics['old_clause_filtered_count']}/{metrics['penetration_count']})")
    print(f"Naive RAG 毒化率       {metrics['naive_poison_rate']:.1%}  "
          f"({metrics['naive_poisoned']}/{metrics['n']})")
    print(f"Temporal 毒化率        {metrics['agent_poison_rate']:.1%}  "
          f"({metrics['agent_poisoned']}/{metrics['n']})")
    print(f"严格拯救率             {metrics['rescue_rate_strict']:.1%}  "
          f"({metrics['rescued_to_correct']}/{metrics['naive_poisoned']})")
    print(f"毒化消除率             {metrics['de_poison_rate']:.1%}  "
          f"({metrics['de_poisoned']}/{metrics['naive_poisoned']})")
    print(f"条件毒化率(Naive)      {metrics['naive_conditionally_poisoned_rate']:.1%}  "
          f"({metrics['naive_conditionally_poisoned_count']}/{metrics['penetration_count']}，穿透题内毒化)")
    print(f"条件毒化率(Temporal)   {metrics['agent_conditionally_poisoned_rate']:.1%}  "
          f"({metrics['agent_conditionally_poisoned_count']}/{metrics['penetration_count']}，穿透题内毒化)")
    print(f"四分类 Naive          正确 {metrics['naive_correct']} / 毒化 {metrics['naive_poisoned']} / "
          f"拒答 {metrics['naive_refused']} / 其他错误 {metrics['naive_other_error']}")
    print(f"四分类 Temporal       正确 {metrics['agent_correct']} / 毒化 {metrics['agent_poisoned']} / "
          f"拒答 {metrics['agent_refused']} / 其他错误 {metrics['agent_other_error']}")
    print(f"毒化加权得分           Naive {metrics['naive_weighted_score']:+.3f} → "
          f"Temporal {metrics['agent_weighted_score']:+.3f}  (增益 {metrics['weighted_gain']:+.3f})  "
          f"[正确+1 拒答+0.2 其他0 毒化-1]")
    print(f"Naive 正确率           {metrics['naive_correct_rate']:.1%}  |  "
          f"Temporal 正确率 {metrics['agent_correct_rate']:.1%}")
    print(f"Naive 拒答率           {metrics['naive_refusal_rate']:.1%}  |  "
          f"Temporal 拒答率 {metrics['agent_refusal_rate']:.1%}")
    print("=" * 64)
    print(f"逐题结果 → {out_path}")

    if records:
        fig_path = plot_metrics(metrics, Path(args.figure), n=metrics["n"])
        print(f"论文图   → {fig_path}")

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
