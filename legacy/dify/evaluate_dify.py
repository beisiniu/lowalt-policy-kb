#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段四 · 时效性衰减评估（LLM-as-a-Judge）

跑一次评估：
    # 知识库更新「前」
    python evaluate.py --tag before --backend dify --test-set test_set.json
    # 入库新政策后，知识库更新「后」
    python evaluate.py --tag after  --backend dify --test-set test_set.json

出对比图（论文用）：
    python evaluate.py --compare results/eval_before.json results/eval_after.json \
                       --figure output/figures/timeliness_decay.png

不需要任何 Key，先看看流程和图表长什么样：
    python evaluate.py --mock

环境变量：
    DEEPSEEK_API_KEY    裁判/答题用（--backend llm 时也用于答题）
    DIFY_APP_API_KEY    Dify **应用**的 API Key（app-xxx，不是知识库的 dataset-xxx）
    DIFY_BASE_URL       默认 https://api.dify.ai/v1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.dify import DifyClient, DifyError
from policycrawler.llm import ChatClient, LLMError

ROOT = Path(__file__).resolve().parent
VERDICTS = ("Correct", "Stale_Fact_Error", "Refusal", "Incorrect_Other")
VERDICT_CN = {
    "Correct": "正确",
    "Stale_Fact_Error": "过期政策误导",
    "Refusal": "拒答",
    "Incorrect_Other": "其他错误",
}

JUDGE_SYSTEM = """你是一个评估专家，请对比【标准答案】和【模型回答】，判断模型回答是否正确。

只能从下面四类中选一个，不要自创标签：
- Correct：核心结论与标准答案一致。措辞不同、信息更简略但结论相同，也算 Correct。
- Stale_Fact_Error：模型给出了错误的审批建议，而且错误原因是使用了已经失效、已被修订或被取代的旧政策。
  典型表现：沿用旧办法的审批门槛（如"一律提前 5 个工作日申请"）、忽略新规的豁免条款、
  引用已废止文件或过期有效期、把旧的地域适用范围套到新场景。
- Refusal：模型拒答，或明确表示无法确定（如"建议咨询主管部门""知识库中没有相关信息""我无法判断"）。
- Incorrect_Other：结论错误，但不是因为用过期政策（如理解偏差、答非所问、凭空编造条款）。

判定优先级：先看结论对错 → 再判是否属于"过期政策导致" → 拒答单独归类。

只输出 JSON，不要任何解释文字：
{"verdict": "Correct|Stale_Fact_Error|Refusal|Incorrect_Other", "reason": "不超过 50 字，说明判定依据"}
"""


# ---------------------------------------------------------------- 工具
def log(msg: str) -> None:
    print(msg, file=sys.stderr if msg.startswith("  ") else sys.stdout, flush=True)


NEED_WORDS = ("需要审批", "需要申请", "需要提前", "需要报批", "需提前", "须提前", "需报批", "须报批",
              "应当提前", "应提前", "应当申请", "应当提出", "必须申请", "需要办理", "需要批准",
              "需经批准", "须经批准")
FREE_WORDS = ("无需审批", "无需申请", "不需要申请", "不需要审批", "免于申请", "无需提出", "不需要提前",
              "不用申请", "无需报备", "无需办理")


def detect_status(answer: str) -> str:
    """从自由文本里粗判审批结论，作为 judge 之外的**客观**辅助指标。"""
    a = answer or ""
    need = any(w in a for w in NEED_WORDS)
    free = any(w in a for w in FREE_WORDS)
    if need and not free:
        return "需要审批"
    if free and not need:
        return "无需审批"
    if need and free:
        return "不确定"  # 分场景回答（如"微型无需、大型需申请"），不强行归类
    return "不确定"


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 裁判
def judge_answer(judge: ChatClient, question: str, ground_truth: str, expected_status: str, answer: str) -> dict:
    user = (
        f"【问题】\n{question}\n\n"
        f"【标准答案】\n{ground_truth}\n\n"
        f"【标准结论】\n{expected_status}\n\n"
        f"【模型回答】\n{answer[:2000]}\n"
    )
    try:
        raw = judge.chat_json(JUDGE_SYSTEM, user, temperature=0.0, max_tokens=300)
        verdict = str(raw.get("verdict", "")).strip()
        if verdict not in VERDICTS:
            # 模型偶尔输出中文或别名，做一次宽松映射
            alias = {"正确": "Correct", "过期": "Stale_Fact_Error", "拒答": "Refusal", "错误": "Incorrect_Other"}
            verdict = next((v for k, v in alias.items() if k in verdict), "Incorrect_Other")
        return {"verdict": verdict, "reason": str(raw.get("reason", ""))[:200]}
    except (LLMError, ValueError) as e:
        return {"verdict": "Incorrect_Other", "reason": f"裁判调用失败：{e}"[:200]}


# ---------------------------------------------------------------- 答题后端
class DifyAnswerer:
    def __init__(self, api_key: str | None, base_url: str | None):
        self.client = DifyClient(api_key=api_key or os.getenv("DIFY_APP_API_KEY"), base_url=base_url)

    def ask(self, question: str, item: dict) -> str:
        resp = self.client.chat_message(query=question, user=f"eval-{item['id']}")
        return str(resp.get("answer", ""))


class LLMAnswerer:
    """对照组：不带检索、纯大模型直答（用于证明知识库本身的价值）。"""

    def __init__(self, api_key: str | None, base_url: str | None, model: str | None):
        self.client = ChatClient(api_key=api_key, base_url=base_url, model=model)

    def ask(self, question: str, item: dict) -> str:
        return self.client.chat(
            [
                {"role": "system", "content": "你是低空经济政策顾问，请依据中国现行法规简洁回答，并说明依据的政策名称与条款。"},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=600,
            json_mode=False,
        )


# ---------------------------------------------------------------- 指标
def summarize(items: list[dict]) -> dict:
    total = len(items) or 1
    counts = {v: sum(1 for i in items if i["verdict"] == v) for v in VERDICTS}
    judged = [i for i in items if i.get("status_match") is not None]
    return {
        "total": len(items),
        **{v: counts[v] for v in VERDICTS},
        "accuracy": round(counts["Correct"] / total, 4),
        "stale_rate": round(counts["Stale_Fact_Error"] / total, 4),
        "refusal_rate": round(counts["Refusal"] / total, 4),
        "incorrect_rate": round(counts["Incorrect_Other"] / total, 4),
        "status_detected_rate": round(len(judged) / total, 4),
        "status_accuracy": round(sum(1 for i in judged if i["status_match"]) / len(judged), 4) if judged else 0.0,
        "stale_trap_total": sum(1 for i in items if i.get("trap") == "stale"),
        "stale_trap_hit": sum(1 for i in items if i.get("trap") == "stale" and i["verdict"] == "Stale_Fact_Error"),
    }


# ---------------------------------------------------------------- 主评估
def run_eval(args) -> dict:
    test_set = load_json(Path(args.test_set))
    if args.limit:
        test_set = test_set[: args.limit]

    judge = ChatClient(api_key=args.judge_key or args.api_key, base_url=args.base_url,
                       model=args.judge_model or args.model)
    answerer = DifyAnswerer(args.app_key, args.dify_base) if args.backend == "dify" else LLMAnswerer(
        args.api_key, args.base_url, args.model
    )

    items = []
    for i, case in enumerate(test_set, 1):
        q = case["question"]
        log(f"[{i}/{len(test_set)}] {case['id']} {q[:38]}…")
        try:
            answer = answerer.ask(q, case)
        except (DifyError, LLMError) as e:
            answer = f"【调用失败】{e}"
        verdict = judge_answer(judge, q, case["ground_truth"], case.get("expected_status", ""), answer)
        detected = detect_status(answer)
        status_match = None if detected == "不确定" else (detected == case.get("expected_status"))
        items.append({
            "id": case["id"],
            "question": q,
            "expected_status": case.get("expected_status", ""),
            "ground_truth": case["ground_truth"],
            "policy_ref": case.get("policy_ref", ""),
            "trap": case.get("trap", "none"),
            "answer": answer,
            "status_detected": detected,
            "status_match": status_match,
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
        })
        log(f"  → {verdict['verdict']}（{verdict['reason'][:40]}）")
        if args.sleep:
            time.sleep(args.sleep)

    result = {
        "tag": args.tag,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "model": args.model if args.backend == "llm" else "dify-app",
        "judge_model": args.judge_model or args.model,
        "simulated": False,
        "metrics": summarize(items),
        "items": items,
    }
    return result


# ---------------------------------------------------------------- Mock（无需 Key）
MOCK_ANSWERS = {
    "Correct": "根据《深圳经济特区低空经济产业促进条例》第九条，最大起飞重量不超过 25 千克的微型、轻型无人机在适飞空域内飞行无需提出飞行计划申请，只需完成实名登记，高度不超过真高 120 米。",
    "Stale_Fact_Error": "根据《深圳市低空飞行活动管理办法》第六条，在本市使用民用无人驾驶航空器飞行，无论机型大小均应当提前 5 个工作日向市低空管理部门提出飞行计划申请，经批准后方可实施。",
    "Refusal": "知识库中没有检索到与该问题直接相关的政策条款，建议咨询深圳市低空经济主管部门确认。",
    "Incorrect_Other": "无人机飞行一般不需要办理任何手续，直接起飞即可，注意避开人群。",
}


def simulate(tag: str, test_set: list[dict], dist: dict, seed: int = 7) -> dict:
    """生成一份"看起来像真的"的评估结果，仅用于验证指标与出图链路。

    为了让模拟结果自洽：Stale_Fact_Error 优先落在 trap=="stale" 的题目上
    （这些题本来就是"旧政策会带偏"的题），剩下的标签再随机分配。
    """
    rng = random.Random(seed)
    n = len(test_set)
    stale_idx = [i for i, c in enumerate(test_set) if c.get("trap") == "stale"]
    other_idx = [i for i in range(n) if i not in stale_idx]

    assign: dict[int, str] = {}
    need_stale = min(dist.get("Stale_Fact_Error", 0), n)
    targets = stale_idx[:need_stale]
    if len(targets) < need_stale:
        targets += rng.sample(other_idx, need_stale - len(targets))
    for i in targets:
        assign[i] = "Stale_Fact_Error"

    pool = [v for v, k in dist.items() if v != "Stale_Fact_Error" for _ in range(k)]
    rest = [i for i in range(n) if i not in assign]
    rng.shuffle(rest)
    for i, v in zip(rest, pool):
        assign[i] = v

    items = []
    for i, case in enumerate(test_set):
        verdict = assign.get(i, "Correct")
        expected = case.get("expected_status", "需要审批")
        detected = expected if verdict == "Correct" else ("不确定" if verdict in ("Refusal", "Incorrect_Other")
                                                          else ("需要审批" if expected == "无需审批" else "无需审批"))
        items.append({
            "id": case["id"], "question": case["question"],
            "expected_status": expected,
            "ground_truth": case["ground_truth"], "policy_ref": case.get("policy_ref", ""),
            "trap": case.get("trap", "none"),
            "answer": MOCK_ANSWERS[verdict], "status_detected": detected,
            "status_match": detected == expected,
            "verdict": verdict,
            "reason": {"Correct": "结论与标准答案一致",
                       "Stale_Fact_Error": "套用了已废止/已修订旧办法的审批要求",
                       "Refusal": "未给出结论，属于拒答",
                       "Incorrect_Other": "结论错误且无政策依据"}[verdict],
        })
    return {
        "tag": tag,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "mock",
        "model": "mock",
        "judge_model": "mock",
        "simulated": True,
        "metrics": summarize(items),
        "items": items,
    }


# ---------------------------------------------------------------- 出图
def setup_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["font.size"] = 11
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
    else:
        print("[WARN] 没找到中文字体，图中中文可能显示为方块。"
              "可在系统安装 SimHei / Microsoft YaHei 后重试。")
    matplotlib.rcParams["axes.unicode_minus"] = False


COLORS = ("#B0B7C3", "#2F6FB2")  # 更新前（灰）/ 更新后（蓝）


def plot_compare(before: dict, after: dict, out: Path, labels: tuple[str, str], note: str = "") -> Path:
    import matplotlib.pyplot as plt

    setup_font()
    groups = [("正确率", "accuracy"), ("过期政策误导率", "stale_rate"),
              ("拒答率", "refusal_rate"), ("其他错误率", "incorrect_rate")]
    b = [before["metrics"][k] * 100 for _, k in groups]
    a = [after["metrics"][k] * 100 for _, k in groups]

    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=300)
    x = range(len(groups))
    w = 0.36

    bars1 = ax.bar([i - w / 2 for i in x], b, w, label=labels[0], color=COLORS[0], edgecolor="white", linewidth=0.6)
    bars2 = ax.bar([i + w / 2 for i in x], a, w, label=labels[1], color=COLORS[1], edgecolor="white", linewidth=0.6)

    for bars in (bars1, bars2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.0f}%", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")

    # 正确率的提升幅度，论文里最重要的一句话
    delta = a[0] - b[0]
    ax.annotate(f"↑{delta:.0f} 个百分点", xy=(0 + w / 2, a[0]), xytext=(0 + w / 2, a[0] + 9),
                ha="center", fontsize=9, color="#1F4E79", fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_ylabel("占比（%）")
    ax.set_ylim(0, 100)
    ax.set_title(f"知识库动态更新前后时效性对比（N={before['metrics']['total']}）", fontsize=12, pad=12)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color="#888888")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_single(result: dict, out: Path, title: str = "") -> Path:
    import matplotlib.pyplot as plt

    setup_font()
    order = ["Correct", "Stale_Fact_Error", "Refusal", "Incorrect_Other"]
    vals = [result["metrics"][v] / max(1, result["metrics"]["total"]) * 100 for v in order]
    color_map = {"Correct": "#2F6FB2", "Stale_Fact_Error": "#C0504D", "Refusal": "#B0B7C3", "Incorrect_Other": "#E3B23C"}

    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=300)
    bars = ax.bar([VERDICT_CN[v] for v in order], vals, color=[color_map[v] for v in order], width=0.55)
    for rect, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}%", (rect.get_x() + rect.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("占比（%）")
    ax.set_ylim(0, 100)
    ax.set_title(title or f"评估结果（{result.get('tag', '')}，N={result['metrics']['total']}）", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def write_csv(before: dict, after: dict, path: Path) -> None:
    rows = [("指标", f"{before.get('tag', 'before')}", f"{after.get('tag', 'after')}", "变化(pp)")]
    for label, key in (("正确率", "accuracy"), ("过期政策误导率", "stale_rate"),
                       ("拒答率", "refusal_rate"), ("其他错误率", "incorrect_rate"),
                       ("审批结论一致率", "status_accuracy")):
        b, a = before["metrics"][key] * 100, after["metrics"][key] * 100
        rows.append((label, f"{b:.1f}%", f"{a:.1f}%", f"{a - b:+.1f}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(",".join(r) for r in rows), encoding="utf-8-sig")
    print(f"指标表已保存：{path}")


# ---------------------------------------------------------------- CLI
def parse_args():
    ap = argparse.ArgumentParser(description="阶段四 · 时效性衰减评估")
    ap.add_argument("--test-set", default=str(ROOT / "test_set.json"))
    ap.add_argument("--tag", default="run", help="本次运行标签，如 before / after")
    ap.add_argument("--backend", default="dify", choices=["dify", "llm", "mock"])
    ap.add_argument("--mock", action="store_true", help="等价于 --backend mock（无需任何 Key）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--sleep", type=float, default=0.5, help="每题之间的间隔秒数")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="只出对比图，不跑评估")
    ap.add_argument("--labels", nargs=2, default=("更新前", "更新后"))
    ap.add_argument("--figure", default=str(ROOT / "output" / "figures" / "timeliness_decay.png"))
    ap.add_argument("--api-key", default=None, help="DeepSeek Key（覆盖 DEEPSEEK_API_KEY）")
    ap.add_argument("--base-url", default=None, help="DeepSeek base_url")
    ap.add_argument("--model", default=None, help="默认 deepseek-flash")
    ap.add_argument("--judge-key", default=None, help="裁判用 Key（默认同上）")
    ap.add_argument("--judge-model", default=None, help="裁判模型（默认同上）")
    ap.add_argument("--app-key", default=None, help="Dify 应用 API Key（app-xxx）")
    ap.add_argument("--dify-base", default=None, help="Dify base_url，默认 https://api.dify.ai/v1")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.mock:
        args.backend = "mock"
    out_dir = Path(args.out_dir)
    figure = Path(args.figure)

    # --- 只出图 ---
    if args.compare:
        before, after = load_json(Path(args.compare[0])), load_json(Path(args.compare[1]))
        note = "注：模拟数据，仅用于验证出图链路。" if (before.get("simulated") or after.get("simulated")) else ""
        plot_compare(before, after, figure, tuple(args.labels), note)
        write_csv(before, after, figure.with_suffix(".csv"))
        print(f"对比图已生成：{figure}")
        print(f"  {args.labels[0]}：正确率 {before['metrics']['accuracy']:.1%}，"
              f"过期误导率 {before['metrics']['stale_rate']:.1%}")
        print(f"  {args.labels[1]}：正确率 {after['metrics']['accuracy']:.1%}，"
              f"过期误导率 {after['metrics']['stale_rate']:.1%}")
        return 0

    test_set = load_json(Path(args.test_set))

    # --- Mock：无需任何 Key ---
    if args.backend == "mock":
        # 更新前：旧办法主导，一半问题被"一律提前 5 个工作日"带偏
        before = simulate("before", test_set, {"Correct": 8, "Stale_Fact_Error": 8, "Refusal": 3, "Incorrect_Other": 1})
        after = simulate("after", test_set, {"Correct": 17, "Stale_Fact_Error": 1, "Refusal": 1, "Incorrect_Other": 1}, seed=11)
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in (before, after):
            (out_dir / f"eval_{r['tag']}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_compare(before, after, figure, tuple(args.labels), "注：模拟数据，仅用于验证出图链路。")
        plot_single(before, figure.with_name("eval_before_single.png"), "知识库更新前")
        write_csv(before, after, figure.with_suffix(".csv"))
        print(json.dumps({"before": before["metrics"], "after": after["metrics"]}, ensure_ascii=False, indent=2))
        print(f"\n模拟结果与图表已生成：{figure}")
        return 0

    # --- 真实评估 ---
    try:
        result = run_eval(args)
    except LLMError as e:
        print(f"[ERROR] {e}")
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    m = result["metrics"]
    print("\n" + "=" * 56)
    print(f"评估完成：{args.tag}（N={m['total']}，backend={args.backend}）")
    print(f"  正确率          {m['accuracy']:>6.1%}  ({m['Correct']}/{m['total']})")
    print(f"  过期政策误导率  {m['stale_rate']:>6.1%}  ({m['Stale_Fact_Error']}/{m['total']})")
    print(f"  拒答率          {m['refusal_rate']:>6.1%}  ({m['Refusal']}/{m['total']})")
    print(f"  其他错误率      {m['incorrect_rate']:>6.1%}  ({m['Incorrect_Other']}/{m['total']})")
    print(f"  审批结论一致率  {m['status_accuracy']:>6.1%}（按关键词客观统计）")
    if m["stale_trap_total"]:
        print(f"  过期陷阱命中    {m['stale_trap_hit']}/{m['stale_trap_total']} 题")
    print("=" * 56)
    print(f"明细：{path}")
    print("下一次跑完后用 --compare 出对比图：")
    print(f"  python evaluate.py --compare {out_dir}/eval_before.json {out_dir}/eval_after.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
