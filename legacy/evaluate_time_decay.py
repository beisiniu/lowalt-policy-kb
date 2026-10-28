#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段四 · 时效性衰减评估（LLM-as-a-Judge）

三种答题后端，方便做对照实验：
    local  —— 本地知识库 + LangGraph 问答图（默认，答案来自你自己的索引）
    llm    —— 纯大模型直答，不检索（对照组，用来证明知识库本身的价值）
    mock   —— 不调任何服务，生成模拟数据（验证指标与出图链路）

跑一次评估：
    python evaluate.py --tag before --backend local --no-llm
    python evaluate.py --tag after  --backend local          # 库里更新后再跑

出对比图（论文用）：
    python evaluate.py --compare results/eval_before.json results/eval_after.json \
                       --figure output/figures/timeliness_decay.png

环境变量（只有走 LLM 时才需要）：
    DEEPSEEK_API_KEY
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb.graph import PolicyQA
from kb.retriever import PolicyRetriever
from kb.schema import freshness
from kb.store import load_index
from policycrawler.llm import ChatClient, LLMError
from policycrawler.utils import load_env_file

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
- Stale_Fact_Error：模型给出了错误事实或审批建议，而且错误原因是使用了已经失效、已被修订、
  已被取代或并非题目所问最新版本的旧政策。信息查询题同样适用。
  典型表现：沿用旧办法的审批门槛、忽略新规豁免、引用已废止文件、把旧版产业目标/补贴标准/
  规划期限当成最新答案、把旧的地域适用范围套到新场景。
  只有回答中确有旧政策依据或旧版数值时才判这一类；仅仅答错不能都归为过期误导。
- Refusal：模型拒答，或明确表示无法确定（如"建议咨询主管部门""知识库中没有相关信息""我无法判断"）。
- Incorrect_Other：结论错误，但不是因为用过期政策（如理解偏差、答非所问、凭空编造条款、混用其他城市的数值）。

判定优先级：先看结论对错 → 再判是否属于"过期政策导致" → 拒答单独归类。

只输出 JSON，不要任何解释文字：
{"verdict": "Correct|Stale_Fact_Error|Refusal|Incorrect_Other", "reason": "不超过 50 字，说明判定依据"}
"""


# ---------------------------------------------------------------- 工具
def log(msg: str) -> None:
    print(msg, file=sys.stderr if msg.startswith("  ") else sys.stdout, flush=True)


NEED_WORDS = ("需要审批", "需要申请", "需要提前", "需要报批", "需提前", "须提前", "需报批", "须报批",
              "应当提前", "应提前", "应当申请", "应当提出", "必须申请", "需要办理", "需要批准",
              "需经批准", "须经批准", "应当取得", "须取得", "需要取得", "应当依法取得",
              "未经批准", "未经空中交通管理机构批准", "不得飞入", "不得擅自")
FREE_WORDS = ("无需审批", "无需申请", "不需要申请", "不需要审批", "免于申请", "无需提出", "不需要提前",
              "不用申请", "无需报备", "无需办理")
REFUSAL_WORDS = ("无法给出确定", "无法确定", "建议咨询", "知识库中没有", "未检索到", "不能作为当前审批依据")


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
def judge_llm(client: ChatClient, question: str, ground_truth: str, expected_status: str, answer: str,
              case: dict | None = None) -> dict:
    case = case or {}
    user = (
        f"【问题】\n{question}\n\n"
        f"【标准答案】\n{ground_truth}\n\n"
        f"【标准依据】\n{case.get('policy_ref', '未提供')}\n\n"
        f"【标准结论】\n{expected_status or '事实查询题，无审批结论'}\n\n"
        f"【模型回答】\n{answer[:2000]}\n"
    )
    try:
        raw = client.chat_json(JUDGE_SYSTEM, user, temperature=0.0, max_tokens=300)
        verdict = str(raw.get("verdict", "")).strip()
        if verdict not in VERDICTS:
            alias = {"正确": "Correct", "过期": "Stale_Fact_Error", "拒答": "Refusal", "错误": "Incorrect_Other"}
            verdict = next((v for k, v in alias.items() if k in verdict), "Incorrect_Other")
        return {"verdict": verdict, "reason": str(raw.get("reason", ""))[:200]}
    except (LLMError, ValueError) as e:
        return {"verdict": "Incorrect_Other", "reason": f"裁判调用失败：{e}"[:200]}


def _majority(votes: list[dict]) -> dict:
    """多数票；平票时按 VERDICTS 顺序里更"严格"的那个（宁可判错，不虚高正确率）。"""
    if not votes:
        return {"verdict": "Incorrect_Other", "reason": "无裁判结果"}
    tally: dict[str, int] = {}
    for v in votes:
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    best = max(tally.items(), key=lambda kv: (kv[1], -VERDICTS.index(kv[0])))[0]
    reason = next(v["reason"] for v in votes if v["verdict"] == best)
    return {"verdict": best, "reason": reason}


class RuleJudge:
    """没有 API Key 时的规则裁判（只用来跑通链路，不能当论文指标）。

    判据：拒答话术 → Refusal；答案引用了库里标记为失效的政策 → Stale_Fact_Error；
    审批结论与 expected_status 一致 → Correct；否则 Incorrect_Other。
    """

    def __init__(self, stale_policy_names: set[str]):
        self.stale_names = {n for n in stale_policy_names if n}

    def judge(self, question: str, ground_truth: str, expected_status: str, answer: str, item: dict | None = None) -> dict:
        if any(w in answer for w in REFUSAL_WORDS):
            return {"verdict": "Refusal", "reason": "回答未给出结论（规则判定）"}
        # 必须带书名号精确匹配：否则「深圳市低空飞行活动管理办法」会命中
        # 「深圳市低空飞行活动管理办法（2026年修订）」造成误判
        if any(f"《{n}》" in answer for n in self.stale_names):
            return {"verdict": "Stale_Fact_Error", "reason": "引用了已失效政策（规则判定）"}

        # 事实查询类（问日期、金额、高度）不看审批结论，只看有没有命中关键事实
        item = item or {}
        if item.get("answer_type") == "信息查询":
            facts = [f for f in (item.get("key_facts") or []) if f]
            if facts and any(f in answer for f in facts):
                return {"verdict": "Correct", "reason": "命中关键事实（规则判定）"}
            return {"verdict": "Incorrect_Other", "reason": "未命中关键事实（规则判定）"}

        detected = detect_status(answer)
        if detected == expected_status:
            return {"verdict": "Correct", "reason": "审批结论与标准一致（规则判定）"}
        return {"verdict": "Incorrect_Other", "reason": f"结论为{detected}，与标准不符（规则判定）"}


# ---------------------------------------------------------------- 答题后端
class LocalAnswerer:
    """本地知识库 + LangGraph 问答图。"""

    def __init__(self, index_dir: str, mode: str = "extractive", client=None, top_k: int = 4,
                 newer_bias: float = 0.35):
        docs, tokens, manifest = load_index(index_dir)
        self.docs = docs
        self.manifest = manifest
        self.retriever = PolicyRetriever(docs, tokens, newer_bias=newer_bias)
        self.qa = PolicyQA(self.retriever, mode=mode, client=client, top_k=top_k)
        self.mode = mode

    def ask(self, question: str, item: dict) -> tuple[str, dict]:
        out = self.qa.ask(question)
        return out["answer"], out

    def stale_policy_names(self) -> set[str]:
        names = set()
        for d in self.docs:
            ok, _ = freshness(d.metadata)
            if not ok:
                names.add(d.metadata.get("policy_name", ""))
        return names


class LLMAnswerer:
    """对照组：不带检索、纯大模型直答（用于证明知识库本身的价值）。"""

    def __init__(self, api_key: str | None, base_url: str | None, model: str | None):
        self.client = ChatClient(api_key=api_key, base_url=base_url, model=model)

    def ask(self, question: str, item: dict) -> tuple[str, dict]:
        ans = self.client.chat(
            [{"role": "system", "content": "你是低空经济政策顾问，请依据中国现行法规简洁回答，并说明依据的政策名称与条款。"},
             {"role": "user", "content": question}],
            temperature=0.0, max_tokens=600, json_mode=False,
        )
        return ans, {"route": "llm-direct", "steps": ["direct"], "cited": [], "blocked_stale": []}


# ---------------------------------------------------------------- 指标
def summarize(items: list[dict]) -> dict:
    total = len(items) or 1
    counts = {v: sum(1 for i in items if i["verdict"] == v) for v in VERDICTS}
    judged = [i for i in items if i.get("status_match") is not None]
    blocked = sum(1 for i in items if i.get("blocked_stale"))

    # 双口径：严格口径把拒答算错，有效应答口径把拒答排除在分母外。
    # 只报其中一个都能被质疑（严格口径惩罚谨慎的系统，有效口径又有美化嫌疑），所以两个都报。
    answered = len(items) - counts["Refusal"]

    # 按题型分开统计：抽取式在"信息查询"上接近满分，在"审批判定"上受限于推理能力，
    # 混在一起报一个 65% 看不出问题在哪。
    by_type: dict[str, dict] = {}
    for it in items:
        t = it.get("answer_type", "审批判定")
        bucket = by_type.setdefault(t, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += 1 if it["verdict"] == "Correct" else 0
    for t, b in by_type.items():
        b["accuracy"] = round(b["correct"] / max(1, b["n"]), 4)

    agreements = [i["judge_agreement"] for i in items if i.get("judge_agreement") is not None]

    return {
        "total": len(items),
        **{v: counts[v] for v in VERDICTS},
        "accuracy": round(counts["Correct"] / total, 4),
        "answered": answered,
        "effective_accuracy": round(counts["Correct"] / answered, 4) if answered else 0.0,
        "by_type": by_type,
        "judge_agreement": round(sum(agreements) / len(agreements), 4) if agreements else None,
        "stale_rate": round(counts["Stale_Fact_Error"] / total, 4),
        "refusal_rate": round(counts["Refusal"] / total, 4),
        "incorrect_rate": round(counts["Incorrect_Other"] / total, 4),
        "status_detected_rate": round(len(judged) / total, 4),
        "status_accuracy": round(sum(1 for i in judged if i["status_match"]) / len(judged), 4) if judged else 0.0,
        "stale_blocked": blocked,
        "stale_trap_total": sum(1 for i in items if i.get("trap") == "stale"),
        "stale_trap_hit": sum(1 for i in items if i.get("trap") == "stale" and i["verdict"] == "Stale_Fact_Error"),
    }


# ---------------------------------------------------------------- 主评估
def run_eval(args) -> dict:
    test_set = load_json(Path(args.test_set))
    if args.limit:
        test_set = test_set[: args.limit]

    judge_client = None
    if args.judge == "llm":
        judge_client = ChatClient(api_key=args.judge_key or args.api_key,
                                  base_url=args.base_url, model=args.judge_model or args.model)

    # 裁判判定"是否过期"必须依据**参考索引**（客观事实），不能用被测知识库自己的状态，
    # 否则系统把旧法标成"现行有效"就能自我免责，测不出真实错误
    stale_ref = set()
    try:
        ref_docs, _, _ = load_index(args.stale_ref_index)
        stale_ref = {d.metadata.get("policy_name", "") for d in ref_docs if not freshness(d.metadata)[0]}
    except FileNotFoundError:
        pass
    rule_judge = RuleJudge(stale_ref)

    if args.backend == "llm":
        answerer = LLMAnswerer(args.api_key, args.base_url, args.model)
    else:
        mode = "extractive" if args.no_llm else "llm"
        client = None if args.no_llm else ChatClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
        answerer = LocalAnswerer(args.index_dir, mode=mode, client=client, top_k=args.top_k,
                                 newer_bias=args.newer_bias)

    items = []
    for i, case in enumerate(test_set, 1):
        q = case["question"]
        log(f"[{i}/{len(test_set)}] {case['id']} {q[:38]}…")
        try:
            answer, trace = answerer.ask(q, case)
        except (LLMError, ValueError, RuntimeError) as e:
            answer, trace = f"【调用失败】{e}", {"route": "error", "steps": [], "cited": [], "blocked_stale": []}

        if judge_client is not None:
            # 同一份回答判 N 次取多数：LLM 裁判本身有随机性，单次结论未必稳
            votes = [judge_llm(judge_client, q, case["ground_truth"],
                               case.get("expected_status", ""), answer, case)
                     for _ in range(max(1, args.judge_repeat))]
            verdict = _majority(votes)
            agreement = round(sum(1 for v in votes if v["verdict"] == verdict["verdict"]) / len(votes), 3)
        else:
            verdict = rule_judge.judge(q, case["ground_truth"], case.get("expected_status", ""), answer, case)
            agreement = None

        detected = detect_status(answer)
        # 只有"审批判定"类题目才做结论一致性比对；事实查询类没有审批结论可言
        if case.get("answer_type") == "信息查询" or not case.get("expected_status"):
            status_match = None
        else:
            status_match = None if detected == "不确定" else (detected == case.get("expected_status"))
        items.append({
            "id": case["id"],
            "question": q,
            "expected_status": case.get("expected_status", ""),
            "ground_truth": case["ground_truth"],
            "policy_ref": case.get("policy_ref", ""),
            "trap": case.get("trap", "none"),
            "group": case.get("group", "all"),
            "answer_type": case.get("answer_type", "审批判定"),
            "answer": answer,
            "status_detected": detected,
            "status_match": status_match,
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
            "judge_agreement": agreement,
            "route": trace.get("route", ""),
            "steps": trace.get("steps", []),
            "cited": trace.get("cited", []),
            "blocked_stale": trace.get("blocked_stale", []),
        })
        log(f"  → {verdict['verdict']}（{verdict['reason'][:40]}）")
        if args.sleep:
            time.sleep(args.sleep)

    return {
        "tag": args.tag,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "answer_mode": "extractive" if args.no_llm else ("llm" if args.backend == "local" else "direct"),
        "judge": args.judge,
        "simulated": False,
        "metrics": summarize(items),
        "items": items,
    }


# ---------------------------------------------------------------- Mock（无需 Key）
MOCK_ANSWERS = {
    "Correct": "根据《深圳经济特区低空经济产业促进条例》第九条，最大起飞重量不超过 25 千克的微型、轻型无人机在适飞空域内飞行无需提出飞行计划申请，只需完成实名登记，高度不超过真高 120 米。",
    "Stale_Fact_Error": "根据《深圳市低空飞行活动管理办法》第六条，在本市使用民用无人驾驶航空器飞行，无论机型大小均应当提前 5 个工作日向市低空管理部门提出飞行计划申请，经批准后方可实施。",
    "Refusal": "知识库中与问题最相关的政策已失效，不能作为当前审批依据，建议咨询深圳市低空经济主管部门确认。",
    "Incorrect_Other": "无人机飞行一般不需要办理任何手续，直接起飞即可，注意避开人群。",
}


def simulate(tag: str, test_set: list[dict], dist: dict, seed: int = 7) -> dict:
    """生成一份"看起来像真的"的评估结果，仅用于验证指标与出图链路。

    为了让模拟结果自洽：Stale_Fact_Error 优先落在 trap=="stale" 的题目上。
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
            "route": {"Correct": "answer", "Stale_Fact_Error": "answer", "Refusal": "refuse"}.get(verdict, "refuse"),
            "steps": [], "cited": [], "blocked_stale": [],
        })
    return {
        "tag": tag,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "mock", "answer_mode": "mock", "judge": "mock",
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
    """左图：四个核心指标；右图：按题型拆分（审批判定 vs 信息查询）。

    为什么把"其他错误率"拿掉：四类判定互斥，过期误导率降到 0 之后，
    残余错误会全部归到"其他错误"，看起来像"变差了"。图里只放单调可比的口径，
    其他错误率仍保留在 CSV 和终端输出里。
    """
    import matplotlib.pyplot as plt

    setup_font()
    groups = [("正确率", "accuracy"), ("有效应答\n正确率", "effective_accuracy"),
              ("过期政策\n误导率", "stale_rate"), ("拒答率", "refusal_rate")]
    b = [before["metrics"][k] * 100 for _, k in groups]
    a = [after["metrics"][k] * 100 for _, k in groups]

    by_type = before["metrics"].get("by_type") or {}
    has_type = bool(by_type) and bool(after["metrics"].get("by_type"))

    if has_type:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3), dpi=300,
                                      gridspec_kw={"width_ratios": [1.4, 1]})
    else:
        fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=300)
        ax2 = None

    def draw(axis, cats, bvals, avals, title, annotate_first=False):
        x = range(len(cats))
        w = 0.36
        bars1 = axis.bar([i - w / 2 for i in x], bvals, w, label=labels[0],
                         color=COLORS[0], edgecolor="white", linewidth=0.6)
        bars2 = axis.bar([i + w / 2 for i in x], avals, w, label=labels[1],
                         color=COLORS[1], edgecolor="white", linewidth=0.6)
        for bars in (bars1, bars2):
            for rect in bars:
                h = rect.get_height()
                axis.annotate(f"{h:.0f}%", (rect.get_x() + rect.get_width() / 2, h),
                              ha="center", va="bottom", fontsize=9, xytext=(0, 2),
                              textcoords="offset points")
        if annotate_first:
            delta = avals[0] - bvals[0]
            top = max(bvals[0], avals[0])
            axis.annotate(f"+{delta:.0f} 个百分点", xy=(0 + w / 2, avals[0]),
                          xytext=(0 + w / 2, min(top + 13, 96)), ha="center",
                          fontsize=9, color="#1F4E79", fontweight="bold")
        axis.set_xticks(list(x))
        axis.set_xticklabels(cats, fontsize=10)
        axis.set_ylabel("占比（%）")
        axis.set_ylim(0, 100)
        axis.set_title(title, fontsize=11.5, pad=10)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
        axis.set_axisbelow(True)
        for sp in ("top", "right"):
            axis.spines[sp].set_visible(False)

    draw(ax, [g[0] for g in groups], b, a, "整体指标", annotate_first=True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    if ax2 is not None:
        types = list(by_type.keys())
        bt_b = [before["metrics"]["by_type"].get(t, {}).get("accuracy", 0) * 100 for t in types]
        bt_a = [after["metrics"]["by_type"].get(t, {}).get("accuracy", 0) * 100 for t in types]
        draw(ax2, types, bt_b, bt_a, "按题型拆分")

    fig.suptitle(f"知识库动态更新前后时效性对比（N={before['metrics']['total']}）",
                 fontsize=12.5, y=1.0)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color="#888888")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_single(result: dict, out: Path, title: str = "") -> Path:
    import matplotlib.pyplot as plt

    setup_font()
    order = ["Correct", "Stale_Fact_Error", "Refusal", "Incorrect_Other"]
    total = max(1, result["metrics"]["total"])
    vals = [result["metrics"][v] / total * 100 for v in order]
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
    for label, key in (("正确率(严格)", "accuracy"), ("有效应答正确率", "effective_accuracy"),
                       ("过期政策误导率", "stale_rate"), ("拒答率", "refusal_rate"),
                       ("其他错误率", "incorrect_rate"), ("审批结论一致率", "status_accuracy")):
        b, a = before["metrics"][key] * 100, after["metrics"][key] * 100
        rows.append((label, f"{b:.1f}%", f"{a:.1f}%", f"{a - b:+.1f}"))
    for t in {**before["metrics"].get("by_type", {}), **after["metrics"].get("by_type", {})}:
        bh = before["metrics"].get("by_type", {}).get(t, {}).get("accuracy", 0) * 100
        ah = after["metrics"].get("by_type", {}).get(t, {}).get("accuracy", 0) * 100
        rows.append((f"其中·{t}", f"{bh:.1f}%", f"{ah:.1f}%", f"{ah - bh:+.1f}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(",".join(r) for r in rows), encoding="utf-8-sig")
    print(f"指标表已保存：{path}")


# ---------------------------------------------------------------- CLI
def parse_args():
    ap = argparse.ArgumentParser(description="阶段四 · 时效性衰减评估")
    ap.add_argument("--test-set", default=str(ROOT / "test_set.json"))
    ap.add_argument("--tag", default="run", help="本次运行标签，如 before / after")
    ap.add_argument("--backend", default="local", choices=["local", "llm", "mock"])
    ap.add_argument("--mock", action="store_true", help="等价于 --backend mock（无需任何 Key）")
    ap.add_argument("--no-llm", action="store_true", help="抽取式回答，不调用任何大模型")
    ap.add_argument("--judge-repeat", type=int, default=1,
                    help="同一份回答让裁判判几次（>1 则多数投票，并输出 judge_agreement）")
    ap.add_argument("--judge", default="auto", choices=["auto", "llm", "rules"],
                    help="裁判方式；auto = 有 Key 用 LLM，没有则用规则裁判")
    ap.add_argument("--index-dir", default=str(ROOT / "kb" / "index"))
    ap.add_argument("--stale-ref-index", default=str(ROOT / "kb" / "index"),
                    help="判定「哪些政策已失效」的参考索引（客观事实，默认用当前全量库）")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--newer-bias", type=float, default=0.35, help="新法优先权重（0=只看相关性，1=只看时间）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--sleep", type=float, default=0.3, help="每题之间的间隔秒数")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="只出对比图，不跑评估")
    ap.add_argument("--labels", nargs=2, default=("更新前", "更新后"))
    ap.add_argument("--figure", default=str(ROOT / "output" / "figures" / "timeliness_decay.png"))
    ap.add_argument("--api-key", default=None, help="DeepSeek Key（覆盖 DEEPSEEK_API_KEY）")
    ap.add_argument("--base-url", default=None, help="DeepSeek base_url")
    ap.add_argument("--model", default=None, help="默认 deepseek-flash")
    ap.add_argument("--judge-key", default=None, help="裁判用 Key（默认同上）")
    ap.add_argument("--judge-model", default=None, help="裁判模型（默认同上）")
    return ap.parse_args()


def main() -> int:
    load_env_file(ROOT / ".env")   # .env 与系统环境变量都支持
    args = parse_args()
    if args.mock:
        args.backend = "mock"
    if args.judge == "auto":
        args.judge = "llm" if (args.api_key or os.getenv("DEEPSEEK_API_KEY")) else "rules"
    if args.backend == "mock":
        args.judge = "mock"

    out_dir = Path(args.out_dir)
    figure = Path(args.figure)

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

    if args.backend == "mock":
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

    try:
        result = run_eval(args)
    except LLMError as e:
        print(f"[ERROR] {e}")
        return 3
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    m = result["metrics"]
    print("\n" + "=" * 56)
    print(f"评估完成：{args.tag}（N={m['total']}，backend={result['backend']}/{result['answer_mode']}，judge={result['judge']}）")
    print(f"  正确率          {m['accuracy']:>6.1%}  ({m['Correct']}/{m['total']})")
    print(f"  过期政策误导率  {m['stale_rate']:>6.1%}  ({m['Stale_Fact_Error']}/{m['total']})")
    print(f"  拒答率          {m['refusal_rate']:>6.1%}  ({m['Refusal']}/{m['total']})")
    print(f"  其他错误率      {m['incorrect_rate']:>6.1%}  ({m['Incorrect_Other']}/{m['total']})")
    print(f"  有效应答正确率  {m['effective_accuracy']:>6.1%}  ({m['Correct']}/{m['answered']}，拒答不计入分母)")
    for t, b in (m.get("by_type") or {}).items():
        print(f"    · {t:<6}     {b['accuracy']:>6.1%}  ({b['correct']}/{b['n']})")
    print(f"  审批结论一致率  {m['status_accuracy']:>6.1%}（关键词客观统计）")
    if m.get("judge_agreement") is not None:
        print(f"  裁判一致性      {m['judge_agreement']:>6.1%}（同一回答多判取多数）")
    print(f"  时效门禁拦截    {m['stale_blocked']} 题（检索只命中已失效政策，被拒答而不是硬答）")
    if m["stale_trap_total"]:
        print(f"  过期陷阱命中    {m['stale_trap_hit']}/{m['stale_trap_total']} 题")
    print("=" * 56)
    print(f"明细：{path}")
    if result["judge"] == "rules":
        print("提示：当前用的是**规则裁判**（无 API Key），指标只够跑通链路，不能写进论文。")
    print("下一次跑完后用 --compare 出对比图：")
    print(f"  python evaluate.py --compare {out_dir}/eval_before.json {out_dir}/eval_after.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
