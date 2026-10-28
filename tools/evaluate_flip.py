#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层 2 评估：答案翻转实验。

同一批题在 T1（2026-12-30，条款现行）与 T2（2027-01-01，条款失效）各跑一遍：
  Naive    上下文不带时点信息 → 时间盲，T1/T2 输入完全相同
  Temporal 带时点过滤 → T2 时目标条款被剔除

立场三分类（对"该事项能否办理"）：
  AFFIRM  认为可以办理 / 仍有效
  DENY    明确否定（不能、已失效、已到期）
  OTHER   拒答 / 无法确定 / 答非所问

判分：
  T1（正确答案=能）：AFFIRM=CORRECT，DENY=WRONG，OTHER=REFUSED
  T2（正确答案=不能）：DENY=CORRECT，AFFIRM=POISONED，OTHER=REFUSED

用法：
    python tools/evaluate_flip.py --benchmark data/national/temporal_flip.json \
        --out results/flip_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from policycrawler.llm import LLMError, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402
from rag_eval.backends import build_backend  # noqa: E402
from rag_eval.temporal import clause_matches, usable  # noqa: E402

T1 = date(2026, 12, 30)
T2 = date(2027, 1, 1)

ANSWER_SYSTEM = """你是政策问答助手。请仅依据提供的政策条款回答用户问题。
若条款未涉及所问事项，或条款均不能作为现行依据，请明确说明无法回答。
回答两句话以内，直接给结论。"""

STANCE_SYSTEM = """你是评测判分员。给你【用户问题】和【模型回答】，判断回答对"该事项现在能否办理/申请"的立场。
只输出一个 JSON 对象：
{"stance": "AFFIRM|DENY|OTHER", "reason": "不超过 25 字"}

- AFFIRM：认为可以办理、能申请、仍可享受，或给出了办理标准/条件。
- DENY：明确否定——不能申请、已失效、已到期、不再有效、已取消。
- OTHER：拒答、表示无法确定、答非所问。

只输出 JSON。"""

DENY_RE = re.compile(r"失效|届满|到期|不再(有效|受理|执行|享受)|不能(申请|办理|享受)|无法(申请|办理|享受)|已取消|不予")
AFFIRM_RE = re.compile(r"可以(申请|享受|办理)|能(申请|办理|享受)|仍(可|能)|按.{0,12}(标准|比例|上限)|补贴.{0,8}为|最高.{0,8}(万元|元)")


def log(msg: str) -> None:
    print(msg, flush=True)


def gen(client, question: str, docs, as_of: date) -> str:
    lines = [f"【政策条款 {i}】《{d.policy_name}》{d.article}\n{d.text}"
             for i, d in enumerate(docs, 1)]
    body = "\n\n".join(lines) if lines else "（无可用条款）"
    return client.chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{body}\n\n【用户问题】\n{question}"},
        ],
        temperature=0.0, max_tokens=500, json_mode=False, thinking=False,
    )


def stance(client, question: str, answer: str) -> tuple[str, str]:
    a = answer or ""
    if DENY_RE.search(a):
        return "DENY", "规则：含失效/否定表述"
    if AFFIRM_RE.search(a):
        return "AFFIRM", "规则：含肯定性表述"
    try:
        raw = client.chat_json(
            STANCE_SYSTEM,
            f"【用户问题】\n{question}\n\n【模型回答】\n{a}",
            temperature=0.0, max_tokens=150, thinking=False,
        )
        s = str(raw.get("stance") or "").strip().upper()
        if s in ("AFFIRM", "DENY", "OTHER"):
            return s, str(raw.get("reason") or "")[:50]
    except LLMError:
        pass
    return "OTHER", "规则与裁判均无法判定"


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description="答案翻转实验")
    ap.add_argument("--benchmark", default=str(ROOT / "data" / "national" / "temporal_flip.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "flip_eval.json"))
    ap.add_argument("--top-k", type=int, default=9)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    print(f"翻转题 {len(items)} 道；T1={T1}（现行）  T2={T2}（失效）")

    backend = build_backend(argparse.Namespace(
        backend="chroma", chroma_dir=str(ROOT / "chroma_lowalt"), chroma_host="",
        chroma_port="8000", collection="lowalt_policy", embedding_function="bge-zh",
        index_dir=str(ROOT / "kb" / "index"),
    ))
    client = provider_client(args.provider)

    records = []
    for i, it in enumerate(items, 1):
        docs = backend.similarity_search(it["question"], k=args.top_k)
        target = next((d for d in docs if clause_matches(
            d, expected_id=it.get("clause_id", ""), expected_text=it["clause_text"])), None)
        in_topk = target is not None

        keep_t1 = [d for d in docs if usable(d, T1)[0]]
        keep_t2 = [d for d in docs if usable(d, T2)[0]]
        dropped_t2 = len(docs) - len(keep_t2)

        rec = {"id": it["id"], "region": it["region"], "question": it["question"],
               "policy_name": it["policy_name"], "in_topk": in_topk,
               "t1_keep": len(keep_t1), "t2_keep": len(keep_t2), "dropped_t2": dropped_t2}
        try:
            rec["t1_naive"] = " ".join(gen(client, it["question"], docs, T1).split())
            rec["t1_agent"] = " ".join(gen(client, it["question"], keep_t1, T1).split())
            rec["t2_naive"] = " ".join(gen(client, it["question"], docs, T2).split())
            rec["t2_agent"] = " ".join(gen(client, it["question"], keep_t2, T2).split())
            for key in ("t1_naive", "t1_agent", "t2_naive", "t2_agent"):
                s, r = stance(client, it["question"], rec[key])
                rec[key + "_stance"] = s
                rec[key + "_reason"] = r
        except LLMError as e:
            log(f"  [{i}] {it['id']} 失败：{str(e)[:80]}")
            continue
        records.append(rec)
        log(f"  [{i}/{len(items)}] {it['id']} T1:N={rec['t1_naive_stance']}/A={rec['t1_agent_stance']}"
            f"  T2:N={rec['t2_naive_stance']}/A={rec['t2_agent_stance']}")

    n = len(records)

    from collections import Counter

    def cnt(side: str, t: str) -> Counter:
        return Counter(r[f"{t}_{side}_stance"] for r in records)

    tn1, ta1 = cnt("naive", "t1"), cnt("agent", "t1")
    tn2 = cnt("naive", "t2")
    ta2 = cnt("agent", "t2")
    summary = {
        "n": n,
        "t1": {"naive_affirm": tn1.get("AFFIRM", 0), "agent_affirm": ta1.get("AFFIRM", 0)},
        "t2_naive": dict(tn2),
        "t2_agent": dict(ta2),
        # 核心指标
        "naive_cliff": f"T1 正确(AFFIRM) {tn1.get('AFFIRM',0)}/{n} → T2 毒化(AFFIRM) {tn2.get('AFFIRM',0)}/{n}",
        "agent_behavior": f"T2 正确(DENY) {ta2.get('DENY',0)}/{n}，安全拒答(OTHER) {ta2.get('OTHER',0)}/{n}，毒化 {ta2.get('AFFIRM',0)}/{n}",
    }
    if n:
        summary["naive_affirm_t1_rate"] = round(tn1.get("AFFIRM", 0) / n, 4)
        summary["naive_poison_t2_rate"] = round(tn2.get("AFFIRM", 0) / n, 4)
        summary["agent_deny_t2_rate"] = round(ta2.get("DENY", 0) / n, 4)
        summary["agent_safe_t2_rate"] = round((ta2.get("DENY", 0) + ta2.get("OTHER", 0)) / n, 4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "items": records},
                              ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    log(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
