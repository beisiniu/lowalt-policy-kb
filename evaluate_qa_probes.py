#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用真实 Chroma 跑覆盖性 QA 探针：检索命中 + 回答正确性。

这不是时效毒化实验，而是数据工程质量检查：
  - 政策是否被正确切分、索引
  - 按地域过滤后，标准条款能否进 Top-5
  - 回答是否忠实于条款（不编造）

输出：
  - 逐题 JSON：results/qa_probe_eval.json
  - 终端指标：条款 Recall@1/5、政策 Recall@5、回答正确率、拒答率

用法：
    python evaluate_qa_probes.py --chroma-dir ./chroma_lowalt \
      --collection lowalt_policy --embedding-function bge-zh \
      --probes qa_probes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.llm import ChatClient, LLMError
from policycrawler.utils import load_env_file
from rag_eval.backends import add_backend_args, build_backend
from rag_eval.temporal import clause_matches
from rag_eval.temporal_graph import generate_answer

ROOT = Path(__file__).resolve().parent
STATUSES = ("CORRECT", "REFUSED", "OTHER_ERROR")
JUDGE_SYSTEM = """你是政策问答评分员。请对比【标准答案】和【模型回答】，判断回答是否正确。
只输出 JSON：
{"status":"CORRECT|REFUSED|OTHER_ERROR","reason":"不超过40字"}

- CORRECT：核心事实/数字与标准答案一致；措辞不同也算正确。
- REFUSED：模型说资料不足、无法确定，没有给出结论。
- OTHER_ERROR：给出了错误结论，或答非所问。
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def judge(client: ChatClient, probe: dict, answer: str) -> dict:
    user = (
        f"【问题】\n{probe['question']}\n\n"
        f"【标准答案】\n{probe['ground_truth']}\n\n"
        f"【模型回答】\n{answer[:2000]}"
    )
    raw = client.chat_json(JUDGE_SYSTEM, user, temperature=0.0, max_tokens=250, thinking=False)
    status = str(raw.get("status") or "").strip().upper()
    if status not in STATUSES:
        raise LLMError(f"非法 status: {status!r}")
    return {"status": status, "reason": str(raw.get("reason") or "")[:160]}


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description="真实 Chroma 覆盖性 QA 验证")
    add_backend_args(ap)
    ap.add_argument("--probes", default=str(ROOT / "qa_probes.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "qa_probe_eval.json"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    probes = json.loads(Path(args.probes).read_text(encoding="utf-8"))
    if args.limit:
        probes = probes[: args.limit]
    if not probes:
        log("[ERROR] 没有探针")
        return 2
    try:
        backend = build_backend(args)
    except Exception as e:
        log(f"[ERROR] 知识库连接失败：{e}")
        return 2
    try:
        client = ChatClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    except LLMError as e:
        log(f"[ERROR] {e}")
        return 3

    records = []
    for i, p in enumerate(probes, 1):
        log(f"[{i}/{len(probes)}] {p['region']:8} {p['question'][:52]}")
        where = None
        if p.get("region") not in ("全国", "国家", ""):
            where = {"region": p["region"]}
        try:
            hits = backend.similarity_search(p["question"], k=args.top_k, where=where)
        except Exception as e:
            log(f"  [FAIL] 检索：{e}")
            continue

        clause_rank = None
        policy_rank = None
        for rank, h in enumerate(hits, 1):
            if clause_matches(h, expected_text=p["source_clause_text"]):
                clause_rank = clause_rank or rank
            if h.policy_name == p.get("source_policy_name") or (
                p.get("source_url") and h.metadata.get("source_url") == p["source_url"]
            ):
                policy_rank = policy_rank or rank

        try:
            answer = generate_answer(client, p["question"], hits, thinking=args.thinking)
            judged = judge(client, p, answer)
        except (LLMError, RuntimeError) as e:
            log(f"  [ABORT] 生成/裁判失败：{e}")
            return 4

        rec = {
            **{k: p.get(k, "") for k in (
                "id", "region", "topic", "question", "ground_truth",
                "source_policy_name", "source_article", "source_url",
            )},
            "retrieved": [{
                "policy_name": h.policy_name, "article": h.article,
                "region": h.region, "source_url": h.metadata.get("source_url", ""),
                "text": h.text,
            } for h in hits],
            "clause_rank": clause_rank,
            "policy_rank": policy_rank,
            "answer": answer,
            "judge": judged,
        }
        records.append(rec)
        log(f"  条款@{clause_rank or '-'} 政策@{policy_rank or '-'} | {judged['status']}")

    total = len(records) or 1
    m = {
        "n": len(records),
        "clause_recall_at_1": round(sum(1 for r in records if r["clause_rank"] == 1) / total, 4),
        "clause_recall_at_5": round(sum(1 for r in records if r["clause_rank"] is not None) / total, 4),
        "policy_recall_at_5": round(sum(1 for r in records if r["policy_rank"] is not None) / total, 4),
        "answer_accuracy": round(sum(1 for r in records if r["judge"]["status"] == "CORRECT") / total, 4),
        "statuses": dict(Counter(r["judge"]["status"] for r in records)),
    }
    out = {
        "built_at": date.today().isoformat(),
        "backend": backend.describe(),
        "metrics": m,
        "items": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    backend.close()

    print("\n" + "=" * 60)
    print(f"样本 {m['n']} | 条款 Recall@1 {m['clause_recall_at_1']:.1%} | Recall@5 {m['clause_recall_at_5']:.1%}")
    print(f"政策 Recall@5 {m['policy_recall_at_5']:.1%} | 回答正确率 {m['answer_accuracy']:.1%} | {m['statuses']}")
    print(f"结果 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
