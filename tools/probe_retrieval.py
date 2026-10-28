#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索层诊断：新版条款为什么进不了 Top-K，以及不同检索策略能救回多少。

只跑检索、不调 LLM。对每道题取 k=20 的候选，然后在几种策略下重建 Top-K，
比较"目标旧条款"与"目标新条款"的召回排名。

策略：
  base5      当前基线：原始 Top-5
  top10      单纯拉高 k 到 10
  top20      拉到 20
  quota5     政策级配额：同一 policy_name 最多占 2 个名额，凑满 5
  quota10    政策级配额，凑满 10

用法：
    python tools/probe_retrieval.py --benchmark data/national/benchmark_tq3_final.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_eval.backends import build_backend  # noqa: E402
from rag_eval.temporal import clause_matches, normalize_for_match  # noqa: E402


def rank_of(docs, target_id: str, target_text: str) -> int | None:
    for i, c in enumerate(docs, 1):
        if clause_matches(c, expected_id=target_id, expected_text=target_text):
            return i
    return None


def apply_quota(docs, k: int, per_policy: int = 2):
    """政策级配额：同一 policy_name 最多 per_policy 条，凑满 k。"""
    seen: Counter[str] = Counter()
    out = []
    for d in docs:
        key = normalize_for_match(getattr(d, "policy_name", "") or "")
        if seen[key] >= per_policy:
            continue
        seen[key] += 1
        out.append(d)
        if len(out) >= k:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="检索层诊断")
    ap.add_argument("--benchmark", default=str(ROOT / "data" / "national" / "benchmark_tq3_final.json"))
    ap.add_argument("--chroma-dir", default=str(ROOT / "chroma_lowalt"))
    ap.add_argument("--collection", default="lowalt_policy")
    ap.add_argument("--embedding-function", default="bge-zh")
    ap.add_argument("--depth", type=int, default=20, help="候选池深度")
    ap.add_argument("--out", default=str(ROOT / "data" / "national" / "retrieval_probe.json"))
    args = ap.parse_args()

    payload = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload

    backend = build_backend(argparse.Namespace(
        backend="chroma", chroma_dir=args.chroma_dir, chroma_host="", chroma_port="8000",
        collection=args.collection, embedding_function=args.embedding_function,
        index_dir=str(ROOT / "kb" / "index"),
    ))
    print(f"库：{backend.describe()}；题：{len(items)}；候选深度 k={args.depth}")

    strategies = {
        "base5": lambda docs: docs[:5],
        "top10": lambda docs: docs[:10],
        "top20": lambda docs: docs[:20],
        "quota5": lambda docs: apply_quota(docs, 5, 2),
        "quota10": lambda docs: apply_quota(docs, 10, 2),
    }
    stats = {name: {"new": [], "old": []} for name in strategies}
    rows = []

    for it in items:
        docs = backend.similarity_search(it["question"], k=args.depth)
        row = {"id": it["id"], "conflict_type": it.get("conflict_type")}
        for name, fn in strategies.items():
            picked = fn(docs)
            rn = rank_of(picked, it.get("new_clause_id", ""), it["new_clause_text"])
            ro = rank_of(picked, it.get("old_clause_id", ""), it["old_clause_text"])
            row[f"{name}_new"] = rn
            row[f"{name}_old"] = ro
            stats[name]["new"].append(rn)
            stats[name]["old"].append(ro)
        rows.append(row)

    n = len(items)
    print()
    print(f"{'策略':10} {'新版召回':>10} {'旧版召回':>10} {'新版@1':>8} {'Top5政策数':>10}")
    for name in strategies:
        rn = stats[name]["new"]
        ro = stats[name]["old"]
        hit_n = sum(1 for r in rn if r is not None)
        hit_o = sum(1 for r in ro if r is not None)
        at1 = sum(1 for r in rn if r == 1)
        print(f"{name:10} {hit_n:>4}/{n:<5} {hit_o:>4}/{n:<5} {at1:>8}")

    # 每题的 Top-5 里有多少个不同政策（看霸榜程度）
    print()
    print("=== 基线 Top-5 的政策集中度（每题的 Top-5 里有几个不同政策）===")
    dist = []
    for it in items:
        docs = backend.similarity_search(it["question"], k=5)
        names = {normalize_for_match(getattr(d, "policy_name", "") or "") for d in docs}
        dist.append(len(names))
    print("  ", dict(Counter(dist)))
    print("   1 个政策占满 Top-5 的题数:", sum(1 for x in dist if x == 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"depth": args.depth, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"\n明细 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
