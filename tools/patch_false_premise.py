"""补跑"前提错误题"并合并进已有题库。

背景：主出题脚本 build_benchmark_v2.py 在 DeepSeek 上调用前提错误题时失败——
要求 response_format=json_object 的提示词里必须出现 "json" 字样。修正提示词后用本脚本补跑，
避免重跑整批（主流程约 30 分钟）。

用法：
  python tools/patch_false_premise.py --benchmark data/national/benchmark_tq2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import build_benchmark_v2 as v2  # noqa: E402
from body_quality import CJK  # noqa: E402
from policycrawler.llm import provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402
from rag_eval.backends import build_backend  # noqa: E402
from rag_eval.temporal import clause_status, iso_or_none  # noqa: E402
import build_benchmark as bm  # noqa: E402


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(ROOT / "data" / "national" / "benchmark_tq2.json"))
    ap.add_argument("--chroma-dir", default=str(ROOT / "chroma_lowalt"))
    ap.add_argument("--as-of", default="2026-10-27")
    ap.add_argument("--per-region", type=int, default=1)
    a = ap.parse_args()

    path = Path(a.benchmark)
    items = json.loads(path.read_text(encoding="utf-8"))
    have_fp = {x["region"] for x in items if x.get("conflict_type") == "false_premise"}
    start = sum(1 for x in items if x.get("conflict_type") == "false_premise")

    backend = build_backend(argparse.Namespace(
        backend="chroma", chroma_dir=a.chroma_dir, chroma_host="", chroma_port="8000",
        collection="lowalt_policy", embedding_function="bge-zh", index_dir=str(ROOT / "kb" / "index"),
    ))
    clauses = backend.all_clauses()
    as_of = __import__("datetime").date.fromisoformat(a.as_of)
    inv_names = {c.policy_name for c in clauses if (bm.clause_valid_to(c) or __import__("datetime").date.max) < as_of}

    titles: dict[str, list[str]] = {}
    pool: dict[str, list] = {}
    for c in clauses:
        if c.policy_name in inv_names:
            continue
        ok, _ = bm.usable(c, as_of)
        if not ok:
            continue
        r = c.region or ""
        titles.setdefault(r, [])
        if c.policy_name not in titles[r]:
            titles[r].append(c.policy_name)
        pool.setdefault(r, []).append(c)

    client = provider_client("deepseek")
    added = 0
    idx = 0
    for region in sorted(titles):
        if len(pool.get(region) or []) < 2:
            continue
        basis = max(pool[region], key=lambda c: len(c.text))
        raw = v2.gen_false_premise(client, region, titles[region], basis, as_of,
                                   scene=v2.SCENES[idx % len(v2.SCENES)]); idx += 1
        if not raw:
            continue
        fake = str(raw.get("fake_name") or "").strip()
        q = str(raw.get("question") or "").strip()
        gt = str(raw.get("ground_truth") or "").strip()
        bq = str(raw.get("basis_quote") or "").strip()
        all_titles = " ".join(titles[region])
        all_text = " ".join(c.text for c in pool[region])
        if not fake or len(fake) < 4 or fake in all_titles or v2.norm(fake) in v2.norm(all_text):
            print(f"[弃用] {region}: 编造名「{fake}」不满足条件")
            continue
        if v2.norm(bq) not in v2.norm(basis.text) or len(bq) < 15:
            print(f"[弃用] {region}: 证据句不在原文或过短")
            continue
        if v2.norm(bq) in v2.norm(basis.policy_name) or v2.norm(basis.policy_name) in v2.norm(bq):
            print(f"[弃用] {region}: 证据句是标题")
            continue
        span = v2.find_span(basis.text, bq)
        start += 1
        items.append({
            "id": f"TQ2-FP{start:02d}",
            "source": "TQ2",
            "region": region,
            "topic": "前提错误",
            "conflict_type": "false_premise",
            "question": q,
            "ground_truth": gt,
            "answer_type": "refuse",
            "gold_evidence": {
                "chunk_id": basis.id,
                "policy_name": basis.policy_name,
                "article": basis.article,
                "quote": bq,
                "char_span": list(span) if span else None,
                "source_url": "",
                "negative": True,
                "fake_name": fake,
                "checked_policies": titles[region][:20],
                "note": "该地区现行政策中不存在此名称；正确行为是明确指出没有依据",
            },
            "old_clause_id": "",
            "new_clause_id": basis.id,
            "old_clause_text": "",
            "new_clause_text": basis.text,
            "old_policy": "",
            "new_policy": basis.policy_name,
            "old_status": "",
            "new_status": clause_status(basis),
            "old_valid_to": "",
            "new_valid_from": iso_or_none(bm.clause_valid_from(basis)) or "",
            "as_of": a.as_of,
            "llm_reasoning": str(raw.get("reasoning") or ""),
            "checks": {"rule": "pass", "fake_absent": True, "quote_span_found": bool(span)},
        })
        added += 1
        print(f"[+] {region} 编造名「{fake}」→ {q[:40]}")

    bm.atomic_write_json(path, items)
    print(f"\n新增前提错误题 {added} 道；题库共 {len(items)} 道 → {path}")
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
