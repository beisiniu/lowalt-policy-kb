"""对 K=1..20 扫描的已有答案补 Qwen/GLM 裁判，不重跑回答。

K 扫描原始结果在 results/ksweep/k*.json，每个 K 有 31 题 ×2 支答案。
本脚本只重判答案，输出到 rejudge/ksweep/。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(r"D:/lowalt-policy-kb_backup_20261028")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rejudge" / "ksweep"
BENCH = PROJECT / "data/national/benchmark_tq3_final.json"
LABELS = ("POISONED", "CORRECT", "REFUSED", "OTHER_ERROR")
ALIASES = {
    "毒化": "POISONED", "被毒化": "POISONED", "中毒": "POISONED",
    "正确": "CORRECT", "答对": "CORRECT",
    "拒答": "REFUSED", "无法确定": "REFUSED",
    "其他错误": "OTHER_ERROR", "其他": "OTHER_ERROR",
}
BRANCHES = (("naive", "naive_answer", "naive_judge"), ("temporal", "agent_answer", "agent_judge"))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_judge(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("裁判输出不是对象")
    status = str(raw.get("status") or "").strip().upper()
    if status not in LABELS:
        for zh, en in ALIASES.items():
            if zh in str(raw.get("status") or ""):
                status = en
                break
    if status not in LABELS:
        raise ValueError(f"非法 status: {raw.get('status')!r}")
    if type(raw.get("uses_old_clause")) is not bool:
        raise ValueError("uses_old_clause 不是布尔")
    reason = str(raw.get("reason") or "").strip()
    if not reason:
        raise ValueError("reason 为空")
    return {"status": status, "uses_old_clause": raw["uses_old_clause"], "reason": reason[:300]}


def make_key(k: int, item_id: str, branch: str, answer: str) -> str:
    return fingerprint({"suite": "ksweep", "k": k, "item_id": item_id,
                        "branch": branch, "answer": answer})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=["qwen", "glm"])
    ap.add_argument("--ks", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--max-failures", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--only-disagree", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(PROJECT))
    from policycrawler.llm import LLMError, provider_client
    from policycrawler.utils import load_env_file
    from rag_eval.experiments import judge_messages

    load_env_file(PROJECT / ".env")
    client = provider_client(args.judge)
    bench = {x["id"]: x for x in load_json(BENCH)}
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cases = []
    for k in ks:
        data = load_json(PROJECT / "results/ksweep" / f"k{k}.json")
        for it in data["items"]:
            item = bench[it["id"]]
            for branch, ans_key, judge_key in BRANCHES:
                answer = it.get(ans_key)
                ds = it.get(judge_key)
                if not answer or not isinstance(ds, dict) or ds.get("status") not in LABELS:
                    raise ValueError(f"k{k} {it['id']} {branch} 缺少答案或 DS 标签")
                key = make_key(k, it["id"], branch, answer)
                cases.append({
                    "case_key": key, "k": k, "branch": branch,
                    "item_id": it["id"], "answer": answer, "ds_judge": ds,
                    "condition": {"suite": "ksweep", "k": k, "branch": branch},
                })

    if args.only_disagree:
        kept = []
        for r in cases:
            qp = OUT / "records/qwen" / f"{r['case_key']}.json"
            if qp.exists() and load_json(qp)["judge"]["status"] == r["ds_judge"]["status"]:
                continue
            kept.append(r)
        cases = kept
        print(f"only-disagree: {len(cases)} cases", flush=True)

    if args.shard_count > 1:
        cases = [r for r in cases
                 if int(hashlib.sha256(r["case_key"].encode()).hexdigest(), 16) % args.shard_count == args.shard_index]
    if args.limit:
        cases = cases[: args.limit]

    out_dir = OUT / "records" / args.judge
    cache_dir = OUT / "cache" / args.judge
    fail_dir = OUT / "failures" / args.judge
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    done = failed = cache_hits = 0
    for i, r in enumerate(cases, 1):
        key = r["case_key"]
        out_path = out_dir / f"{key}.json"
        fail_path = fail_dir / f"{key}.json"
        if out_path.exists():
            done += 1
            continue
        if fail_path.exists() and not args.retry_failed:
            failed += 1
            continue
        item = bench[r["item_id"]]
        messages = judge_messages(item, r["answer"])
        system, user = messages[0]["content"], messages[1]["content"]
        cache_key = fingerprint({
            "provider": args.judge, "model": client.model,
            "system": system, "user": user,
            "temperature": 0.0, "max_tokens": 400,
        })
        cache_path = cache_dir / f"{cache_key}.json"
        judged = None
        if cache_path.exists():
            try:
                judged = normalize_judge(load_json(cache_path))
                cache_hits += 1
            except Exception:
                cache_path.unlink(missing_ok=True)
        if judged is None:
            last_err = ""
            for attempt in range(3):
                try:
                    raw = client.chat_json(system, user, temperature=0.0, max_tokens=400, thinking=False)
                    judged = normalize_judge(raw)
                    atomic_write(cache_path, judged)
                    break
                except (LLMError, ValueError) as e:
                    last_err = f"{type(e).__name__}: {str(e)[:240]}"
                    time.sleep(args.sleep * (attempt + 1))
            if judged is None:
                atomic_write(fail_path, {
                    "case_key": key, "k": r["k"], "branch": r["branch"],
                    "item_id": r["item_id"], "judge": args.judge,
                    "error": last_err, "created_at": now(),
                })
                failed += 1
                print(f"[FAIL {failed}] k{r['k']} {r['item_id']} {r['branch']} {args.judge}: {last_err}", flush=True)
                if failed >= args.max_failures:
                    print("达到失败上限，停止", flush=True)
                    break
                continue
        atomic_write(out_path, {
            "case_key": key, "k": r["k"], "branch": r["branch"], "item_id": r["item_id"],
            "condition": r["condition"], "answer": r["answer"], "ds_judge": r["ds_judge"],
            "judge_name": args.judge, "judge_model": client.model,
            "judge_system_version": "controlled-v1",
            "judge": judged, "created_at": now(),
        })
        fail_path.unlink(missing_ok=True)
        done += 1
        if i % 20 == 0:
            print(f"[{i}/{len(cases)}] done={done} failed={failed} cache={cache_hits} "
                  f"k{r['k']} {r['item_id']} {r['branch']} {args.judge} -> {judged['status']}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)
    print(f"FINISH {args.judge}: shard={args.shard_index}/{args.shard_count} cases={len(cases)} done={done} failed={failed} cache_hits={cache_hits}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
