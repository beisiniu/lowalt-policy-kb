#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可控实验入口：冻结检索快照、离线审计、基线/噪声运行、检查和汇总。

与 evaluate.py 分开：这里的条件、快照、缓存和续跑身份都是显式绑定的，
不会复用旧评估的 resume 或旧裁判结果。

用法：
    python tools/run_controlled_experiments.py freeze
    python tools/run_controlled_experiments.py audit --suite baseline
    python tools/run_controlled_experiments.py run --suite baseline --plan-only
    python tools/run_controlled_experiments.py run --suite baseline
    python tools/run_controlled_experiments.py check --suite baseline
    python tools/run_controlled_experiments.py summarize --suite baseline
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from policycrawler.llm import LLMError, extract_json, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402
from rag_eval.backends import build_backend  # noqa: E402
from rag_eval.experiments import (  # noqa: E402
    EXTENDED_METHODS,
    LABELS,
    METHODS,
    NOISE_MODES,
    RATES,
    SEEDS,
    case_key,
    conditions,
    context_audit,
    corrupt_docs,
    dump_clause,
    eligible_policies,
    eligible_policies_in_retrieval,
    file_digest,
    fingerprint,
    gate_messages,
    judge_messages,
    load_clause,
    noise_selection,
    prepare_answer,
    validate_gate,
    validate_judge,
    validate_snapshot,
)
from rag_eval.policy_family import build_policy_family, load_overrides  # noqa: E402
from rag_eval.temporal import usable  # noqa: E402

EXP_DIR = ROOT / "results" / "controlled_20261028"
DEFAULT_SNAPSHOT = EXP_DIR / "snapshot.json"
DEFAULT_BENCHMARK = ROOT / "data" / "national" / "benchmark_tq3_final.json"
DEFAULT_CHROMA = ROOT / "chroma_lowalt"
PROTOCOL = "controlled-v1"
WEIGHTS = {"CORRECT": 1.0, "REFUSED": 0.2, "OTHER_ERROR": 0.0, "POISONED": -1.0}


def log(msg: str) -> None:
    print(msg, flush=True)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def code_digest() -> str:
    files = (
        ROOT / "rag_eval" / "experiments.py",
        ROOT / "rag_eval" / "temporal_graph.py",
        ROOT / "rag_eval" / "validity.py",
        ROOT / "rag_eval" / "policy_family.py",
        ROOT / "tools" / "run_controlled_experiments.py",
    )
    return fingerprint({p.relative_to(ROOT).as_posix(): file_digest(p) for p in files})


def family_for(snapshot: dict) -> Any:
    clauses = [load_clause(r) for r in snapshot["corpus"]]
    return build_policy_family(clauses, load_overrides(ROOT / "data" / "national" / "validity_overrides.json"))


def load_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} 没有题目")
    ids = [str(x.get("id") or "") for x in items]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("题目ID为空或重复")
    return items


def load_snapshot(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    info = validate_snapshot(payload)
    log(f"快照校验通过：{info['n']} 题 / {info['chunks']} 切片 / "
        f"旧目标召回 {info['old_recalled']} / 新目标召回 {info['new_recalled']}")
    return payload


def snapshot_items(snapshot: dict) -> list[dict]:
    return snapshot["items"]


def snapshot_docs(snapshot: dict, item_id: str) -> list[Any]:
    return [load_clause(r) for r in snapshot["retrieval"][item_id]]


def selected_items(snapshot: dict, args: argparse.Namespace) -> list[dict]:
    items = snapshot_items(snapshot)
    if args.item_ids:
        wanted = [x.strip() for x in args.item_ids.split(",") if x.strip()]
        by_id = {x["id"]: x for x in items}
        missing = [x for x in wanted if x not in by_id]
        if missing:
            raise ValueError(f"未知题目ID：{missing}")
        items = [by_id[x] for x in wanted]
    if args.limit_items:
        items = items[: args.limit_items]
    return items


def selected_conditions(args: argparse.Namespace) -> list[dict]:
    if args.suite in ("baseline", "reliability"):
        result = conditions(args.suite)
        allowed = set(METHODS if args.suite == "baseline" else EXTENDED_METHODS)
        if args.methods:
            wanted = {x.strip() for x in args.methods.split(",") if x.strip()}
            unknown = wanted - allowed
            if unknown:
                raise ValueError(f"未知方法：{sorted(unknown)}")
            result = [c for c in result if c["method"] in wanted]
    else:
        raw_methods = args.noise_methods or "temporal_filter"
        noise_methods = tuple(x.strip() for x in raw_methods.split(",") if x.strip())
        result = conditions(args.suite, noise_methods)
        if args.modes:
            wanted = {x.strip() for x in args.modes.split(",") if x.strip()}
            unknown = wanted - set(NOISE_MODES)
            if unknown:
                raise ValueError(f"未知噪声：{sorted(unknown)}")
            result = [c for c in result if c["noise"] in wanted]
        if args.rates:
            wanted = {float(x) for x in args.rates.split(",") if x.strip()}
            unknown = wanted - set(RATES)
            if unknown:
                raise ValueError(f"未知比例：{sorted(unknown)}")
            result = [c for c in result if c["rate"] in wanted]
        if args.seeds:
            wanted = {int(x) for x in args.seeds.split(",") if x.strip()}
            unknown = wanted - set(SEEDS)
            if unknown:
                raise ValueError(f"未知种子：{sorted(unknown)}")
            result = [c for c in result if c["seed"] in wanted]
    return result


def default_run_dir(args: argparse.Namespace) -> Path:
    return Path(args.run_dir) if args.run_dir else EXP_DIR / f"run_{args.suite}"


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise RuntimeError(
                f"运行目录已有锁文件 {self.path}；确认没有其他进程后手动删除再重试"
            ) from e
        os.write(self.fd, f"pid={os.getpid()} started={now()}\n".encode("utf-8"))
        os.fsync(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except OSError:
            pass


def request_key(client: Any, messages: list[dict], *, json_mode: bool, max_tokens: int,
                thinking: bool, temperature: float) -> str:
    return fingerprint({
        "provider": client.provider,
        "style": client.style,
        "model": client.model,
        "base_url": client.base_url,
        "messages": messages,
        "json_mode": json_mode,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "temperature": temperature,
    })


def cached_chat(client: Any, messages: list[dict], cache_dir: Path, *, expected_model: str,
                json_mode: bool, max_tokens: int, thinking: bool, temperature: float,
                origin: str, sleep: float = 0.0) -> tuple[str, bool, str]:
    key = request_key(client, messages, json_mode=json_mode, max_tokens=max_tokens,
                      thinking=thinking, temperature=temperature)
    path = cache_dir / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("request_key") != key or data.get("model") != expected_model:
            raise RuntimeError(f"缓存身份不一致：{path}")
        content = str(data.get("content") or "")
        if not content.strip():
            raise RuntimeError(f"缓存内容为空：{path}")
        return content, True, key
    if client.model != expected_model:
        raise RuntimeError(f"模型已从 {expected_model} 切换为 {client.model}，停止")
    try:
        content = client.chat(messages, temperature=temperature, max_tokens=max_tokens,
                              json_mode=json_mode, thinking=thinking)
    except LLMError:
        raise
    if client.model != expected_model:
        raise RuntimeError(f"调用期间模型切换：{expected_model} -> {client.model}，拒绝写入缓存")
    if not (content or "").strip():
        raise RuntimeError("模型返回空答案")
    atomic_write_json(path, {
        "request_key": key, "model": expected_model, "provider": client.provider,
        "base_url": client.base_url, "messages_digest": fingerprint(messages),
        "json_mode": json_mode, "max_tokens": max_tokens, "thinking": thinking,
        "temperature": temperature, "content": content, "origin": origin, "created_at": now(),
    })
    if sleep:
        time.sleep(sleep)
    return content, False, key


def cached_chat_json(client: Any, messages: list[dict], cache_dir: Path, *,
                     expected_model: str, max_tokens: int, thinking: bool,
                     temperature: float, origin: str, sleep: float = 0.0) -> tuple[dict, bool, str]:
    raw, hit, key = cached_chat(client, messages, cache_dir, expected_model=expected_model,
                                json_mode=True, max_tokens=max_tokens, thinking=thinking,
                                temperature=temperature, origin=origin, sleep=sleep)
    try:
        return extract_json(raw), hit, key
    except LLMError:
        repair = [
            {"role": "system", "content": "你是 JSON 修复器。只输出修正后的合法 JSON 对象，不要任何解释。"},
            {"role": "user", "content": f"下面的内容应当是一个 JSON 对象，请修复格式后原样返回字段：\n{raw[:4000]}"},
        ]
        fixed, hit2, key2 = cached_chat(client, repair, cache_dir, expected_model=expected_model,
                                        json_mode=True, max_tokens=max_tokens, thinking=thinking,
                                        temperature=temperature, origin=origin, sleep=sleep)
        try:
            return extract_json(fixed), hit or hit2, f"{key};{key2}"
        except LLMError:
            remove_cache_keys(cache_dir, f"{key};{key2}")
            raise


def build_manifest(args: argparse.Namespace, snapshot: dict, items: list[dict],
                   conds: list[dict], clients: dict[str, Any]) -> dict:
    def client_info(client: Any) -> dict:
        return {"provider": client.provider, "model": client.model,
                "base_url": client.base_url, "style": client.style}
    return {
        "protocol": PROTOCOL,
        "suite": args.suite,
        "snapshot_fingerprint": snapshot["fingerprint"],
        "benchmark_fingerprint": snapshot["benchmark_fingerprint"],
        "as_of": snapshot["as_of"],
        "top_k": snapshot["top_k"],
        "items": [x["id"] for x in items],
        "conditions": [{"condition": c, "condition_key": fingerprint(c)} for c in conds],
        "method_protocol": ("date_rerank: valid_to DESC(null first), valid_from DESC; "
                            "llm_gate: ID-validated per-clause temporal decisions; "
                            "temporal_filter: usable() hard filter; "
                            "uncertainty_filter: valid/invalid/uncertain gate; "
                            "family_*: policy-family successor expansion before gate"),
        "family_extra": getattr(args, "family_extra", 0),
        "answer": {**client_info(clients["answer"]), "temperature": args.temperature,
                   "max_tokens": args.answer_max_tokens, "thinking": False},
        "gate": {**client_info(clients["gate"]), "temperature": args.temperature,
                 "max_tokens": args.gate_max_tokens, "thinking": False},
        "judge": {**client_info(clients["judge"]), "temperature": args.temperature,
                  "max_tokens": args.judge_max_tokens, "thinking": False},
        "noise": {"unit": "policy_id", "modes": list(NOISE_MODES), "rates": list(RATES),
                  "seeds": list(SEEDS), "pool_mode": args.pool_mode,
                  "methods": [x.strip() for x in (args.noise_methods or "temporal_filter").split(",") if x.strip()],
                  "selection": "seeded hash order, nested prefixes"},
        "code_digest": code_digest(),
        "git_head": git_head(),
        "created_at": now(),
    }


def manifest_config(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k != "created_at"}


def read_manifest(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_records(directory: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = str(data.get("case_key") or "")
        if not key or key != path.stem:
            raise ValueError(f"结果文件身份不匹配：{path}")
        if key in result:
            raise ValueError(f"结果重复：{key}")
        result[key] = data
    return result


def read_failures(directory: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = str(data.get("case_key") or "")
        if not key or key != path.stem:
            raise ValueError(f"失败记录身份不匹配：{path}")
        result[key] = data
    return result


def write_case(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def noise_for_condition(cond: dict, snapshot: dict, pool_mode: str) -> tuple[list[str], int]:
    pool = (eligible_policies_in_retrieval(snapshot) if pool_mode == "retrieval"
            else eligible_policies(snapshot))
    selected = noise_selection(pool, cond["seed"], cond["rate"])
    return selected, len(pool)


def remove_cache_keys(cache_dir: Path, keys: str) -> None:
    for key in str(keys or "").split(";"):
        if not key:
            continue
        try:
            (cache_dir / f"{key}.json").unlink()
        except FileNotFoundError:
            pass


def changed_summary(changes: list[dict]) -> dict:
    return {
        "actual_policies": sorted({x["policy_id"] for x in changes}),
        "actual_chunks": len(changes),
    }


def freeze(args: argparse.Namespace) -> int:
    snapshot_path = Path(args.snapshot)
    if snapshot_path.exists() and not args.overwrite:
        raise RuntimeError(f"快照已存在：{snapshot_path}；如需重冻结请加 --overwrite")
    source = Path(args.chroma_dir)
    if not (source / "chroma.sqlite3").exists():
        raise RuntimeError(f"Chroma 目录不存在或缺少 chroma.sqlite3：{source}")
    source_hash = file_digest(source / "chroma.sqlite3")
    temp_parent = Path(args.temp_dir)
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = temp_parent / f"chroma_freeze_{datetime.now():%Y%m%d_%H%M%S_%f}"
    items = load_items(Path(args.benchmark))
    backend = None
    started = time.time()
    try:
        shutil.copytree(source, temp_root)
        backend = build_backend(SimpleNamespace(
            backend="chroma", chroma_dir=str(temp_root), chroma_host="",
            chroma_port="8000", collection=args.collection,
            embedding_function=args.embedding_function,
            index_dir=str(ROOT / "kb" / "index"),
        ))
        clauses = backend.all_clauses()
        if not clauses:
            raise RuntimeError("检索库为空")
        corpus_ids = [c.id for c in clauses]
        if len(corpus_ids) != len(set(corpus_ids)):
            raise RuntimeError("检索库切片ID重复")
        retrieval: dict[str, list[dict]] = {}
        for i, item in enumerate(items, 1):
            docs = backend.similarity_search(item["question"], k=args.top_k)
            ids = [c.id for c in docs]
            if len(docs) != args.top_k or len(ids) != len(set(ids)):
                raise RuntimeError(f"{item['id']} 检索结果数量/ID异常")
            retrieval[item["id"]] = [dump_clause(c) for c in docs]
            log(f"[{i}/{len(items)}] {item['id']} {item['question'][:36]}")
        snapshot = {
            "as_of": args.as_of,
            "top_k": args.top_k,
            "items": items,
            "corpus": [dump_clause(c) for c in clauses],
            "retrieval": retrieval,
            "benchmark_fingerprint": fingerprint(items),
            "source": {
                "chroma_dir": str(source.resolve()),
                "collection": args.collection,
                "embedding_function": args.embedding_function,
                "count": len(clauses),
                "sqlite_sha256": source_hash,
                "git_head": git_head(),
                "code_digest": code_digest(),
                "frozen_at": now(),
            },
        }
        snapshot["fingerprint"] = fingerprint({k: v for k, v in snapshot.items() if k != "fingerprint"})
        info = validate_snapshot(snapshot)
        if source_hash != file_digest(source / "chroma.sqlite3"):
            raise RuntimeError("冻结期间源 Chroma 被修改")
        if snapshot_path.exists():
            backup = snapshot_path.with_name(f"snapshot_backup_{datetime.now():%Y%m%d_%H%M%S}.json")
            shutil.move(str(snapshot_path), str(backup))
            log(f"旧快照已备份：{backup}")
        atomic_write_json(snapshot_path, snapshot)
        atomic_write_json(snapshot_path.with_name("freeze_log.json"), {
            "created_at": now(), "elapsed_s": round(time.time() - started, 2),
            "snapshot_fingerprint": snapshot["fingerprint"], "info": info,
            "source_sqlite_sha256": source_hash, "temp_dir": str(temp_root),
            "kept_temp": bool(args.keep_temp),
        })
        log(f"快照写入：{snapshot_path}")
        log(f"题目 {info['n']} / 切片 {info['chunks']} / 旧目标召回 {info['old_recalled']} / "
            f"新目标召回 {info['new_recalled']}")
    finally:
        if backend is not None:
            backend.close()
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
    return 0


def audit(args: argparse.Namespace) -> int:
    snapshot = load_snapshot(Path(args.snapshot))
    items = selected_items(snapshot, args)
    conds = selected_conditions(args)
    as_of = date.fromisoformat(snapshot["as_of"])
    pool_mode = args.pool_mode
    pool = (eligible_policies_in_retrieval(snapshot) if pool_mode == "retrieval"
            else eligible_policies(snapshot))
    family = family_for(snapshot) if args.suite == "reliability" else None
    out_cases: list[dict] = []
    summaries: list[dict] = []
    for cond in conds:
        selected: list[str] = []
        pool_size = 0
        if cond["suite"] == "noise":
            selected, pool_size = noise_for_condition(cond, snapshot, pool_mode)
        cases = []
        for item in items:
            clean = snapshot_docs(snapshot, item["id"])
            changed = []
            family_added = []
            if cond["suite"] == "noise":
                docs, changed = corrupt_docs(clean, cond["noise"], selected, as_of)
            else:
                docs = clean
            if cond["suite"] == "reliability" and cond["method"].startswith("family_"):
                ex = family.expand(clean, item["question"], as_of, max_extra=args.family_extra)
                family_added = [{"id": x.clause.id, "old_policy_id": x.old_policy_id,
                                 "new_policy_id": x.new_policy_id, "score": x.score}
                                for x in ex]
                docs = clean + [x.clause for x in ex]
            retrieved_clean = clean if not family_added else docs
            if cond["method"] == "llm_gate":
                cases.append({
                    "item_id": item["id"], "case_key": case_key(item, cond),
                    "requires_api": True, "noise": None,
                })
                continue
            prep = prepare_answer(cond["method"], item["question"], docs, as_of, family=family)
            audit_info = context_audit(retrieved_clean, prep["docs"], item, as_of)
            cases.append({
                "item_id": item["id"], "case_key": case_key(item, cond),
                "requires_api": False, "method": cond["method"],
                "family_added": family_added,
                "noise": ({"mode": cond["noise"], "rate": cond["rate"], "seed": cond["seed"],
                           "pool_size": pool_size, "selected_policies": selected,
                           **changed_summary(changed)} if cond["suite"] == "noise" else None),
                "route": prep["route"], "audit": audit_info,
            })
        valid = [x for x in cases if not x["requires_api"]]
        summaries.append({
            "condition": cond, "condition_key": fingerprint(cond),
            "pool_size": pool_size, "selected_policies": selected,
            "n_cases": len(cases), "n_requires_api": len(cases) - len(valid),
            "n_valid": len(valid),
            "old_in_context": sum(1 for x in valid if x["audit"]["old_in_context"]),
            "new_in_context": sum(1 for x in valid if x["audit"]["new_in_context"]),
            "true_invalid_kept": sum(len(x["audit"]["true_invalid_kept"]) for x in valid),
            "true_invalid_blocked": sum(len(x["audit"]["true_invalid_blocked"]) for x in valid),
            "family_added": sum(len(x.get("family_added") or []) for x in valid),
        })
        out_cases.extend(cases)
        log(f"{cond} -> {len(cases)} 条，离线可知 {len(valid)} 条")
    out = {
        "created_at": now(), "snapshot_fingerprint": snapshot["fingerprint"],
        "as_of": snapshot["as_of"], "pool_mode": pool_mode, "pool_size": len(pool),
        "conditions": summaries, "cases": out_cases,
    }
    out_path = Path(args.out or (EXP_DIR / "audit" / f"audit_{args.suite}.json"))
    atomic_write_json(out_path, out)
    valid_summaries = [x for x in summaries if x["n_valid"]]
    if valid_summaries:
        log("\n条件摘要：")
        for x in valid_summaries:
            log(f"  {x['condition']} 旧在上下文 {x['old_in_context']} / "
                f"无效切片入上下文 {x['true_invalid_kept']} / 阻断 {x['true_invalid_blocked']}")
    log(f"审计写入：{out_path}")
    return 0


def run(args: argparse.Namespace) -> int:
    load_env_file(ROOT / ".env")
    snapshot = load_snapshot(Path(args.snapshot))
    items = selected_items(snapshot, args)
    conds = selected_conditions(args)
    run_dir = default_run_dir(args)
    records_dir, failures_dir, cache_dir = run_dir / "records", run_dir / "failures", run_dir / "cache"
    for d in (run_dir, records_dir, failures_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    answer_client = provider_client(args.provider, model=args.answer_model or None)
    gate_client = provider_client(args.provider, model=args.answer_model or None)
    judge_client = provider_client(args.judge_provider, model=args.judge_model or None)
    clients = {"answer": answer_client, "gate": gate_client, "judge": judge_client}
    manifest = build_manifest(args, snapshot, items, conds, clients)
    manifest_path = run_dir / "manifest.json"
    old = read_manifest(manifest_path)
    if old is not None and manifest_config(old) != manifest_config(manifest):
        raise RuntimeError("运行目录已有不同配置的 manifest；请换 --run-dir 或先核对")
    if old is None:
        atomic_write_json(manifest_path, manifest)
    records = read_records(records_dir)
    failures = read_failures(failures_dir)
    as_of = date.fromisoformat(snapshot["as_of"])
    family = family_for(snapshot) if args.suite == "reliability" else None
    expected: list[tuple[dict, dict, str]] = []
    for cond in conds:
        for item in items:
            expected.append((cond, item, case_key(item, cond)))
    if args.max_cases:
        expected = expected[: args.max_cases]
    log(f"运行目录：{run_dir}")
    log(f"模型：答案={answer_client.model} 门禁={gate_client.model} 裁判={judge_client.model}")
    log(f"条件 {len(conds)} / 题目 {len(items)} / 计划 {len(expected)} 条；"
        f"已完成 {len(records)}，失败 {len(failures)}")
    if args.plan_only:
        for cond, item, key in expected:
            log(f"  {key[:10]} {cond} {item['id']}")
        return 0
    failure_count = 0
    with RunLock(run_dir / "run.lock"):
        for index, (cond, item, key) in enumerate(expected, 1):
            if key in records:
                continue
            if key in failures and not args.retry_failed:
                continue
            clean = snapshot_docs(snapshot, item["id"])
            selected: list[str] = []
            pool_size = 0
            changes: list[dict] = []
            family_added: list[dict] = []
            if cond["suite"] == "noise":
                selected, pool_size = noise_for_condition(cond, snapshot, args.pool_mode)
                docs, changes = corrupt_docs(clean, cond["noise"], selected, as_of)
                noise = {"mode": cond["noise"], "rate": cond["rate"], "seed": cond["seed"],
                         "pool_mode": args.pool_mode, "pool_size": pool_size,
                         "selected_policies": selected, **changed_summary(changes)}
            else:
                docs, noise = clean, None
            if cond["suite"] == "reliability" and cond["method"].startswith("family_"):
                ex = family.expand(clean, item["question"], as_of, max_extra=args.family_extra)
                family_added = [{"id": x.clause.id, "old_policy_id": x.old_policy_id,
                                 "new_policy_id": x.new_policy_id, "score": x.score}
                                for x in ex]
                docs = clean + [x.clause for x in ex]
            retrieved_clean = docs if family_added else clean
            stage = "gate"
            try:
                decisions = None
                gate_info = None
                if cond["method"] == "llm_gate":
                    raw, gate_hit, gate_key = cached_chat_json(
                        gate_client, gate_messages(docs, as_of), cache_dir,
                        expected_model=gate_client.model, max_tokens=args.gate_max_tokens,
                        thinking=False, temperature=args.temperature, origin=key,
                    )
                    try:
                        decisions = validate_gate(raw, docs)
                    except ValueError:
                        remove_cache_keys(cache_dir, gate_key)
                        raise
                    gate_info = {"decisions": decisions, "request_key": gate_key,
                                 "cache_hit": gate_hit}
                stage = "prepare"
                prep = prepare_answer(cond["method"], item["question"], docs, as_of,
                                      decisions=decisions, family=family)
                stage = "answer"
                if prep["route"] == "refuse":
                    answer = prep["answer"]
                    answer_source = "refusal"
                    answer_key = None
                    answer_hit = False
                else:
                    answer, answer_hit, answer_key = cached_chat(
                        answer_client, prep["messages"], cache_dir,
                        expected_model=answer_client.model, json_mode=False,
                        max_tokens=args.answer_max_tokens, thinking=False,
                        temperature=args.temperature, origin=key, sleep=args.sleep,
                    )
                    answer_source = "model"
                stage = "judge"
                judged, judge_hit, judge_key = cached_chat_json(
                    judge_client, judge_messages(item, answer), cache_dir,
                    expected_model=judge_client.model, max_tokens=args.judge_max_tokens,
                    thinking=False, temperature=args.temperature, origin=key,
                    sleep=args.sleep,
                )
                try:
                    judged = validate_judge(judged)
                except ValueError:
                    remove_cache_keys(cache_dir, judge_key)
                    raise
                stage = "audit"
                audit_info = context_audit(retrieved_clean, prep["docs"], item, as_of)
                record = {
                    "case_key": key, "protocol": PROTOCOL,
                    "item_id": item["id"], "item_fingerprint": fingerprint(item),
                    "condition": cond, "condition_key": fingerprint(cond),
                    "snapshot_fingerprint": snapshot["fingerprint"],
                    "noise": noise, "family_added": family_added, "gate": gate_info,
                    "answer": answer, "answer_source": answer_source,
                    "judge": judged, "audit": audit_info,
                    "request": {"answer_key": answer_key, "judge_key": judge_key,
                                "answer_cache_hit": answer_hit, "judge_cache_hit": judge_hit},
                    "created_at": now(),
                }
                write_case(records_dir / f"{key}.json", record)
                records[key] = record
                failures.pop(key, None)
                try:
                    (failures_dir / f"{key}.json").unlink()
                except FileNotFoundError:
                    pass
                log(f"[{index}/{len(expected)}] {item['id']} {cond} -> "
                    f"{judged['status']} (无效入上下文 {len(audit_info['true_invalid_kept'])})")
            except (LLMError, RuntimeError, ValueError, OSError) as e:
                failure = {
                    "case_key": key, "item_id": item["id"], "condition": cond,
                    "condition_key": fingerprint(cond), "stage": stage,
                    "error": f"{type(e).__name__}: {str(e)[:500]}", "created_at": now(),
                }
                write_case(failures_dir / f"{key}.json", failure)
                failures[key] = failure
                failure_count += 1
                log(f"[FAIL {failure_count}] {item['id']} {cond} @ {stage}: {e}")
                if failure_count >= args.max_failures:
                    log(f"失败达到 {failure_count}，停止；修复后带 --retry-failed 续跑")
                    break
    completed = sum(1 for _, _, key in expected if key in records)
    log(f"完成 {completed}/{len(expected)}；失败记录 {len(failures)}；结果目录 {run_dir}")
    return 0 if not failures else 2


def check(args: argparse.Namespace) -> int:
    snapshot = load_snapshot(Path(args.snapshot))
    items_by_id = {x["id"]: x for x in snapshot_items(snapshot)}
    run_dir = default_run_dir(args)
    manifest = read_manifest(run_dir / "manifest.json")
    if manifest is not None:
        conds = [x["condition"] for x in manifest.get("conditions", [])]
        items = [items_by_id[x] for x in manifest.get("items", [])]
    else:
        conds = conditions(args.suite) if args.suite else conditions("baseline") + conditions("noise")
        items = list(items_by_id.values())
    expected_keys = {case_key(item, cond): (cond, item) for cond in conds for item in items}
    records = read_records(run_dir / "records")
    failures = read_failures(run_dir / "failures")
    problems: list[str] = []
    for key, record in records.items():
        if key not in expected_keys:
            problems.append(f"未知 result key：{key}")
            continue
        cond, item = expected_keys[key]
        if record.get("item_id") != item["id"]:
            problems.append(f"{key}: item_id 错位")
        if record.get("item_fingerprint") != fingerprint(item):
            problems.append(f"{key}: 题内容指纹错位")
        if record.get("condition_key") != fingerprint(cond):
            problems.append(f"{key}: 条件指纹错位")
        if record.get("snapshot_fingerprint") != snapshot["fingerprint"]:
            problems.append(f"{key}: 快照指纹错位")
        if record.get("judge", {}).get("status") not in LABELS:
            problems.append(f"{key}: 裁判状态非法")
        if not str(record.get("answer") or "").strip():
            problems.append(f"{key}: 回答为空")
        audit_info = record.get("audit") or {}
        base_ids = [r["id"] for r in snapshot["retrieval"].get(item["id"], [])]
        added_ids = [str(x.get("id") or "") for x in (record.get("family_added") or [])]
        expected_ids = base_ids + added_ids
        if audit_info.get("retrieved_ids") != expected_ids:
            problems.append(f"{key}: 检索快照/族谱补入ID错位")
        if len(set(expected_ids)) != len(expected_ids):
            problems.append(f"{key}: 族谱补入出现重复ID")
        if added_ids and not set(added_ids) <= set(audit_info.get("context_ids") or []):
            problems.append(f"{key}: 族谱补入条款未进入上下文")
        if not set(audit_info.get("context_ids") or []) <= set(audit_info.get("retrieved_ids") or []):
            problems.append(f"{key}: 上下文ID越界")
        if audit_info.get("old_in_context") != (item["old_clause_id"] in (audit_info.get("context_ids") or [])):
            problems.append(f"{key}: old_in_context不一致")
        if audit_info.get("new_in_context") != (item["new_clause_id"] in (audit_info.get("context_ids") or [])):
            problems.append(f"{key}: new_in_context不一致")
    missing = set(expected_keys) - set(records) - set(failures)
    if missing and not args.allow_incomplete:
        problems.append(f"缺少 {len(missing)} 条结果（加 --allow-incomplete 可只报告）")
    log(f"记录 {len(records)} / 失败 {len(failures)} / 预期 {len(expected_keys)} / 缺少 {len(missing)}")
    if problems:
        for p in problems[:50]:
            log(f"[BAD] {p}")
        if len(problems) > 50:
            log(f"... 其余 {len(problems) - 50} 条")
        return 1
    log("检查通过：记录身份、上下文、题号和裁判字段一致")
    return 0


def summarize(args: argparse.Namespace) -> int:
    run_dir = default_run_dir(args)
    records = list(read_records(run_dir / "records").values())
    failures = read_failures(run_dir / "failures")
    if not records:
        raise RuntimeError(f"{run_dir} 没有结果，无法汇总")
    if args.suite in ("baseline", "reliability"):
        methods = METHODS if args.suite == "baseline" else EXTENDED_METHODS
        groups = []
        naive = {r["item_id"]: r["judge"]["status"] for r in records
                 if r["condition"]["suite"] == args.suite and r["condition"]["method"] == "naive"}
        for method in methods:
            rs = [r for r in records if r["condition"]["suite"] == args.suite
                  and r["condition"]["method"] == method]
            if not rs:
                continue
            st = {k: sum(1 for r in rs if r["judge"]["status"] == k) for k in LABELS}
            n = len(rs)
            poisoned_naive = [i for i, s in naive.items() if s == "POISONED"]
            strict = sum(1 for r in rs if r["item_id"] in poisoned_naive
                         and r["judge"]["status"] == "CORRECT")
            groups.append({
                "method": method, "n": n, **st,
                "poison_rate": round(st["POISONED"] / n, 4),
                "correct_rate": round(st["CORRECT"] / n, 4),
                "refuse_rate": round(st["REFUSED"] / n, 4),
                "weighted_score": round(sum(WEIGHTS[r["judge"]["status"]] for r in rs) / n, 4),
                "old_in_context": sum(1 for r in rs if r["audit"]["old_in_context"]),
                "new_in_context": sum(1 for r in rs if r["audit"]["new_in_context"]),
                "invalid_kept": sum(len(r["audit"]["true_invalid_kept"]) for r in rs),
                "family_added": sum(len(r.get("family_added") or []) for r in rs),
                "strict_rescue": (round(strict / len(poisoned_naive), 4) if poisoned_naive else None),
            })
        out = {"suite": args.suite, "run_dir": str(run_dir), "records": len(records),
               "failures": len(failures), "groups": groups}
        atomic_write_json(run_dir / "summary.json", out)
        title = "# 基线对照汇总" if args.suite == "baseline" else "# 时序可靠性扩展汇总"
        lines = [title, "",
                 f"记录 {len(records)}，失败 {len(failures)}；开发轮单裁判，未作最终论文主表。", "",
                 "| 方法 | n | 毒化 | 正确 | 拒答 | 其他 | 加权分 | 旧在上下文 | 无效入上下文 | 族谱补入 | 严格拯救 |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in groups:
            rescue = ("—" if g["method"] == "naive" or g["strict_rescue"] is None
                      else f"{g['strict_rescue']:.1%}")
            lines.append(f"| {g['method']} | {g['n']} | {g['poison_rate']:.1%} | "
                         f"{g['correct_rate']:.1%} | {g['refuse_rate']:.1%} | "
                         f"{g['OTHER_ERROR']} | {g['weighted_score']:+.3f} | "
                         f"{g['old_in_context']} | {g['invalid_kept']} | "
                         f"{g['family_added']} | {rescue} |")
        (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"汇总写入：{run_dir / 'summary.md'}")
        return 0
    groups = []
    for method in METHODS:
        for mode in NOISE_MODES:
            for rate in RATES:
                rs = [r for r in records if r["condition"]["suite"] == "noise"
                      and r["condition"]["method"] == method
                      and r["condition"]["noise"] == mode and r["condition"]["rate"] == rate]
                if not rs:
                    continue
                st = {k: sum(1 for r in rs if r["judge"]["status"] == k) for k in LABELS}
                n = len(rs)
                actual = sorted({p for r in rs for p in (r.get("noise") or {}).get("actual_policies", [])})
                groups.append({
                    "method": method, "mode": mode, "rate": rate, "n": n, **st,
                    "actual_policies": len(actual),
                    "actual_chunks": sum((r.get("noise") or {}).get("actual_chunks", 0) for r in rs),
                    "poison_rate": round(st["POISONED"] / n, 4),
                    "correct_rate": round(st["CORRECT"] / n, 4),
                    "refuse_rate": round(st["REFUSED"] / n, 4),
                    "weighted_score": round(sum(WEIGHTS[r["judge"]["status"]] for r in rs) / n, 4),
                    "old_in_context": sum(1 for r in rs if r["audit"]["old_in_context"]),
                    "invalid_kept": sum(len(r["audit"]["true_invalid_kept"]) for r in rs),
                })
    out = {"suite": "noise", "run_dir": str(run_dir), "records": len(records),
           "failures": len(failures), "groups": groups}
    atomic_write_json(run_dir / "summary.json", out)
    lines = ["# 元数据噪声汇总", "",
             f"记录 {len(records)}，失败 {len(failures)}；开发轮单裁判，未作最终论文主表。", "",
             "| 方法 | 噪声 | 比例 | n | 毒化 | 正确 | 拒答 | 加权分 | 实际政策 | 实际切片 | 无效入上下文 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g['method']} | {g['mode']} | {g['rate']:.0%} | {g['n']} | "
                     f"{g['poison_rate']:.1%} | {g['correct_rate']:.1%} | {g['refuse_rate']:.1%} | "
                     f"{g['weighted_score']:+.3f} | {g['actual_policies']} | {g['actual_chunks']} | "
                     f"{g['invalid_kept']} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"汇总写入：{run_dir / 'summary.md'}")
    return 0


def add_common(parser: argparse.ArgumentParser, *, run: bool = False) -> None:
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--suite", choices=["baseline", "noise", "reliability"], default="baseline")
    parser.add_argument("--item-ids", default="")
    parser.add_argument("--limit-items", type=int, default=0)
    parser.add_argument("--methods", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--rates", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--pool-mode", choices=["retrieval", "corpus"], default="retrieval",
                        help="噪声注入池：retrieval=31题Top-K中实际出现的失效政策（主口径）；"
                             "corpus=库内全部失效政策（敏感性口径）")
    parser.add_argument("--noise-methods", default="temporal_filter",
                        help="噪声实验参与的方法，逗号分隔；默认仅temporal_filter")
    parser.add_argument("--family-extra", type=int, default=2,
                        help="reliability 中每条旧政策最多补入的继任条款数")
    if run:
        parser.add_argument("--run-dir", default="")
        parser.add_argument("--provider", default="deepseek")
        parser.add_argument("--judge-provider", default="deepseek")
        parser.add_argument("--answer-model", default="")
        parser.add_argument("--judge-model", default="")
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--answer-max-tokens", type=int, default=1200)
        parser.add_argument("--gate-max-tokens", type=int, default=1200)
        parser.add_argument("--judge-max-tokens", type=int, default=400)
        parser.add_argument("--sleep", type=float, default=0.1)
        parser.add_argument("--max-failures", type=int, default=5)
        parser.add_argument("--max-cases", type=int, default=0)
        parser.add_argument("--retry-failed", action="store_true")
        parser.add_argument("--plan-only", action="store_true")


def main() -> int:
    ap = argparse.ArgumentParser(description="可控基线/元数据噪声实验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("freeze", help="冻结一次检索快照")
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA))
    p.add_argument("--collection", default="lowalt_policy")
    p.add_argument("--embedding-function", default="bge-zh")
    p.add_argument("--temp-dir", default=str(ROOT / "data" / "_tmp"))
    p.add_argument("--top-k", type=int, default=9)
    p.add_argument("--as-of", default="2026-10-27")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-temp", action="store_true")

    p = sub.add_parser("audit", help="离线审计条件/上下文/噪声（不调用API）")
    add_common(p)
    p.add_argument("--out", default="", help="默认 audit_<suite>.json")

    p = sub.add_parser("run", help="运行基线或噪声实验")
    add_common(p, run=True)

    p = sub.add_parser("check", help="检查结果身份/上下文/覆盖（不调用API）")
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.add_argument("--suite", choices=["baseline", "noise", "reliability"], default="baseline")
    p.add_argument("--run-dir", default="")
    p.add_argument("--allow-incomplete", action="store_true")

    p = sub.add_parser("summarize", help="汇总 baseline/noise")
    p.add_argument("--suite", choices=["baseline", "noise", "reliability"], required=True)
    p.add_argument("--run-dir", default="")

    args = ap.parse_args()
    try:
        if args.cmd == "freeze":
            return freeze(args)
        if args.cmd == "audit":
            return audit(args)
        if args.cmd == "run":
            return run(args)
        if args.cmd == "check":
            return check(args)
        if args.cmd == "summarize":
            return summarize(args)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as e:
        log(f"[ERROR] {type(e).__name__}: {e}")
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
