from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from .models import Clause
from .temporal import clause_valid_from, clause_valid_to, usable
from .temporal_graph import ANSWER_SYSTEM, TemporalRAGGraph, build_answer_user_prompt

METHODS = ("naive", "prompt_only", "date_rerank", "downweight", "llm_gate", "temporal_filter")
EXTENDED_METHODS = ("naive", "prompt_only", "temporal_filter", "uncertainty_filter",
                    "family_filter", "family_uncertainty")
NOISE_MODES = ("expiry_missing", "status_missing", "joint_missing", "joint_wrong_future")
END_KEYS = ("valid_to", "expiry_date", "expires_at", "end_date", "invalid_at")
RATES = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (17, 29, 43)
LABELS = ("POISONED", "CORRECT", "REFUSED", "OTHER_ERROR")

PROMPT_TEMPORAL = """\n评估日期与时效元数据随资料提供。请选择评估日期已生效且尚未失效的依据。
已废止、失效、无效、停止执行，或生效日晚于评估日期，或失效日早于评估日期的条款不可作为现行依据。
失效日等于评估日期时仍在有效期内。没有有效依据或依据不足时明确无法确定，不猜测。"""
GATE_SYSTEM = """你是政策条款时效核查员。仅判断每条资料在给定评估日能否作为有效依据，不回答用户问题，不判断相关性。
状态明确为废止、失效、无效或停止执行，或生效日晚于评估日，或失效日早于评估日：usable=false。
失效日等于评估日仍有效。日期为空时不因缺失而判过期，但仍须检查状态和另一侧日期。
只输出JSON对象：{"decisions":[{"id":"输入切片ID","usable":true,"reason":"简短理由"}]}。
必须对输入的每个ID恰好返回一次，不增加、遗漏或重复ID。"""
JUDGE_SYSTEM = """你是政策问答评估员。根据问题、标准答案、新旧证据，判断模型回答。
CORRECT：核心结论、关键数值和适用条件与现行证据及标准答案一致。仅引用旧文件但最终结论正确，不算毒化。
POISONED：核心结论或数值错误，且错误明确来自旧条款或已失效标准。
REFUSED：资料不足、无法确定，没有给出明确可判错的结论。有依据地否定某项要求不是拒答。
OTHER_ERROR：有错误结论但并非旧条款导致，例如串地区、主体错误、计算错误、无证据编造。
不要仅因答案含有旧数字就判毒化：可能是在否定旧标准。不得把业务整体不存在从单条证据沉默中推出。
只输出JSON对象：{"status":"POISONED|CORRECT|REFUSED|OTHER_ERROR","uses_old_clause":false,"reason":"100字内"}。"""


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_clause(c: Clause) -> dict:
    return {"id": c.id, "text": c.text, "metadata": copy.deepcopy(c.metadata), "score": c.score}


def load_clause(row: dict) -> Clause:
    return Clause(row["id"], row["text"], copy.deepcopy(row["metadata"]), row.get("score"))


def policy_key(c: Clause) -> str:
    key = c.metadata.get("policy_id")
    if not isinstance(key, str) or not key:
        raise ValueError(f"切片缺少policy_id: {c.id}")
    return key


def norm_quote(s: str) -> str:
    return "".join(s.split())


def validate_snapshot(snapshot: dict) -> dict:
    payload = {k: v for k, v in snapshot.items() if k != "fingerprint"}
    if fingerprint(payload) != snapshot.get("fingerprint"):
        raise ValueError("快照指纹不匹配")
    corpus = [load_clause(r) for r in snapshot["corpus"]]
    lookup = {c.id: c for c in corpus}
    if len(lookup) != len(corpus):
        raise ValueError("语料切片ID重复")
    items = snapshot["items"]
    qids = [x["id"] for x in items]
    if not qids or len(set(qids)) != len(qids):
        raise ValueError("题目ID重复或题库为空")
    if fingerprint(items) != snapshot["benchmark_fingerprint"]:
        raise ValueError("题目内容指纹不匹配")
    if set(snapshot["retrieval"]) != set(qids):
        raise ValueError("检索记录与题目ID集合不匹配")
    as_of = date.fromisoformat(snapshot["as_of"])
    old_hits = new_hits = 0
    for item in items:
        for side, expected_ok in (("old", False), ("new", True)):
            cid = item[f"{side}_clause_id"]
            c = lookup[cid]
            if c.text != item[f"{side}_clause_text"]:
                raise ValueError(f"{item['id']} {side}证据与语料不同")
            if usable(c, as_of)[0] != expected_ok:
                raise ValueError(f"{item['id']} {side}时效与题库定义冲突")
        ge, be = item["gold_evidence"], item["bait_evidence"]
        if ge["chunk_id"] != item["new_clause_id"]:
            raise ValueError(f"{item['id']} 金标ID错位")
        span = ge["char_span"]
        if (not isinstance(span, list) or len(span) != 2
                or not all(type(v) is int for v in span)
                or not 0 <= span[0] < span[1] <= len(item["new_clause_text"])):
            raise ValueError(f"{item['id']} 引文位置不合法")
        if norm_quote(item["new_clause_text"][span[0]:span[1]]) != norm_quote(ge["quote"]):
            raise ValueError(f"{item['id']} 引文位置不吻合")
        if not norm_quote(be["a_quote"]) or norm_quote(be["a_quote"]) not in norm_quote(item["old_clause_text"]):
            raise ValueError(f"{item['id']} 毒饵引文不吻合")
        rows = snapshot["retrieval"][item["id"]]
        ids = [r["id"] for r in rows]
        if len(ids) != snapshot["top_k"] or len(set(ids)) != len(ids):
            raise ValueError(f"{item['id']} Top-K数量/ID异常")
        for row in rows:
            c = lookup[row["id"]]
            if row["text"] != c.text or row["metadata"] != c.metadata:
                raise ValueError(f"{item['id']} 检索快照与语料不同")
        old_hits += item["old_clause_id"] in ids
        new_hits += item["new_clause_id"] in ids
    return {"n": len(items), "chunks": len(corpus), "old_recalled": old_hits, "new_recalled": new_hits,
            "regions": sorted({it["region"] for it in items})}


def eligible_policies(snapshot: dict) -> list[str]:
    as_of = date.fromisoformat(snapshot["as_of"])
    pool = set()
    for row in snapshot["corpus"]:
        c = load_clause(row)
        vf = clause_valid_from(c)
        if not usable(c, as_of)[0] and (vf is None or vf <= as_of):
            pool.add(policy_key(c))
    return sorted(pool)


def eligible_policies_in_retrieval(snapshot: dict) -> list[str]:
    """只取真正出现在31题Top-K中的失效政策，避免把无关政策计入注入池。"""
    as_of = date.fromisoformat(snapshot["as_of"])
    corpus = {r["id"]: load_clause(r) for r in snapshot["corpus"]}
    pool = set()
    for rows in snapshot["retrieval"].values():
        for row in rows:
            c = corpus[row["id"]]
            vf = clause_valid_from(c)
            if not usable(c, as_of)[0] and (vf is None or vf <= as_of):
                pool.add(policy_key(c))
    return sorted(pool)


def noise_selection(pool: list[str], seed: int, rate: float) -> list[str]:
    if not 0 <= rate <= 1 or len(set(pool)) != len(pool):
        raise ValueError("噪声比例或政策集合不合法")
    order = sorted(pool, key=lambda pid: (fingerprint([seed, pid]), pid))
    return order[:math.floor(rate * len(order))]


def corrupt_docs(docs: list[Clause], mode: str, selected: list[str], as_of: date) -> tuple[list[Clause], list[dict]]:
    if mode not in NOISE_MODES:
        raise ValueError(f"未知噪声: {mode}")
    selected_set = set(selected)
    output, changes = [], []
    for original in docs:
        c = load_clause(dump_clause(original))
        vf = clause_valid_from(c)
        if policy_key(c) in selected_set and not usable(c, as_of)[0] and (vf is None or vf <= as_of):
            before = copy.deepcopy(c.metadata)
            if mode in ("expiry_missing", "joint_missing", "joint_wrong_future"):
                for key in END_KEYS:
                    c.metadata.pop(key, None)
            if mode in ("status_missing", "joint_missing"):
                c.metadata.pop("status", None)
            if mode == "joint_wrong_future":
                c.metadata.update(status="现行有效", valid_to="2099-12-31")
            changed = {k: {"before_present": k in before, "before": before.get(k),
                           "after_present": k in c.metadata, "after": c.metadata.get(k)}
                       for k in set(before) | set(c.metadata) if (k in before, before.get(k)) != (k in c.metadata, c.metadata.get(k))}
            if changed:
                changes.append({"id": c.id, "policy_id": policy_key(c), "fields": changed})
        output.append(c)
    return output, changes


def metadata_context(docs: list[Clause]) -> str:
    rows = [{"id": c.id, "policy_name": c.policy_name, "article": c.article,
             "status": c.metadata.get("status", ""),
             "valid_from": str(clause_valid_from(c) or ""), "valid_to": str(clause_valid_to(c) or ""),
             "text": c.text} for c in docs]
    return json.dumps(rows, ensure_ascii=False)


def gate_messages(docs: list[Clause], as_of: date) -> list[dict]:
    return [{"role": "system", "content": GATE_SYSTEM},
            {"role": "user", "content": f"评估日：{as_of.isoformat()}\n资料：{metadata_context(docs)}"}]


def validate_gate(raw: dict, docs: list[Clause]) -> list[dict]:
    rows = raw.get("decisions") if isinstance(raw, dict) else None
    if not isinstance(rows, list) or len(rows) != len(docs):
        raise ValueError("LLM门禁未覆盖全部切片")
    ids = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or type(row.get("usable")) is not bool:
            raise ValueError("LLM门禁返回非法ID或布尔值")
        ids.append(row["id"])
    if len(set(ids)) != len(ids) or set(ids) != {c.id for c in docs}:
        raise ValueError("LLM门禁返回重复、遗漏或未知ID")
    by_id = {r["id"]: r for r in rows}
    return [by_id[c.id] for c in docs]


def prepare_answer(method: str, question: str, docs: list[Clause], as_of: date,
                   decisions: list[dict] | None = None, family: Any = None) -> dict:
    if method not in METHODS + EXTENDED_METHODS:
        raise ValueError(f"未知方法: {method}")
    keep, dropped = list(docs), []
    gate_mode = {
        "temporal_filter": "filter",
        "downweight": "downweight",
        "family_filter": "filter",
        "uncertainty_filter": "uncertainty",
        "family_uncertainty": "uncertainty",
    }.get(method)
    if gate_mode:
        graph = TemporalRAGGraph(mode=gate_mode, family=family)
        state = {"as_of": as_of.isoformat(), "retrieved_docs": docs}
        checked = graph._temporal_check(state)
        keep = checked["filtered_docs"]
        if method != "downweight":
            dropped = checked["dropped"]
        if not keep:
            answer = graph._refuse({**state, **checked})["answer"]
            return {"docs": keep, "dropped": dropped, "messages": None, "answer": answer,
                    "route": "refuse", "uncertain": checked.get("uncertain_docs", [])}
    elif method == "date_rerank":
        keep = sorted(docs, key=lambda c: (clause_valid_to(c) or date.max,
                                          clause_valid_from(c) or date.min), reverse=True)
    elif method == "llm_gate":
        decisions = validate_gate({"decisions": decisions}, docs)
        flags = {r["id"]: r for r in decisions}
        keep = [c for c in docs if flags[c.id]["usable"]]
        dropped = [{"id": c.id, "reason": flags[c.id].get("reason", "")} for c in docs if not flags[c.id]["usable"]]
        if not keep:
            answer = TemporalRAGGraph()._refuse({"as_of": as_of.isoformat(), "retrieved_docs": docs, "dropped": dropped})["answer"]
            return {"docs": keep, "dropped": dropped, "messages": None, "answer": answer, "route": "refuse"}
    system = ANSWER_SYSTEM
    user = build_answer_user_prompt(question, keep)
    if method == "prompt_only":
        system += PROMPT_TEMPORAL
        user = f"评估日：{as_of.isoformat()}\n【资料】\n{metadata_context(keep)}\n\n【问题】\n{question}"
    return {"docs": keep, "dropped": dropped, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": user}], "answer": None, "route": "generate"}


def judge_messages(item: dict, answer: str) -> list[dict]:
    facts = {"question": item["question"], "ground_truth": item["ground_truth"],
             "old_clause_text": item["old_clause_text"], "new_clause_text": item["new_clause_text"],
             "bait_evidence": item["bait_evidence"], "gold_quote": item["gold_evidence"]["quote"],
             "answer": answer}
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]


def validate_judge(raw: dict) -> dict:
    if (not isinstance(raw, dict) or raw.get("status") not in LABELS
            or type(raw.get("uses_old_clause")) is not bool
            or not isinstance(raw.get("reason"), str) or not raw["reason"].strip()):
        raise ValueError("裁判输出schema非法")
    return raw


def context_audit(clean: list[Clause], kept: list[Clause], item: dict, as_of: date) -> dict:
    clean_by_id = {c.id: c for c in clean}
    ids = [c.id for c in kept]
    if len(set(ids)) != len(ids) or not set(ids) <= set(clean_by_id):
        raise ValueError("生成上下文切片ID非法")
    if any(c.text != clean_by_id[c.id].text for c in kept):
        raise ValueError("实验意外改动了正文")
    invalid = [c.id for c in clean if not usable(c, as_of)[0]]
    invalid_kept = [cid for cid in ids if cid in invalid]
    return {"retrieved_ids": [c.id for c in clean], "context_ids": ids,
            "true_invalid_retrieved": invalid, "true_invalid_kept": invalid_kept,
            "true_invalid_blocked": [cid for cid in invalid if cid not in ids],
            "old_recalled": item["old_clause_id"] in clean_by_id,
            "new_recalled": item["new_clause_id"] in clean_by_id,
            "old_in_context": item["old_clause_id"] in ids,
            "new_in_context": item["new_clause_id"] in ids}


def case_key(item: dict, condition: dict) -> str:
    return fingerprint({"item_id": item["id"], "item_fingerprint": fingerprint(item), "condition": condition})


def conditions(suite: str, noise_methods: tuple[str, ...] = ("temporal_filter",)) -> list[dict]:
    if suite == "baseline":
        return [{"suite": suite, "method": method} for method in METHODS]
    if suite == "reliability":
        return [{"suite": suite, "method": method} for method in EXTENDED_METHODS]
    if suite == "noise":
        unknown = set(noise_methods) - set(METHODS)
        if unknown:
            raise ValueError(f"噪声条件中有未知方法: {sorted(unknown)}")
        return [{"suite": suite, "method": method, "noise": mode, "rate": rate, "seed": seed}
                for method in noise_methods for mode in NOISE_MODES for seed in SEEDS for rate in RATES]
    raise ValueError(f"未知实验组: {suite}")
