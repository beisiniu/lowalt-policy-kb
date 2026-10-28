from __future__ import annotations

import copy
import json
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_eval.experiments import (  # noqa: E402
    METHODS,
    case_key,
    conditions,
    context_audit,
    corrupt_docs,
    dump_clause,
    eligible_policies,
    eligible_policies_in_retrieval,
    fingerprint,
    judge_messages,
    noise_selection,
    prepare_answer,
    validate_gate,
    validate_judge,
    validate_snapshot,
)
from rag_eval.models import Clause  # noqa: E402

AS_OF = date(2026, 10, 27)


def make_clause(cid: str, text: str, policy_id: str, status: str,
                valid_from: str = "2025-01-01", valid_to: str = "",
                policy_name: str = "", article: str = "第一条") -> Clause:
    return Clause(cid, text, {
        "policy_id": policy_id,
        "policy_name": policy_name or policy_id,
        "status": status,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "article": article,
    })


def make_item() -> dict:
    old_text = "旧条款：中型起降场补贴上限二百万元。"
    new_text = "现行条款：中型起降场补贴上限三百万元。"
    quote = "现行条款：中型起降场"
    return {
        "id": "Q-001", "region": "测试区", "question": "中型起降场补贴上限是多少？",
        "ground_truth": "三百万元", "answer_type": "numeric",
        "old_clause_id": "old-1", "new_clause_id": "new-1",
        "old_clause_text": old_text, "new_clause_text": new_text,
        "gold_evidence": {"chunk_id": "new-1", "quote": quote,
                          "char_span": [0, len(quote)], "article": "第一条",
                          "policy_name": "现行政策", "negative": False, "source_url": ""},
        "bait_evidence": {"issue": "补贴上限", "a_answer": "二百万元", "b_answer": "三百万元",
                          "a_quote": old_text[:12], "mechanism": "contradiction"},
        "old_policy": "旧政策", "new_policy": "现行政策",
    }


def base_snapshot() -> dict:
    old = make_clause("old-1", "旧条款：中型起降场补贴上限二百万元。", "P-old", "已废止",
                      "2025-01-01", "2026-01-01", "旧政策")
    new = make_clause("new-1", "现行条款：中型起降场补贴上限三百万元。", "P-new", "现行有效",
                      "2026-01-01", "", "现行政策")
    other_old = make_clause("old-2", "旧条款：另一项旧补贴。", "P-old-2", "已失效",
                            "2025-01-01", "2026-02-01", "另一旧政策")
    other_new = make_clause("new-2", "现行条款：另一项现行补贴。", "P-new-2", "现行有效",
                            "2025-01-01", "", "另一现行政策")
    future = make_clause("future-1", "未来条款：尚未生效。", "P-future", "现行有效",
                         "2027-01-01", "", "未来政策")
    item = make_item()
    snapshot = {
        "as_of": AS_OF.isoformat(), "top_k": 2,
        "items": [item],
        "corpus": [dump_clause(c) for c in (old, new, other_old, other_new, future)],
        "retrieval": {item["id"]: [dump_clause(old), dump_clause(new)]},
        "benchmark_fingerprint": "",
        "fingerprint": "",
    }
    return seal(snapshot)


def seal(snapshot: dict) -> dict:
    snapshot = copy.deepcopy(snapshot)
    snapshot["benchmark_fingerprint"] = fingerprint(snapshot["items"])
    snapshot["fingerprint"] = fingerprint({k: v for k, v in snapshot.items() if k != "fingerprint"})
    return snapshot


def docs_of(snapshot: dict, item_id: str) -> list[Clause]:
    return [Clause(r["id"], r["text"], copy.deepcopy(r["metadata"]), r.get("score"))
            for r in snapshot["retrieval"][item_id]]


class SnapshotTests(unittest.TestCase):
    def test_valid_snapshot(self):
        info = validate_snapshot(base_snapshot())
        self.assertEqual(info["n"], 1)
        self.assertEqual(info["chunks"], 5)
        self.assertEqual(info["old_recalled"], 1)
        self.assertEqual(info["new_recalled"], 1)

    def test_duplicate_item(self):
        s = base_snapshot()
        s["items"].append(copy.deepcopy(s["items"][0]))
        with self.assertRaisesRegex(ValueError, "题目ID重复"):
            validate_snapshot(seal(s))

    def test_duplicate_chunk(self):
        s = base_snapshot()
        s["corpus"].append(copy.deepcopy(s["corpus"][0]))
        with self.assertRaisesRegex(ValueError, "语料切片ID重复"):
            validate_snapshot(seal(s))

    def test_retrieval_id_set_mismatch(self):
        s = base_snapshot()
        s["retrieval"] = {}
        with self.assertRaisesRegex(ValueError, "检索记录与题目ID集合不匹配"):
            validate_snapshot(seal(s))

    def test_retrieval_text_mismatch(self):
        s = base_snapshot()
        s["retrieval"]["Q-001"][0]["text"] = "被篡改的正文"
        with self.assertRaisesRegex(ValueError, "检索快照与语料不同"):
            validate_snapshot(seal(s))

    def test_bad_span(self):
        s = base_snapshot()
        s["items"][0]["gold_evidence"]["char_span"] = [0, 999]
        with self.assertRaisesRegex(ValueError, "引文位置不合法"):
            validate_snapshot(seal(s))

    def test_quote_mismatch(self):
        s = base_snapshot()
        s["items"][0]["gold_evidence"]["quote"] = "现行条款：并不存在的句子"
        with self.assertRaisesRegex(ValueError, "引文位置不吻合"):
            validate_snapshot(seal(s))

    def test_temporal_side_inverted(self):
        s = base_snapshot()
        s["items"][0]["old_clause_id"] = "new-1"
        s["items"][0]["old_clause_text"] = s["items"][0]["new_clause_text"]
        with self.assertRaisesRegex(ValueError, "old时效与题库定义冲突"):
            validate_snapshot(seal(s))

    def test_fingerprint_tamper(self):
        s = base_snapshot()
        s["items"][0]["question"] += "被改了"
        with self.assertRaisesRegex(ValueError, "快照指纹不匹配"):
            validate_snapshot(s)
        # 重签顶层快照指纹、但不重签题目指纹：必须仍被单独拦住
        s["fingerprint"] = fingerprint({k: v for k, v in s.items() if k != "fingerprint"})
        with self.assertRaisesRegex(ValueError, "题目内容指纹不匹配"):
            validate_snapshot(s)


class NoiseTests(unittest.TestCase):
    def test_selection_reproducible_and_nested(self):
        pool = [f"P-{i}" for i in range(10)]
        low = noise_selection(pool, 17, 0.25)
        high = noise_selection(pool, 17, 0.75)
        self.assertEqual(low, noise_selection(pool, 17, 0.25))
        self.assertTrue(set(low) <= set(high))
        self.assertEqual(len(low), 2)
        self.assertEqual(len(high), 7)

    def test_eligible_pool_excludes_future(self):
        pool = eligible_policies(base_snapshot())
        self.assertEqual(pool, ["P-old", "P-old-2"])

    def test_retrieval_pool_only_includes_policies_in_topk(self):
        self.assertEqual(eligible_policies_in_retrieval(base_snapshot()), ["P-old"])

    def test_corrupt_docs_is_copy_and_policy_consistent(self):
        snapshot = base_snapshot()
        docs = docs_of(snapshot, "Q-001") + [make_clause("old-2", "旧条款：另一项旧补贴。", "P-old-2", "已失效",
                                                          "2025-01-01", "2026-02-01", "另一旧政策")]
        original = copy.deepcopy([dump_clause(c) for c in docs])
        changed, changes = corrupt_docs(docs, "expiry_missing", ["P-old"], AS_OF)
        self.assertEqual([dump_clause(c) for c in docs], original)
        self.assertEqual([c.id for c in changed], [c.id for c in docs])
        self.assertEqual([dump_clause(c) for c in changed if c.id != "old-1"],
                         [r for r in original if r["id"] != "old-1"])
        old = next(c for c in changed if c.id == "old-1")
        self.assertNotIn("valid_to", old.metadata)
        self.assertEqual(old.metadata["status"], "已废止")
        self.assertTrue(changes)
        self.assertEqual(changes[0]["policy_id"], "P-old")

    def test_single_field_noise_still_blocked_but_joint_released(self):
        snapshot = base_snapshot()
        docs = docs_of(snapshot, "Q-001")
        expiry, _ = corrupt_docs(docs, "expiry_missing", ["P-old"], AS_OF)
        status, _ = corrupt_docs(docs, "status_missing", ["P-old"], AS_OF)
        joint, _ = corrupt_docs(docs, "joint_missing", ["P-old"], AS_OF)
        wrong, _ = corrupt_docs(docs, "joint_wrong_future", ["P-old"], AS_OF)
        self.assertFalse(__import__("rag_eval.temporal", fromlist=["usable"]).usable(expiry[0], AS_OF)[0])
        self.assertFalse(__import__("rag_eval.temporal", fromlist=["usable"]).usable(status[0], AS_OF)[0])
        self.assertTrue(__import__("rag_eval.temporal", fromlist=["usable"]).usable(joint[0], AS_OF)[0])
        self.assertTrue(__import__("rag_eval.temporal", fromlist=["usable"]).usable(wrong[0], AS_OF)[0])
        self.assertEqual(wrong[0].metadata["valid_to"], "2099-12-31")

    def test_future_policy_not_corrupted(self):
        docs = [make_clause("future-1", "未来条款：尚未生效。", "P-future", "现行有效",
                            "2027-01-01", "", "未来政策")]
        changed, changes = corrupt_docs(docs, "joint_wrong_future", ["P-future"], AS_OF)
        self.assertEqual(dump_clause(changed[0]), dump_clause(docs[0]))
        self.assertEqual(changes, [])


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = base_snapshot()
        self.item = self.snapshot["items"][0]
        self.docs = docs_of(self.snapshot, "Q-001")

    def test_methods_constant(self):
        self.assertEqual(METHODS, ("naive", "prompt_only", "date_rerank", "downweight", "llm_gate", "temporal_filter"))

    def test_filter_and_downweight(self):
        filtered = prepare_answer("temporal_filter", self.item["question"], self.docs, AS_OF)
        self.assertEqual([c.id for c in filtered["docs"]], ["new-1"])
        self.assertEqual([d["id"] for d in filtered["dropped"]], ["old-1"])
        self.assertEqual(filtered["route"], "generate")
        self.assertIn("现行条款", filtered["messages"][1]["content"])
        self.assertNotIn("旧条款", filtered["messages"][1]["content"])
        weighted = prepare_answer("downweight", self.item["question"], self.docs, AS_OF)
        self.assertEqual([c.id for c in weighted["docs"]], ["new-1", "old-1"])
        self.assertEqual(weighted["dropped"], [])

    def test_no_keep_refuses(self):
        old = self.docs[0]
        filtered = prepare_answer("temporal_filter", self.item["question"], [old], AS_OF)
        self.assertEqual(filtered["route"], "refuse")
        self.assertIn("均已失效", filtered["answer"])
        self.assertIsNone(filtered["messages"])
        gated = prepare_answer("llm_gate", self.item["question"], [old], AS_OF,
                               decisions=[{"id": "old-1", "usable": False, "reason": "已废止"}])
        self.assertEqual(gated["route"], "refuse")
        self.assertIn("均已失效", gated["answer"])

    def test_prompt_only_has_metadata_and_naive_does_not(self):
        naive = prepare_answer("naive", self.item["question"], self.docs, AS_OF)
        prompt = prepare_answer("prompt_only", self.item["question"], self.docs, AS_OF)
        self.assertNotIn("评估日", naive["messages"][1]["content"])
        self.assertNotIn("valid_to", naive["messages"][1]["content"])
        self.assertIn("评估日期", prompt["messages"][0]["content"])
        self.assertIn("2026-10-27", prompt["messages"][1]["content"])
        self.assertIn("status", prompt["messages"][1]["content"])

    def test_date_rerank_differs_from_downweight_for_two_current_docs(self):
        near = make_clause("near", "近期到期现行条款。", "P-near", "现行有效", "2025-01-01", "2026-12-31")
        long = make_clause("long", "长期有效现行条款。", "P-long", "现行有效", "2025-01-01", "")
        reranked = prepare_answer("date_rerank", "问题", [near, long], AS_OF)
        downweighted = prepare_answer("downweight", "问题", [near, long], AS_OF)
        self.assertEqual([c.id for c in reranked["docs"]], ["long", "near"])
        self.assertEqual([c.id for c in downweighted["docs"]], ["near", "long"])

    def test_llm_gate_by_id_and_order(self):
        decisions = [{"id": "new-1", "usable": True, "reason": "现行"},
                     {"id": "old-1", "usable": False, "reason": "已废止"}]
        out = prepare_answer("llm_gate", self.item["question"], self.docs, AS_OF, decisions=decisions)
        self.assertEqual([c.id for c in out["docs"]], ["new-1"])
        self.assertEqual(out["dropped"][0]["id"], "old-1")


class GateJudgeAuditTests(unittest.TestCase):
    def test_gate_order_and_validation(self):
        docs = [make_clause("a", "a", "P-a", "现行有效"), make_clause("b", "b", "P-b", "现行有效")]
        rows = validate_gate({"decisions": [{"id": "b", "usable": True, "reason": "ok"},
                                            {"id": "a", "usable": False, "reason": "no"}]}, docs)
        self.assertEqual([r["id"] for r in rows], ["a", "b"])
        for bad in (
            {"decisions": [{"id": "a", "usable": True, "reason": "ok"}]},
            {"decisions": [{"id": "a", "usable": True, "reason": "ok"},
                            {"id": "a", "usable": False, "reason": "no"}]},
            {"decisions": [{"id": "a", "usable": "true", "reason": "ok"},
                            {"id": "b", "usable": True, "reason": "ok"}]},
            {"decisions": [{"id": "x", "usable": True, "reason": "ok"},
                            {"id": "b", "usable": True, "reason": "ok"}]},
        ):
            with self.assertRaises(ValueError):
                validate_gate(bad, docs)

    def test_judge_messages_are_blind_and_validate(self):
        item = make_item()
        messages = judge_messages(item, "三百万元")
        text = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("naive", text.lower())
        self.assertNotIn("temporal_filter", text.lower())
        self.assertNotIn("noise", text.lower())
        good = validate_judge({"status": "CORRECT", "uses_old_clause": False, "reason": "一致"})
        self.assertEqual(good["status"], "CORRECT")
        for bad in ({"status": "正确", "uses_old_clause": False, "reason": "x"},
                    {"status": "CORRECT", "uses_old_clause": "false", "reason": "x"},
                    {"status": "CORRECT", "uses_old_clause": False, "reason": ""}):
            with self.assertRaises(ValueError):
                validate_judge(bad)

    def test_context_audit_uses_clean_truth(self):
        snapshot = base_snapshot()
        item = snapshot["items"][0]
        clean = docs_of(snapshot, "Q-001")
        corrupted, _ = corrupt_docs(clean, "joint_missing", ["P-old"], AS_OF)
        audit = context_audit(clean, corrupted, item, AS_OF)
        self.assertTrue(audit["old_in_context"])
        self.assertTrue(audit["new_in_context"])
        self.assertEqual(audit["true_invalid_kept"], ["old-1"])
        self.assertEqual(audit["true_invalid_blocked"], [])
        filtered = prepare_answer("temporal_filter", item["question"], clean, AS_OF)
        audit2 = context_audit(clean, filtered["docs"], item, AS_OF)
        self.assertFalse(audit2["old_in_context"])
        self.assertEqual(audit2["true_invalid_blocked"], ["old-1"])
        # 污染后同一过滤器会把旧条款当成可用，印证单字段/联合噪声的行为差异
        released = prepare_answer("temporal_filter", item["question"], corrupted, AS_OF)
        released_audit = context_audit(clean, released["docs"], item, AS_OF)
        self.assertTrue(released_audit["old_in_context"])
        self.assertEqual(released_audit["true_invalid_kept"], ["old-1"])

    def test_case_key_and_conditions(self):
        item = make_item()
        self.assertNotEqual(case_key(item, {"method": "naive"}), case_key(item, {"method": "prompt_only"}))
        changed = copy.deepcopy(item)
        changed["question"] += "改"
        self.assertNotEqual(case_key(item, {"method": "naive"}), case_key(changed, {"method": "naive"}))
        self.assertEqual(len(conditions("baseline")), 6)
        self.assertEqual(len(conditions("noise")), 60)
        self.assertEqual(len({fingerprint(c) for c in conditions("noise")}), 60)
        self.assertEqual(len(conditions("noise", ("naive", "temporal_filter"))), 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
