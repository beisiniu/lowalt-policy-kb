from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_eval.experiments import (  # noqa: E402
    context_audit,
    corrupt_docs,
    eligible_policies_in_retrieval,
    load_clause,
    prepare_answer,
)
from rag_eval.models import Clause  # noqa: E402
from rag_eval.policy_family import (  # noqa: E402
    FamilyEdge,
    PolicyFamily,
    PolicyNode,
    build_policy_family,
    load_overrides,
)
from rag_eval.temporal_graph import TemporalRAGGraph  # noqa: E402
from rag_eval.validity import INVALID, UNCERTAIN, VALID, assess_validity  # noqa: E402

AS_OF = date(2026, 10, 27)


def C(cid: str, text: str, pid: str, status: str, vf: str = "2025-01-01",
      vt: str = "", name: str = "", region: str = "测试区") -> Clause:
    return Clause(cid, text, {
        "policy_id": pid, "policy_name": name or pid, "region": region,
        "status": status, "valid_from": vf, "valid_to": vt, "article": "第一条",
    })


def node(pid: str, name: str, status: str, vf: str = "2025-01-01",
         vt: str = "", url: str = "") -> PolicyNode:
    return PolicyNode(pid, name, "测试区", status,
                      date.fromisoformat(vf) if vf else None,
                      date.fromisoformat(vt) if vt else None,
                      url)


class ValidityTests(unittest.TestCase):
    def test_clean_states(self):
        old = C("old", "旧条款", "P-old", "已废止", vt="2026-01-01")
        new = C("new", "新条款", "P-new", "现行有效", vt="2028-01-01")
        self.assertEqual(assess_validity(old, AS_OF).state, INVALID)
        self.assertEqual(assess_validity(new, AS_OF).state, VALID)

    def test_date_states(self):
        future = C("future", "未来条款", "P-f", "现行有效", vf="2027-01-01")
        expired = C("expired", "过期条款", "P-e", "现行有效", vt="2026-01-01")
        self.assertEqual(assess_validity(future, AS_OF).state, INVALID)
        self.assertEqual(assess_validity(expired, AS_OF).state, INVALID)

    def test_missing_without_family_is_uncertain(self):
        missing = C("missing", "缺字段条款", "P-x", "", vf="", vt="")
        self.assertEqual(assess_validity(missing, AS_OF).state, UNCERTAIN)

    def test_missing_with_policy_registry(self):
        old = C("old", "旧条款", "P-old", "", vf="", vt="")
        new = C("new", "新条款", "P-new", "", vf="", vt="")
        fam = PolicyFamily(
            {"P-old": node("P-old", "旧政策", "已废止", vt="2026-01-01"),
             "P-new": node("P-new", "新政策", "现行有效", vt="2028-01-01")},
            [],
        )
        self.assertEqual(assess_validity(old, AS_OF, fam).state, INVALID)
        self.assertEqual(assess_validity(new, AS_OF, fam).state, VALID)

    def test_date_can_invalidate_even_without_status(self):
        c = C("c", "缺状态", "P", "", vf="", vt="2026-01-01")
        self.assertEqual(assess_validity(c, AS_OF).state, INVALID)


class FamilyTests(unittest.TestCase):
    def test_manual_edge_and_expansion(self):
        old_name = "广州开发区（黄埔区）促进低空经济高质量发展的若干措施实施细则"
        new_name = "广州开发区（黄埔区）促进低空经济与航空航天高质量发展的若干措施"
        old = C("old", "旧中型起降场补贴200万元", "P-old", "已废止", vt="2026-01-01", name=old_name)
        new = C("new", "新中型起降场补贴300万元", "P-new", "现行有效", vt="2028-01-01", name=new_name)
        fam = build_policy_family([old, new], manual=True)
        self.assertEqual(len(fam.edges), 1)
        self.assertEqual(fam.edges[0].old_id, "P-old")
        self.assertTrue(fam.is_old("P-old", AS_OF))
        out = fam.expand([old], "中型起降场补贴上限", AS_OF, max_extra=1)
        self.assertEqual(out[0].clause.id, "new")
        self.assertEqual(out[0].new_policy_id, "P-new")

    def test_override_edge(self):
        old = C("old", "旧条款", "P-old", "已废止", vt="2026-01-01",
                name="旧政策", )
        old.metadata["source_url"] = "https://old.example"
        new = C("new", "新条款", "P-new", "现行有效", name="新政策")
        ov = {"https://old.example": {"superseded_by": "新政策"}}
        fam = build_policy_family([old, new], ov, manual=False)
        self.assertEqual(len(fam.edges), 1)
        self.assertEqual(fam.edges[0].new_id, "P-new")

    @unittest.skipUnless((ROOT / "results/controlled_20261028/snapshot.json").exists(),
                         "需要冻结快照")
    def test_tq3_hard_recall_cases_are_expanded(self):
        snap = json.loads((ROOT / "results/controlled_20261028/snapshot.json").read_text(encoding="utf-8"))
        fam = build_policy_family(
            [load_clause(r) for r in snap["corpus"]],
            load_overrides(ROOT / "data/national/validity_overrides.json"),
        )
        for qid in ("TQ3-030", "TQ3-032", "TQ3-041"):
            item = next(x for x in snap["items"] if x["id"] == qid)
            base = [load_clause(r) for r in snap["retrieval"][qid]]
            self.assertNotIn(item["new_clause_id"], [d.id for d in base])
            out = fam.expand(base, item["question"], date.fromisoformat(snap["as_of"]), max_extra=3)
            self.assertIn(item["new_clause_id"], [x.clause.id for x in out])

    @unittest.skipUnless((ROOT / "results/controlled_20261028/snapshot.json").exists(),
                         "需要冻结快照")
    def test_family_uncertainty_survives_joint_missing(self):
        snap = json.loads((ROOT / "results/controlled_20261028/snapshot.json").read_text(encoding="utf-8"))
        as_of = date.fromisoformat(snap["as_of"])
        fam = build_policy_family(
            [load_clause(r) for r in snap["corpus"]],
            load_overrides(ROOT / "data/national/validity_overrides.json"),
        )
        item = next(x for x in snap["items"] if x["id"] == "TQ3-041")
        base = [load_clause(r) for r in snap["retrieval"]["TQ3-041"]]
        pool = eligible_policies_in_retrieval(snap)
        corrupted, _ = corrupt_docs(base, "joint_missing", pool, as_of)
        added = [x.clause for x in fam.expand(base, item["question"], as_of, max_extra=2)]
        prep = prepare_answer("family_uncertainty", item["question"], corrupted + added,
                              as_of, family=fam)
        audit = context_audit(base + added, prep["docs"], item, as_of)
        self.assertEqual(prep["route"], "generate")
        self.assertIn(item["new_clause_id"], audit["context_ids"])
        self.assertEqual(audit["true_invalid_kept"], [])
        self.assertTrue(audit["new_in_context"])


class FakeChat:
    model = "fake"
    provider = "fake"

    def chat(self, messages, **kwargs):
        return "根据现行条款，答案如下。"


class GraphTests(unittest.TestCase):
    def test_uncertainty_unknown_refuses(self):
        c = C("c", "缺字段", "P", "", vf="", vt="")
        out = TemporalRAGGraph(None, mode="uncertainty").ask("问题", [c], AS_OF)
        self.assertEqual(out["route"], "refuse")
        self.assertIn("时效无法核实", out["answer"])

    def test_uncertainty_registry_invalid_refuses(self):
        c = C("c", "缺字段", "P-old", "", vf="", vt="")
        fam = PolicyFamily({"P-old": node("P-old", "旧政策", "已废止", vt="2026-01-01")}, [])
        out = TemporalRAGGraph(None, mode="uncertainty", family=fam).ask("问题", [c], AS_OF)
        self.assertEqual(out["route"], "refuse")
        self.assertIn("均已失效", out["answer"])

    def test_uncertainty_keeps_valid_after_dropping_old(self):
        old = C("old", "旧", "P-old", "", vf="", vt="")
        new = C("new", "新", "P-new", "", vf="", vt="")
        fam = PolicyFamily(
            {"P-old": node("P-old", "旧政策", "已废止", vt="2026-01-01"),
             "P-new": node("P-new", "新政策", "现行有效", vt="2028-01-01")},
            [],
        )
        out = TemporalRAGGraph(FakeChat(), mode="uncertainty", family=fam).ask("问题", [old, new], AS_OF)
        self.assertEqual(out["route"], "generate")
        self.assertEqual([c.id for c in out["filtered_docs"]], ["new"])
        self.assertEqual(out["dropped"][0]["id"], "old")

    def test_legacy_filter_still_treats_missing_as_usable(self):
        c = C("c", "缺字段", "P", "", vf="", vt="")
        out = TemporalRAGGraph(FakeChat(), mode="filter").ask("问题", [c], AS_OF)
        self.assertEqual(out["route"], "generate")
        self.assertEqual([x.id for x in out["filtered_docs"]], ["c"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
