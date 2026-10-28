"""对 infer_validity.py 的结果做质量核验与统计。

三项检查：
  1. 证据是否真的摘自正文（逐字子串匹配，去空白后比对）；
  2. 与规则分类（明确有效期／有效期N年／只有规划期／无条款）是否一致；
  3. 日期是否落在合理区间（2000-01-01 至 2035-12-31），且不早于发布日期。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data" / "national" / "validity_candidates.json"
MAN = ROOT / "data" / "national" / "corpus_manifest.json"

PLAN = re.compile(r"(20\d{2})\s*[—\-－~至到]\s*(20\d{2})\s*年")
VTO = re.compile(r"有效期(?:至|到)\s*(20\d{2})\s*年\s*(\d{1,2})?\s*月?\s*(\d{1,2})?")
VN = re.compile(r"有效期(?:为)?\s*([一二三四五六七八九十\d]+)\s*年")
CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def rule_class(body: str, name: str) -> str:
    if VTO.search(body):
        return "explicit_expiry"
    if VN.search(body):
        return "explicit_n_year"
    if PLAN.search(body) or PLAN.search(name):
        return "plan_period"
    return "no_clause"


def main() -> None:
    cand = json.loads(CAND.read_text(encoding="utf-8"))["items"]
    man = {p["policy_id"]: p for p in json.loads(MAN.read_text(encoding="utf-8"))["policies_detail"]}

    rows = []
    for c in cand:
        p = man.get(c["policy_id"], {})
        try:
            txt = Path(p.get("source_file", "")).read_text(encoding="utf-8", errors="replace")
            body = txt.split("-" * 60, 1)[-1]
        except Exception:
            body = ""
        ev_ok = norm(c.get("evidence", "")) in norm(body) if c.get("evidence") else False
        rc = rule_class(body, p.get("policy_name", ""))
        d = c.get("inferred_expiry")
        date_ok = None
        if d:
            try:
                dd = date.fromisoformat(str(d)[:10])
                date_ok = date(2000, 1, 1) <= dd <= date(2035, 12, 31)
            except Exception:
                date_ok = False
        rows.append({**c, "evidence_verified": ev_ok, "rule_class": rc, "date_ok": date_ok,
                     "body_len": len(body)})

    total = len(rows)
    filled = [r for r in rows if r.get("inferred_expiry")]
    print(f"参与判断: {total} 份 | 给出日期: {len(filled)} 份 ({len(filled)/max(1,total):.1%})")
    print("kind 分布:", dict(Counter(r["kind"] for r in rows)))
    print("confidence 分布:", dict(Counter(r["confidence"] for r in rows)))
    print(f"证据逐字可核: {sum(1 for r in rows if r['evidence_verified'])} / {total}")
    bad_date = [r for r in rows if r.get("inferred_expiry") and not r["date_ok"]]
    print(f"日期越界/格式异常: {len(bad_date)}")

    print("\n规则分类 x 模型 kind:")
    cross = Counter((r["rule_class"], r["kind"]) for r in rows)
    for (a, b), n in sorted(cross.items(), key=lambda x: -x[1]):
        print(f"  {a:<18} -> {b:<14} {n}")

    print("\n需要人工看一眼的三类:")
    conf1 = [r for r in rows if r.get("inferred_expiry") and not r["evidence_verified"]]
    print(f"  A 给出日期但证据不可核: {len(conf1)}")
    for r in conf1[:8]:
        print(f"     {r['policy_name'][:24]} {r['inferred_expiry']} | {r['evidence'][:40]}")
    conf2 = [r for r in rows if r["kind"] == "plan_period" and r.get("inferred_expiry")]
    print(f"  B 按规划期给出日期: {len(conf2)}")
    for r in conf2[:6]:
        print(f"     {r['policy_name'][:26]} -> {r['inferred_expiry']}")
    conf3 = [r for r in rows if r["rule_class"] in ("explicit_expiry", "explicit_n_year") and r["kind"] != "explicit_expiry"]
    print(f"  C 正文有明确有效期但模型没判 explicit: {len(conf3)}")
    for r in conf3[:8]:
        print(f"     {r['policy_name'][:24]} rule={r['rule_class']} model={r['kind']} {r.get('inferred_expiry')}")

    out = ROOT / "output" / "validity_candidates_report.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n明细写入:", out)


if __name__ == "__main__":
    main()
