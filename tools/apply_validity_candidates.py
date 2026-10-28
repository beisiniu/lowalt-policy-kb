"""把 A 级候选写入覆盖表（仅补 expiry_date，不改 status）。

安全边界：
  · 只写 kind=plan_period / explicit_expiry、证据逐字可核、置信 high、
    且日期与区间末年 12-31 对齐的条目；
  · 剔除名称含纲要/指南/公告/意见/标准/通知的条目——它们的年份是目标年，不是时效终点；
  · 不写 status，避免覆盖已有的人工裁定；
  · 写入前备份覆盖表，写入条目带 inference 字段，便于与人工条目区分与回滚。
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROWS = ROOT / "output" / "validity_candidates_report.json"
OV = ROOT / "data" / "national" / "validity_overrides.json"
MAN = ROOT / "data" / "national" / "corpus_manifest.json"

RANGE = re.compile(r"(20\d{2})\s*[—\-－~至到]\s*(20\d{2})")
EXPL = re.compile(r"有效期|至\s*20\d{2}\s*年")
TARGET_ONLY = re.compile(r"纲要|指南|公告|意见|标准|通知")
DRY = False


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    man = {p["policy_id"]: p for p in json.loads(MAN.read_text(encoding="utf-8"))["policies_detail"]}
    ov = json.loads(OV.read_text(encoding="utf-8"))

    picked, skipped = [], []
    for r in rows:
        d = r.get("inferred_expiry")
        if not d:
            continue
        txt = r["policy_name"] + " " + (r.get("evidence") or "")
        ok = False
        if r["kind"] == "plan_period":
            m = RANGE.search(txt)
            ok = bool(m) and str(d)[:4] == m.group(2) and str(d)[5:] == "12-31"
        elif r["kind"] == "explicit_expiry":
            ok = bool(EXPL.search(txt))
        if not (ok and r["evidence_verified"] and r["confidence"] == "high"):
            continue
        if TARGET_ONLY.search(r["policy_name"]):
            skipped.append(r)
            continue
        picked.append(r)

    print(f"可写入 {len(picked)} 条；因目标年剔除 {len(skipped)} 条")
    for r in skipped:
        print("  剔除:", r["policy_name"][:34], r["inferred_expiry"])

    if not DRY:
        shutil.copy2(OV, OV.with_name(f"validity_overrides_backup_{datetime.now():%Y%m%d_%H%M%S}.json"))
        n_new = n_upd = 0
        for r in picked:
            p = man.get(r["policy_id"], {})
            url = p.get("source_url", "")
            if not url:
                continue
            item = ov.get(url, {})
            if item.get("expiry_date"):
                n_upd += 0
                continue
            item["expiry_date"] = r["inferred_expiry"]
            item["inference"] = "llm_plan_period" if r["kind"] == "plan_period" else "llm_explicit_expiry"
            item["evidence"] = (r.get("evidence") or "")[:120]
            item["note"] = (
                "失效日为规划期末推算（大模型判定，未经人工逐条复核）；"
                "若该文件已被新版替代，请以带 evidence_url 的人工条目为准。"
            )
            item["inferred_at"] = "2026-10-27"
            ov[url] = item
            n_new += 1
        OV.write_text(json.dumps(ov, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {n_new} 条（跳过已有人工到期日的 {n_upd} 条）；覆盖表共 {len(ov)} 条")


if __name__ == "__main__":
    main()
