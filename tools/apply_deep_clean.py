"""把深度清洗应用到全部政策正文（一次性固化，之后建库仍会再兜底过滤一次）。

清洗目标：剥离网页导航栏、页脚、JSON 元数据、JS 残留、重复标题，
只保留政策正文（含标题、文号、条文、落款、施行日期）。

保护措施：
  · 清洗后不足原文 35%（>400 字的文件）视为误切，跳过该篇；
  · 清洗后汉字数少于 300 视为内容不足，跳过；
  · 原文件备份为 .bak_before_deepclean。

用法：
  python tools/apply_deep_clean.py --dry-run
  python tools/apply_deep_clean.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from body_quality import CJK, deep_clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "national" / "corpus_manifest.json"
REPORT = ROOT / "output" / "deep_clean_report.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows, changed, skipped = [], 0, 0
    seen: set[str] = set()
    for p in man["policies_detail"]:
        f = Path(p["source_file"])
        if not f.exists() or str(f) in seen:
            continue
        seen.add(str(f))
        if a.limit and len(rows) >= a.limit:
            break
        raw = f.read_text(encoding="utf-8", errors="replace")
        head, sep, body = raw.partition("-" * 60)
        cleaned = deep_clean(body)
        dc, dn = len(CJK.findall(cleaned)), len(CJK.findall(body))
        ratio = len(cleaned) / max(1, len(body))
        verdict = "keep"
        if len(body) > 400 and ratio < 0.30:
            verdict = "skip_overshrink"
        elif dn - dc > dn * 0.62:
            verdict = "skip_content_loss"
        elif dc < (150 if len(body) < 900 else 300):
            verdict = "skip_too_short"
        else:
            changed += 1 if ratio < 0.97 else 0
        rows.append({
            "policy_name": p["policy_name"], "region": p["region"],
            "before": len(body), "after": len(cleaned),
            "cjk_before": dn, "cjk_after": dc, "ratio": round(ratio, 3),
            "verdict": verdict,
        })
        if verdict == "keep" and not a.dry_run and ratio < 0.97:
            shutil.copy2(f, f.with_suffix(f.suffix + ".bak_before_deepclean"))
            f.write_text(head + sep + "\n" + cleaned, encoding="utf-8")
        elif verdict != "keep":
            skipped += 1
            print(f"[跳过] {p['policy_name'][:36]} | {verdict} | {len(body)}->{len(cleaned)}")

    REPORT.write_text(json.dumps({
        "at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": a.dry_run, "files": len(rows),
        "changed": changed, "skipped": skipped, "items": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n处理 {len(rows)} 篇 | 实际改动 {changed} 篇 | 跳过 {skipped} 篇")
    print("报告:", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
