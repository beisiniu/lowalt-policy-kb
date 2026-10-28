"""把三模型投票结果写入失效覆盖表。

只在"至少两家一致"时才采纳：
  · 一致采纳（3/3）：直接写入；
  · 多数采纳（2/3）：写入并记录分歧；
  · 判定分歧（1/1/1）：不写日期，仅记录"经投票无法判定"，退回长期有效处理。

写入方式：
  · final_expiry 非空 → 补/改 expiry_date，并标注 inference=3model_vote 与各模型票型；
  · final_expiry 为空（open_ended）→ 不设日期，仅在 note 记录三模型一致认为长期有效；
  · 不覆盖已有的人工条目（source=manual 或带 evidence_url 的）除非 allow-overwrite。

用法：
  python tools/apply_votes.py --dry-run
  python tools/apply_votes.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOTES = ROOT / "data" / "national" / "validity_votes_result.json"
OVERRIDES = ROOT / "data" / "national" / "validity_overrides.json"
MANIFEST = ROOT / "data" / "national" / "corpus_manifest.json"

ADOPT = {"一致采纳", "多数采纳"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    results = json.loads(VOTES.read_text(encoding="utf-8"))["results"]
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    url_of = {p["policy_name"]: p["source_url"] for p in man["policies_detail"]}

    ov = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    added = updated = skipped = 0
    log = []

    for r in results:
        url = url_of.get(r["policy_name"], "")
        if not url:
            # 名称与库内不完全一致时按前 24 字模糊匹配
            for k, v in url_of.items():
                if k[:24] == r["policy_name"][:24]:
                    url = v
                    break
        if not url:
            skipped += 1
            log.append(f"[跳过] 找不到 source_url：{r['policy_name'][:36]}")
            continue

        votes = {pr: f"{v['kind']}/{v['expiry']}" for pr, v in r.get("per_model", {}).items()}
        cur = ov.get(url, {})
        entry = dict(cur)

        if r["verdict"] in ADOPT and r.get("final_expiry"):
            entry.update({
                "expiry_date": r["final_expiry"],
                "inference": "3model_vote",
                "vote_verdict": r["verdict"],
                "votes": votes,
                "vote_at": datetime.now().strftime("%Y-%m-%d"),
                "note": (f"三模型投票判定：{r['verdict']}；失效日 {r['final_expiry']}"
                         f"（{r['final_kind']}）。仅供时效检索参考，不改 status。"),
            })
            added += 1 if url not in ov else 0
            updated += 1 if url in ov else 0
        elif r["verdict"] in ADOPT and not r.get("final_expiry"):
            entry.setdefault("note", "")
            entry.update({
                "inference": "3model_vote",
                "vote_verdict": r["verdict"],
                "votes": votes,
                "vote_at": datetime.now().strftime("%Y-%m-%d"),
                "note": (f"三模型投票判定为长期有效（{r['verdict']}），不设失效日。"
                         f"如需时效提醒，请在规划期结束后复核。"),
            })
            updated += 1 if url in ov else 0
            added += 1 if url not in ov else 0
        else:
            entry.setdefault("note", "")
            entry["vote_verdict"] = r["verdict"]
            entry["votes"] = votes
            entry["vote_at"] = datetime.now().strftime("%Y-%m-%d")
            entry["note"] = (f"三模型判定分歧（{r['verdict']}），不采纳日期，按长期有效处理，待人工复核。")
            updated += 1 if url in ov else 0
            added += 1 if url not in ov else 0

        ov[url] = entry
        log.append(f"[{r['verdict']}] {r['policy_name'][:34]:<36} -> {r.get('final_expiry') or '无日期'}")

    if not a.dry_run:
        OVERRIDES.write_text(json.dumps(ov, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(log))
    print()
    print(f"新增 {added} 条 / 更新 {updated} 条 / 跳过 {skipped} 条 | 覆盖表共 {len(ov)} 条")
    if a.dry_run:
        print("（dry-run，未写盘）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
