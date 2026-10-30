"""合并 K=1..20 扫描的 DeepSeek/Qwen/GLM 裁判，重算每个 K 的指标。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(r"D:/lowalt-policy-kb_backup_20261028")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rejudge" / "ksweep"
LABELS = ("POISONED", "CORRECT", "REFUSED", "OTHER_ERROR")
WEIGHTS = {"CORRECT": 1.0, "REFUSED": 0.2, "OTHER_ERROR": 0.0, "POISONED": -1.0}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_key(k: int, item_id: str, branch: str, answer: str) -> str:
    return fingerprint({"suite": "ksweep", "k": k, "item_id": item_id,
                        "branch": branch, "answer": answer})


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> int:
    qwen = {p.stem: load_json(p) for p in (OUT / "records/qwen").glob("*.json")}
    glm = {p.stem: load_json(p) for p in (OUT / "records/glm").glob("*.json")}
    human = load_json(OUT / "human_review.json")["overrides"] if (OUT / "human_review.json").exists() else {}
    merged = []
    unresolved = []
    ds_qwen_pairs = []
    glm_pairs = []
    no_majority = 0
    human_reviewed_count = 0
    human_override_count = 0
    for k in range(1, 21):
        data = load_json(PROJECT / "results/ksweep" / f"k{k}.json")
        items = data["items"]
        for it in items:
            for branch, ans_key, ds_key in (("naive", "naive_answer", "naive_judge"),
                                             ("temporal", "agent_answer", "agent_judge")):
                answer = it[ans_key]
                ds = it[ds_key]
                key = make_key(k, it["id"], branch, answer)
                q = qwen.get(key)
                if q is None:
                    unresolved.append({"k": k, "item_id": it["id"], "branch": branch, "reason": "missing qwen"})
                    continue
                qj = q["judge"]
                ds_qwen_pairs.append((ds["status"], qj["status"]))
                g = glm.get(key)
                if qj["status"] == ds["status"]:
                    status, vote, nm = ds["status"], {ds["status"]: 2}, False
                    gout = None
                elif g is not None:
                    gout = g["judge"]
                    votes = [ds, qj, gout]
                    c = Counter(v["status"] for v in votes)
                    top, n = c.most_common(1)[0]
                    if n >= 2:
                        status, vote, nm = top, dict(c), False
                    else:
                        status, vote, nm = ds["status"], dict(c), True
                    glm_pairs.extend([(ds["status"], gout["status"]), (qj["status"], gout["status"])])
                else:
                    unresolved.append({"k": k, "item_id": it["id"], "branch": branch,
                                       "reason": "missing glm", "ds": ds["status"], "qwen": qj["status"]})
                    continue
                no_majority += int(nm)
                h = human.get(key)
                auto_status = status
                final_status = h["status"] if h else status
                human_reviewed_count += int(bool(h))
                human_override_count += int(bool(h) and h["status"] != status)
                merged.append({
                    "case_key": key, "k": k, "branch": branch, "item_id": it["id"],
                    "answer": answer, "ds_judge": ds, "qwen_judge": qj, "glm_judge": gout,
                    "judge_majority": {
                        "status": final_status, "vote": vote, "no_majority": nm,
                        "auto_status": auto_status if h else None,
                        "human_reviewed": bool(h),
                    },
                })

    merged_dir = OUT / "merged/records"
    for r in merged:
        atomic_write(merged_dir / f"{r['case_key']}.json", r)

    def metrics(k: int, branch: str, comparison: str):
        rs = [r for r in merged if r["k"] == k and r["branch"] == branch]
        c = Counter(r[f"{comparison}_judge"]["status"] if comparison == "ds" else r["judge_majority"]["status"]
                    for r in rs)
        n = len(rs)
        return {
            "n": n, **{x: c.get(x, 0) for x in LABELS},
            "poison_rate": round(c.get("POISONED", 0) / n, 4) if n else None,
            "correct_rate": round(c.get("CORRECT", 0) / n, 4) if n else None,
            "refuse_rate": round(c.get("REFUSED", 0) / n, 4) if n else None,
            "weighted_score": round(sum(WEIGHTS[r[f"{comparison}_judge"]["status"] if comparison == "ds" else r["judge_majority"]["status"]] for r in rs) / n, 4) if n else None,
        }

    groups = []
    for k in range(1, 21):
        data = load_json(PROJECT / "results/ksweep" / f"k{k}.json")
        items = data["items"]
        new_recall = sum(1 for x in items if x.get("new_clause_penetrated"))
        old_pen = sum(1 for x in items if x.get("old_clause_penetrated"))
        for branch in ("naive", "temporal"):
            out = {"k": k, "branch": branch, "new_recall": new_recall, "old_penetration": old_pen}
            for comp in ("ds", "majority"):
                out[comp] = metrics(k, branch, comp)
            if branch == "temporal":
                for comp in ("ds", "majority"):
                    naive_poison = {r["item_id"] for r in merged
                                    if r["k"] == k and r["branch"] == "naive"
                                    and (r["ds_judge"]["status"] if comp == "ds" else r["judge_majority"]["status"]) == "POISONED"}
                    temporal_correct = {r["item_id"] for r in merged
                                        if r["k"] == k and r["branch"] == "temporal"
                                        and (r["ds_judge"]["status"] if comp == "ds" else r["judge_majority"]["status"]) == "CORRECT"}
                    out[f"strict_rescue_{comp}"] = round(len(naive_poison & temporal_correct) / len(naive_poison), 4) if naive_poison else None
            groups.append(out)

    ds_qwen_agree = sum(a == b for a, b in ds_qwen_pairs) / len(ds_qwen_pairs) if ds_qwen_pairs else None
    summary = {
        "created_at": now(),
        "merged": len(merged),
        "unresolved": len(unresolved),
        "no_majority": no_majority,
        "human_reviewed": human_reviewed_count,
        "human_overrides": human_override_count,
        "ds_qwen_agreement": round(ds_qwen_agree, 4) if ds_qwen_agree is not None else None,
        "ds_qwen_kappa": round(kappa(ds_qwen_pairs), 4) if ds_qwen_pairs and kappa(ds_qwen_pairs) is not None else None,
        "glm_pair_agreement": round(sum(a == b for a, b in glm_pairs) / len(glm_pairs), 4) if glm_pairs else None,
        "glm_pair_kappa": round(kappa(glm_pairs), 4) if glm_pairs and kappa(glm_pairs) is not None else None,
        "glm_used": len(glm_pairs) // 2,
        "groups": groups,
        "unresolved_cases": unresolved[:50],
    }
    atomic_write(OUT / "merged_summary.json", summary)
    lines = ["# K 扫描三裁判复核", "",
             f"合并 {len(merged)} 条，未决 {len(unresolved)} 条；DS/Qwen 一致率 {summary['ds_qwen_agreement']}，"
             f"κ={summary['ds_qwen_kappa']}；GLM 用于 {summary['glm_used']} 条分歧（成对一致率 {summary['glm_pair_agreement']}、κ={summary['glm_pair_kappa']}）；无多数 {no_majority} 条，人工复核 {human_reviewed_count} 条、改判 {human_override_count} 条。", "",
             "| K | 分支 | 新召回 | 旧穿透 | 三裁判毒化 | 三裁判正确 | 三裁判拒答 | 其他 | 加权 | 严格拯救 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        m = g["majority"]
        lines.append(f"| {g['k']} | {g['branch']} | {g['new_recall']}/31 | {g['old_penetration']}/31 | "
                     f"{m['POISONED']} | {m['CORRECT']} | {m['REFUSED']} | {m['OTHER_ERROR']} | "
                     f"{m['weighted_score']:+.3f} | {g.get('strict_rescue_majority', '—')} |")
    (OUT / "merged_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"merged {len(merged)} unresolved {len(unresolved)} no_majority {no_majority} "
          f"ds_qwen_agree {summary['ds_qwen_agreement']} glm_used {summary['glm_used']}")
    return 0 if not unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
