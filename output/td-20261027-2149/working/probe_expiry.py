import json
import re
from pathlib import Path

m = json.loads(Path("data/national/corpus_manifest.json").read_text(encoding="utf-8"))
pol = m["policies_detail"]
missing = [p for p in pol if not str(p.get("valid_to", "")).strip()]

PLAN = re.compile(r"(20\d{2})\s*[—\-－~至到]\s*(20\d{2})\s*年")
VALID_TO = re.compile(r"有效期(?:至|到)\s*(20\d{2})\s*年\s*(\d{1,2})?\s*月?")
VALID_N = re.compile(r"有效期(?:为)?\s*([一二三四五六七八九十\d]+)\s*年")

stat = {"A_明确有效期至": 0, "B_只有规划期": 0, "C_两者都无": 0, "D_有效期N年": 0}
rows = []
for p in missing:
    try:
        txt = Path(p["source_file"]).read_text(encoding="utf-8", errors="replace")
    except Exception:
        txt = ""
    body = txt.split("-" * 60, 1)[-1]
    name = p["policy_name"]
    has_valid_to = bool(VALID_TO.search(body))
    has_valid_n = bool(VALID_N.search(body))
    has_plan = bool(PLAN.search(body)) or bool(PLAN.search(name))
    if has_valid_to:
        stat["A_明确有效期至"] += 1
        rows.append(("A", p["region"], name[:36]))
    elif has_valid_n:
        stat["D_有效期N年"] += 1
        rows.append(("D", p["region"], name[:36]))
    elif has_plan:
        stat["B_只有规划期"] += 1
    else:
        stat["C_两者都无"] += 1

print("无 valid_to 的政策数:", len(missing))
for k, v in stat.items():
    print(f"  {k}: {v}")
print()
print("A/D 类（本可推算但未填）示例:")
for r in rows[:12]:
    print("  ", r)
