"""让大模型判断"这份政策从哪天起不应再作为现行依据"。

与 extract_policy.py 的区别：
  · 只处理 valid_to 为空的文件；
  · 提示词把"规划期"纳入判断（规划期最后一年 12-31 视为时效终点），
    同时保留"长期有效法规不给日期"的约束，避免把条例类文件误判为过期；
  · 必须回引原文证据，无法确定则返回 null，禁止编造。

用法：
  python tools/infer_validity.py --limit 5            # 小样本试跑
  python tools/infer_validity.py                      # 全量（支持断点续跑）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policycrawler.llm import LLMError, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "national" / "validity_candidates.json"

SYSTEM = """你是中国低空经济与民航政策的研究助理。你的任务不是复述法条，而是判断一份政策文件
"从哪一天起不应再被当作现行依据"（时效终点）。请只输出一个 JSON 对象。

输出字段：
{
  "kind": "explicit_expiry | plan_period | open_ended | repealed | unclear",
  "inferred_expiry": "YYYY-MM-DD 或 null",
  "confidence": "high | medium | low",
  "evidence": "支撑判断的原文片段，不超过 60 字，必须逐字摘自给定正文",
  "reasoning": "不超过 40 字的判断依据"
}

判断规则（按顺序）：
1. 正文写了确定日期，如"有效期至 2026 年 8 月 31 日"→ kind=explicit_expiry，直接取该日期。
2. 正文写了"有效期 X 年"且能确定施行日 → kind=explicit_expiry，取施行日 + X 年再减一天。
3. 正文或标题只给了规划期／行动期，如"2024—2026 年""三年行动方案（2024—2026）""到 2027 年"
   且没有更明确的失效条款 → kind=plan_period，取该期间的最后一年的 12 月 31 日。
   例：2024—2026 年 → 2026-12-31；2025—2027 年 → 2027-12-31。
4. 属于条例、规定、办法、规则、强制性标准等长期有效文件，正文没有任何有效期或规划期表述
   → kind=open_ended，inferred_expiry=null。不要为了给答案而编日期。
5. 正文明确写了废止、停止执行或被新文件替代 → kind=repealed，若给了废止日期则取之，否则 null。
6. 以上都不能确定 → kind=unclear，inferred_expiry=null。

硬性要求：
- evidence 必须是给定正文中的连续原文，不得改写、不得使用标题以外的信息。
- 不得使用"没有写就等于取消"这类推断。
- 同一句话里同时出现规划期和有效期时，以有效期为准。
- 只输出 JSON，不要 Markdown，不要代码块。
"""


def body_of(path: str) -> str:
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    return txt.split("-" * 60, 1)[-1].strip()


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    man = json.loads((ROOT / "data/national/corpus_manifest.json").read_text(encoding="utf-8"))
    targets = [p for p in man["policies_detail"] if not str(p.get("valid_to", "")).strip()]

    out_path = Path(args.out)
    done: dict[str, dict] = {}
    if out_path.exists():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        done = {x["policy_id"]: x for x in old.get("items", [])}
        print(f"[resume] 已有 {len(done)} 条结果", flush=True)

    todo = [p for p in targets if p["policy_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"待判断 {len(todo)} / 共 {len(targets)} 份无失效日文件", flush=True)

    client = provider_client(args.provider)
    items = list(done.values())
    for i, p in enumerate(todo, 1):
        try:
            body = body_of(p["source_file"])
            user = f"【政策名称】{p['policy_name']}\n【适用地区】{p['region']}\n\n【正文】\n{body[:6000]}\n……\n{body[-1200:]}"
            raw = client.chat_json(SYSTEM, user, temperature=0.0, max_tokens=600)
            items.append({
                "policy_id": p["policy_id"],
                "policy_name": p["policy_name"],
                "region": p["region"],
                "level": p.get("level"),
                "status": p.get("status"),
                "issue_date": p.get("issue_date"),
                "kind": str(raw.get("kind", "")).strip(),
                "inferred_expiry": raw.get("inferred_expiry"),
                "confidence": str(raw.get("confidence", "")).strip(),
                "evidence": str(raw.get("evidence", ""))[:120],
                "reasoning": str(raw.get("reasoning", ""))[:120],
                "checked_at": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"[{i}/{len(todo)}] {p['policy_name'][:30]} -> {raw.get('kind')} {raw.get('inferred_expiry')}", flush=True)
        except (LLMError, OSError, ValueError) as e:
            print(f"[FAIL] {p['policy_name'][:30]}: {e}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(args.sleep)

    print(f"完成，共 {len(items)} 条 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
