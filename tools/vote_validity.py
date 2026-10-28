"""三家大模型对 B 级失效日候选做独立判定 + 多数票裁决。

为什么需要投票：
  B 级的共同特征是"给了日期但证据不够硬"（多为只写规划期、或年份疑为目标年）。
  单模型在这种灰区上容易受提示词措辞影响，三家独立判断可以：
    · 一致（3/3）：可信度接近 A 级，可直接采纳；
    · 二比一（2/3）：采纳多数，并保留分歧记录供回溯；
    · 三家各异（1/1/1）：说明正文确实不足以判定，退回 C 级（不给日期）。

投票口径：
  主票 = kind（判定类型），副票 = inferred_expiry（具体日期）。
  两者都被 ≥2 家认同才算"一致采纳"；kind 一致但日期分歧的记为"日期分歧"。

用法：
  python tools/vote_validity.py --limit 3      # 小样本验证
  python tools/vote_validity.py                # 全量（断点续跑）
  python tools/vote_validity.py --report       # 只用已有投票结果重算报表
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policycrawler.llm import LLMError, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "output" / "validity_candidates_report.json"
VOTES = ROOT / "data" / "national" / "validity_votes.json"
TABLE = ROOT / "output" / "validity_votes_table.md"
MANIFEST = ROOT / "data" / "national" / "corpus_manifest.json"

PROVIDERS = ("deepseek", "qwen", "glm")

# 与 infer_validity.py 保持同一套判定口径，保证可比
SYSTEM = """你是中国低空经济与民航政策的研究助理。判断一份政策文件"从哪一天起不应再被当作现行依据"（时效终点）。
请只输出一个 JSON 对象。

输出字段：
{
  "kind": "explicit_expiry | plan_period | open_ended | repealed | unclear",
  "inferred_expiry": "YYYY-MM-DD 或 null",
  "confidence": "high | medium | low",
  "evidence": "支撑判断的原文片段，不超过 60 字，必须逐字摘自给定正文",
  "reasoning": "不超过 40 字的判断依据"
}

判断规则（按顺序）：
1. 正文写了确定日期，如"有效期至 2026 年 8 月 31 日"→ kind=explicit_expiry，取该日期。
2. 正文写了"有效期 X 年"且能确定施行日 → kind=explicit_expiry，取施行日 + X 年再减一天。
3. 正文或标题只给了规划期／行动期，如"2024—2026 年""三年行动方案（2024—2026）"→ kind=plan_period，
   取该期间最后一年 12 月 31 日。
4. 属于条例、规定、办法、规则、强制性标准等长期有效文件，正文没有任何有效期或规划期表述
   → kind=open_ended，inferred_expiry=null。不要为了给答案而编日期。
5. 正文明确写了废止、停止执行或被新文件替代 → kind=repealed。
6. 以上都不能确定 → kind=unclear，inferred_expiry=null。

特别注意（本批文件的主要陷阱）：
- "到 2027 年建成……"、"2030 年产业规模达到……"这类是**目标年**，不是有效期终点。
  若文件只有目标年而无规划期区间，应判 open_ended 或 unclear，不要给日期。
- 只有区间表述（如 2024—2026 年）时，区间末年 12 月 31 日才是答案，不要取目标年。
- 意见、纲要、指南、方案若写明"本方案自印发之日起施行，有效期至……"则以该条为准。

硬性要求：
- evidence 必须是给定正文中的连续原文，不得改写。
- 不得使用"没有写就等于取消"这类推断。
- 只输出 JSON，不要 Markdown，不要代码块。
"""


def body_of(path: str, limit_head: int = 6000, limit_tail: int = 1200) -> str:
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    body = txt.split("-" * 60, 1)[-1].strip()
    if len(body) <= limit_head + limit_tail:
        return body
    return body[:limit_head] + "\n……\n" + body[-limit_tail:]


def b_level_targets() -> list[dict]:
    """复现 tier 分层的 B 级（排除 repealed，那类不需要投票）。"""
    rows = json.loads(CAND.read_text(encoding="utf-8"))
    man = {p["policy_id"]: p for p in json.loads(MANIFEST.read_text(encoding="utf-8"))["policies_detail"]}
    out = []
    for r in rows:
        # 沿用 tier 的 A 级判据：给了日期但不是 A 的即 B
        d = r.get("inferred_expiry")
        if not d:
            continue
        if r.get("rule_class") == "no_clause" and r.get("kind") != "repealed":
            pass
        # B 级 = 有日期 且 不属于"证据可核+high+日期对齐"的 A 级
        txt = r["policy_name"] + " " + (r.get("evidence") or "")
        import re
        RANGE = re.compile(r"(20\d{2})\s*[—\-－~至到]\s*(20\d{2})")
        EXPL = re.compile(r"有效期|至\s*20\d{2}\s*年")
        ok = False
        if r["kind"] == "plan_period":
            m = RANGE.search(txt)
            ok = bool(m) and str(d)[:4] == m.group(2) and str(d)[5:] == "12-31"
        elif r["kind"] == "explicit_expiry":
            ok = bool(EXPL.search(txt))
        is_a = ok and r.get("evidence_verified") and r.get("confidence") == "high"
        if is_a:
            continue
        if r["kind"] == "repealed":
            continue
        p = man.get(r["policy_id"], {})
        out.append({**r, "source_file": p.get("source_file", ""), "chunks": p.get("chunks", 0)})
    return out


def vote_one(targets: list[dict], providers: tuple[str, ...], sleep: float, limit: int = 0) -> dict:
    """对每条目标分别调用各模型（互不可见），结果按 policy_id + provider 落盘。"""
    store: dict[str, dict] = {}
    if VOTES.exists():
        old = json.loads(VOTES.read_text(encoding="utf-8"))
        store = old.get("votes", {})
        print(f"[resume] 已有 {sum(len(v) for v in store.values())} 条模型判定", flush=True)

    clients = {}
    for pr in providers:
        try:
            clients[pr] = provider_client(pr)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {pr} 不可用：{e}", flush=True)

    todo = targets[: limit] if limit else targets
    for i, t in enumerate(todo, 1):
        bucket = store.setdefault(t["policy_id"], {})
        need = [pr for pr in clients if pr not in bucket]
        if not need:
            continue
        body = body_of(t["source_file"]) if t.get("source_file") else ""
        if not body or len(body) < 200:
            print(f"[SKIP] 正文过短/缺失：{t['policy_name'][:30]}", flush=True)
            continue
        user = (
            f"【政策名称】{t['policy_name']}\n"
            f"【适用地区】{t['region']}\n"
            f"【参考信息】另有模型初判为 {t['kind']} / {t.get('inferred_expiry')}，请独立判断，不必与之一致。\n\n"
            f"【正文】\n{body}"
        )
        for pr in need:
            try:
                raw, tries = {}, 0
                # GLM 等模型思考可能吃满预算导致返回空结构，空 kind 时重试
                while tries < 3:
                    tries += 1
                    raw = clients[pr].chat_json(SYSTEM, user, temperature=0.0, max_tokens=5000)
                    if str(raw.get("kind", "")).strip():
                        break
                    print(f"  [retry {tries}] {pr} 返回空 kind，重试", flush=True)
                    time.sleep(1.0)
                bucket[pr] = {
                    "kind": str(raw.get("kind", "")).strip(),
                    "inferred_expiry": raw.get("inferred_expiry"),
                    "confidence": str(raw.get("confidence", "")).strip(),
                    "evidence": str(raw.get("evidence", ""))[:120],
                    "reasoning": str(raw.get("reasoning", ""))[:120],
                    "at": datetime.now().isoformat(timespec="seconds"),
                }
                print(f"[{i}/{len(todo)}] {pr:<8} {t['policy_name'][:24]:<26} -> {bucket[pr]['kind']} {bucket[pr]['inferred_expiry']}", flush=True)
            except (LLMError, OSError, ValueError) as e:
                print(f"[FAIL] {pr} {t['policy_name'][:24]}: {e}", flush=True)
            VOTES.parent.mkdir(parents=True, exist_ok=True)
            VOTES.write_text(
                json.dumps({"updated_at": datetime.now().isoformat(timespec="seconds"), "votes": store},
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            time.sleep(sleep)
    return store


def tally(store: dict[str, dict], targets: list[dict], providers: tuple[str, ...]) -> list[dict]:
    """多数票裁决：kind 与 date 各投一票，≥2 家认同才算一致。"""
    results = []
    for t in targets:
        bucket = store.get(t["policy_id"], {})
        got = [(pr, bucket[pr]) for pr in providers if pr in bucket]
        if not got:
            results.append({**t, "verdict": "无有效投票", "n_votes": 0})
            continue
        kinds = Counter(v["kind"] for _, v in got)
        dates = Counter(str(v["inferred_expiry"]) for _, v in got)
        k_top, k_n = kinds.most_common(1)[0]
        d_top, d_n = dates.most_common(1)[0]
        n = len(got)
        if k_n >= 2 and d_n >= 2:
            verdict = "一致采纳" if n == 3 and k_n == 3 and d_n == 3 else "多数采纳"
        elif k_n >= 2 and d_n < 2:
            verdict = "日期分歧"
        elif k_n < 2:
            verdict = "判定分歧"
        else:
            verdict = "待复核"
        results.append({
            **t,
            "verdict": verdict,
            "n_votes": n,
            "vote_kind": dict(kinds),
            "vote_date": dict(dates),
            "final_kind": k_top,
            "final_expiry": None if d_top in ("None", "null", "") else d_top,
            "per_model": {pr: {"kind": v["kind"], "expiry": v["inferred_expiry"], "conf": v["confidence"],
                               "evidence": v["evidence"]} for pr, v in got},
        })
    return results


def write_table(results: list[dict], providers: tuple[str, ...]) -> None:
    cnt = Counter(r["verdict"] for r in results)
    lines = [
        "# B 级失效日候选：三家模型投票结果",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　|　参与模型：{'、'.join(providers)}　|　条目：{len(results)}",
        "",
        "## 汇总",
        "",
        "| 裁决 | 条数 | 处理方式 |",
        "| --- | --- | --- |",
        f"| 一致采纳（3/3） | {cnt.get('一致采纳', 0)} | 直接写入覆盖表 |",
        f"| 多数采纳（2/3） | {cnt.get('多数采纳', 0)} | 写入覆盖表并标注分歧 |",
        f"| 日期分歧（类型同、日期不同） | {cnt.get('日期分歧', 0)} | 取多数日期，标注待核 |",
        f"| 判定分歧（类型都不同） | {cnt.get('判定分歧', 0)} | 不采纳，退回长期有效 |",
        "",
        "## 逐条结果",
        "",
        "| 政策 | 地区 | 初判 | " + " | ".join(providers) + " | 裁决 | 最终日期 |",
        "| --- | --- | --- | " + " | ".join(["---"] * len(providers)) + " | --- | --- |",
    ]
    for r in sorted(results, key=lambda x: x["verdict"]):
        cells = []
        for pr in providers:
            v = r.get("per_model", {}).get(pr)
            cells.append(f"{v['kind'][:14]} / {v['expiry']}" if v else "—")
        lines.append(
            f"| {r['policy_name'][:30]} | {r['region']} | {r['kind'][:14]} / {r.get('inferred_expiry')} | "
            + " | ".join(cells)
            + f" | **{r['verdict']}** | {r.get('final_expiry')} |"
        )
    lines += ["", "## 证据摘录（每条的模型回引原文）", ""]
    for r in results:
        lines.append(f"### {r['policy_name'][:40]}（{r['verdict']}）")
        for pr in providers:
            v = r.get("per_model", {}).get(pr)
            if v:
                lines.append(f"- **{pr}**：{v['kind']} / {v['expiry']} / {v['conf']}　证据：「{v['evidence'][:70]}」")
        lines.append("")
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    print("投票表写入:", TABLE)
    print("裁决分布:", dict(cnt))


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--report", action="store_true", help="跳过调用，只用已有投票重算报表")
    args = ap.parse_args()

    targets = b_level_targets()
    print(f"B 级目标 {len(targets)} 条（已排除 repealed）", flush=True)

    if args.report:
        store = json.loads(VOTES.read_text(encoding="utf-8"))["votes"] if VOTES.exists() else {}
    else:
        store = vote_one(targets, PROVIDERS, args.sleep, args.limit)

    results = tally(store, targets, PROVIDERS)
    VOTES.with_name("validity_votes_result.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "results": results},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    write_table(results, PROVIDERS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
