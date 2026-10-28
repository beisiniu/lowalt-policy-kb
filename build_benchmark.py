#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从真实政策库自动构建“时序冲突”对抗基准。

流程：
    读取 Chroma 全部条款
      → 按 (region, topic) 分组
      → 找 valid_to 已过期 的旧条款 × valid_to 为空 的现行条款
      → 用词面相似度挑候选对（省钱，不把无关条款送去 LLM）
      → LLM 判断是否有实质差异，并生成“容易召回旧条款”的问题与标准答案
      → benchmark.json + benchmark.manifest.json

关键约束：
  1. 旧条款是否过期只看 valid_to / status，不看 benchmark 标签。
  2. LLM 只负责“判断+出题”，不负责决定哪条是旧、哪条是新。
  3. 生成的问题不得出现新旧政策名、日期或“已废止”等提示，否则评估组
     会因为问题本身暴露答案而虚高。

用法：
    python build_benchmark.py \
        --chroma-dir ./chroma_db --collection lowalt_policy \
        --as-of 2026-10-26 --max-pairs 80 --max-items 30

    # 没有 Chroma 时，用现有本地索引跑通流程（旧/新共存数据需自行准备）
    python build_benchmark.py --backend local --index-dir kb/index
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.llm import ChatClient, LLMError, provider_client
from policycrawler.utils import load_env_file
from rag_eval.backends import add_backend_args, build_backend
from rag_eval.models import Clause
from rag_eval.temporal import (
    bigram_jaccard,
    clause_matches,
    clause_valid_from,
    clause_valid_to,
    effective_region,
    effective_topic,
    iso_or_none,
    usable,
)

ROOT = Path(__file__).resolve().parent
CONFLICT_TYPES = (
    "threshold_changed", "approval_removed", "approval_added", "duration_changed",
    "scope_changed", "authority_changed", "subsidy_changed", "other",
)

VERIFY_SYSTEM = """你是时序 RAG 测试集的质量审核员。请严格审核一条自动生成的冲突题。

只输出 JSON：
{"valid": true/false, "same_specific_issue": true/false, "gt_supported_by_new": true/false, "reason": "不超过50字"}

审核标准：
1. 新条款必须**直接回答题目所问的具体事项**；只是同一主题、但管的是另一项补贴/另一类项目 → false。
2. 标准答案必须能由新条款原文直接推出，不能靠“新条款没写 = 取消了”这种缺失推断。
3. 如果标准答案声称“不再补贴/不再保留”，但新条款没有明确写废止、删除或替代，判 false。
4. 旧条款与新条款必须指向同一现实问题，并给出不同答案。
5. 题目不得含政策名、日期、“旧/新/废止”等提示。
任何一条不满足就 valid=false。
"""

BENCH_SYSTEM = """你是时序 RAG 的对抗测试集设计专家，熟悉中国低空经济与无人机政策。
我会给你两条同一主题、同一地域的条款：
  A：已经在某一时点失效的旧条款
  B：在评估时点仍然有效的现行条款

你的任务不是改写法条，而是判断 A 与 B 是否构成“实质性规则变化”，并在构成时设计一道
会让普通向量检索很容易召回 A 的用户问题。

只输出一个 JSON 对象：
{
  "has_substantive_difference": true/false,
  "conflict_type": "threshold_changed|approval_removed|approval_added|duration_changed|scope_changed|authority_changed|subsidy_changed|other",
  "question": "用户问题",
  "ground_truth": "只依据 B 的标准答案",
  "reasoning": "不超过 40 字，说明差异点"
}

判定规则：
1. 只有“对同一个现实问题会给出不同答案”才算实质差异。
   纯措辞调整、增加原则性表述、职责分工变化、没有改变适用条件的补充说明，一律 false。
2. question 必须：
   - 像真实用户会问的口语问题，不得出现政策名称、文号、日期、“旧/新/废止/修订”等提示；
   - 包含地域或机型/场景等限定，使问题足够具体；
   - **刻意复用旧条款 A 的独特措辞、数字或门槛**，以便它自然排在 Top-5 前面；
   - 问的是“现在/当前”的规则，而不是历史沿革。
3. ground_truth 必须只依据 B，不得混入 A；关键数字、单位、适用条件要完整。
   **不得使用“B 没写 = 政策取消了”的缺失推断**：如果 B 没有直接规定题目所问的那项补贴/门槛，直接返回 false。
4. 如果 A 并不是被 B 取代，或者两条只是不同事项，返回 false，不要硬编问题。
5. 不得编造原文没有的数字或豁免条件。
"""


@dataclass
class CandidatePair:
    old: Clause
    new: Clause
    topic: str
    region: str
    similarity: float
    note: dict[str, Any] = field(default_factory=dict)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.suffix == ".json":
        # 旧题集自动备份:出题含 LLM 随机性,重跑会淘汰旧题,必须可回溯
        import shutil
        ts = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, path.with_name(f"{path.stem}_backup_{ts}{path.suffix}"))
        log(f"[备份] 旧题集 -> {path.stem}_backup_{ts}{path.suffix}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v or "").strip().lower() in ("true", "1", "yes", "y", "是")


def _clean_one_line(v: Any) -> str:
    return " ".join(str(v or "").split()).strip()


def classify_clauses(clauses: list[Clause], as_of: date) -> tuple[list[Clause], list[Clause]]:
    """按时间元数据分出“真旧”和“真现行”。

    旧：valid_to 有值且早于 as_of（或状态明确废止）
    新：as_of 时点仍有效（valid_to 为空 **或未来到期**，且未废止）

    不能用 `valid_to is None` 判新政策：很多现行政策有明确未来到期日
    （如“有效期至2028-06-25”），把这类排除会漏掉大量新旧对照。
    """
    old: list[Clause] = []
    new: list[Clause] = []
    for c in clauses:
        vt = clause_valid_to(c)
        ok, _ = usable(c, as_of)
        if vt and vt < as_of:
            old.append(c)
        elif ok:
            new.append(c)
    return old, new


def build_candidate_pairs(
    clauses: list[Clause],
    as_of: date,
    *,
    min_similarity: float = 0.04,
    max_pairs_per_topic: int = 6,
) -> tuple[list[CandidatePair], dict[str, Any]]:
    old_docs, new_docs = classify_clauses(clauses, as_of)
    groups_new: dict[tuple[str, str], list[Clause]] = {}
    for c in new_docs:
        groups_new.setdefault((effective_region(c), effective_topic(c)), []).append(c)

    pairs: list[CandidatePair] = []
    skipped = {"no_new_counterpart": 0, "same_text": 0, "low_similarity": 0}
    for o in old_docs:
        key = (effective_region(o), effective_topic(o))
        if key[1] == "未分类":
            continue
        counterparts = groups_new.get(key) or []
        if not counterparts:
            skipped["no_new_counterpart"] += 1
            continue

        local: list[CandidatePair] = []
        for n in counterparts:
            if o.id == n.id:
                continue
            if "".join(o.text.split()) == "".join(n.text.split()):
                skipped["same_text"] += 1
                continue
            sim = bigram_jaccard(o.text, n.text)
            if sim < min_similarity:
                skipped["low_similarity"] += 1
                continue
            # 同一条款的不同版本往往更值得检查；不同政策也允许，但同等相似度下略降权
            penalty = 0.95 if (o.policy_name and o.policy_name == n.policy_name) else 1.0
            local.append(CandidatePair(o, n, key[1], key[0], round(sim * penalty, 4)))
        local.sort(key=lambda p: p.similarity, reverse=True)
        pairs.extend(local[:max_pairs_per_topic])

    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs, {
        "clauses_total": len(clauses),
        "old_clauses": len(old_docs),
        "current_clauses": len(new_docs),
        "candidate_pairs": len(pairs),
        "topics": len({(p.region, p.topic) for p in pairs}),
        "skipped": skipped,
    }


def diversify_pairs(pairs: list[CandidatePair]) -> list[CandidatePair]:
    """按 (region, topic) 轮转，避免单地区的高相似度候选吃满 LLM 预算。

    全国库不能只出福州的题；即使某个地区条款字面更接近，也要保证地域覆盖。
    """
    groups: dict[tuple[str, str], list[CandidatePair]] = {}
    for p in pairs:
        groups.setdefault((p.region, p.topic), []).append(p)
    for g in groups.values():
        g.sort(key=lambda x: x.similarity, reverse=True)
    order = sorted(groups, key=lambda k: groups[k][0].similarity, reverse=True)
    out: list[CandidatePair] = []
    depth = 0
    while True:
        added = False
        for key in order:
            bucket = groups[key]
            if depth < len(bucket):
                out.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1
    return out


def _pair_prompt(pair: CandidatePair, as_of: date) -> str:
    ovf = iso_or_none(clause_valid_from(pair.old)) or "无"
    ovt = iso_or_none(clause_valid_to(pair.old)) or "无"
    nvf = iso_or_none(clause_valid_from(pair.new)) or "无"
    nvt = iso_or_none(clause_valid_to(pair.new)) or "无（现行有效）"
    return (
        f"评估时点：{as_of.isoformat()}\n"
        f"主题：{pair.topic}\n地域：{pair.region}\n\n"
        f"【条款 A · 旧】《{pair.old.policy_name}》{pair.old.article}\n"
        f"valid_from={ovf}, valid_to={ovt}\n{pair.old.text}\n\n"
        f"【条款 B · 现行】《{pair.new.policy_name}》{pair.new.article}\n"
        f"valid_from={nvf}, valid_to={nvt}\n{pair.new.text}"
    )


def generate_item(
    client: ChatClient,
    pair: CandidatePair,
    as_of: date,
    *,
    index: int,
) -> dict[str, Any] | None:
    raw = client.chat_json(
        BENCH_SYSTEM, _pair_prompt(pair, as_of),
        temperature=0.0, max_tokens=1200, thinking=False,
    )
    if not _as_bool(raw.get("has_substantive_difference")):
        # 必须留痕：否则日志里只看到"通过/二审否决"，看不出初审淘汰了多少组，
        # 论文里没法报告"候选→成题"的完整漏斗
        log(f"  [SKIP] 初审：两条不构成实质规则变化（{_clean_one_line(raw.get('reasoning'))[:60]}）")
        return None

    q = _clean_one_line(raw.get("question"))
    gt = _clean_one_line(raw.get("ground_truth"))
    if len(q) < 8 or len(gt) < 6:
        log(f"  [SKIP] 问题或答案过短：{q[:40]!r} / {gt[:40]!r}")
        return None

    # 反泄漏校验：问题本身不能告诉模型哪条是旧的
    banned = ("旧政策", "新政策", "已废止", "已失效", "现行有效", "修订前", "修订后")
    if any(w in q for w in banned):
        log(f"  [SKIP] 问题泄漏时效标签：{q[:60]}")
        return None
    for name in (pair.old.policy_name, pair.new.policy_name):
        if name and len(name) >= 5 and name in q:
            log(f"  [SKIP] 问题直接点名政策：{q[:60]}")
            return None

    ctype = str(raw.get("conflict_type") or "other").strip()
    if ctype not in CONFLICT_TYPES:
        ctype = "other"

    return {
        "id": f"TQ-{index:04d}",
        "topic": pair.topic,
        "region": pair.region,
        "question": q,
        "ground_truth": gt,
        # ---- 以下为可审计溯源字段，评估时不会传给 Temporal_Check ----
        "old_clause_text": pair.old.text,
        "new_clause_text": pair.new.text,
        "old_clause_id": pair.old.id,
        "new_clause_id": pair.new.id,
        "old_policy_name": pair.old.policy_name,
        "new_policy_name": pair.new.policy_name,
        "old_valid_from": iso_or_none(clause_valid_from(pair.old)),
        "old_valid_to": iso_or_none(clause_valid_to(pair.old)),
        "new_valid_from": iso_or_none(clause_valid_from(pair.new)),
        "new_valid_to": iso_or_none(clause_valid_to(pair.new)),
        "conflict_type": ctype,
        "llm_reasoning": _clean_one_line(raw.get("reasoning"))[:160],
        "as_of": as_of.isoformat(),
    }


def verify_item(client: ChatClient, pair: CandidatePair, item: dict) -> tuple[bool, str]:
    user = (
        f"【问题】\n{item['question']}\n\n"
        f"【标准答案】\n{item['ground_truth']}\n\n"
        f"【旧条款】\n{pair.old.text[:1800]}\n\n"
        f"【新条款】\n{pair.new.text[:1800]}"
    )
    raw = client.chat_json(VERIFY_SYSTEM, user, temperature=0.0, max_tokens=350, thinking=False)
    valid = _as_bool(raw.get("valid")) and _as_bool(raw.get("same_specific_issue")) \
        and _as_bool(raw.get("gt_supported_by_new"))
    return valid, _clean_one_line(raw.get("reason"))[:180]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="从真实政策库构建时序冲突 benchmark")
    add_backend_args(ap)
    ap.add_argument("--as-of", default=date.today().isoformat(), help="评估时点 YYYY-MM-DD")
    ap.add_argument("--output", default=str(ROOT / "benchmark.json"))
    ap.add_argument("--manifest", default="", help="默认 <output 同名>.manifest.json")
    ap.add_argument("--max-pairs", type=int, default=80, help="最多送多少候选对给 LLM（成本上限）")
    ap.add_argument("--max-pairs-per-topic", type=int, default=1,
                    help="每条旧条款最多配几个新条款；默认1，避免把同主题的不同条款误当替代关系")
    ap.add_argument("--max-items", type=int, default=30, help="生成够多少题就停止；0=不限")
    ap.add_argument("--min-similarity", type=float, default=0.12,
                    help="候选对的二元组 Jaccard 下限；过低会把同主题不同条款配在一起")
    ap.add_argument("--require-old-in-topk", type=int, default=0, metavar="K",
                    help="生成后先检索一次，旧条款不在 Top-K 则丢弃（建议首轮人工核验时用）")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "qwen", "glm"],
                    help="出题模型供应商")
    return ap.parse_args()


def main() -> int:
    load_env_file(ROOT / ".env")
    args = parse_args()
    as_of = date.fromisoformat(args.as_of)
    output = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")

    try:
        backend = build_backend(args)
    except Exception as e:
        log(f"[ERROR] {e}")
        return 2

    log(f"读取条款：{backend.describe()}")
    try:
        clauses = backend.all_clauses()
    except Exception as e:
        log(f"[ERROR] 读取全部条款失败：{e}")
        return 2
    if not clauses:
        log("[ERROR] 库里没有条款")
        return 2

    pairs, pair_stats = build_candidate_pairs(
        clauses, as_of,
        min_similarity=args.min_similarity,
        max_pairs_per_topic=args.max_pairs_per_topic,
    )
    pairs = diversify_pairs(pairs)
    log(f"候选对：{len(pairs)}；{json.dumps(pair_stats, ensure_ascii=False)}")
    if not pairs:
        log(
            "[ERROR] 没有可配对的新旧条款。请检查：\n"
            "  1) 旧条款 metadata 是否有 valid_to（或 expiry_date）且早于 as-of；\n"
            "  2) 新条款 valid_to 是否为空；\n"
            "  3) topic/region 是否一致（topic 读取 metadata.topic/category/section，缺失时按主题词典推断）。"
        )
        return 2

    try:
        client = (ChatClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
                  if (args.api_key or args.base_url or args.model)
                  else provider_client(args.provider))
    except LLMError as e:
        log(f"[ERROR] {e}")
        return 3

    items: list[dict[str, Any]] = []
    attempted = 0
    skipped_by_llm = 0
    skipped_by_retrieval = 0
    failed = 0
    seen_questions: set[str] = set()
    # 同一条旧条款只出一道题，否则“6个月租金”会被重复生成 3 次，虚增样本量
    seen_old_clauses: set[str] = set()

    for pair in pairs:
        if args.max_pairs and attempted >= args.max_pairs:
            break
        if args.max_items and len(items) >= args.max_items:
            break
        attempted += 1
        log(
            f"[{attempted}/{min(len(pairs), args.max_pairs or len(pairs))}] "
            f"{pair.region}·{pair.topic} sim={pair.similarity:.3f}"
        )
        try:
            item = generate_item(client, pair, as_of, index=len(items) + 1)
        except (LLMError, ValueError) as e:
            failed += 1
            log(f"  [FAIL] {str(e)[:180]}")
            continue

        if item is None:
            skipped_by_llm += 1
            if args.sleep:
                time.sleep(args.sleep)
            continue

        if item["old_clause_id"] in seen_old_clauses:
            log("  [SKIP] 同一旧条款已出过题")
            skipped_by_llm += 1
            continue
        norm_q = "".join(item["question"].split())
        if norm_q in seen_questions:
            log("  [SKIP] 问题重复")
            skipped_by_llm += 1
            continue

        if args.require_old_in_topk > 0:
            top = backend.similarity_search(item["question"], k=args.require_old_in_topk)
            if not any(clause_matches(
                c,
                expected_id=item["old_clause_id"],
                expected_text=item["old_clause_text"],
            ) for c in top):
                skipped_by_retrieval += 1
                log(f"  [SKIP] 旧条款未进入 Top-{args.require_old_in_topk}，问题不够迷惑")
                continue

        # 第二道 LLM 审核：防止“同主题不同条款”被误配，防止缺失推断编造标准答案
        try:
            valid, why = verify_item(client, pair, item)
        except (LLMError, ValueError) as e:
            failed += 1
            log(f"  [FAIL] 审核调用失败：{str(e)[:100]}")
            continue
        if not valid:
            log(f"  [SKIP] 审核不通过：{why}")
            skipped_by_llm += 1
            continue

        seen_questions.add(norm_q)
        seen_old_clauses.add(item["old_clause_id"])
        item["id"] = f"TQ-{len(items) + 1:04d}"
        items.append(item)
        log(f"  [NEW] {item['id']} {item['question'][:56]}")

        # 每生成一题就落盘，限流/中断时不必从头烧 token
        atomic_write_json(output, items)
        if args.sleep:
            time.sleep(args.sleep)

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of": as_of.isoformat(),
        "backend": backend.describe(),
        "model": getattr(client, "model", ""),
        "pair_stats": pair_stats,
        "attempted_pairs": attempted,
        "accepted": len(items),
        "skipped_by_llm": skipped_by_llm,
        "skipped_by_retrieval": skipped_by_retrieval,
        "failed_calls": failed,
        "params": {
            "min_similarity": args.min_similarity,
            "max_pairs": args.max_pairs,
            "max_pairs_per_topic": args.max_pairs_per_topic,
            "require_old_in_topk": args.require_old_in_topk,
        },
    }
    atomic_write_json(manifest_path, manifest)
    if not items:
        atomic_write_json(output, [])
    backend.close()

    print(f"\n已生成 {len(items)} 题 → {output}")
    print(f"manifest → {manifest_path}")
    if failed:
        print(f"[WARN] {failed} 次 LLM 调用失败，详见日志；可调整 --max-pairs 后重跑")
    return 0 if items else 2


if __name__ == "__main__":
    raise SystemExit(main())
