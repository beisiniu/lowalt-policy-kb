#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从真实 Chroma 库生成“有原文依据”的覆盖性问答探针。

它和 build_benchmark.py 的分工：
  - build_benchmark.py：挖新旧冲突，测“过期政策毒化”
  - build_qa_probes.py：按地区/主题抽样现行条款，测“库能不能被正确检索和回答”

自动出题同样要求问题不含政策名/条款号，避免答案泄漏；ground_truth 只来自所给条款。

用法：
    python build_qa_probes.py --per-region 1 --max-items 16 \
      --chroma-dir ./chroma_lowalt --collection lowalt_policy \
      --embedding-function bge-zh --output qa_probes.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.llm import ChatClient, LLMError
from policycrawler.utils import load_env_file
from rag_eval.backends import add_backend_args, build_backend
from rag_eval.models import Clause

ROOT = Path(__file__).resolve().parent
SYSTEM = """你是低空政策问答测试集的命题专家。
我会给你一条现行政策条款。请只依据该条款生成一道真实用户会问的问题和标准答案。

只输出 JSON：
{
  "question": "自然的中文问题",
  "ground_truth": "只依据条款的标准答案，保留关键数字、单位、适用条件",
  "key_fact": "答案中最关键的一个数字或短语"
}

要求：
1. question 必须包含地域或具体场景；不得出现政策名称、条款号、发布日期；
2. 不得出现“旧政策/新政策/已废止/修订”等时效提示；
3. 问题必须能由该条款回答，不要问条款之外的知识；
4. 条款如果只是目录、职责分工、原则性表述，返回 {"skip": true}；
5. 不得编造数字或豁免条件。
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def atomic_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def select_clauses(clauses: list[Clause], per_region: int, per_policy: int = 1) -> list[Clause]:
    """每个地区选信息量较高、不重复政策的条款。"""
    by_region: dict[str, list[Clause]] = defaultdict(list)
    for c in clauses:
        text = c.text.strip()
        if len(text) < 120 or len(text) > 900:
            continue
        if len(c.metadata.get("article", "")) == 0 and len(text) < 250:
            continue
        if not re.search(r"\d|应当|无需|不得|可以|申请|补贴|奖励|审批", text):
            continue
        by_region[c.region or "未标注"].append(c)

    picked: list[Clause] = []
    for region in sorted(by_region):
        # 先按 policy 去重，再取文字信息量高的
        seen_policy: dict[str, int] = {}
        candidates = sorted(
            by_region[region],
            key=lambda c: (
                len(re.findall(r"\d", c.text)), len(c.text)
            ),
            reverse=True,
        )
        got = 0
        for c in candidates:
            key = c.metadata.get("policy_id") or c.policy_name
            if seen_policy.get(key, 0) >= per_policy:
                continue
            seen_policy[key] = seen_policy.get(key, 0) + 1
            picked.append(c)
            got += 1
            if got >= per_region:
                break
    return picked


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description="从真实 Chroma 库生成覆盖性 QA 探针")
    add_backend_args(ap)
    ap.add_argument("--output", default=str(ROOT / "qa_probes.json"))
    ap.add_argument("--per-region", type=int, default=1)
    ap.add_argument("--per-policy", type=int, default=1,
                    help="同一篇政策最多出几题；调大可在政策数不变时扩充题量")
    ap.add_argument("--max-items", type=int, default=16)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    try:
        backend = build_backend(args)
        clauses = backend.all_clauses()
    except Exception as e:
        log(f"[ERROR] 连接知识库失败：{e}")
        return 2
    if not clauses:
        log("[ERROR] 知识库为空")
        return 2

    candidates = select_clauses(clauses, args.per_region, args.per_policy)
    log(f"条款 {len(clauses)} 条；候选 {len(candidates)} 条，覆盖 {len(set(c.region for c in candidates))} 个地域")
    if not candidates:
        return 2
    try:
        client = ChatClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    except LLMError as e:
        log(f"[ERROR] {e}")
        return 3

    probes: list[dict] = []
    for i, c in enumerate(candidates, 1):
        if len(probes) >= args.max_items:
            break
        log(f"[{i}/{len(candidates)}] {c.region:8} {c.policy_name[:36]} {c.article}")
        user = (
            f"地域：{c.region}\n主题：{c.topic or '未标注'}\n"
            f"条款：\n{c.text}"
        )
        try:
            raw = client.chat_json(SYSTEM, user, temperature=0.0, max_tokens=700, thinking=False)
        except LLMError as e:
            log(f"  [FAIL] {str(e)[:120]}")
            continue
        if raw.get("skip") is True:
            log("  [SKIP] 目录/原则性条款")
            continue
        q = " ".join(str(raw.get("question") or "").split())
        gt = " ".join(str(raw.get("ground_truth") or "").split())
        if len(q) < 8 or len(gt) < 6:
            log("  [SKIP] 问题或答案过短")
            continue
        if any(w in q for w in ("旧政策", "新政策", "已废止", "修订", "第几条")):
            log("  [SKIP] 问题泄漏")
            continue
        if c.policy_name and len(c.policy_name) >= 6 and c.policy_name in q:
            log("  [SKIP] 问题点名政策")
            continue
        probes.append({
            "id": f"QA-{len(probes)+1:03d}",
            "region": c.region,
            "topic": c.topic or "未标注",
            "question": q,
            "ground_truth": gt,
            "key_fact": str(raw.get("key_fact") or "")[:100],
            "source_policy_name": c.policy_name,
            "source_article": c.article,
            "source_url": c.metadata.get("source_url", ""),
            "source_clause_text": c.text,
            "as_of": c.metadata.get("valid_from", ""),
        })
        log(f"  [NEW] {q[:64]}")
        atomic_write(Path(args.output), probes)
        if args.sleep:
            time.sleep(args.sleep)

    atomic_write(Path(args.output), probes)
    backend.close()
    print(f"\n生成 {len(probes)} 题 → {args.output}")
    print(f"覆盖地域：{len(set(x['region'] for x in probes))} 个")
    return 0 if probes else 2


if __name__ == "__main__":
    raise SystemExit(main())
