#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""毒饵有效性检验：判断旧条款 A 是否构成对某道题的有效毒饵。

背景：条款对照题的正确答案锚定现行条款 B，旧条款 A 的角色是毒饵——模型若采纳 A
就会答错，这才是「时序毒化」。但出题环节只判定了「A 与 B 有实质差异」（文段级），
并未验证「题目所问的点正落在差异上」（题目级）。若题目问在 A 与 B 结论相同的点上，
A 就是无效毒饵：穿透了也不会毒化，等于白测。

两种判据：
  --mode design（默认，推荐）：静态判定 A 与「本题标准答案」的关系，不跑模型。
      INDUCES_ERROR    A 对本题给出明确立场且与标准答案相反 → 有效毒饵
      LACKS_INFO       A 完全未涉及本题 → 不构成毒饵（读了也只能拒答或猜）
      SAME_CONCLUSION  A 对本题的立场与标准答案一致 → 无效毒饵（读 A 也能答对）
      优点：判定的是「题目属性」，不受某个模型先验知识的影响，可复现、可审计。

  --mode behavior：行为检验，只把 A 喂给模型作答，再判是否与标准答案冲突。
      判定的是「该模型是否被带偏」，会混入模型先验知识，只作参考。

用法：
    python tools/check_bait.py                    # 静态判定全部条款对照题
    python tools/check_bait.py --mode behavior    # 行为检验
    python tools/check_bait.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from policycrawler.llm import LLMError, provider_client  # noqa: E402
from policycrawler.utils import load_env_file  # noqa: E402

DESIGN_SYSTEM = """你是时序政策评测的题目审核员。给你一道题、它的标准答案、以及一条**已失效的旧条款 A**。
请判断：一个只看得到 A、看不到其他任何材料的模型，面对这道题会怎样。

只输出一个 JSON 对象：
{"verdict": "INDUCES_ERROR|LACKS_INFO|SAME_CONCLUSION", "a_stance": "A 对本题的立场，不超过 25 字", "reason": "不超过 40 字"}

三类的定义（互斥，必须选最贴切的一类）：
- INDUCES_ERROR：A 对这道题给出了**明确立场**，且该立场与标准答案**相反**。
  例：标准答案「必须纳入规划才能申请」，A 写「建成即按投入补贴」（A 隐含无需规划）→ A 会诱导出错误答案。
  数值题：A 的标准与答案不同（答案 50 万元、A 写 15 万元）也属此类。
- SAME_CONCLUSION：A 对这道题的立场与标准答案**一致**，或 A 已包含标准答案的关键限定。
  例：标准答案「须实名登记才能飞」，A 也写「必须完成实名登记」→ 读 A 也能答对。
- LACKS_INFO：A 完全没有涉及这道题所问的事项，读了也答不出。
  例：题目问「飞行计划通过什么渠道提交」，而 A 讲的是禁飞区范围。

判断要点：**只看 A 是否覆盖了本题所问的那个点**，不要用你自己的政策知识补全 A 没写的内容。
A 没写不等于 A 说了相反的话——没写就是 LACKS_INFO。"""

ANSWER_SYSTEM = """你是政策问答助手。请仅依据用户提供的那一条政策条款作答，不要引入外部知识。
若该条款没有涉及所问事项，直接说明"该条款未作规定"。
回答控制在两句话以内，直接给结论，不要复述条款。"""

RELATION_SYSTEM = """你是评测判分员。给你【问题】【标准答案】【依据某一条政策条款给出的回答】，
判断该回答与标准答案的关系，只输出一个 JSON 对象：

{"relation": "CONFLICT|SAME|INSUFFICIENT", "reason": "不超过 30 字"}

- CONFLICT：回答对题目所问事项的**实际立场**与标准答案相反。注意：判断的是立场，
  不是有没有出现"不"字。回答"不是，只需按投入比例补贴"实际上是在说"可以申请"，
  属于认可前提，应判 SAME。
- SAME：对题目所问事项的实际立场与标准答案一致。
- INSUFFICIENT：没有给出明确立场，或表示该条款未作规定、无法回答。

只输出 JSON，不要任何其他文字。"""


def norm_verdict(v: str) -> str:
    t = str(v or "").strip().upper()
    return t if t in ("INDUCES_ERROR", "LACKS_INFO", "SAME_CONCLUSION") else "UNKNOWN"


def build_design_prompt(item: dict) -> str:
    return (
        f"【题目】\n{item['question']}\n\n"
        f"【标准答案】\n{item['ground_truth']}\n\n"
        f"【旧条款 A（已失效）】《{item.get('old_policy') or '相关政策'}》"
        f"{item.get('old_status') or ''}\n{item['old_clause_text']}"
    )


def build_answer_prompt(item: dict) -> str:
    policy = item.get("old_policy") or "相关政策"
    return (
        f"【政策条款】《{policy}》\n{item['old_clause_text']}\n\n"
        f"【用户问题】\n{item['question']}"
    )


def build_relation_prompt(item: dict, answer: str) -> str:
    return (
        f"【问题】\n{item['question']}\n\n"
        f"【标准答案】\n{item['ground_truth']}\n\n"
        f"【依据某一条政策条款给出的回答】\n{answer}"
    )


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    load_env_file(ROOT / ".env")
    ap = argparse.ArgumentParser(description="毒饵有效性检验")
    ap.add_argument("--benchmark", default=str(ROOT / "data" / "national" / "benchmark_tq2.json"))
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", default="design", choices=["design", "behavior"])
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    out_path = Path(args.out or (ROOT / "data" / "national" /
                                 f"bait_check_{args.mode}.json"))

    payload = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, dict) else payload
    items = [i for i in items if i.get("conflict_type") != "false_premise"]
    items = [i for i in items if (i.get("old_clause_text") or "").strip()]
    if args.limit:
        items = items[: args.limit]
    log(f"模式={args.mode}，条款对照题 {len(items)} 道")

    client = provider_client(args.provider)
    log(f"模型：{client.model}")

    records = []
    for i, item in enumerate(items, 1):
        rec = {
            "id": item["id"],
            "region": item.get("region"),
            "conflict_type": item.get("conflict_type"),
            "answer_type": item.get("answer_type"),
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "old_policy": item.get("old_policy"),
            "old_clause_text_head": item["old_clause_text"][:120],
        }
        try:
            if args.mode == "design":
                raw = client.chat_json(
                    DESIGN_SYSTEM, build_design_prompt(item),
                    temperature=0.0, max_tokens=260, thinking=False,
                )
                rec["verdict"] = norm_verdict(raw.get("verdict"))
                rec["a_stance"] = str(raw.get("a_stance") or "")[:60]
                rec["reason"] = str(raw.get("reason") or "")[:80]
                rec["bait_effective"] = rec["verdict"] == "INDUCES_ERROR"
            else:
                answer = client.chat(
                    [
                        {"role": "system", "content": ANSWER_SYSTEM},
                        {"role": "user", "content": build_answer_prompt(item)},
                    ],
                    temperature=0.0, max_tokens=400, json_mode=False, thinking=False,
                )
                raw = client.chat_json(
                    RELATION_SYSTEM, build_relation_prompt(item, answer),
                    temperature=0.0, max_tokens=200, thinking=False,
                )
                relation = str(raw.get("relation") or "").strip().upper()
                rec["a_only_answer"] = " ".join(str(answer).split())
                rec["relation"] = relation
                rec["reason"] = str(raw.get("reason") or "")[:80]
                rec["bait_effective"] = relation == "CONFLICT"
        except LLMError as e:
            log(f"  [{i}/{len(items)}] {item['id']} 调用失败：{str(e)[:100]}")
            continue

        records.append(rec)
        tag = {"INDUCES_ERROR": "有效", "SAME_CONCLUSION": "无效·结论相同",
               "LACKS_INFO": "无效·条款无关", "CONFLICT": "被带偏",
               "SAME": "未被带偏", "INSUFFICIENT": "答不出"}.get(
            rec.get("verdict") or rec.get("relation"), "?")
        log(f"  [{i}/{len(items)}] {rec['id']} {str(rec['conflict_type'])[:18]:18} → {tag}")
        if args.sleep:
            time.sleep(args.sleep)

    summary = {"mode": args.mode, "checked": len(records), "by_conflict_type": {}}
    if args.mode == "design":
        eff = sum(1 for r in records if r.get("bait_effective"))
        summary["bait_effective"] = eff
        summary["bait_ineffective"] = len(records) - eff
        summary["effective_rate"] = round(eff / len(records), 4) if records else 0.0
        for r in records:
            v = r.get("verdict")
            d = summary["by_conflict_type"].setdefault(r["conflict_type"], {})
            d[v] = d.get(v, 0) + 1
    else:
        eff = sum(1 for r in records if r.get("bait_effective"))
        summary["model_got_misled"] = eff
        summary["misled_rate"] = round(eff / len(records), 4) if records else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"summary": summary, "items": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out_path)

    log("")
    if args.mode == "design":
        log(f"有效毒饵 {summary['bait_effective']}/{len(records)}"
            f"（{summary['effective_rate']:.1%}）")
        for k, d in sorted(summary["by_conflict_type"].items()):
            log(f"  {k:20} {d}")
    else:
        log(f"模型被带偏 {summary['model_got_misled']}/{len(records)}"
            f"（{summary['misled_rate']:.1%}）")
    log(f"结果写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
