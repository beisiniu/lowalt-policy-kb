"""TQ-v2 题库定型：控制题型配比 + 生成人读报表。

为什么需要这一步：
  前提错误题按地区逐个生成（67 个地区 → 67 道），会把题库变成单一题型。
  按"场景不重复 + 地区分散"抽稀到 22 道，使任何一类都不超过总数四成
  （文档 7.2 验收标准第三条）。

用法：
  python tools/curate_tq2.py                       # 定型 + 出报表
  python tools/curate_tq2.py --fp-limit 22
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "national" / "benchmark_tq2.json"
OUT_TABLE = ROOT / "output" / "benchmark_tq2_table.md"

TYPE_CN = {
    "subsidy_changed": "补贴标准", "threshold_changed": "门槛数值",
    "scope_changed": "适用范围", "eligibility_changed": "资格资质",
    "procedure_changed": "办理流程", "repeal_recognition": "失效识别",
    "false_premise": "前提错误",
}
SCENE_KEYS = ("婚庆", "植保", "巡检", "测绘", "灯光", "外卖", "急救", "影视",
              "防火", "救援", "清洗", "监测", "快递", "观光", "理货", "渔业",
              "桥梁", "中继", "气象", "疏导")


def scene_of(item: dict) -> str:
    name = str((item.get("gold_evidence") or {}).get("fake_name") or "")
    for k in SCENE_KEYS:
        if k in name:
            return k
    return name[:6] or "其他"


def pick_fp(fp_items: list[dict], limit: int) -> list[dict]:
    """前提错误题抽稀：先按场景去重，再按地区去重，最后按剩余顺序补足。"""
    chosen, seen_scene, seen_region = [], set(), set()
    for it in fp_items:
        sc, rg = scene_of(it), it["region"]
        if sc in seen_scene:
            continue
        chosen.append(it)
        seen_scene.add(sc)
        seen_region.add(rg)
        if len(chosen) >= limit:
            return chosen
    for it in fp_items:
        if it in chosen:
            continue
        if it["region"] in seen_region:
            continue
        chosen.append(it)
        seen_region.add(it["region"])
        if len(chosen) >= limit:
            break
    for it in fp_items:
        if len(chosen) >= limit:
            break
        if it not in chosen:
            chosen.append(it)
    return chosen[:limit]


def write_table(items: list[dict], path: Path) -> None:
    by_type = Counter(x["conflict_type"] for x in items)
    by_region = Counter(x["region"] for x in items)
    total = len(items)
    lines = [
        "# TQ-v2 题库：问题、标准答案与金标证据",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　|　基准日：2026-10-27　|　**{total} 道**",
        "",
        f"地区 **{len(by_region)}** 个　|　题型 **{len(by_type)}** 类　|　"
        f"最大单类占比 **{max(by_type.values())/total:.0%}**　|　最大单地区占比 **{max(by_region.values())/total:.0%}**",
        "",
        "## 分布",
        "",
        "| 题型 | 题数 | 占比 |",
        "| --- | --- | --- |",
    ]
    for k, v in by_type.most_common():
        lines.append(f"| {TYPE_CN.get(k, k)}（{k}） | {v} | {v/total:.0%} |")
    lines += ["", "| 地区 | 题数 |", "| --- | --- |"]
    for k, v in by_region.most_common():
        lines.append(f"| {k} | {v} |")

    # 总览表：一屏看完所有题的问题/答案/证据
    lines += ["", "## 总览（一题一行）", "",
              "| ID | 地区 | 题型 | 问题 | 标准答案（截断） | 金标证据切片 | 摘录（截断） |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for it in items:
        ge = it.get("gold_evidence") or {}
        q = str(it["question"]).replace("|", "｜")[:44]
        gt = str(it["ground_truth"]).replace("|", "｜").replace("\n", " ")[:40]
        cid = str(ge.get("chunk_id") or "")[:18]
        if it["conflict_type"] == "false_premise":
            n = len(ge.get("checked_policies") or [])
            quote = f"已核对 {n} 份现行政策，均无此名称"
        else:
            quote = str(ge.get("quote") or "").replace("|", "｜").replace("\n", " ")[:34]
        lines.append(f"| {it['id']} | {it['region']} | {TYPE_CN.get(it['conflict_type'], it['conflict_type'])} "
                     f"| {q} | {gt} | `{cid}` | {quote} |")
    lines += ["", "## 逐题明细", ""]
    for it in items:
        ge = it.get("gold_evidence") or {}
        tag = TYPE_CN.get(it["conflict_type"], it["conflict_type"])
        lines += [
            f"### {it['id']}　[{tag}]　{it['region']}",
            "",
            f"**问题**：{it['question']}",
            "",
            f"**标准答案**：{it['ground_truth']}",
            "",
            "**金标证据**：",
            f"- 切片编号：`{ge.get('chunk_id', '')}`",
            f"- 出处：《{ge.get('policy_name', '')}》{ge.get('article', '') or ''}",
            f"- 原文摘录：「{ge.get('quote', '')}」",
            f"- 字符区间：{ge.get('char_span')}",
        ]
        if ge.get("negative"):
            lines.append(f"- 类型：**否定证据** —— 编造名「{ge.get('fake_name', '')}」在该地区现行政策中不存在")
            cps = ge.get("checked_policies") or []
            if cps:
                lines.append(f"- 已核对的现行政策：{'；'.join(cps[:6])}{' 等' if len(cps) > 6 else ''}")
                if len(cps) < 2:
                    lines.append("- ⚠ 证据强度说明：该地区库内仅收录 1 份现行政策，"
                                 "只能断言「本项目收录的政策中没有」，不能断言「该地区不存在」")
        lines += ["", "<details><summary>新旧条款对照</summary>", ""]
        if it.get("old_policy"):
            lines += [f"**旧（已失效）**：《{it['old_policy']}》　失效日 {it.get('old_valid_to') or '—'}",
                      f"> {(it.get('old_clause_text') or '')[:350]}", ""]
        lines += [f"**现行**：《{it['new_policy']}》　生效日 {it.get('new_valid_from') or '—'}",
                  f"> {(it.get('new_clause_text') or '')[:350]}", "", "</details>", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-limit", type=int, default=22, help="前提错误题保留上限")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--table", default=str(OUT_TABLE))
    a = ap.parse_args()

    items = json.loads(Path(a.src).read_text(encoding="utf-8"))
    # 去重：同一证据切片 + 同一摘录句 → 同一事实，重复出题只会虚增样本
    seen, dedup = set(), []
    for it in items:
        ge = it.get("gold_evidence") or {}
        key = (ge.get("chunk_id"), str(ge.get("quote") or "")[:40])
        if key in seen:
            print(f"[去重] {it['id']} 与已有题目共用同一证据（{ge.get('chunk_id','')[:16]}）")
            continue
        seen.add(key)
        dedup.append(it)
    items = dedup
    fp = [x for x in items if x["conflict_type"] == "false_premise"]
    rest = [x for x in items if x["conflict_type"] != "false_premise"]
    kept_fp = pick_fp(fp, a.fp_limit)
    final = rest + kept_fp
    final.sort(key=lambda x: (x["conflict_type"], x["region"], x["id"]))

    Path(a.src).write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    write_table(final, Path(a.table))

    by_type = Counter(x["conflict_type"] for x in final)
    by_region = Counter(x["region"] for x in final)
    print(f"题库定型：{len(rest)} 道冲突题 + {len(kept_fp)}/{len(fp)} 道前提错误题 = **{len(final)} 道**")
    print("题型：", {TYPE_CN.get(k, k): v for k, v in by_type.most_common()})
    print(f"地区 {len(by_region)} 个，最大单地区 {by_region.most_common(1)[0][0]} "
          f"{by_region.most_common(1)[0][1]} 道（{by_region.most_common(1)[0][1]/len(final):.0%}）")
    print(f"最大单题型占比 {max(by_type.values())/len(final):.0%}")
    print("报表 →", a.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
