#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段三边界用例测试（切分 / 过滤 / 图）

  F1 超长条款被二次切分，但保留条号且不超长
  F2 省级政策能命中市级提问（上位规定适用于下级行政区）
  F3 尚未生效的政策被过滤（不能拿来回答"现在要不要审批"）
  F4 已废止的政策被过滤，即使没写失效日期
  F5 同一篇政策重复入库不产生重复 chunk
  F6 图不会无限循环：expand 最多 1 次，steps 有界
  F7 没有「第X条」的文本（通知/函）也能切分且不丢句子
  F8 正文为空时明确告警，而不是静默产出 0 个 chunk

运行：python tests/verify_phase3_edges.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.graph import PolicyQA                      # noqa: E402
from kb.retriever import PolicyRetriever           # noqa: E402
from kb.schema import detect_region, freshness, is_usable, region_matches   # noqa: E402
from kb.splitter import PolicyChunkSplitter, build_documents  # noqa: E402
from kb.store import load_index, save_index        # noqa: E402

PY = sys.executable


def ok(flag: bool, label: str, extra: str = "") -> bool:
    print(f"[{'PASS' if flag else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return flag


def doc(text: str, name: str, region: str = "深圳市", status: str = "现行有效",
        effective: str = "2026-01-01", expiry: str = "无", pid: str | None = None):
    return build_documents(text, {
        "policy_name": name, "policy_id": pid or name, "issue_date": "2025-12-01",
        "effective_date": effective, "expiry_date": expiry, "apply_region": region,
        "status": status, "source_url": "http://example.com/x",
    })


# ---------------------------------------------------------------- F1
def case_f1() -> bool:
    long_body = "第一条 " + "".join(f"第{i}款规定无人机飞行应当遵守相关管理要求。" for i in range(1, 60))
    docs = doc(long_body, "超长条款办法")
    longest = max(len(d.page_content) for d in docs)
    arts = {d.metadata["article"] for d in docs}
    flag = len(docs) > 1 and longest <= 700 and arts == {"第一条"}
    return ok(flag, "F1 超长条款二次切分（每块 ≤ 700 字，条号保留）",
              f"{len(docs)} 块，最长 {longest} 字")


# ---------------------------------------------------------------- F2
def case_f2() -> bool:
    direct = region_matches("广东省", "深圳市")
    province_hit = is_usable({"apply_region": "广东省", "status": "现行有效",
                              "effective_date": "2026-01-01", "expiry_date": "无"}, None, "深圳市")
    other = region_matches("广东省", "上海市")
    flag = direct and province_hit and not other
    return ok(flag, "F2 省级政策命中市内提问，但不串到外省",
              f"粤→深 {direct} / 粤→沪 {other}")


# ---------------------------------------------------------------- F3 / F4
def case_f3_f4() -> bool:
    future = {"apply_region": "深圳市", "status": "现行有效",
              "effective_date": "2099-01-01", "expiry_date": "无"}
    dead = {"apply_region": "深圳市", "status": "已废止",
            "effective_date": "2020-01-01", "expiry_date": "无"}
    expired = {"apply_region": "深圳市", "status": "现行有效",
               "effective_date": "2020-01-01", "expiry_date": "2021-01-01"}
    ok3 = not freshness(future)[0] and "尚未生效" in freshness(future)[1]
    ok4 = (not freshness(dead)[0]) and (not freshness(expired)[0])
    flag = ok3 and ok4
    return ok(flag, "F3/F4 未生效、已废止、已过期三类都被拦下",
              f"未生效={freshness(future)[1]}；已废止={freshness(dead)[1]}；过期={freshness(expired)[1]}")


# ---------------------------------------------------------------- F5
def case_f5() -> bool:
    tmp = Path(tempfile.mkdtemp())
    index_dir = tmp / "idx"
    src = ROOT / "data" / "extracted" / "2026-03-15_深圳经济特区低空经济产业促进条例.json"
    if not src.exists():
        return ok(False, "F5 重复入库不产生重复 chunk", "缺少样例 json")

    def run(*extra: str) -> int:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        p = subprocess.run([PY, "kb_ingest.py", str(src), "--index-dir", str(index_dir), *extra],
                           cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
        return p.returncode

    run("--rebuild")
    n1 = len(load_index(index_dir)[0])
    run("--append")
    n2 = len(load_index(index_dir)[0])
    flag = n1 > 0 and n1 == n2
    return ok(flag, "F5 同一政策重复入库 → chunk 数不变", f"{n1} → {n2}")


# ---------------------------------------------------------------- F6
def case_f6() -> bool:
    docs = doc("第一条 " + "低空飞行基础设施建设由市交通运输部门负责。" * 12, "无关政策A")
    docs += doc("第二条 " + "航空器维修人员培训纳入职业技能提升计划。" * 12, "无关政策B")
    qa = PolicyQA(PolicyRetriever(docs), mode="extractive", max_expand=1)
    out = qa.ask("火星上飞无人机需要审批吗？")
    expands = sum(1 for s in out["steps"] if s == "expand")
    flag = expands <= 1 and len(out["steps"]) <= 8 and out["route"].startswith("refuse")
    return ok(flag, "F6 图有界：expand 最多 1 次，不会死循环",
              f"steps={out['steps']}")


# ---------------------------------------------------------------- F7
def test_f7() -> bool:
    # 故意写一篇 1500+ 字、完全没有「第X条」的通知
    filler = "各单位应当对照整治重点逐项开展自查自纠，建立问题台账，明确整改责任人和整改时限。\n" * 30
    text = ("关于开展低空飞行活动专项整治的通知\n"
            "各有关单位：\n"
            "为进一步规范低空飞行秩序，决定开展专项整治。\n"
            "一、整治范围为全市真高1000米以下空域。\n"
            + filler +
            "二、各单位应当于2026年3月1日前完成自查。\n")
    docs = doc(text, "专项整治通知")
    joined = "".join(d.page_content for d in docs)
    longest = max(len(d.page_content) for d in docs)
    keys = ["各有关单位", "进一步规范低空飞行秩序", "整治范围", "2026年3月1日"]
    flag = len(docs) >= 2 and longest <= 700 and all(k in joined for k in keys)
    return ok(flag, "F7 无条号长文本按段落装包切分且不丢句子",
              f"{len(docs)} 块，最长 {longest} 字")


# ---------------------------------------------------------------- F8
def case_f8() -> bool:
    tmp = Path(tempfile.mkdtemp())
    body = tmp / "empty.txt"
    body.write_text("政策名称：空政策\n" + "-" * 60 + "\n   \n", encoding="utf-8")
    meta = {
        "policy_name": "空政策", "policy_id": "empty-1", "issue_date": "2026-01-01",
        "effective_date": "2026-01-01", "expiry_date": "无", "apply_region": "深圳市",
        "status": "现行有效", "_meta": {"source_file": str(body)},
    }
    j = tmp / "empty.json"
    j.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.run([PY, "kb_ingest.py", str(j), "--index-dir", str(tmp / "idx"), "--rebuild"],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
    out = (p.stdout or "") + (p.stderr or "")
    flag = "WARN" in out and "0 个 chunk" in out
    return ok(flag, "F8 空正文 → 明确告警", "有 WARN" if flag else out.strip().splitlines()[-1][:60])


# ---------------------------------------------------------------- F9 / F10
def case_f9_f10() -> bool:
    regions_ok = (
        detect_region("中山市到2028年低空产业规模突破多少？") == "中山市"
        and detect_region("在惠州飞无人机要注意什么？") == "惠州市"
        and detect_region("清远市的行动方案是哪三年？") == "清远市"
        and detect_region("广州市黄埔区的 TC 奖励是多少？") == "广州市黄埔区"
    )
    hierarchy_ok = (
        region_matches("广东省", "中山市")
        and region_matches("广州市", "广州市黄埔区")
        and region_matches("广州市黄埔区", "广州市黄埔区")
        and not region_matches("珠海市", "广州市黄埔区")
    )

    # expand 只能删掉检索词里的地域，不能删掉 region 过滤本身。
    qa = PolicyQA(PolicyRetriever(doc("第一条 中山市低空基础设施建设要求。", "中山政策", region="中山市")),
                  mode="extractive")
    before = {"question": "中山低空基础设施怎么建？", "query": "中山低空基础设施怎么建？",
              "region": "中山市", "attempts": 0}
    # LangGraph 会把节点返回值合并进原 state；直接调用节点时要模拟这一合并。
    state = {**before, **qa._expand(before)}
    expand_ok = state["region"] == "中山市" and "中山" not in state["query"]
    flag = regions_ok and hierarchy_ok and expand_ok
    return ok(flag, "F9/F10 广东全域与区县识别；expand 保留地域硬过滤",
              f"地域={regions_ok}，层级={hierarchy_ok}，expand={state}")


def main() -> int:
    results = [case_f1(), case_f2(), case_f3_f4(), case_f5(), case_f6(), test_f7(), case_f8(), case_f9_f10()]
    print("=" * 60)
    print("阶段三边界用例：全部通过 ✅" if all(results) else "阶段三边界用例：存在失败项 ❌")
    print("=" * 60)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
