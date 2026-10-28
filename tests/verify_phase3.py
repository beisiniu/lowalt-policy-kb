#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段三自检：本地知识库（切分 → 索引 → 检索 → 时效门禁）

  A. 建库：4 篇政策切成条文级 chunk，元数据齐全
  B. 检索：地域过滤、新法优先
  C. 时效门禁：已废止/过期政策不会被拿来硬答，而是明确拒答
  D. 图路径：正常问题走 answer，无关问题走 expand→refuse

运行：python tests/verify_phase3.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.graph import PolicyQA                      # noqa: E402
from kb.retriever import PolicyRetriever           # noqa: E402
from kb.splitter import build_documents            # noqa: E402
from kb.store import load_index, save_index        # noqa: E402

PY = sys.executable
REQUIRED_META = ("policy_name", "policy_id", "issue_date", "effective_date",
                 "expiry_date", "apply_region", "status", "source_url")


def ok_line(flag: bool, label: str, extra: str = "") -> bool:
    print(f"[{'PASS' if flag else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return flag


def run(args: list[str]) -> tuple[int, str, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    p = subprocess.run([PY, *args], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="lowalt_kb_"))
    index_dir = tmp / "index"

    # ---------------- A. 建库 ----------------
    code, out, err = run(["kb_ingest.py", "data/extracted", "--rebuild", "--index-dir", str(index_dir)])
    a1 = code == 0 and index_dir.joinpath("chunks.jsonl").exists()
    ok &= ok_line(a1, "A1 建库成功", f"exit={code}")
    if not a1:
        print(out[-800:], err[-800:])
        return 1

    docs, tokens, manifest = load_index(index_dir)
    ok &= ok_line(len(docs) > 15 and manifest["policies"] == 4, "A2 政策数与 chunk 数合理",
                  f"{manifest['policies']} 篇 / {len(docs)} chunk")

    missing = [i for i, d in enumerate(docs) if not all(d.metadata.get(k) for k in REQUIRED_META)]
    ok &= ok_line(not missing, "A3 每个 chunk 都带齐 8 个元数据字段", f"缺失 {len(missing)} 个")

    arts = {d.metadata.get("article") for d in docs if d.metadata.get("article")}
    ok &= ok_line(len(arts) >= 8, "A4 按「第X条」切分（不是按字数硬切）", f"{len(arts)} 个条文块")

    # ---------------- B. 检索 ----------------
    r = PolicyRetriever(docs, tokens)
    hits = r.retrieve("在深圳市用 249 克微型无人机飞行需要申请吗", k=3, region="深圳市")
    top = hits[0] if hits else None
    ok &= ok_line(bool(top), "B1 能召回结果", f"{len(hits)} 条")
    if top:
        # 新法优先：2026 年的两部政策（条例 / 修订办法）都算新法，只要不是 2025 旧办法即可
        ok &= ok_line("2025-05-10" != top.metadata["effective_date"]
                      and top.metadata["effective_date"].startswith("2026"),
                      "B2 新法优先：top1 是 2026 年的政策，不是 2025 旧办法",
                      f"{top.metadata['policy_name'][:20]} {top.metadata['article']}")

    regional = r.retrieve("无人机飞行需要申请吗", k=4, region="上海市")
    regions = {d.metadata["apply_region"] for d in regional}
    ok &= ok_line(all(x in ("全国", "上海市") for x in regions), "B3 地域过滤：问上海不漏深圳文件",
                  f"命中地域 {regions}")

    # ---------------- C. 时效门禁 ----------------
    stale_meta = {
        "policy_name": "某市低空飞行管理暂行办法", "policy_id": "stale-test-0001",
        "issue_date": "2024-01-01", "effective_date": "2024-02-01", "expiry_date": "2025-01-31",
        "apply_region": "深圳市", "status": "现行有效", "source_url": "http://example.com/stale",
    }
    stale_docs = build_documents(
        "第一条 在本市范围内使用民用无人驾驶航空器从事飞行活动的，无论机型大小，均应当提前5个工作日提出飞行计划申请。",
        stale_meta,
    )
    # 再放几篇无关的现行政策：BM25 在文档数太少时 IDF 会退化成 0，测不出真实排序
    filler_texts = [
        ("某市低空基础设施规划指引", "first-0002",
         "第一条 本市低空飞行基础设施建设规划由市交通运输部门会同市规划和自然资源部门编制。"),
        ("某市低空产业专项资金管理办法", "first-0003",
         "第二条 市财政应当安排低空经济产业发展专项资金，用于支持企业研发和场景应用。"),
        ("某市航空器维修人员培训规定", "first-0004",
         "第三条 市人力资源部门应当组织开展航空器维修技能培训，并纳入职业技能提升计划。"),
    ]
    filler_docs = []
    for name, pid, text in filler_texts:
        filler_docs.extend(build_documents(text, {
            **stale_meta, "policy_name": name, "policy_id": pid,
            "issue_date": "2026-01-01", "effective_date": "2026-02-01", "expiry_date": "无",
        }))
    stale_dir = tmp / "index_stale"
    save_index(stale_docs + filler_docs, stale_dir)
    sdocs, stoks, _ = load_index(stale_dir)
    sr = PolicyRetriever(sdocs, stoks)
    qa = PolicyQA(sr, mode="extractive")

    filtered = sr.retrieve("无人机需要提前申请吗", k=3, region="深圳市")
    ok &= ok_line(not filtered, "C1 已过有效期的政策被硬过滤", f"召回 {len(filtered)} 条")

    out_stale = qa.ask("无人机需要提前申请吗")
    ok &= ok_line(out_stale["route"] == "refuse" and bool(out_stale["blocked_stale"]),
                  "C2 只剩失效政策时明确拒答，而不是照着旧办法硬答",
                  f"route={out_stale['route']} blocked={out_stale['blocked_stale']}")
    ok &= ok_line("失效" in out_stale["answer"] or "有效期" in out_stale["answer"],
                  "C3 拒答理由说明是「时效问题」")

    # ---------------- D. 图路径 ----------------
    main_qa = PolicyQA(PolicyRetriever(docs, tokens), mode="extractive")
    normal = main_qa.ask("在深圳用起飞重量 30 千克的无人机做电力巡检，要提前几天申请？")
    ok &= ok_line(normal["route"] == "answer" and bool(normal["cited"]), "D1 正常问题走 answer 分支",
                  f"{len(normal['cited'])} 条引用")

    unrelated = main_qa.ask("火星上飞无人机需要审批吗？")
    ok &= ok_line(unrelated["route"] in ("refuse_empty", "refuse") and "expand" in unrelated["steps"],
                  "D2 无关问题先放宽一次再拒答（图里有回路）",
                  " > ".join(unrelated["steps"]))

    cited_ok = all(c.get("status") == "现行有效" and c.get("effective_date") for c in normal["cited"])
    ok &= ok_line(cited_ok, "D3 引用来源都带 status/effective_date")

    print("=" * 60)
    print("阶段三自检：全部通过 ✅" if ok else "阶段三自检：存在失败项 ❌")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
