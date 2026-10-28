#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键跑完四个阶段的自检：python tests/run_all.py"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES = [
    ("阶段一 增量爬虫（主流程）", "tests/verify_phase1.py"),
    ("阶段一 增量爬虫（边界）", "tests/verify_phase1_edges.py"),
    ("阶段二 大模型打标", "tests/verify_phase2.py"),
    ("阶段三 本地知识库", "tests/verify_phase3.py"),
    ("阶段三 本地知识库（边界）", "tests/verify_phase3_edges.py"),
    ("阶段四 时序RAG对抗评估", "tests/verify_temporal_benchmark.py"),
]


def main() -> int:
    results = []
    for name, script in PHASES:
        print("\n" + "=" * 70)
        print(f"▶ {name}  ({script})")
        print("=" * 70)
        p = subprocess.run([sys.executable, script], cwd=str(ROOT))
        results.append((name, p.returncode == 0))

    print("\n" + "=" * 70)
    print("汇总")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(ok for _, ok in results)
    print("=" * 70)
    print("全部通过 ✅" if all_ok else "存在失败项 ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
