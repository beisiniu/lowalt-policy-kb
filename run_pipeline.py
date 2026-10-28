#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一条命令跑完 抓取 → 打标 → 建库 →（可选）评估。

    python run_pipeline.py --config config.guangdong.json --limit 20   # 前三个阶段
    python run_pipeline.py --all --benchmark benchmark.json            # 有冲突集时四阶段全跑
    python run_pipeline.py --evaluate-only --benchmark benchmark.json  # 只重新评估
    python run_pipeline.py --all --offline                             # 无 Key 只跑前三个阶段

特点：
  - 打标只处理**新增或更新过**的正文（对比 txt 与已产出 json 的 mtime），
    不会每次都把整个库重新烧一遍 token。
  - 任一步失败立即停止并给出原因，不会带着坏数据继续往下走。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(title: str, args: list[str], allow_fail: bool = False) -> int:
    print("\n" + "=" * 74)
    print(f"▶ {title}")
    print("=" * 74, flush=True)
    t0 = time.time()
    code = subprocess.call([PY, *args], cwd=str(ROOT))
    dt = time.time() - t0
    flag = "OK" if code == 0 else f"FAILED(exit={code})"
    print(f"— {title} {flag}  {dt:.1f}s", flush=True)
    if code != 0 and not allow_fail:
        print(f"\n[ABORT] {title} 失败，后续步骤不再执行。")
        raise SystemExit(code)
    return code


def pending_policies(policy_dir: Path, extracted_dir: Path) -> list[Path]:
    """挑出「还没打标」或「正文比结果新」的政策 txt。"""
    out: list[Path] = []
    for txt in sorted(policy_dir.glob("*.txt")):
        result = extracted_dir / f"{txt.stem}.json"
        if not result.exists() or txt.stat().st_mtime > result.stat().st_mtime:
            out.append(txt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="低空政策知识库 · 一键流水线")
    ap.add_argument("--config", default=str(ROOT / "config.json"), help="爬虫配置")
    ap.add_argument("--limit", type=int, default=20, help="本轮最多抓多少条")
    ap.add_argument("--all", action="store_true", help="跑完后接着评估")
    ap.add_argument("--evaluate-only", action="store_true", help="跳过前三个阶段，只评估")
    ap.add_argument("--offline", action="store_true", help="打标走规则版（不需要 Key）")
    # ---- 阶段四（新）：需要先用 build_benchmark.py 生成冲突测试集 ----
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark.json"))
    ap.add_argument("--eval-backend", choices=["chroma", "local"], default="chroma")
    ap.add_argument("--chroma-dir", default=str(ROOT / "chroma_lowalt"))
    ap.add_argument("--collection", default="lowalt_policy")
    ap.add_argument("--embedding-function", default="bge-zh")
    ap.add_argument("--index-dir", default=str(ROOT / "kb" / "index"))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] 配置文件不存在：{cfg_path}")
        return 1
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    policy_dir = (cfg_path.parent / cfg["storage"]["policy_dir"]).resolve()
    extracted_dir = ROOT / "data" / "extracted"

    if not args.evaluate_only:
        run("阶段一 增量抓取", ["incremental_crawler.py", "-c", str(cfg_path), "--limit", str(args.limit)])

        pending = pending_policies(policy_dir, extracted_dir)
        if not pending:
            print("\n[SKIP] 没有待打标的新政策（全部已有结果且未变更）")
        else:
            print(f"\n待打标 {len(pending)} 篇：")
            for p in pending[:8]:
                print(f"  - {p.name}")
            if len(pending) > 8:
                print(f"  … 还有 {len(pending) - 8} 篇")
            extra = ["--offline"] if args.offline else []
            run("阶段二 结构化打标", ["extract_policy.py", str(policy_dir), *extra])

        run("阶段三 建本地知识库", ["kb_ingest.py", str(extracted_dir), "--rebuild",
                                    "--index-dir", args.index_dir])

    if args.all or args.evaluate_only:
        bench = Path(args.benchmark)
        if not bench.exists():
            print(
                f"\n[SKIP] 阶段四：未找到 {bench}。\n"
                "先构建冲突测试集（需要 Chroma 与 DeepSeek Key）：\n"
                "  python build_benchmark.py --chroma-dir ./chroma_db "
                "--collection lowalt_policy\n"
                "然后重跑 --all，或直接：python evaluate.py --benchmark benchmark.json"
            )
        else:
            run("阶段四 时序 RAG 双分支评估", [
                "evaluate.py", "--backend", args.eval_backend,
                "--benchmark", str(bench),
                "--chroma-dir", args.chroma_dir,
                "--collection", args.collection,
                "--embedding-function", args.embedding_function,
                "--index-dir", args.index_dir,
            ])

    print("\n" + "=" * 74)
    print("流水线完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
