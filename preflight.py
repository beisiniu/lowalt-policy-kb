#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""开跑前的环境自检：把「待会儿一定会报错」的问题提前说清楚。

    python preflight.py
    python preflight.py --config config.beijing.json

检查项：
  1. 依赖包是否装齐、版本够不够
  2. .env / 环境变量里有没有 DEEPSEEK_API_KEY（只查存在，不验证有效性）
  3. 爬虫配置：URL 是否还是示例值、选择器是否为空、verify_ssl
  4. 本地知识库索引：是否已建、多少 chunk、多少已失效
  5. 测试集：题目数、题型分布
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "OK  ", "WARN", "FAIL"


from policycrawler.utils import load_env_file  # noqa: E402


def check_deps() -> list[tuple[str, str, str]]:
    need = {"requests": "2.31", "bs4": "4.12", "lxml": "5.0", "openai": "1.30",
            "langgraph": "0.2", "langchain_core": "0.3", "rank_bm25": "0.2", "jieba": "0.42",
            "matplotlib": "3.8", "chromadb": "1.0"}
    rows = []
    for mod, minimum in need.items():
        pkg = {"bs4": "beautifulsoup4", "langchain_core": "langchain-core",
               "rank_bm25": "rank-bm25"}.get(mod, mod)
        try:
            importlib = __import__(mod)
            ver = getattr(importlib, "__version__", None) or md.version(pkg)
            rows.append((OK, pkg, ver))
        except ImportError:
            rows.append((BAD, pkg, f"未安装（pip install {pkg}>={minimum}）"))
    return rows


def check_keys() -> list[tuple[str, str, str]]:
    rows = []
    env_file = ROOT / ".env"
    loaded = load_env_file(env_file)
    rows.append((OK if env_file.exists() else WARN, ".env",
                 f"已读取 {loaded} 项" if env_file.exists() else "不存在（可选，也可以在系统环境变量里配）"))
    key = os.getenv("DEEPSEEK_API_KEY", "")
    rows.append((OK if key else WARN, "DEEPSEEK_API_KEY",
                 f"{key[:6]}…{key[-4:]}（{len(key)} 位）" if key else
                 "未配置 → 只能 --offline 打标；无法自动出题/双分支评估"))
    return rows


def check_config(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    if not path.exists():
        return [(BAD, "配置文件", f"不存在：{path}")]
    cfg = json.loads(path.read_text(encoding="utf-8"))
    url = cfg.get("source", {}).get("list_url", "")
    if "caac.gov.cn" in url:
        rows.append((WARN, "list_url", f"{url}\n         这是示例地址：该栏目已改为 JS 渲染，实测抓不到条目。\n"
                                       f"         可直接用的真实示例见 config.beijing.json"))
    else:
        rows.append((OK if url.startswith("http") else BAD, "list_url", url or "空"))

    sel = cfg.get("list_selectors", {})
    empty = [k for k in ("item", "title", "link", "date") if not sel.get(k)]
    rows.append((OK if not empty else BAD, "list_selectors",
                 "齐全" if not empty else f"缺少 {empty}"))

    det = cfg.get("detail_selectors", {})
    empty = [k for k in ("title", "publish_date", "content") if not det.get(k)]
    rows.append((OK if not empty else WARN, "detail_selectors",
                 "齐全" if not empty else f"缺少 {empty}（会走兜底，可能抓不准）"))

    v = cfg.get("crawl", {}).get("verify_ssl", True)
    rows.append((WARN if v is False else OK, "verify_ssl",
                 "已关闭证书校验（政府站点常见，注意只在可信网络里用）" if v is False else "开启"))
    return rows


def check_index(index_dir: Path) -> list[tuple[str, str, str]]:
    from kb.schema import freshness
    from kb.store import load_index

    rows = []
    try:
        docs, _, manifest = load_index(index_dir)
    except FileNotFoundError:
        return [(WARN, "知识库索引", f"未建：{index_dir}（先跑 kb_ingest.py）")]
    stale = sum(1 for d in docs if not freshness(d.metadata)[0])
    rows.append((OK if docs else BAD, "知识库索引",
                 f"{manifest.get('policies', 0)} 篇 / {len(docs)} chunk，"
                 f"其中 {stale} 个已失效（检索时会被过滤）"))
    regions = sorted({d.metadata.get("apply_region", "") for d in docs})
    rows.append((OK, "覆盖地域", "、".join(regions) or "空"))
    return rows


def check_benchmark(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    if not path.exists():
        demo = ROOT / "demo" / "benchmark_temporal_demo.json"
        rows.append((WARN, "冲突测试集", f"不存在：{path}\n"
                                        f"         先生成：python build_benchmark.py "
                                        f"--chroma-dir ./chroma_db --collection lowalt_policy"
                                        + (f"\n         仓库内有 2 题演示：{demo}" if demo.exists() else "")))
        return rows
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items") if isinstance(payload, dict) else payload
    except (OSError, json.JSONDecodeError):
        return [(BAD, "冲突测试集", f"无法解析：{path}")]
    regions: dict[str, int] = {}
    topics: dict[str, int] = {}
    for it in items:
        regions[it.get("region", "?")] = regions.get(it.get("region", "?"), 0) + 1
        topics[it.get("topic", "?")] = topics.get(it.get("topic", "?"), 0) + 1
    rows.append((OK if len(items) >= 10 else WARN, "冲突测试集", f"{len(items)} 题"))
    rows.append((OK, "覆盖", f"{len(regions)} 地域 / {len(topics)} 主题"))
    return rows


def render(title: str, rows: list[tuple[str, str, str]]) -> int:
    print(f"\n{title}")
    worst = 0
    for level, name, detail in rows:
        print(f"  [{level}] {name:<22} {detail}")
        worst = max(worst, {OK: 0, WARN: 1, BAD: 2}[level])
    return worst


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--index-dir", default=str(ROOT / "kb" / "index"))
    ap.add_argument("--benchmark", default=str(ROOT / "benchmark.json"))
    args = ap.parse_args()

    print("=" * 78)
    print("低空政策知识库 · 环境预检")
    print("=" * 78)

    worst = 0
    worst = max(worst, render("依赖", check_deps()))
    worst = max(worst, render("密钥", check_keys()))
    worst = max(worst, render(f"爬虫配置（{Path(args.config).name}）", check_config(Path(args.config))))
    worst = max(worst, render("知识库", check_index(Path(args.index_dir))))
    worst = max(worst, render("时序评估", check_benchmark(Path(args.benchmark))))

    print("\n" + "-" * 78)
    if worst == 0:
        print("全部就绪，可以评估：python evaluate.py --benchmark benchmark.json")
    elif worst == 1:
        print("有警告但可运行。生成冲突集：python build_benchmark.py ...")
    else:
        print("有阻塞项，先按上面 [FAIL] 的提示修。")
    print("-" * 78)
    return 0 if worst < 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
