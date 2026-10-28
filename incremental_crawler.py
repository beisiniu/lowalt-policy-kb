#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段一 · 增量监控爬虫（带本地状态记录 + 哈希去重）

用法：
    python incremental_crawler.py                      # 按 config.json 抓一轮
    python incremental_crawler.py --dry-run            # 只判定不落盘（预演增量效果）
    python incremental_crawler.py --stats              # 只看历史库存统计
    python incremental_crawler.py --limit 10           # 本轮最多处理 10 条
    python incremental_crawler.py --no-check-updates   # 只抓新链接，不校验老政策是否修订
    python incremental_crawler.py --reset              # 清空状态（重新全量抓）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许从任意目录执行（如 cron / 计划任务）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.config import load_config
from policycrawler.pipeline import IncrementalCrawler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="低空政策知识库 · 增量爬虫")
    p.add_argument("-c", "--config", default=str(Path(__file__).parent / "config.json"), help="配置文件路径")
    p.add_argument("--limit", type=int, default=None, help="本轮最多处理多少条列表记录")
    p.add_argument("--dry-run", action="store_true", help="预演：判定新增/更新但不写文件不改状态")
    p.add_argument("--no-check-updates", action="store_true", help="不校验老政策正文是否修订（更快、请求更少）")
    p.add_argument("--stats", action="store_true", help="只打印历史状态统计后退出")
    p.add_argument("--reset", action="store_true", help="清空 crawled_history.json（不可恢复）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.reset:
        if cfg.history_file.exists():
            backup = cfg.history_file.with_suffix(".json.bak")
            cfg.history_file.replace(backup)
            print(f"已清空状态文件，旧文件备份为 {backup}")
        else:
            print("状态文件不存在，无需清空")
        return 0

    crawler = IncrementalCrawler(
        cfg,
        check_updates=False if args.no_check_updates else None,
        limit=args.limit,
    )

    if args.stats:
        print(crawler.store.summary())
        return 0

    print("=" * 78)
    print(f"源站：{cfg.source.get('name')}   状态文件：{cfg.history_file}")
    print("=" * 78)

    result = crawler.run(dry_run=args.dry_run)

    print("-" * 78)
    print(f"列表 {result.listed} 条 → 新增 {len(result.new_items)} | 更新 {len(result.updated_items)} "
          f"| 跳过 {len(result.skipped)} | 失败 {len(result.failed)} | 耗时 {result.elapsed_sec:.1f}s")
    if args.dry_run:
        print("（dry-run：未写入任何文件，也未更新 crawled_history.json）")

    if result.new_items:
        print("\n本轮新增：")
        for i in result.new_items:
            print(f"  + {i.publish_date} {i.title}\n      → {i.saved_to or '(dry-run)'}")
    if result.updated_items:
        print("\n本轮修订：")
        for i in result.updated_items:
            print(f"  ~ {i.publish_date} {i.title} (rev{i.revision})\n      → {i.saved_to or '(dry-run)'}")

    if result.failed:
        print("\n抓取失败（下轮会自动重试）：")
        for i in result.failed:
            print(f"  ! {i.title} {i.url}\n      {i.note}")
        return 2

    print(f"\n历史库存：{crawler.store.count} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
