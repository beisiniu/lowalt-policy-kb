#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""站点选择器探测：给一个列表页 URL，自动猜出 item/title/date/link 选择器。

写 config.json 时最费时间的就是这一步。用法：

    python tools/inspect_site.py https://www.gov.cn/zhengce/zuixin/
    python tools/inspect_site.py <url> --grep 低空        # 只看含关键词的条目
    python tools/inspect_site.py <url> --json             # 直接输出可粘贴的配置片段

原理：把常见的列表容器（li / tr / 带 class 的 div）全试一遍，
统计「包含一个可点标题 + 一个日期」的子节点数量，命中最多的那个就是 item 选择器。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup, Tag

from policycrawler.fetcher import Fetcher, FetchError

DATE_IN_TEXT = re.compile(r"\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}")
DATE_LIKE_ATTR = re.compile(r"date|time|riqi|pub", re.I)


def css_of(tag: Tag) -> str:
    """给 Tag 生成一个尽量稳定的 CSS 选择器（tag + 一级 class）。"""
    if tag.get("id"):
        return f"#{tag['id']}"
    classes = [c for c in (tag.get("class") or []) if len(c) < 24 and not c.isdigit()]
    base = tag.name
    if classes:
        return base + "".join(f".{c}" for c in classes[:2])
    return base


def score_container(node: Tag) -> tuple[int, list[dict]]:
    """一个容器下有多少条「像政策列表项」的子节点。"""
    picks: list[dict] = []
    for child in node.find_all(recursive=False):
        a = child.find("a", href=True)
        if not a:
            continue
        title = a.get_text(" ", strip=True) or (a.get("title") or "")
        if len(title) < 6 or title in ("更多", "下一页", "首页"):
            continue
        text = child.get_text(" ", strip=True)
        # 日期可能写在下级节点里，也可能在属性（如 data-date）上
        has_date = bool(DATE_IN_TEXT.search(text))
        date_node = None
        if not has_date:
            for sub in child.find_all(True, recursive=True):
                if DATE_IN_TEXT.search(sub.get_text(" ", strip=True)) and len(sub.get_text(strip=True)) <= 24:
                    has_date, date_node = True, sub
                    break
            if not has_date:
                for sub in child.find_all(True, recursive=True):
                    if any(DATE_LIKE_ATTR.search(k) or DATE_LIKE_ATTR.search(str(v)) for k, v in sub.attrs.items()):
                        has_date, date_node = True, sub
                        break
        if not has_date:
            continue
        picks.append({"title": title, "href": a.get("href"), "date_node": date_node, "child": child})
    return len(picks), picks


def guess(html: str, top: int = 5) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    for bad in soup.find_all(("script", "style")):
        bad.decompose()

    results = []
    seen: Counter[str] = Counter()
    for node in soup.find_all(["ul", "ol", "tbody", "div", "table"]):
        sel = css_of(node)
        # 生成「父选择器 子标签」形式，更贴近实际用法
        for child_tag in ("li", "tr", "div", "dd"):
            children = node.find_all(child_tag, recursive=False)
            if len(children) < 3:
                continue
            n, picks = score_container(node)
            if n < 3:
                continue
            full = f"{sel} {child_tag}" if sel != node.name else f"{node.name} {child_tag}"
            if seen[full]:
                continue
            seen[full] += 1
            results.append({"selector": full, "count": n, "picks": picks[:3]})

    results.sort(key=lambda r: r["count"], reverse=True)
    # 去掉明显互相包含的重复项，保留 count 最高的若干
    return results[:top]


def date_selector(sample: dict, html_soup: BeautifulSoup) -> str:
    node = sample.get("date_node")
    return css_of(node) if isinstance(node, Tag) else "span"


def main() -> int:
    ap = argparse.ArgumentParser(description="探测政策列表页的选择器")
    ap.add_argument("url")
    ap.add_argument("--grep", default="", help="只展示标题含该关键词的候选")
    ap.add_argument("--json", action="store_true", help="输出可直接粘进 config.json 的片段")
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--insecure", action="store_true", help="跳过 SSL 校验（政府站点证书问题）")
    args = ap.parse_args()

    try:
        html = Fetcher(timeout=args.timeout, retries=2, delay=0.3, verify_ssl=not args.insecure).get_text(args.url)
    except FetchError as e:
        print(f"[ERROR] {e}")
        return 1

    cands = guess(html)
    if not cands:
        print("没找到像样的列表结构。可能是 JS 渲染，或需要登录/Cookie。")
        print("建议：F12 → Network 里找返回 JSON 的接口，直接请求接口比解析 HTML 稳。")
        return 2

    print(f"页面：{args.url}（{len(html)} 字）\n")
    for i, c in enumerate(cands, 1):
        print(f"[{i}] {c['selector']}   命中 {c['count']} 条")
        for p in c["picks"]:
            if args.grep and args.grep not in p["title"]:
                continue
            print(f"     - {p['title'][:52]}")
            print(f"       {p['href']}")
        print()

    best = cands[0]
    sample = next((p for p in best["picks"] if (not args.grep or args.grep in p["title"])), best["picks"][0])
    snippet = {
        "list_selectors": {
            "item": [best["selector"]],
            "title": ["a"],
            "link": ["a"],
            "date": [date_selector(sample, None), "span", "td"],
        }
    }
    if args.json:
        print(json.dumps(snippet, ensure_ascii=False, indent=2))
    else:
        print("→ 建议写入 config.json：")
        print(json.dumps(snippet, ensure_ascii=False, indent=2))
        print("\n（date 选择器是猜的，跑一次 `python incremental_crawler.py --dry-run` 看日期是否都能解出来）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
