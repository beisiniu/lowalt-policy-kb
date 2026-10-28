#!/usr/bin/env python3
"""从中国民用航空局（CAAC）站内检索接口补充全国性政策种子。

CAAC 的信息公开栏目是 JS 渲染的，列表页 HTML 里没有条目。但它的站内检索走的是
老式 TRS WAS 接口（`/was5/web/search`），**返回的就是静态 HTML 列表**，可以自由翻页。

这一步只负责"发现 URL"，正文抓取、去重、台账仍由 tools/crawl_policy_urls.py 完成，
保证所有来源走同一条落盘与审计链路。

用法：
    python tools/harvest_caac.py                       # 抓取并合并进种子清单
    python tools/harvest_caac.py --max-pages 3         # 每个关键词最多翻 3 页
    python tools/harvest_caac.py --dry-run             # 只看会新增多少条
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = ROOT / "sources" / "lowalt_policy_urls.json"

SEARCH_URL = "https://www.caac.gov.cn/was5/web/search"
CHANNEL_ALL = "211383"          # 信息公开全栏目
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Referer": "https://www.caac.gov.cn/XXGK/XXGK/",
}

# 检索词：既覆盖无人机专门规章，也覆盖通用航空/空域这类上位规范
KEYWORDS = (
    "无人驾驶航空器",
    "无人机",
    "低空",
    "通用航空",
    "空域",
    "适航",
)

# 标题必须命中其一，否则视为与低空无关（如航班正常率、机场收费）
LOWALT_KWS = (
    "无人驾驶航空器", "无人机", "低空", "通用航空", "空域",
    "eVTOL", "飞行汽车", "民用航空器", "航空器适航", "飞行标准",
)

# 解读/新闻/统计/答复不是规范性文件，进库会制造"貌似权威"的错误依据
EXCLUDE_KWS = (
    "解读", "答记者问", "新闻发布会", "统计", "公报", "年度报告",
    "建议", "提案", "征求意见", "公示", "名单", "评价结果", "工作情况的报告",
)


def normalize_href(href: str) -> str:
    """CAAC 检索结果里的链接形如 `/www.caac.gov.cn/XXGK/...`，需要还原成绝对地址。"""
    href = (href or "").strip()
    if href.startswith("/www."):
        return "https://" + href[1:]
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return urljoin("https://www.caac.gov.cn/", href)
    return href


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()


def search_page(keyword: str, page: int, timeout: float = 25) -> tuple[list[tuple[str, str]], bool]:
    """返回 ([(title, url)], 是否还有下一页)。"""
    params = {"channelid": CHANNEL_ALL, "searchword": keyword, "page": str(page)}
    r = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=timeout, verify=False)
    r.encoding = r.apparent_encoding or "utf-8"
    soup = BeautifulSoup(r.text, "lxml")

    out: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = normalize_href(a["href"])
        if "caac.gov.cn" not in href:
            continue
        if not re.search(r"/t\d{8}_\d+\.(?:html?|shtml)$", href):
            continue
        title = clean(a.get_text(" ", strip=True))
        if len(title) < 6:
            continue
        out.append((title, href))

    # 该接口忽略 perpage，固定 10 条/页；是否继续翻页由调用方按"本页是否还有结果"判断
    return out, bool(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="从民航局检索接口补充政策 URL 种子")
    ap.add_argument("--seed", default=str(DEFAULT_SEED))
    ap.add_argument("--max-pages", type=int, default=2, help="每个关键词最多翻几页（每页 30 条）")
    ap.add_argument("--delay", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    requests.packages.urllib3.disable_warnings()
    seed_path = Path(args.seed)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    items: list[dict] = payload["items"]
    known = {x["url"] for x in items}

    found: dict[str, tuple[str, str]] = {}
    stats = []
    for kw in KEYWORDS:
        got = 0
        prev_sig: tuple[str, ...] = ()
        for page in range(1, max(1, args.max_pages) + 1):
            try:
                rows, has_next = search_page(kw, page)
            except Exception as e:
                print(f"[WARN] 检索失败 kw={kw} page={page}: {type(e).__name__} {str(e)[:80]}")
                break
            sig = tuple(sorted(u for _, u in rows))
            # 某些关键词翻到后面会重复返回同一页，用签名判重避免空转
            if not rows or sig == prev_sig:
                break
            prev_sig = sig
            new_rows = [(t, u) for t, u in rows if u not in found]
            for t, u in new_rows:
                found[u] = (t, kw)
            got += len(new_rows)
            time.sleep(args.delay)
        stats.append((kw, got))
        print(f"  检索「{kw}」→ 新增候选 {got}")
        time.sleep(args.delay)

    added = skipped = 0
    skipped_reasons: list[dict] = []
    for url, (title, kw) in found.items():
        if url in known:
            continue
        if not any(k in title for k in LOWALT_KWS):
            skipped += 1
            skipped_reasons.append({"title": title[:80], "reason": "标题非低空主题", "keyword": kw})
            continue
        if any(k in title for k in EXCLUDE_KWS):
            skipped += 1
            skipped_reasons.append({"title": title[:80], "reason": "非规范性文件（解读/统计/公示）", "keyword": kw})
            continue
        known.add(url)
        items.append({
            "id": f"URL-{len(items) + 1:04d}",
            "title": title,
            "url": url,
            "region": "全国",
            "level": "national",
            "doc_type": "民航规章/规范性文件",
            "source_name": "中国民用航空局",
            "source_page": f"{SEARCH_URL}?searchword={kw}",
            "status": "pending",
        })
        added += 1

    print(f"\n候选链接 {len(found)} 条 → 命中低空主题 {added} 条，跳过 {skipped} 条")
    if args.dry_run:
        print("（dry-run，不写文件）")
        return 0

    payload["items"] = items
    payload["count"] = len(items)
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    payload.setdefault("harvester_stats", {})["caac"] = {
        "keywords": dict(stats), "added": added, "skipped": skipped,
    }
    payload["skipped"] = (payload.get("skipped") or [])[-40:] + skipped_reasons[-40:]
    seed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"种子清单已更新：{seed_path}（共 {len(items)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
