#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从公开低空政策索引页生成可审计的 URL 种子清单。

索引源：
  人民网 828 企业服务平台 · 低空经济产业数据库
    - 省级政策
    - 地方级政策

索引页本身是静态 HTML，但每条政策指向不同政府站点，不能用一个 config 统一抓。
因此先抽出「标题 + 原始 URL + 层级 + 地域」，再由 tools/crawl_policy_urls.py
逐条抓取并记录 HTTP 状态、失败原因、正文哈希。

用法：
    python tools/build_url_seed.py
    python tools/build_url_seed.py --output sources/lowalt_policy_urls.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INDEX_PAGES = (
    ("人民网低空经济产业数据库-省级政策", "http://828.people.com.cn/GB/459707/459834/459836/index.html", "province"),
    ("人民网低空经济产业数据库-地方级政策", "http://828.people.com.cn/GB/459707/459834/459837/index.html", "city"),
)

# 国家层面法规手工补两条（索引页有省级/地方级，没有单独国家页）
CURATED_SEEDS = (
    {
        "title": "广州开发区（黄埔区）促进低空经济高质量发展的若干措施（2023年）",
        "url": "https://www.hp.gov.cn/gkmlpt/content/9/9280/post_9280468.html",
        "region": "广州市黄埔区", "level": "city", "doc_type": "已废止政策",
        "source_name": "广州市黄埔区人民政府",
    },
    {
        "title": "福州市人民政府关于推进民用无人驾驶航空器产业高质量发展的若干意见",
        "url": "https://www.fuzhou.gov.cn/zfxxgkzl/szfbmjxsqxxgk/szfbmxxgk/fzsrmzf/yfzwjk/202301/t20230112_4520403.htm",
        "region": "福州市", "level": "city", "doc_type": "已废止政策",
        "source_name": "福州市人民政府·已废止文件库",
    },
    {
        "title": "福州市加快推动低空产业发展行动方案（2024—2026年）",
        "url": "https://zqztc.fujian.gov.cn/zcy/policy/policy/content.html?uniqid=66e165603a121",
        "region": "福州市", "level": "city", "doc_type": "政策文件",
        "source_name": "福建政企直通车",
    },
    {
        "title": "江苏省航空航天产业发展三年行动计划（2023—2025年）",
        "url": "https://xcqfzx.szxc.gov.cn/xcservplat/Apps/psp/index.php?s=/Home/PolicySearch/policydetail/id/111535",
        "region": "江苏省", "level": "province", "doc_type": "已到期政策",
        "source_name": "相城区惠企通服务平台",
    },
    {
        "title": "西藏自治区支持低空经济高质量发展的若干政策（试行）",
        "url": "https://drc.xizang.gov.cn/zsyz/zc/202604/P020260413388348845537.pdf",
        "region": "西藏自治区", "level": "province", "doc_type": "政策文件",
        "source_name": "西藏自治区发展和改革委员会",
    },
)

NATIONAL_SEEDS = (
    {
        "title": "无人驾驶航空器飞行管理暂行条例",
        "url": "https://politics.people.com.cn/n1/2023/0629/c1001-40023528.html",
        "region": "全国",
        "level": "national",
        "doc_type": "行政法规",
        "source_name": "人民网·时政",
        "publish_date_hint": "2023-06-28",
    },
    {
        "title": "民用无人驾驶航空器运行安全管理规则",
        # 政府网正文页已是前端模板渲染，静态 HTML 只剩导航；部委公开的 docx 是同一文本
        "url": "https://xxgk.mot.gov.cn/2020/jigou/fgs/202401/P020240103558815898025.docx",
        "region": "全国",
        "level": "national",
        "doc_type": "部门规章",
        "source_name": "交通运输部",
        "publish_date_hint": "2024-01-01",
    },
)

LOWALT_KWS = ("低空", "无人机", "无人驾驶航空器", "通用航空", "eVTOL", "飞行汽车", "空天信息")
# 索引里混入的泛产业文件，标题不含低空/无人机就不收
EXCLUDE_KWS = ("户外运动", "制造业高质量发展", "未来产业创新高地", "六新突破", "场景创新和应用")
PROVINCES = (
    "北京市", "上海市", "天津市", "重庆市", "河北省", "山西省", "辽宁省", "吉林省", "黑龙江省",
    "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省", "河南省", "湖北省", "湖南省",
    "广东省", "海南省", "四川省", "贵州省", "云南省", "陕西省", "甘肃省", "青海省", "台湾省",
    "内蒙古自治区", "广西壮族自治区", "西藏自治区", "宁夏回族自治区", "新疆维吾尔自治区",
)
CITY_HINTS = (
    "深圳市", "广州市", "珠海市", "佛山市", "东莞市", "惠州市", "中山市", "清远市", "汕头市",
    "嘉兴市", "青岛市", "沈阳市", "十堰市", "福州市", "绍兴市", "荆州市", "石河子市", "南昌市",
    "常州市", "南通市", "西安市", "郑州市", "杭州市", "武汉市", "长沙市", "日照市", "长治市",
    "成都市", "济南市", "漳州市", "苏州市", "合肥市", "南京市", "宁波市", "无锡市", "雄安新区",
    "广州市南沙区", "天津市宁河区",
)
REGION_ALIASES = (
    ("广西", "广西壮族自治区"), ("内蒙古", "内蒙古自治区"), ("西藏", "西藏自治区"),
    ("宁夏", "宁夏回族自治区"), ("新疆", "新疆维吾尔自治区"),
    ("广州南沙", "广州市南沙区"), ("南沙", "广州市南沙区"),
)


def infer_region(title: str) -> str:
    for r in sorted(CITY_HINTS + PROVINCES, key=len, reverse=True):
        if r in title:
            return r
    for alias, region in REGION_ALIASES:
        if alias in title:
            return region
    if title.startswith("广东省"):
        return "广东省"
    return ""


def infer_level(region: str, fallback: str) -> str:
    if region == "全国":
        return "national"
    if region.endswith(("省", "自治区")):
        return "province"
    if region.endswith(("市", "区", "新区")):
        return "city"
    return fallback


def clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_index(url: str) -> str:
    r = requests.get(
        url, timeout=25, verify=False,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"},
    )
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def iter_index_pages(page_url: str, max_pages: int = 8):
    """索引页有分页：第一页是目录本身，之后是 index2.html、index3.html…

    人民网的「地方级政策」实际有 2 页（42 条），只抓第一页会漏掉三分之一。
    翻到 404 或没有新条目就停。
    """
    yield page_url
    # 目录页两种写法：以 / 结尾，或直接是 index.html；都要还原到目录再拼 index2.html
    if page_url.endswith(".html"):
        base = page_url.rsplit("/", 1)[0] + "/"
    else:
        base = page_url if page_url.endswith("/") else page_url + "/"
    for i in range(2, max_pages + 1):
        yield f"{base}index{i}.html"


def collect_from_index(source_name: str, page_url: str, level: str,
                       items: list[dict], seen_urls: set[str], skipped: list[dict]) -> int:
    added = 0
    for url in iter_index_pages(page_url):
        try:
            html = fetch_index(url)
        except Exception as e:
            # 第一页失败要显式报错；后续页 404 属于正常结束
            if url == page_url:
                print(f"[FAIL] 索引页 {url}: {e}")
            break
        soup = BeautifulSoup(html, "lxml")
        page_hits = 0
        page_candidates = 0     # 本页通过主题筛选的条目数（含已收录的）
        for a in soup.find_all("a", href=True):
            title = clean_title(a.get_text(" ", strip=True))
            href = urldefrag(a["href"].strip())[0]
            if len(title) < 8 or not href.startswith("http"):
                continue
            if title in ("低空经济产业数据库", "省级政策", "地方级政策"):
                continue
            if not any(k in title for k in LOWALT_KWS):
                if len(title) > 8:
                    skipped.append({"title": title[:80], "reason": "非低空主题", "source_page": url})
                continue
            if any(k in title for k in EXCLUDE_KWS) and "低空" not in title:
                skipped.append({"title": title[:80], "reason": "泛产业文件", "source_page": url})
                continue
            page_candidates += 1
            if href in seen_urls:
                continue
            region = infer_region(title)
            seen_urls.add(href)
            items.append({
                "id": f"URL-{len(items)+1:04d}",
                "title": title,
                "url": href,
                "region": region or "待识别",
                "level": infer_level(region, level),
                "doc_type": "政策文件",
                "source_name": source_name,
                "source_page": url,
                "status": "pending",
            })
            page_hits += 1
        added += page_hits
        # 用「本页候选数」判断是否到底：已收录的条目也会命中筛选，
        # 拿新增数判断会导致第一页全是旧数据时直接停止翻页
        if page_candidates == 0:
            break
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description="生成低空政策 URL 种子清单")
    ap.add_argument("--output", default=str(ROOT / "sources" / "lowalt_policy_urls.json"))
    args = ap.parse_args()

    out = Path(args.output)
    # 先读已有种子：里面可能有民航局检索、手工种子等非索引页来源，重跑不能把它们冲掉
    items: list[dict] = []
    skipped: list[dict] = []
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        items = prev.get("items") or []
        skipped = prev.get("skipped") or []
    seen_urls: set[str] = {x["url"] for x in items if x.get("url")}
    before = len(items)

    for source_name, page_url, level in INDEX_PAGES:
        n = collect_from_index(source_name, page_url, level, items, seen_urls, skipped)
        print(f"  索引「{source_name}」新增 {n} 条（累计 {len(items)}）")

    for seed in (*NATIONAL_SEEDS, *CURATED_SEEDS):
        if seed["url"] in seen_urls:
            continue
        seen_urls.add(seed["url"])
        items.append({
            "id": f"URL-{len(items)+1:04d}",
            **seed,
            "source_page": "手工种子（国家/真实新旧对照）",
            "status": "pending",
        })

    items.sort(key=lambda x: ({"national": 0, "province": 1, "city": 2}.get(x["level"], 3), x["region"], x["title"]))
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "index_pages": [p[1] for p in INDEX_PAGES],
        "count": len(items),
        "items": items,
        "skipped": skipped,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    levels: dict[str, int] = {}
    regions: dict[str, int] = {}
    for x in items:
        levels[x["level"]] = levels.get(x["level"], 0) + 1
        regions[x["region"]] = regions.get(x["region"], 0) + 1
    print(f"种子 {len(items)} 条 → {out}")
    print(f"层级：{levels}")
    print(f"地域：{len(regions)} 个；{regions}")
    print(f"过滤掉非低空条目 {len(skipped)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
