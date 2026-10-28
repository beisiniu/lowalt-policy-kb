#!/usr/bin/env python3
"""从深圳市交通运输局「政策法规」栏目补充深圳政策种子。

该站 https 用 ECC 证书，本地 Python OpenSSL 握手报 BAD_ECPOINT，
因此这里统一走系统 curl（Schannel），与 crawl_policy_urls 的兜底逻辑一致。

用法：
    python tools/harvest_sz.py --pages 6
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "sources" / "lowalt_policy_urls.json"
BASE = "https://jtys.sz.gov.cn/zwgk/xxgkml/zcfgjjd/zcfg/"
KWS = ("低空", "无人机", "无人驾驶航空器", "通用航空", "eVTOL", "空域")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def curl(url: str, timeout: int = 25) -> str:
    out = subprocess.run(
        ["curl", "-k", "-L", "-sS", "--compressed", "--max-time", str(timeout),
         "-A", UA, "-H", "Accept-Language: zh-CN,zh;q=0.9", url],
        capture_output=True, timeout=timeout + 10,
    )
    return out.stdout.decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--seed", default=str(SEED))
    args = ap.parse_args()

    payload = json.loads(Path(args.seed).read_text(encoding="utf-8"))
    items: list[dict] = payload["items"]
    known = {x["url"] for x in items}
    added = 0

    for page in range(1, args.pages + 1):
        url = BASE if page == 1 else f"{BASE}index_{page}.html"
        try:
            html = curl(url)
        except Exception as e:
            print(f"[WARN] 第 {page} 页失败：{type(e).__name__}")
            break
        if len(html) < 3000:
            break
        soup = BeautifulSoup(html, "lxml")
        hits = 0
        for a in soup.find_all("a", href=True):
            title = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
            href = a["href"].strip()
            if not href.startswith("http"):
                continue
            if len(title) < 8 or not any(k in title for k in KWS):
                continue
            if href in known:
                continue
            known.add(href)
            items.append({
                "id": f"URL-{len(items) + 1:04d}",
                "title": title,
                "url": href,
                "region": "深圳市",
                "level": "city",
                "doc_type": "政策文件",
                "source_name": "深圳市交通运输局-政策法规",
                "source_page": url,
                "status": "pending",
            })
            added += 1
            hits += 1
        print(f"  第 {page} 页：新增 {hits}")
        # 没有下一页就停
        if f"index_{page + 1}.html" not in html:
            break

    payload["items"] = items
    payload["count"] = len(items)
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    Path(args.seed).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"新增 {added} 条，种子共 {len(items)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
