#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人工数据导入：把浏览器里存下来/复制下来的政策，变成流水线能直接吃的格式。

给反爬或前端渲染的站点（中国政府网、民航局、深圳政府在线…）留的入口。
导进来的文件和爬虫抓的格式**完全一致**，后面 extract_policy.py / kb_ingest.py 不用改。

用法：
    # 单个文件（HTML 或纯文本）
    python tools/import_manual.py 下载/某政策.html
    python tools/import_manual.py 某政策.txt --date 2025-06-01 --region 深圳市

    # 整个目录批量
    python tools/import_manual.py 下载目录/ --out data/manual_policies

    # 先看会导入什么，不落盘
    python tools/import_manual.py 下载目录/ --dry-run

支持：
    .html/.htm  浏览器"另存为"的整页 → 自动抽标题、日期、正文
    .txt/.md    已经复制好的纯文本 → 首行当标题（或 --title），日期从正文里找
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

from policycrawler.parser import parse_detail
from policycrawler.state import HistoryStore
from policycrawler.storage import save_policy
from policycrawler.utils import normalize_text, parse_date_to_iso, sha256_text, slugify

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".html", ".htm", ".txt", ".md"}
# 破折号必须是「空格+破折号+空格」才算分隔符，否则会把"（2026—2028年）"这一刀切了
_SITE_TAIL = re.compile(r"\s{2,}|\u3000|\s+[-_|—]+\s+")


def collect(src: Path) -> list[Path]:
    if src.is_dir():
        return sorted(p for p in src.rglob("*") if p.suffix.lower() in SUFFIXES)
    return [src] if src.exists() else []


def from_html(path: Path) -> tuple[str, str, str]:
    """返回 (标题, 日期, 正文)。选择器传空 = 全部走兜底逻辑。"""
    html = path.read_text(encoding="utf-8", errors="replace")
    d = parse_detail(html, {}, f"manual://{path.name}")
    title = d.title or ""
    if not title:
        title = path.stem
    soup = BeautifulSoup(html, "lxml")
    # 优先用页面自己声明的 canonical / og:url，其次才用伪 URL
    url = ""
    node = soup.find("link", rel="canonical") or soup.find("meta", attrs={"property": "og:url"})
    if node is not None:
        url = (node.get("href") or node.get("content") or "").strip()
    return title, d.publish_date, d.content


def from_text(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "--------------------" in text:  # 已经是标准格式，直接取正文
        text = text.split("--------------------", 1)[1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = lines[0] if lines and len(lines[0]) <= 60 else path.stem
    title = re.sub(r"^政策名称[：:]\s*", "", title)
    # 纯文本没有 DOM 可依赖，取正文里**最后一个**日期 —— 公文落款日期一般就在末尾
    dates = re.findall(r"\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}", "\n".join(lines))
    date = parse_date_to_iso(dates[-1]) if dates else ""
    return title, date, normalize_text("\n".join(lines))


def clean_title(raw: str, fallback: str) -> str:
    """去掉站点后缀（"…的通知  广东省人民政府"）。"""
    t = _SITE_TAIL.split(raw)[0].strip() if raw else ""
    t = t.strip("《》 ")
    return t if len(t) >= 6 else (fallback or raw or "未命名政策")


def main() -> int:
    ap = argparse.ArgumentParser(description="人工导入政策文件")
    ap.add_argument("src", help="文件或目录")
    ap.add_argument("--out", default=str(ROOT / "data" / "manual_policies"), help="输出目录")
    ap.add_argument("--history", default="", help="状态文件（默认 <out>/manual_history.json）")
    ap.add_argument("--date", default="", help="统一指定发布日期 YYYY-MM-DD")
    ap.add_argument("--title", default="", help="统一指定标题（单文件时用）")
    ap.add_argument("--url", default="", help="来源链接（单文件时用）")
    ap.add_argument("--dry-run", action="store_true", help="只预览，不落盘")
    args = ap.parse_args()

    files = collect(Path(args.src))
    if not files:
        print(f"[ERROR] 没找到可导入的文件：{args.src}（支持 {'/'.join(sorted(SUFFIXES))}）")
        return 1

    out_dir = Path(args.out)
    hist_path = Path(args.history) if args.history else out_dir / "manual_history.json"
    store = HistoryStore(hist_path, source="人工导入")
    print(f"待处理 {len(files)} 个文件 → {out_dir}")
    print(f"状态文件：{hist_path}（已有 {store.count} 条）\n")

    added = skipped = 0
    for i, f in enumerate(files, 1):
        try:
            if f.suffix.lower() in (".html", ".htm"):
                title, date, body = from_html(f)
            else:
                title, date, body = from_text(f)
        except Exception as e:  # 单文件失败不影响其余
            print(f"[{i}] [FAIL] {f.name}: {type(e).__name__}: {e}")
            continue

        title = clean_title(args.title or title, f.stem)
        date = parse_date_to_iso(args.date) or parse_date_to_iso(date) or parse_date_to_iso(f.name) or ""
        url = args.url or f"manual://{slugify(title)}"

        body = (body or "").strip()
        if len(body) < 50:
            print(f"[{i}] [SKIP] {f.name}: 正文只有 {len(body)} 字，可能没抽到正文（试试 --title/手工复制成 txt）")
            skipped += 1
            continue

        chash = sha256_text(normalize_text(body))
        rec = store.get(url)
        if rec and rec.get("content_hash") == chash:
            print(f"[{i}] [DUP ] {title[:40]}（内容未变）")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[{i}] [DRY ] {date or '----'}  {title[:40]}  {len(body)} 字")
            added += 1
            continue

        detail = type("D", (), {"title": title, "url": url, "publish_date": date, "content": body})()
        if rec:
            txt_path, json_path, _ = save_policy(detail, out_dir, revision=int(rec.get("revision", 1)) + 1)
            store.mark_updated(url, chash, title=title, publish_date=date,
                               content_length=len(body), txt_path=str(txt_path), json_path=str(json_path))
            print(f"[{i}] [UPD ] {date or '----'}  {title[:40]}  {len(body)} 字 → {txt_path.name}")
        else:
            txt_path, json_path, _ = save_policy(detail, out_dir)
            store.add_new({"url": url, "title": title, "publish_date": date, "content_hash": chash,
                           "content_length": len(body), "revision": 1,
                           "txt_path": str(txt_path), "json_path": str(json_path)})
            print(f"[{i}] [NEW ] {date or '----'}  {title[:40]}  {len(body)} 字 → {txt_path.name}")
        added += 1

    print("-" * 70)
    print(f"新增/更新 {added} 篇，跳过 {skipped} 篇" + ("（dry-run，未写盘）" if args.dry_run else ""))
    if not args.dry_run and added:
        print(f"\n下一步：\n  python extract_policy.py {out_dir}\n  python kb_ingest.py data/extracted --rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
