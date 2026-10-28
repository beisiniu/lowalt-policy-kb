#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 URL 种子逐条抓取政策正文，并留下可审计台账。

为什么不用 incremental_crawler.py 的单一 config？
  全国政策分散在各省市政府网站，列表结构完全不同。这里改用「已知政策 URL 清单」，
  每条单独抓详情页，更适合跨站采集；列表页增量仍由 incremental_crawler.py 负责。

台账字段（data/national/crawl_manifest.jsonl）：
  seed_id, title, url, final_url, level, region, source_name,
  http_status, content_type, bytes, chars, publish_date,
  status(ok/duplicate_content/...), reason, content_hash, duplicate_of,
  txt_path, fetched_at, elapsed_s

用法：
    python tools/crawl_policy_urls.py --dry-run
    python tools/crawl_policy_urls.py --limit 5
    python tools/crawl_policy_urls.py --only-level national
    python tools/crawl_policy_urls.py               # 全量
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urldefrag, urlparse, urlunparse
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from policycrawler.utils import CST, sha256_text   # noqa: E402

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

DEFAULT_SEED = ROOT / "sources" / "lowalt_policy_urls.json"
DEFAULT_OUT = ROOT / "data" / "national_policies"
DEFAULT_LEDGER = ROOT / "data" / "national" / "crawl_manifest.jsonl"
DEFAULT_CSV = ROOT / "data" / "national" / "crawl_manifest.csv"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
DATE_RE = re.compile(r"(20\d{2})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

CONTENT_SELECTORS = (
    "div.TRS_Editor", "div#zoom", "div.zoom", "div#UCAP-CONTENT", "div.article-content",
    "div.article_con", "div.art-con", "div.content", "div.zw", "div.view", "div#content",
    "div.main-content", "div.detail", "div.article", "div.pages_content", "article",
    "div#js_content", "div.wzcon", "div.text", "td.content",
)
# ⚠ 不要删 <form>：大量老政府站点把正文包在 form 里（四川、部分省政府都是），
# 整块删掉会把正文一起丢掉，只剩导航。
DROP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "iframe", "button")
DROP_CLASS_RE = re.compile(r"(nav|menu|footer|header|crumb|share|related|recommend|advert|banner|sidebar|print|comment)", re.I)
BOILERPLATE = (
    "打印本页", "关闭窗口", "分享到", "字体：", "字体:", "扫一扫", "责任编辑", "浏览次数",
    "网站地图", "主办单位", "版权所有", "ICP备", "上一篇", "下一篇", "相关阅读", "返回顶部",
    "点击量", "访问量", "当前位置", "首页 >", "首页>>", "分享：",
)


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def canonical_url(url: str) -> str:
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")]
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))


def safe_filename(s: str, limit: int = 58) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip(" ._-")
    return s[:limit] or "未命名政策"


def parse_any_date(s: str) -> str:
    m = DATE_RE.search(s or "")
    if not m:
        return ""
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date().isoformat()
    except ValueError:
        return ""


def clean_title(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    # 只剥常见的“_站点名 / -站点名”，不碰标题里的年份区间
    s = re.split(r"\s{2,}|\s[-_|]\s|_[^_]{2,20}$", s)[0].strip()
    return s


def extract_date(html: str, soup: BeautifulSoup, text: str) -> str:
    metas = (
        "PubDate", "pubdate", "publishdate", "publish_date", "firstpublishedtime",
        "article:published_time", "og:release_date", "release_date", "date",
    )
    for name in metas:
        node = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if node and node.get("content"):
            d = parse_any_date(node["content"])
            if d:
                return d

    head = text[:3000]
    for pat in (
        r"(?:发布时间|发布日期|成文日期|印发日期|时间)[：:\s]*([^\n]{0,30})",
        r"(20\d{2}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2})",
    ):
        m = re.search(pat, head)
        if m:
            d = parse_any_date(m.group(1) if m.lastindex else m.group(0))
            if d:
                return d
    return ""


def extract_title(soup: BeautifulSoup, seed_title: str) -> str:
    for sel in ('meta[property="og:title"]', 'meta[name="ArticleTitle"]', "h1"):
        node = soup.select_one(sel)
        if node:
            t = node.get("content") if node.name == "meta" else node.get_text(" ", strip=True)
            if t and len(t) >= 6:
                return clean_title(t)
    if soup.title:
        return clean_title(soup.title.get_text(" ", strip=True))
    return clean_title(seed_title)


def _link_ratio(node: Tag, total_len: int) -> float:
    link_len = sum(len(a.get_text("", strip=True)) for a in node.find_all("a"))
    return link_len / max(1, total_len)


def _node_text(node: Tag) -> str:
    # 保留段落边界，避免整篇粘成一行
    return node.get_text("\n", strip=True)


def extract_content(soup: BeautifulSoup) -> str:
    for bad in soup.find_all(DROP_TAGS):
        bad.decompose()
    for bad in soup.find_all(True, class_=DROP_CLASS_RE):
        bad.decompose()

    candidates: list[tuple[float, Tag]] = []
    seen_ids: set[int] = set()
    for sel in CONTENT_SELECTORS:
        try:
            nodes = soup.select(sel)
        except Exception:
            continue
        for node in nodes:
            if id(node) in seen_ids:
                continue
            seen_ids.add(id(node))
            txt = _node_text(node)
            if len(txt) < 150:
                continue
            p_text = "\n".join(p.get_text(" ", strip=True) for p in node.find_all("p") if len(p.get_text(strip=True)) >= 10)
            body = p_text if len(p_text) >= 200 else txt
            score = len(body) * (1.0 - min(0.8, _link_ratio(node, len(txt))))
            score += min(300, len(node.find_all("p"))) * 20
            candidates.append((score, node))

    if not candidates:
        # 兜底：找段落文字总量最大的容器。很多老政府页没有语义 class。
        for node in soup.find_all(("div", "article", "section", "td")):
            ps = [p for p in node.find_all("p") if len(p.get_text(strip=True)) >= 20]
            if len(ps) < 3:
                continue
            p_text = "\n".join(p.get_text(" ", strip=True) for p in ps)
            total = _node_text(node)
            if len(p_text) < 250:
                continue
            score = len(p_text) * (1.0 - min(0.8, _link_ratio(node, len(total))))
            candidates.append((score, node))

    if not candidates:
        body = soup.body or soup
        raw = _node_text(body)
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        node = candidates[0][1]
        ps = [p.get_text(" ", strip=True) for p in node.find_all("p") if len(p.get_text(strip=True)) >= 10]
        raw = "\n".join(ps) if len("\n".join(ps)) >= 200 else _node_text(node)

    lines: list[str] = []
    for line in re.split(r"\n+", raw):
        line = re.sub(r"[ \t\u3000]+", " ", line).strip()
        if not line:
            continue
        if len(line) <= 40 and any(line.startswith(b) or line == b for b in BOILERPLATE):
            continue
        if len(line) <= 30 and re.fullmatch(r"(来源|作者|编辑)[：:].{0,20}", line):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    # 去掉正文开头重复的“来源/时间”行
    text = re.sub(r"^(来源|发布时间|发布日期|编辑)[：:][^\n]*\n", "", text)
    return text.strip()


def extract_docx(data: bytes) -> str:
    """从 .docx 直接读 word/document.xml，避免为一个格式引入 python-docx。"""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    lines: list[str] = []
    for p in root.iter(f"{ns}p"):
        txt = "".join(t.text or "" for t in p.iter(f"{ns}t")).strip()
        if txt:
            lines.append(txt)
    return "\n".join(lines).strip()


def content_hash(text: str) -> str:
    norm = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return "sha256:" + sha256_text(norm)


def load_known_hashes(dirs: list[Path]) -> dict[str, str]:
    known: dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.txt"):
            try:
                head = p.read_text(encoding="utf-8", errors="replace")[:1200]
            except OSError:
                continue
            m = re.search(r"正文哈希[：:]\s*(sha256:[0-9a-f]+)", head)
            if m:
                known.setdefault(m.group(1), str(p))
    return known


def load_previous(ledger: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not ledger.exists():
        return out
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[canonical_url(row.get("url", ""))] = row
    return out


def append_ledger(ledger: Path, row: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(ledger: Path, csv_path: Path) -> None:
    latest: dict[str, dict] = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[canonical_url(row.get("url", ""))] = row
    fields = [
        "seed_id", "title", "url", "final_url", "level", "region", "source_name",
        "http_status", "status", "reason", "publish_date", "chars", "content_hash",
        "txt_path", "fetched_at",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(latest.values(), key=lambda x: (x.get("level", ""), x.get("region", ""), x.get("title", ""))):
            w.writerow(row)


def save_policy(item: dict, title: str, publish_date: str, body: str,
                final_url: str, ctype: str, out_dir: Path) -> tuple[Path, Path, str]:
    h = content_hash(body)
    prefix = publish_date or "undated"
    base = f"{prefix}_{safe_filename(title)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"{base}.txt"
    if txt.exists():
        base = f"{base}_{h[-8:]}"
        txt = out_dir / f"{base}.txt"
    json_path = txt.with_suffix(".json")

    header = (
        f"政策名称：{title}\n"
        f"发布日期：{publish_date or '无'}\n"
        f"来源链接：{final_url}\n"
        f"抓取时间：{now_iso()}\n"
        f"正文哈希：{h}\n"
        "版本：rev1\n"
        f"采集来源：{item.get('source_name', '')}\n"
        f"区域提示：{item.get('region', '')}\n"
        "------------------------------------------------------------\n"
    )
    txt.write_text(header + body + "\n", encoding="utf-8", newline="\n")
    side = {
        "title": title,
        "url": item["url"],
        "final_url": final_url,
        "publish_date": publish_date or "无",
        "fetched_at": now_iso(),
        "content_hash": h,
        "txt_path": str(txt.resolve()),
        "seed_id": item.get("id", ""),
        "level": item.get("level", ""),
        "region_hint": item.get("region", ""),
        "source_name": item.get("source_name", ""),
        "source_page": item.get("source_page", ""),
        "doc_type": item.get("doc_type", ""),
        "content_type": ctype,
    }
    json_path.write_text(json.dumps(side, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return txt, json_path, h



class _CurlResp:
    """把 curl 的输出包成 requests.Response 的最小可用形状。"""
    __slots__ = ("status_code", "content", "headers", "url", "encoding")

    def __init__(self, status_code: int, content: bytes, ctype: str, url: str):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": ctype}
        self.url = url
        self.encoding = None

    @property
    def apparent_encoding(self) -> str:
        # 政府站以 UTF-8 为主，老站是 GB18030；按能否解码判断，够用且不引依赖
        for enc in ("utf-8", "gb18030"):
            try:
                self.content.decode(enc)
                return enc
            except UnicodeDecodeError:
                continue
        return "utf-8"

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or self.apparent_encoding, "replace")

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def raise_for_status(self) -> None:
        return None


def curl_get(url: str, timeout: float) -> _CurlResp:
    """用系统 curl 兜底：Python 的 OpenSSL 对部分政府站点会 BAD_ECPOINT，
    而系统 curl 走 Schannel/BoringSSL，通常能正常握手（实测深圳政府网可用）。"""
    cmd = [
        "curl", "-k", "-L", "-sS", "--compressed",
        "--max-time", str(max(5, int(timeout))),
        "-A", UA,
        "-H", "Accept-Language: zh-CN,zh;q=0.9",
        "-w", "\n__META__%{http_code}|%{content_type}",
        url,
    ]
    out = subprocess.run(cmd, capture_output=True, timeout=timeout + 15)
    raw = out.stdout or b""
    body, sep, meta = raw.rpartition(b"\n__META__")
    if not sep:
        return _CurlResp(0, raw, "", url)
    code_s, _, ctype = meta.decode("utf-8", "replace").partition("|")
    try:
        code = int(code_s)
    except ValueError:
        code = 0
    return _CurlResp(code, body, ctype, url)


def crawl_one(session: requests.Session, item: dict, out_dir: Path, timeout: float) -> dict:
    t0 = time.time()
    url = canonical_url(item["url"])
    row = {
        "seed_id": item.get("id", ""),
        "title": item.get("title", ""),
        "url": url,
        "final_url": "",
        "level": item.get("level", ""),
        "region": item.get("region", ""),
        "source_name": item.get("source_name", ""),
        "http_status": "",
        "content_type": "",
        "bytes": 0,
        "chars": 0,
        "publish_date": "",
        "status": "pending",
        "reason": "",
        "content_hash": "",
        "duplicate_of": "",
        "txt_path": "",
        "fetched_at": now_iso(),
        "elapsed_s": 0,
    }

    path_lower = url.lower().split("?")[0]
    if path_lower.endswith((".doc", ".xls", ".xlsx", ".zip")):
        row.update(status="unsupported_type", reason="旧版 Office/压缩包，首轮先跳过")
        row["elapsed_s"] = round(time.time() - t0, 2)
        return row

    try:
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True, verify=False)
        except requests.exceptions.SSLError:
            # 深圳等站点 TLS 报 BAD_ECPOINT：本地 OpenSSL 不支持对方 ECC 证书。
            # 先用系统 curl 重试（Schannel 能握手），再退到 http。
            row["reason"] = "HTTPS TLS 握手失败，改用系统 curl 重试"
            resp = curl_get(url, timeout)
            if resp.status_code >= 400 and url.startswith("https://"):
                resp = curl_get("http://" + url[len("https://"):], timeout)
        row["http_status"] = resp.status_code
        row["final_url"] = resp.url
        ctype = (resp.headers.get("Content-Type") or "").lower()
        row["content_type"] = ctype.split(";")[0]
        row["bytes"] = len(resp.content)
        if resp.status_code >= 400:
            row.update(status="fetch_error", reason=f"HTTP {resp.status_code}")
            row["elapsed_s"] = round(time.time() - t0, 2)
            return row

        seed_title = clean_title(item.get("title", ""))
        is_docx = path_lower.endswith(".docx") or "wordprocessingml" in ctype
        is_pdf = path_lower.endswith(".pdf") or "application/pdf" in ctype
        if is_docx:
            body = extract_docx(resp.content)
            title = seed_title or "未命名政策"
            dummy = BeautifulSoup("", "lxml")
            publish_date = extract_date("", dummy, body[:2000])
        elif is_pdf:
            try:
                from pypdf import PdfReader
            except ImportError:
                row.update(status="unsupported_type", reason="需要 pypdf 解析 PDF")
                row["elapsed_s"] = round(time.time() - t0, 2)
                return row
            reader = PdfReader(io.BytesIO(resp.content))
            pages = [(pg.extract_text() or "") for pg in reader.pages]
            body = "\n".join(pages).strip()
            title = seed_title or "未命名政策"
            dummy = BeautifulSoup("", "lxml")
            publish_date = extract_date("", dummy, body[:3000])
        else:
            if any(x in ctype for x in ("pdf", "msword", "officedocument", "excel", "zip")):
                row.update(status="unsupported_type", reason=f"非 HTML：{ctype}")
                row["elapsed_s"] = round(time.time() - t0, 2)
                return row
            if not resp.encoding or resp.encoding.lower() in ("iso-8859-1", "ascii"):
                resp.encoding = resp.apparent_encoding or "utf-8"
            html = resp.text
            soup = BeautifulSoup(html, "lxml")
            title = extract_title(soup, item.get("title", ""))
            # 索引标题是人工维护的政策名，比站点 <title>（常带“门户网站/欢迎您”）可靠
            if seed_title and len(seed_title) >= 8:
                title = seed_title
            body = extract_content(soup)
            publish_date = extract_date(html, soup, body or soup.get_text("\n", strip=True))
        # 少数法规正文含“通过日期/签署日期”，页面日期不可靠；种子可显式指定发布日
        publish_date = item.get("publish_date_hint") or publish_date
        row["chars"] = len(body)
        row["publish_date"] = publish_date

        # 政策全文通常 >1000 字；低于 500 的多是摘要/新闻导语/只抓到导航。
        # 宁可标 extract_failed 并保留 URL 台账，也不要把残缺文本灌进知识库。
        if len(body) < 500 or len(CJK_RE.findall(body)) < 250:
            row.update(status="extract_failed", reason=f"正文过短（{len(body)} 字），可能只抓到摘要/导航")
            row["elapsed_s"] = round(time.time() - t0, 2)
            return row

        txt, _, h = save_policy(item, title, publish_date, body, resp.url, row["content_type"], out_dir)
        row.update(status="ok", content_hash=h, txt_path=str(txt.resolve()))
        if not publish_date:
            row["reason"] = "正文已抓取，但发布日期未识别"
        row["elapsed_s"] = round(time.time() - t0, 2)
        return row
    except requests.Timeout as e:
        row.update(status="fetch_error", reason=f"超时：{str(e)[:100]}")
    except requests.RequestException as e:
        row.update(status="fetch_error", reason=f"网络错误：{type(e).__name__}: {str(e)[:120]}")
    except Exception as e:
        row.update(status="parse_error", reason=f"{type(e).__name__}: {str(e)[:160]}")
    row["elapsed_s"] = round(time.time() - t0, 2)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="按 URL 清单抓取低空政策并记录台账")
    ap.add_argument("--seed", default=str(DEFAULT_SEED))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-level", choices=["national", "province", "city"], default="")
    ap.add_argument("--timeout", type=float, default=25)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true", help="跳过上次已成功/重复的 URL")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seed_path = Path(args.seed)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if args.only_level:
        items = [x for x in items if x.get("level") == args.only_level]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("[ERROR] 没有待抓 URL")
        return 2

    out_dir = Path(args.out_dir)
    ledger = Path(args.ledger)
    previous = load_previous(ledger)
    known_hashes = load_known_hashes([
        out_dir,
        ROOT / "data" / "gd_policies",
        ROOT / "data" / "beijing_policies",
        ROOT / "data" / "manual_policies",
    ])

    if args.dry_run:
        print(f"待抓 {len(items)} 条：")
        for x in items:
            print(f"  [{x.get('level', ''):8}] {x.get('region', ''):10} {x.get('title', '')[:54]}")
        return 0

    session = requests.Session()
    session.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
    })

    ok = dup = 0
    failed = 0
    skipped_resume = 0
    print(f"待抓 {len(items)} 条 → {out_dir}")
    for i, item in enumerate(items, 1):
        cu = canonical_url(item["url"])
        prev = previous.get(cu)
        if args.resume and prev and prev.get("status") in ("ok", "duplicate_content"):
            skipped_resume += 1
            print(f"[{i}/{len(items)}] SKIP(已抓) {item.get('title', '')[:48]}")
            continue

        print(f"[{i}/{len(items)}] {item.get('region', ''):8} {item.get('title', '')[:48]}")
        row = crawl_one(session, item, out_dir, args.timeout)

        if row["status"] == "ok":
            h = row["content_hash"]
            if h in known_hashes:
                # 正文与已有政策相同：把刚写的文件删掉，保留 URL 记录
                p = Path(row["txt_path"])
                side = p.with_suffix(".json")
                try:
                    p.unlink(missing_ok=True)
                    side.unlink(missing_ok=True)
                except OSError:
                    pass
                row.update(status="duplicate_content", duplicate_of=known_hashes[h],
                           txt_path="", reason="正文哈希与已有政策一致")
                dup += 1
            else:
                known_hashes[h] = row["txt_path"]
                ok += 1
        elif row["status"] == "duplicate_content":
            dup += 1
        else:
            failed += 1

        append_ledger(ledger, row)
        write_csv(ledger, Path(args.csv))
        print(f"          → {row['status']:18} {row.get('reason', '')[:60]}  {row['chars']}字")
        if args.delay and i < len(items):
            time.sleep(args.delay)

    write_csv(ledger, Path(args.csv))
    latest: dict[str, dict] = load_previous(ledger)
    status_count: dict[str, int] = {}
    for v in latest.values():
        status_count[v.get("status", "")] = status_count.get(v.get("status", ""), 0) + 1
    print("\n" + "=" * 60)
    print(f"本轮：成功 {ok} / 重复 {dup} / 失败 {failed} / resume跳过 {skipped_resume}")
    print(f"台账最新状态：{status_count}")
    print(f"URL 台账：{ledger}")
    print(f"CSV 台账：{args.csv}")
    print(f"政策目录：{out_dir}")
    return 0 if ok or dup else 2


if __name__ == "__main__":
    raise SystemExit(main())
