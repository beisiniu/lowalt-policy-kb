"""
低空政策收割管线(三段式)

  discover  找种子:官方搜索接口 / 列表页解析,只收 *.gov.cn,写入种子表(pending)
  harvest   批量爬:requests -> 系统 curl 兜底 -> CDP 浏览器渲染,质量校验后落盘标准 txt
  retry     失败重试:仅对 failed 种子,优先浏览器渲染
  report    进度表:种子/成功/失败/入库/打标统计

种子表: sources/lowalt_policy_urls.json(既有结构 + harvest_status/harvest_note 字段)
台账:   data/national/crawl_manifest.jsonl(追加)
成品:   data/national_policies/<日期>_<名称>.txt(与手工采集同格式)

用法:
  python tools/harvest_pipeline.py discover --keyword "低空经济" --pages 3
  python tools/harvest_pipeline.py harvest --limit 30
  python tools/harvest_pipeline.py retry --browser --limit 10
  python tools/harvest_pipeline.py report
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED_FILE = ROOT / "sources" / "lowalt_policy_urls.json"
POLICY_DIR = ROOT / "data" / "national_policies"
MANIFEST = ROOT / "data" / "national" / "crawl_manifest.jsonl"
REGION_MAP_FILE = ROOT / "output" / "china_map" / "region_map.json"
CDP_SKILL = Path("C:/Users/86139/.workbuddy/skills/browser-automation-cdp")
PYEXE = sys.executable

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
KEYWORDS = ("低空", "通用航空", "通航", "无人机", "航空经济", "eVTOL", "空天")
MIN_BODY_CHARS = 1200  # 低于此视为低质(目录页/新闻摘要/扫描版)

# ---------------------------------------------------------------- 基础设施

def _load(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def curl_get(url: str, timeout: float = 30.0) -> tuple[int, bytes]:
    """系统 curl 兜底(政府站 ECC 证书问题)。返回 (status, body)"""
    cmd = ["curl", "-k", "-L", "-sS", "--compressed", "--max-time", str(int(timeout)),
           "-A", UA, "-H", "Accept-Language: zh-CN,zh;q=0.9", url]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout + 15)
        return 200, out.stdout or b""
    except Exception:
        return 0, b""


def http_get(url: str, timeout: float = 30.0) -> tuple[int, bytes]:
    """requests 优先,失败/异常降级 curl。"""
    try:
        import requests
        headers = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}
        r = requests.get(url, timeout=timeout, verify=False, headers=headers)
        if r.status_code < 400:
            return r.status_code, r.content
    except Exception:
        pass
    return curl_get(url, timeout)


def extract_text(html: str) -> str:
    """复用既有解析器;失败退回全文文本。"""
    try:
        from bs4 import BeautifulSoup
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "crawl_policy_urls", str(ROOT / "tools" / "crawl_policy_urls.py"))
        cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cp)
        text = cp.extract_content(BeautifulSoup(html, "lxml"))
        if len(text) >= 400:
            return text
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "lxml").get_text("\n", strip=True)
    except Exception:
        return re.sub(r"<[^>]+>", "\n", html)


def extract_pdf(data: bytes) -> str:
    try:
        import io
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception:
        return ""


def cdp_render(url: str, wait: int = 8) -> str:
    """浏览器渲染(必须单调用内完成:启动+打开+提取)。"""
    chrome = CDP_SKILL / "scripts" / "start_chrome.py"
    cdp = CDP_SKILL / "scripts" / "cdp.py"
    if not chrome.exists():
        return ""
    subprocess.run([PYEXE, str(chrome), "--temp"], capture_output=True, timeout=60)
    import time
    time.sleep(8)
    subprocess.run([PYEXE, str(cdp), "open", url], capture_output=True, timeout=120)
    time.sleep(wait)
    r = subprocess.run([PYEXE, str(cdp), "text", "--max", "30000"],
                       capture_output=True, timeout=180)
    return (r.stdout or b"").decode("utf-8", "replace")


def clean_render_text(t: str) -> str:
    """CDP 全页文本 -> 正文区间:去掉头部导航与尾部备案。"""
    if not t:
        return ""
    lines = t.splitlines()
    # 正文起点:第一个含"印发/通知/措施/条例/方案/意见"且较长的行
    start = 0
    for i, ln in enumerate(lines):
        if len(ln) > 12 and re.search(r"关于印发|若干措施|实施方案|行动方案|条例|若干意见|管理办法|实施细则", ln):
            start = i
            break
    end = len(lines)
    for i, ln in enumerate(lines):
        if re.search(r"网站地图|备案| Government |主办单位|联系我们 \| 网站声明", ln):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


# ---------------------------------------------------------------- 种子表

def seed_items() -> list[dict]:
    d = _load(SEED_FILE, {"items": []})
    return d.get("items", [])


def save_seeds(items: list[dict]):
    d = _load(SEED_FILE, {"items": []})
    d["items"] = items
    d["count"] = len(items)
    d["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    _save(SEED_FILE, d)


def next_seed_id(items: list[dict]) -> str:
    ids = [int(x["id"].split("-")[-1]) for x in items
           if str(x.get("id", "")).startswith("URL-") and str(x["id"]).split("-")[-1].isdigit()]
    return f"URL-{(max(ids) + 1) if ids else 100}"


def upsert_seed(items: list[dict], title: str, url: str, region: str,
                source: str, note: str = "", level: str = "city") -> bool:
    """新种子入表(pending);已存在则跳过。返回是否新增。"""
    if not url or not url.startswith(("http://", "https://")):
        return False
    if "gov.cn" not in url:  # 权威性红线
        return False
    if any(x["url"] == url for x in items):
        return False
    items.append({
        "id": next_seed_id(items),
        "title": title[:120],
        "url": url,
        "region": region or "待识别",
        "level": level,
        "doc_type": "政策文件",
        "source_name": source,
        "added_at": datetime.date.today().isoformat(),
        "harvest_status": "pending",
        "harvest_note": note,
    })
    return True


# ---------------------------------------------------------------- discover

def discover_from_gd_search(keyword: str, pages: int) -> int:
    """广东省政府站内搜索 JSONP 接口(已验证可用),返回新增种子数。"""
    added = 0
    items = seed_items()
    import requests
    for page in range(1, pages + 1):
        api = ("https://search.gd.gov.cn/jsonp/listKeywordJsonp/?callback=CB"
               f"&site=755001&keyword={keyword}&page={page}&pagesize=20")
        try:
            r = requests.get(api, timeout=25, verify=False, headers={"User-Agent": UA})
            m = re.search(r"CB\((.*)\)\s*$", r.text, re.S)
            data = json.loads(m.group(1)) if m else {}
        except Exception as e:
            print(f"  [搜索第{page}页失败] {e}")
            continue
        rows = data.get("data", {}).get("list", []) or data.get("list", [])
        for row in rows:
            url = row.get("url") or ""
            title = re.sub(r"<[^>]+>", "", row.get("title", "")).strip()
            site = (row.get("site_name") or "").strip()
            if "gov.cn" not in url or not any(k in title for k in KEYWORDS):
                continue
            if upsert_seed(items, title, url, "", site or "广东省政府站内搜索", "gd-search"):
                added += 1
                print(f"  + {title[:46]}")
    save_seeds(items)
    return added


def discover_from_list_pages(urls: list[str], keyword: str = "", all_links: bool = False) -> int:
    """通用列表页解析:提取详情链接(标题含关键词;all_links 时整页收录)。"""
    added = 0
    items = seed_items()
    for lp in urls:
        code, body = http_get(lp)
        if code >= 400 or not body:
            print(f"  [列表页失败] {lp[:60]}")
            continue
        html = body.decode("utf-8", "replace")
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        base = lp.split("/")[0] + "//" + lp.split("/")[2]
        for a in soup.find_all("a", href=True):
            href, title = a["href"], re.sub(r"\s+", "", a.get_text(" ", strip=True))
            if not title or len(title) < 10:
                continue
            if not all_links and keyword not in title and not any(k in title for k in KEYWORDS):
                continue
            if not re.search(r"post_|art_|content|t20\d\d|XXGK|zwgk|fgwj|zcwj|document", href, re.I):
                continue
            if href.startswith("/"):
                href = base + href
            elif not href.startswith("http"):
                href = lp.rsplit("/", 1)[0] + "/" + href
            if "gov.cn" not in href or href.rstrip("/") == lp.rstrip("/"):
                continue
            if upsert_seed(items, title, href, "", f"列表页:{lp[:40]}", "list-page"):
                added += 1
                print(f"  + {title[:46]}")
    save_seeds(items)
    return added


# ---------------------------------------------------------------- harvest

def guess_region_from_title(title: str) -> str:
    """按标题前缀猜区域(城市名表)。"""
    m = re.match(r"^(深圳市|广州市|武汉市|成都市|杭州市|南京市|苏州市|无锡市|常州市|南通市|徐州市|扬州市|"
                 r"珠海市|佛山市|东莞市|惠州市|中山市|青岛市|长沙市|合肥市|福州市|厦门市|温州市|绍兴市|"
                 r"嘉兴市|烟台市|潍坊市|洛阳市|宜昌市|襄阳市|株洲市|湘潭市|岳阳市|常德市|汕头市|湛江市|"
                 r"汕头|遵义市|绵阳市|宜宾市|南充市|泸州市|大连市|沈阳市|长春市|哈尔滨市|太原市|石家庄市|"
                 r"呼和浩特市|兰州市|西宁市|银川市|乌鲁木齐市|贵州省|湖南省|河南省|山东省|山西省|河北省|"
                 r"四川省|安徽省|福建省|江西省|湖北省|江苏省|浙江省|广东省|辽宁省|吉林省|黑龙江省|"
                 r"青海省|甘肃省|陕西省|云南省|海南省|贵州省|内蒙古自治区|广西壮族自治区|宁夏回族自治区|"
                 r"新疆维吾尔自治区|西藏自治区|北京市|天津市|上海市|重庆市)", title)
    return m.group(1) if m else "待识别"


def quality_ok(text: str) -> bool:
    if len(text) < MIN_BODY_CHARS:
        return False
    # 条款结构:至少出现若干"第X条/（一）/1."样式
    hits = len(re.findall(r"第[一二三四五六七八九十百]+条|（[一二三四五六七八九十]+）|\(\d+\)", text))
    if hits >= 5:
        return True
    # 通告/通知类文件无条款编号,但含政策文书特征词即接受
    formal = len(re.findall(r"特此通告|特此公告|特此通知|同时废止|同步废止|自即日起|施行|有效期|不另行通知", text))
    return formal >= 2 and "低空" in text or "无人驾驶航空器" in text or "无人机" in text


def save_policy(title: str, url: str, region: str, source: str, text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text).strip()
    date = ""
    m = re.search(r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})", url) or \
        re.search(r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})", text[:600])
    if m:
        date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    else:
        date = datetime.date.today().isoformat()
    name = re.sub(r"[\\/:*?\"<>|\n\r\t]", "", title)[:80]
    head = (f"政策名称：{name}\n发布日期：{date}\n来源链接：{url}\n"
            f"抓取时间：{datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"正文哈希：sha256:{hashlib.sha256(text.encode()).hexdigest()}\n版本：rev1\n"
            f"采集来源：{source}\n区域提示：{region}\n" + "-" * 60 + "\n\n")
    out = POLICY_DIR / f"{date}_{name}.txt"
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(head + text, encoding="utf-8")
    return str(out)


def log_manifest(status: str, seed: dict, reason: str = "", out: str = ""):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.open("a", encoding="utf-8").write(json.dumps({
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": status, "url": seed["url"], "title": seed["title"][:60],
        "reason": reason, "out": out,
    }, ensure_ascii=False) + "\n")


def fetch_one(seed: dict, use_browser: bool = False) -> tuple[str, str]:
    """抓单条种子。返回 (status, note)。status: done/low_quality/failed"""
    url = seed["url"]
    code, body = http_get(url)
    text = ""
    if code and body:
        low = url.lower()
        if low.endswith(".pdf") or body[:4] == b"%PDF":
            text = extract_pdf(body)
        else:
            text = extract_text(body.decode("utf-8", "replace"))
    if not quality_ok(text) and use_browser and not low_pdf(url):
        rendered = clean_render_text(cdp_render(url))
        if quality_ok(rendered):
            text = f"{text}\n{rendered}" if text and len(text) > len(rendered) else rendered
    if not quality_ok(text):
        note = f"低质/不足{MIN_BODY_CHARS}字(现{len(text)})"
        return ("low_quality" if text else "failed"), note
    region = seed.get("region", "")
    if region in ("", "待识别"):
        region = guess_region_from_title(seed["title"])
    out = save_policy(seed["title"], url, region, seed.get("source_name", "收割管线"), text)
    return "done", out


def low_pdf(url: str) -> bool:
    return url.lower().endswith(".pdf")


def run_harvest(limit: int, browser_fallback: bool, only_failed: bool = False):
    items = seed_items()
    if only_failed:
        todo = [x for x in items if x.get("harvest_status") in ("failed", "low_quality")]
    else:
        todo = [x for x in items if x.get("harvest_status") == "pending"]
    todo = todo[:limit]
    print(f"待处理 {len(todo)} 条\n")
    ok = bad = 0
    for i, seed in enumerate(todo, 1):
        status, note = fetch_one(seed, use_browser=browser_fallback)
        seed["harvest_status"] = status
        seed["harvest_note"] = note
        log_manifest(status, seed, note if status != "done" else "", note if status == "done" else "")
        mark = "OK " if status == "done" else ("LOW" if status == "low_quality" else "FAIL")
        print(f"[{i}/{len(todo)}] {mark} {seed['title'][:44]}")
        if status == "done":
            ok += 1
        else:
            bad += 1
        save_seeds(items)  # 逐条落盘,中断可续
    print(f"\n完成: 成功 {ok} / 低质或失败 {bad}")
    # 已落盘的 done 种子清理入休:打标交给 extract_policy.py 批量
    done = [x for x in items if x.get("harvest_status") == "done"
            and x.get("harvest_note", "").endswith(".txt")]
    print(f"累计已入库种子: {len(done)}(后续执行 extract_policy.py 打标)")


# ---------------------------------------------------------------- report

def run_report():
    items = seed_items()
    from collections import Counter
    st = Counter(x.get("harvest_status", "未标记") for x in items)
    tagged = {p.stem for p in (ROOT / "data" / "extracted_national").glob("*.json")}
    txts = {p.stem for p in POLICY_DIR.glob("*.txt")}
    pending = [x for x in items if x.get("harvest_status") == "pending"]
    failed = [x for x in items if x.get("harvest_status") in ("failed", "low_quality")]
    print("=== 种子表状态 ===")
    for k, v in st.most_common():
        print(f"  {k}: {v}")
    print(f"政策 txt: {len(txts)} | 已打标: {len(tagged)}")
    print(f"\n待爬 pending: {len(pending)}")
    for x in pending[:15]:
        print(f"  - {x['title'][:52]}")
    if len(pending) > 15:
        print(f"  ... 等 {len(pending)} 条")
    print(f"\n失败/低质: {len(failed)}(可用 retry --browser 定向处理)")
    for x in failed[:15]:
        print(f"  - [{x.get('harvest_status')}] {x['title'][:44]} | {x.get('harvest_note','')[:40]}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="低空政策收割管线")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="发现种子写入种子表")
    d.add_argument("--keyword", default="低空经济")
    d.add_argument("--pages", type=int, default=3, help="搜索接口页数")
    d.add_argument("--list-page", nargs="*", default=[], help="额外列表页 URL")
    d.add_argument("--all", action="store_true", help="整页收录列表页详情链接(不过滤关键词,由打标阶段判低空相关性)")

    h = sub.add_parser("harvest", help="批量爬 pending 种子")
    h.add_argument("--limit", type=int, default=30)
    h.add_argument("--browser", action="store_true", help="低质时尝试浏览器渲染兜底")

    r = sub.add_parser("retry", help="重试 failed/low_quality 种子")
    r.add_argument("--limit", type=int, default=10)
    r.add_argument("--browser", action="store_true", default=True)

    rp = sub.add_parser("report", help="进度表")

    args = ap.parse_args()
    if args.cmd == "discover":
        n = discover_from_gd_search(args.keyword, args.pages)
        print(f"[搜索接口] 新增种子 {n}")
        if args.list_page:
            n2 = discover_from_list_pages(args.list_page, all_links=args.all)
            print(f"[列表页] 新增种子 {n2}")
    elif args.cmd == "harvest":
        run_harvest(args.limit, browser_fallback=args.browser)
    elif args.cmd == "retry":
        run_harvest(args.limit, browser_fallback=True, only_failed=True)
    elif args.cmd == "report":
        run_report()


if __name__ == "__main__":
    main()
