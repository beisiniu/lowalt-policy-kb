#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段一边界用例测试（异常路径，不是 happy path）

  E1 状态文件损坏      → 自动备份成 .broken，按空历史继续跑，不能崩
  E2 单条详情页 404    → 只记 failed，其余条目照常入库，退出码 2
  E3 列表页有 5 条但日期全缺 → 必须告警，不能静默返回 0 条
  E4 列表页选择器全部失配  → 必须告警提示"可能改版了"
  E5 同一正文出现在两个链接 → 识别为 DUP，只落一份文件
  E6 连接失败 → 抛 FetchError，不能抛 AttributeError（异常表写错会让重试整段失效）
  E7 详情页日期与列表页冲突 → 以列表页（发布日期）为准

运行：python tests/verify_phase1_edges.py
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        return


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Quiet, directory=str(directory)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def write_site(tmp: Path, list_html: str, details: dict[str, str]) -> Path:
    site = tmp / "site"
    (site / "detail").mkdir(parents=True, exist_ok=True)
    (site / "list.html").write_text(list_html, encoding="utf-8")
    for name, html in details.items():
        (site / "detail" / name).write_text(html, encoding="utf-8")
    return site


def cfg(tmp: Path, port: int) -> Path:
    raw = {
        "source": {"name": "边界测试站", "list_url": f"http://127.0.0.1:{port}/list.html",
                   "base_url": f"http://127.0.0.1:{port}/", "charset": "utf-8"},
        "list_selectors": {"item": ["div.list ul li"], "title": ["a"], "link": ["a"], "date": ["span.date"]},
        "detail_selectors": {"title": ["h1"], "publish_date": ["div.info span.date"],
                             "content": ["div.article-content"]},
        "crawl": {"timeout": 5, "retries": 1, "retry_backoff": 1, "delay_seconds": 0.01,
                  "max_items_per_run": 20, "check_content_update": True},
        "storage": {"history_file": "data/h.json", "policy_dir": "data/policies", "log_dir": "data/logs"},
        "politeness": {"user_agents": ["Mozilla/5.0 EdgeTest/1.0"]},
    }
    p = tmp / "config.json"
    p.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def run(config: Path) -> tuple[int, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    p = subprocess.run([PY, "incremental_crawler.py", "-c", str(config)], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def detail_html(title: str, date: str, body: str) -> str:
    return (f"<html><head><meta charset='utf-8'></head><body><h1>{title}</h1>"
            f"<div class='info'><span class='date'>{date}</span></div>"
            f"<div class='article-content'><p>{body}</p></div></body></html>")


def ok(flag: bool, label: str, extra: str = "") -> bool:
    print(f"[{'PASS' if flag else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return flag


def case_e1() -> bool:
    tmp = Path(tempfile.mkdtemp())
    site = write_site(tmp, "<div class='list'><ul><li><a href='detail/a.html'>某低空飞行管理办法</a>"
                           "<span class='date'>2026-01-02</span></li></ul></div>",
                      {"a.html": detail_html("某低空飞行管理办法", "2026-01-02", "第一条 " + "无人机管理。" * 30)})
    srv, port = serve(site)
    try:
        c = cfg(tmp, port)
        # 先跑一次产生状态文件，再手工写坏
        run(c)
        h = tmp / "data" / "h.json"
        h.write_text('{"records": {"broken": ', encoding="utf-8")
        code, out = run(c)
        broken = list((tmp / "data").glob("h.json.broken"))
        flag = code == 0 and broken and "不是合法 JSON" in out
        return ok(flag, "E1 状态文件损坏 → 备份 .broken 并继续", f"exit={code} backup={bool(broken)}")
    finally:
        srv.shutdown()


def case_e2() -> bool:
    tmp = Path(tempfile.mkdtemp())
    body = "第一条 " + "无人机飞行应当遵守本规定。" * 30
    site = write_site(
        tmp,
        "<div class='list'><ul>"
        "<li><a href='detail/ok.html'>正常政策文件</a><span class='date'>2026-02-01</span></li>"
        "<li><a href='detail/missing.html'>缺失的政策文件</a><span class='date'>2026-02-02</span></li>"
        "</ul></div>",
        {"ok.html": detail_html("正常政策文件", "2026-02-01", body)},
    )
    srv, port = serve(site)
    try:
        code, out = run(cfg(tmp, port))
        files = list((tmp / "data" / "policies").glob("*.txt"))
        flag = code == 2 and len(files) == 1 and "FAIL" in out
        return ok(flag, "E2 单条 404 → 只记 failed，其余照常入库", f"exit={code} 落盘 {len(files)} 篇")
    finally:
        srv.shutdown()


def case_e3() -> bool:
    """列表页 5 条都没有日期 → 应该告警，而不是静默 0 条。"""
    tmp = Path(tempfile.mkdtemp())
    items = "".join(f"<li><a href='detail/p{i}.html'>某低空管理政策第{i}号</a></li>" for i in range(5))
    site = write_site(tmp, f"<div class='list'><ul>{items}</ul></div>", {})
    srv, port = serve(site)
    try:
        code, out = run(cfg(tmp, port))
        flag = code == 0 and ("日期" in out and ("丢弃" in out or "跳过" in out))
        return ok(flag, "E3 条目全无日期 → 明确告警（不静默丢数据）", "有日期告警" if flag else out.strip().splitlines()[-1][:60])
    finally:
        srv.shutdown()


def case_e4() -> bool:
    """选择器全失配（站点改版）→ 应该提示可能改版了。"""
    tmp = Path(tempfile.mkdtemp())
    site = write_site(tmp, "<div class='brand-new'><p><a href='detail/p0.html'>某低空政策</a></p></div>", {})
    srv, port = serve(site)
    try:
        code, out = run(cfg(tmp, port))
        flag = code == 0 and ("改版" in out or "选择器" in out)
        return ok(flag, "E4 列表页 0 条 → 提示选择器可能失配", "有提示" if flag else "无提示")
    finally:
        srv.shutdown()


def case_e5() -> bool:
    """同一篇正文挂在两个链接下 → DUP，只落一份。"""
    tmp = Path(tempfile.mkdtemp())
    body = "第一条 " + "同一篇政策被转载到两个栏目。" * 30
    site = write_site(
        tmp,
        "<div class='list'><ul>"
        "<li><a href='detail/a.html'>某低空经济促进条例</a><span class='date'>2026-03-01</span></li>"
        "<li><a href='detail/b.html'>某低空经济促进条例（转载）</a><span class='date'>2026-03-01</span></li>"
        "</ul></div>",
        {"a.html": detail_html("某低空经济促进条例", "2026-03-01", body),
         "b.html": detail_html("某低空经济促进条例（转载）", "2026-03-01", body)},
    )
    srv, port = serve(site)
    try:
        code, out = run(cfg(tmp, port))
        files = list((tmp / "data" / "policies").glob("*.txt"))
        flag = code == 0 and len(files) == 1 and "DUP" in out
        return ok(flag, "E5 跨链接同正文 → 识别 DUP，只落一份", f"落盘 {len(files)} 篇")
    finally:
        srv.shutdown()



def case_e6() -> bool:
    """网络层异常必须是 FetchError，且消息里有原因。

    历史 bug：except 元组里写了 requests.ChunkedEncodingError（顶层不存在），
    Python 求值异常元组时就抛 AttributeError，重试与退避全部失效。
    """
    sys.path.insert(0, str(ROOT))
    from policycrawler.fetcher import Fetcher, FetchError

    try:
        Fetcher(timeout=1, retries=2, delay=0, verify_ssl=False).get_text("http://127.0.0.1:1/")
        return ok(False, "E6 连接失败抛 FetchError", "居然没抛异常")
    except FetchError as e:
        return ok("抓取失败" in str(e), "E6 连接失败抛 FetchError", str(e)[:56])
    except AttributeError as e:
        return ok(False, "E6 连接失败抛 FetchError", f"抛了 AttributeError：{e}")


def case_e7() -> bool:
    """列表页写 2026-10-23，详情页写 2027-01-01（实施日）→ 必须取列表页。"""
    tmp = Path(tempfile.mkdtemp())
    body = "第一条 " + "行政规范性文件应当经合法性审核。" * 30
    site = write_site(
        tmp,
        "<div class='list'><ul><li><a href='detail/x.html'>某行政规范性文件备案审查规定</a>"
        "<span class='date'>2026-10-23</span></li></ul></div>",
        {"x.html": detail_html("某行政规范性文件备案审查规定", "2027-01-01", body)},
    )
    srv, port = serve(site)
    try:
        run(cfg(tmp, port))
        meta_files = list((tmp / "data" / "policies").glob("*.json"))
        if not meta_files:
            return ok(False, "E7 日期冲突时以列表页为准", "没有落盘")
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        return ok(meta.get("publish_date") == "2026-10-23",
                  "E7 日期冲突时以列表页为准", f"publish_date={meta.get('publish_date')}")
    finally:
        srv.shutdown()


def main() -> int:
    results = [case_e1(), case_e2(), case_e3(), case_e4(), case_e5(), case_e6(), case_e7()]
    print("=" * 60)
    print("阶段一边界用例：全部通过 ✅" if all(results) else "阶段一边界用例：存在失败项 ❌")
    print("=" * 60)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
