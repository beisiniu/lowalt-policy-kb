#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段一自检：用本地演示站验证「增量」到底成不成立。

四个场景（全部应通过）：
  S1 首次运行             → 3 条新增
  S2 原样再跑一次         → 0 新增，3 条跳过（URL + 内容哈希双去重生效）
  S3 列表页新增 1 条政策  → 1 条新增，3 条跳过
  S4 老政策正文被修订     → 1 条更新(rev2)，其余跳过
  S5 再跑一次             → 全部跳过（幂等）

运行：
    python tests/verify_phase1.py
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
DEMO = ROOT / "demo"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # 静音，避免刷屏
        return


def start_server(directory: Path, port: int) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(QuietHandler, directory=str(directory))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run_crawler(config: Path) -> tuple[int, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, "incremental_crawler.py", "-c", str(config)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def summary_line(out: str) -> dict:
    """从输出里抠出 `新增 N | 更新 N | 跳过 N | 失败 N`。"""
    import re
    m = re.search(r"新增 (\d+) \| 更新 (\d+) \| 跳过 (\d+) \| 失败 (\d+)", out)
    if not m:
        return {}
    return dict(new=int(m.group(1)), updated=int(m.group(2)), skipped=int(m.group(3)), failed=int(m.group(4)))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lowalt_demo_"))
    site = tmp / "site"
    shutil.copytree(DEMO, site, ignore=shutil.ignore_patterns("data", "config.demo.json"))

    port = free_port()
    cfg_raw = json.loads((DEMO / "config.demo.json").read_text(encoding="utf-8"))
    cfg_raw["source"]["list_url"] = f"http://127.0.0.1:{port}/list.html"
    cfg_raw["source"]["base_url"] = f"http://127.0.0.1:{port}/"
    cfg_path = tmp / "config.test.json"
    cfg_path.write_text(json.dumps(cfg_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    srv = start_server(site, port)
    print(f"演示站已启动：http://127.0.0.1:{port}/list.html  (临时目录 {tmp})\n")

    cases = [
        ("S1 首次全量抓取", 3, 0, 0),
        ("S2 原样再跑（应全部跳过）", 0, 0, 3),
    ]
    ok = True
    try:
        for name, exp_new, exp_upd, exp_skip in cases:
            code, out = run_crawler(cfg_path)
            got = summary_line(out)
            passed = (code == 0 and got.get("new") == exp_new and got.get("updated") == exp_upd
                      and got.get("skipped") == exp_skip and got.get("failed") == 0)
            ok &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {name}  →  实际 {got}  期望 new={exp_new} upd={exp_upd} skip={exp_skip}")
            if not passed:
                print(out)

        # S3：列表页新增一条
        shutil.copyfile(site / "list_v2.html", site / "list.html")
        code, out = run_crawler(cfg_path)
        got = summary_line(out)
        passed = code == 0 and got.get("new") == 1 and got.get("skipped") == 3 and got.get("failed") == 0
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] S3 列表页新增 1 条  →  实际 {got}  期望 new=1 skip=3")
        if not passed:
            print(out)

        # S4：修改老政策正文（模拟政策修订）
        p1 = site / "detail" / "p1.html"
        html = p1.read_text(encoding="utf-8")
        html = html.replace(
            "<p>第二十条 本条例自2026年5月1日起施行。",
            "<p>第十九条 微型、轻型民用无人驾驶航空器在适飞空域内飞行发生轻微违规的，首次予以警告，不予罚款。</p>\n"
            "  <p>第二十条 本条例自2026年5月1日起施行。",
        )
        p1.write_text(html, encoding="utf-8")
        code, out = run_crawler(cfg_path)
        got = summary_line(out)
        passed = code == 0 and got.get("new") == 0 and got.get("updated") == 1 and got.get("skipped") == 3
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] S4 老政策正文修订  →  实际 {got}  期望 upd=1 skip=3")
        if not passed:
            print(out)

        # S5：再次运行，必须完全幂等
        code, out = run_crawler(cfg_path)
        got = summary_line(out)
        passed = code == 0 and got.get("new") == 0 and got.get("updated") == 0 and got.get("skipped") == 4
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] S5 再跑一次（幂等）  →  实际 {got}  期望 全部跳过 4")

        # 状态文件与落盘文件核对
        hist = json.loads((tmp / "data" / "crawled_history.json").read_text(encoding="utf-8"))
        files = sorted(p.name for p in (tmp / "data" / "policies").glob("*.txt"))
        print("\n状态文件 records：%d 条；落盘 txt：%d 个" % (len(hist["records"]), len(files)))
        for f in files:
            print("   -", f)
        rev2 = [r for r in hist["records"].values() if r.get("revision", 1) > 1]
        print("修订记录（revision>1）：", ", ".join(f"{r['title']} rev{r['revision']}" for r in rev2) or "无")

        ok &= len(hist["records"]) == 4 and len(rev2) == 1
    finally:
        srv.shutdown()

    print("\n" + ("=" * 60))
    print("阶段一自检：全部通过 ✅" if ok else "阶段一自检：存在失败项 ❌")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
