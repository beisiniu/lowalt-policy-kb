"""用 CDP 真实浏览器抓取政策原文（解决源码残留与 JS 渲染问题）。

为什么需要它：
  部分政府站返回的 HTML 里正文嵌在脚本/样式之间，直接抓源码会得到
  "整页 JS + CSS"（汉字占比 < 10%）。用浏览器渲染后取 innerText，
  天然只保留可见正文。

关键约束：
  Chrome 进程在沙箱里跨命令调用会被回收，因此**启动、抓取、关闭必须在同一次执行内完成**。

用法：
  python tools/fetch_by_browser.py --manifest data/_tmp/refetch_targets.json
  python tools/fetch_by_browser.py --url "https://..." --out data/national_policies/xxx.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path(r"C:\Users\86139\.workbuddy\skills\browser-automation-cdp")
PY = sys.executable

# 渲染后抽取正文的 JS：优先常见正文容器，回退 body.innerText
EXTRACT_JS = r"""
(() => {
  const pick = [
    '#ivs_content', '.article-content', '.content', '#content', '.TRS_Editor',
    '.pages_content', '.zoom', '#zoom', '.article', '.news_content', '.detail',
    'div.xxgk_content', '#mainText', '.main-content'
  ];
  for (const sel of pick) {
    const el = document.querySelector(sel);
    if (el && el.innerText && el.innerText.trim().length > 400) {
      return el.innerText;
    }
  }
  const body = document.body ? document.body.innerText : '';
  return body;
})()
"""


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout)


def cdp(*args: str, timeout: int = 120) -> str:
    r = run([PY, str(SKILL / "scripts" / "cdp.py"), "--port", "9222", *args], timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    return out.strip()


def fetch_batch(targets: list[dict], out_dir: Path, wait: float = 1.5) -> list[dict]:
    """一次执行内完成：启动浏览器 → 逐个渲染抓取 → 关闭。"""
    results = []
    print("启动浏览器…", flush=True)
    r = run([PY, str(SKILL / "scripts" / "start_chrome.py"), "--temp", "--headless"], timeout=90)
    if "READY" not in (r.stdout or "") and "ALREADY_RUNNING" not in (r.stdout or ""):
        print("[WARN] 启动输出异常:", (r.stdout or r.stderr)[:200], flush=True)
    time.sleep(2)

    js_file = out_dir / "_extract.js"
    js_file.parent.mkdir(parents=True, exist_ok=True)
    js_file.write_text(EXTRACT_JS, encoding="utf-8")

    try:
        for i, t in enumerate(targets, 1):
            url, name = t["url"], t["name"]
            print(f"[{i}/{len(targets)}] {name[:40]}", flush=True)
            op = cdp("open", url, timeout=120)
            if "CDP_UNREACHABLE" in op:
                print("   CDP 掉线，重建浏览器", flush=True)
                run([PY, str(SKILL / "scripts" / "start_chrome.py"), "--temp", "--headless"], timeout=90)
                time.sleep(3)
                op = cdp("open", url, timeout=120)
            if '"isErrorPage": true' in op or '"isErrorPage":true' in op:
                print("   错误页，跳过", flush=True)
                results.append({**t, "ok": False, "reason": "error_page"})
                continue
            time.sleep(wait)
            txt = cdp("eval", "--file", str(js_file), timeout=120)
            # cdp.py 的 eval 输出可能带前后缀，取最长行块
            body = txt
            if body.startswith("{"):
                try:
                    j = json.loads(body)
                    body = j.get("result") or j.get("value") or ""
                except json.JSONDecodeError:
                    pass
            body = body.strip()
            ok = len(body) > 600
            results.append({**t, "ok": ok, "chars": len(body), "reason": "" if ok else "too_short"})
            if ok:
                dst = out_dir / f"{name}.txt"
                dst.write_text(body, encoding="utf-8")
                print(f"   OK {len(body)} 字符 -> {dst.name[:50]}", flush=True)
            else:
                print(f"   过短（{len(body)}），首 120 字：{body[:120]}", flush=True)
    finally:
        run([PY, str(SKILL / "scripts" / "stop_chrome.py")], timeout=60)
        print("浏览器已关闭", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="JSON 文件：[{url, name}]")
    ap.add_argument("--url")
    ap.add_argument("--out")
    ap.add_argument("--wait", type=float, default=1.5)
    a = ap.parse_args()

    out_dir = ROOT / "data" / "_tmp" / "browser_fetch"
    if a.manifest:
        targets = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    elif a.url and a.out:
        targets = [{"url": a.url, "name": Path(a.out).stem}]
        out_dir = Path(a.out).parent
    else:
        print("需要 --manifest 或 (--url + --out)")
        return 2

    res = fetch_batch(targets, out_dir, a.wait)
    print()
    print(json.dumps(res, ensure_ascii=False, indent=1))
    ok = sum(1 for r in res if r["ok"])
    print(f"成功 {ok}/{len(res)}")
    return 0 if ok == len(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
