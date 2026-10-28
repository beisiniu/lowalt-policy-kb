"""批量修复正文质量问题（源码残留 / 页面框架 / PDF 乱码）。

背景：
  154 篇语料里有 39 篇正文不可信——网页源码被当正文（汉字占比 6%~25%）、
  浏览器渲染带出导航栏与页脚、PDF 用 CID 编码提取出乱码。
  这些内容进入向量库后成为无意义切片，污染检索并虚增库规模。

修复策略（按优先级取质量最好者）：
  1. 浏览器重抓结果（data/_tmp/browser_fetch/*.txt）—— 最干净；
  2. 从原文件剥离 <script>/<style>/标签/实体，回收正文；
  3. 在 2 的基础上再去掉页面导航与页脚。
  达标线：汉字占比 ≥ 0.55 且汉字数 ≥ 400。不达标者不改动并列入待重抓。

用法：
  python tools/repair_bodies.py --dry-run     # 只出报告，不写盘
  python tools/repair_bodies.py               # 写盘（原文件备份为 .bak）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from body_quality import (  # noqa: E402
    auto_repair,
    clean_html_noise,
    judge,
    metrics,
    strip_page_furniture,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "national" / "corpus_manifest.json"
FETCH_DIR = ROOT / "data" / "_tmp" / "browser_fetch"
REPORT = ROOT / "output" / "body_repair_report.json"

# 浏览器重抓结果 → 库内政策名（按包含关系匹配）
FETCH_MAP = {
    "上海2025版适飞空域通告": "上海市人民政府关于公布微型轻型小型无人驾驶航空器适飞空域范围的通告",
    "上海2026版适飞空域通告": "上海市人民政府关于公布微型轻型小型无人驾驶航空器适飞空域范围的通告(2026版)",
    "北京3部试行文件通知页": "北京市公安局关于印发《北京市无人驾驶航空器特殊保障规定(试行)》等3部行政规范性文件",
    "天河暂停实施通知": "广州市天河区发展和改革局关于暂停实施",
    "光明暂停实施公告v2": "深圳市光明区发展和改革局关于暂停实施",
    "陕西推动低空制造方案": "推动低空制造产业高质量发展工作方案",
    "北京管理规定2026修订版": "北京市无人驾驶航空器管理规定",
    "北京特殊保障规定9月修订版": "北京市无人驾驶航空器特殊保障规定(试行)(2026年9月修订版",
}

# 公告/通知类文书正文天然极短（如光明区暂停公告仅 150 字），门槛另设
SHORT_DOC_KW = ("公告", "通告", "通知")
SHORT_MIN_CJK = 120

MIN_RATIO = 0.55
MIN_CJK = 400


def passes(m: dict, policy_name: str) -> bool:
    """达标判定：常规要求 400 字；公告/通告/通知类放宽到 120 字。"""
    if m["cjk_ratio"] < MIN_RATIO or m.get("js_per_10k", 0) > 15:
        return False
    if m["cjk"] >= MIN_CJK:
        return True
    return any(k in policy_name for k in SHORT_DOC_KW) and m["cjk"] >= SHORT_MIN_CJK


def candidates(body: str) -> list[tuple[str, str]]:
    """生成若干清洗方案，返回 (方案名, 文本)。"""
    out = [("raw", body)]
    c1 = clean_html_noise(body)
    out.append(("strip_html", c1))
    c2 = strip_page_furniture(c1)
    out.append(("strip_html+furniture", c2))
    c3 = strip_page_furniture(body)
    out.append(("furniture", c3))
    fixed, ok = auto_repair(body)
    if ok:
        out.append(("auto_repair", fixed))
    return out


def best(texts: list[tuple[str, str]], policy_name: str = "") -> tuple[str, str, dict]:
    """按 (达标, 汉字数) 选最优方案。"""
    scored = []
    for name, t in texts:
        m = metrics(t)
        good = passes(m, policy_name)
        scored.append((good, m["cjk"], name, t, m))
    scored.sort(key=lambda x: (-int(x[0]), -x[1]))
    good, _, name, t, m = scored[0]
    return name, t, {**m, "ok": good}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="只处理政策名包含该子串的条目")
    ap.add_argument("--from-extracted", action="store_true",
                    help="直接从打标 JSON 的 source_file 取文件（用于已被质量闸拦下、不在 manifest 的条目）")
    a = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = man["policies_detail"]
    if a.from_extracted:
        seen, rows = set(), []
        for j in sorted((ROOT / "data" / "extracted_national").glob("*.json")):
            if j.name.endswith(".bak"):
                continue
            try:
                d = json.loads(j.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            it = d[0] if isinstance(d, list) else d
            src = str((it.get("_meta") or {}).get("source_file", ""))
            if not src or src in seen:
                continue
            seen.add(src)
            rows.append({"policy_name": it.get("policy_name", ""), "region": it.get("apply_region", ""),
                         "source_file": src, "chunks": 0})
        entries = rows
        print(f"[from-extracted] 候选文件 {len(entries)} 个", flush=True)

    fetched = {p.stem: p.read_text(encoding="utf-8") for p in FETCH_DIR.glob("*.txt")} if FETCH_DIR.exists() else {}

    report, fixed, pending = [], 0, []
    for p in entries:
        f = Path(p["source_file"])
        if not f.exists():
            continue
        if a.only and a.only not in p["policy_name"]:
            continue
        raw = f.read_text(encoding="utf-8", errors="replace")
        head, _, body = raw.partition("-" * 60)
        verdict, m0 = judge(body)
        if verdict == "ok":
            continue

        opts = []
        # 1) 浏览器重抓（若有）
        for key, kw in FETCH_MAP.items():
            if kw in p["policy_name"] and key in fetched:
                ftxt = strip_page_furniture(fetched[key])
                opts.append((f"browser:{key}", ftxt))
        # 2) 本地清洗
        opts += candidates(body)

        name, text, m1 = best(opts, p["policy_name"])
        report.append({
            "policy_name": p["policy_name"],
            "region": p["region"],
            "chunks_before": p["chunks"],
            "verdict_before": verdict,
            "before": m0,
            "chosen": name,
            "after": m1,
            "action": "repair" if m1["ok"] else "pending_refetch",
        })
        if not m1["ok"]:
            pending.append(p["policy_name"])
            continue

        if not a.dry_run:
            shutil.copy2(f, f.with_suffix(f.suffix + ".bak_before_repair"))
            f.write_text(head + "-" * 60 + "\n" + text, encoding="utf-8")
        fixed += 1
        print(f"[修复] {p['policy_name'][:34]:<36} {m0['cjk_ratio']}->{m1['cjk_ratio']} ({m1['cjk']}字) via {name}", flush=True)

    REPORT.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": a.dry_run,
        "fixed": fixed,
        "pending": pending,
        "items": report,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"可修复 {fixed} 篇 | 仍需重抓 {len(pending)} 篇")
    for n in pending:
        print("   待重抓:", n[:50])
    print("报告:", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
