# -*- coding: utf-8 -*-
"""政策状态跟踪工具：从语料库元数据自动生成"即将废止时间表"。

用法:
    python tools/status_tracker.py            # 重新生成 output/政策状态跟踪清单.md 的动态部分
    python tools/status_tracker.py --check    # 仅检查异常(已到期仍标现行等),不写文件
"""
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERRIDES = ROOT / "data/national/validity_overrides.json"
MANIFEST = ROOT / "data/national/corpus_manifest.json"
TRACKING = ROOT / "output/政策状态跟踪清单.md"

VALID_OK = ("现行有效", "有效", "")


def _parse_date(s):
    """从各种日期写法中提取 yyyy-mm-dd。"""
    if not s:
        return None
    m = re.search(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})", str(s))
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def collect_entries():
    """汇总所有状态条目: (来源, region, name, status, expiry_date, valid_from, url, note)。"""
    entries = []
    # 1) corpus_manifest(来自 extracted 元数据,含 LLM 提取的 expiry_date/valid_from)
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for p in m.get("policies_detail", []):
        ed = _parse_date(p.get("expiry_date") or p.get("valid_to"))
        vf = _parse_date(p.get("valid_from") or p.get("effective_date"))
        entries.append({
            "source": "manifest",
            "region": p.get("region", ""),
            "name": p.get("policy_name", ""),
            "status": p.get("status", ""),
            "expiry": ed,
            "vfrom": vf,
            "url": p.get("source_url", ""),
            "note": p.get("status_note", ""),
        })
    # 2) validity_overrides(人工核验层,优先级更高,覆盖同名)
    v = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    for url, item in v.items():
        ed = _parse_date(item.get("expiry_date"))
        vf = _parse_date(item.get("valid_from"))
        # 用覆盖条目替换 manifest 中同 URL 的条目(人工核验优先)
        for e in entries:
            if e["url"] == url:
                e["source"] = "override"
                e["status"] = item.get("status", e["status"])
                e["expiry"] = ed or e["expiry"]
                e["vfrom"] = vf or e["vfrom"]
                e["note"] = (item.get("note") or item.get("evidence", ""))[:80]
                break
        else:
            entries.append({
                "source": "override",
                "region": "",
                "name": "",
                "status": item.get("status", ""),
                "expiry": ed,
                "vfrom": vf,
                "url": url,
                "note": (item.get("note") or item.get("evidence", ""))[:80],
            })
    return entries


def check_anomalies(entries, today):
    """异常检测:已到期仍标现行 / 未生效却标现行且已过生效日。"""
    problems = []
    for e in entries:
        if e["expiry"] and e["status"] in VALID_OK and e["expiry"] < today:
            problems.append(("已到期仍标现行", e))
    return problems


def build_table(entries, today):
    """按到期日分组生成即将废止时间表(markdown 行)。"""
    valid = [e for e in entries if e["status"] in VALID_OK and e["expiry"]]
    valid.sort(key=lambda e: e["expiry"])

    def fmt(e):
        days = (e["expiry"] - today).days
        name = e["name"] or e["note"] or e["url"][:50]
        name = re.sub(r"[|\n]", " ", name)[:42]
        return f"| {e['expiry'].isoformat()} | {e['region'] or '—'} | {name} | {'**' + str(days) + ' 天**' if 0 <= days <= 60 else days} |"

    expired = [e for e in valid if e["expiry"] < today]
    soon = [e for e in valid if today <= e["expiry"] <= date(today.year + 1, today.month, today.day)]
    later = [e for e in valid if e["expiry"] > date(today.year + 1, today.month, today.day)]

    # 即将生效(公布未施行): 标现行但 valid_from 在未来
    incoming = sorted([e for e in entries if e["status"] in VALID_OK and e["vfrom"] and e["vfrom"] > today],
                      key=lambda e: e["vfrom"])

    lines = []
    lines.append(f"### 2.1 已到期但状态仍为『现行』——**需立即改标**({len(expired)} 条)\n")
    if expired:
        lines.append("| 到期日 | 地区 | 文件 | 状态 |")
        lines.append("| --- | --- | --- | --- |")
        lines += [f"| {e['expiry'].isoformat()} | {e['region'] or '—'} | {(e['name'] or e['url'])[:44]} | ⚠️ 待改标 |" for e in expired]
    else:
        lines.append("无——所有已到文件均已正确改标。")
    lines.append("")
    lines.append(f"### 2.2 12 个月内到期({len(soon)} 条,即\"即将废止\"完整名单)\n")
    lines.append("| 到期日 | 地区 | 文件 | 距今 |")
    lines.append("| --- | --- | --- | --- |")
    lines += [fmt(e) for e in soon]
    lines.append("")
    lines.append(f"### 2.3 12 个月以上到期({len(later)} 条,中长期观察)\n")
    lines.append("| 到期日 | 地区 | 文件 | 距今(天) |")
    lines.append("| --- | --- | --- | --- |")
    lines += [fmt(e) for e in later]
    lines.append("")
    lines.append(f"### 2.4 即将生效——已公布未施行({len(incoming)} 条,窗口期内不得作为现行依据召回)\n")
    if incoming:
        lines.append("| 生效日 | 地区 | 文件 | 距今 |")
        lines.append("| --- | --- | --- | --- |")
        for e in incoming:
            days = (e["vfrom"] - today).days
            name = e["name"] or e["note"] or e["url"][:50]
            name = re.sub(r"[|\n]", " ", name)[:42]
            region = e["region"] or re.sub(r"[|\n]", " ", e["note"])[:16] or "—"
            d_str = f"**{days} 天**" if 0 <= days <= 60 else str(days)
            lines.append(f"| {e['vfrom'].isoformat()} | {region} | {name} | {d_str} |")
    else:
        lines.append("无——当前没有处于\"已公布未施行\"窗口期的文件。")
    return "\n".join(lines), expired


TEMPLATE_HEAD = """# 政策状态跟踪清单

> ⚠️ **核心事实:语料库中每份文件的状态(现行/已废止/已失效)都是"截至某时点"的快照,会随时间变化。**
> 本清单的"即将废止时间表"由 `tools/status_tracker.py` 从语料库元数据**自动生成**(共扫描 {total} 条状态记录),非人工列举。
> 最后生成:{today}

---

## 一、状态时效性原则(给出题人和系统维护者)

1. **任何"现行有效"标注都有时效**。有效期的三个来源:①文件附则明文(最可靠)②官方平台标注 ③官方废止/修订通知。
2. **"公布≠施行"**:法规通过后常有施行窗口期(如北京新规 9-11 通过、11-15 施行)。窗口期内:新版已存在但**不是现行依据**,旧版仍是现行依据。时序检索必须同时处理 `valid_from`(未来生效)与 `valid_to`。
3. **同一问题跨状态变化日答案会翻转**(例:"北京现在能飞无人机吗"——11-14 答"能",11-15 答"全域禁飞")。这类题是时序分支的最强考题。
4. **失效原因要精确**:已废止(官方明文废止)≠ 已失效(有效期届满)≠ 暂停实施 ≠ 停止执行。出题时"为什么失效"的答案各不相同。
5. **到期≠一定失效**:部分文件到期后会自动延期或被新版承接(如宝安 2023 措施到期后区发改局仍按其公示资助)。改标前先搜一次"新版/修订/延期"。

---

## 二、即将废止时间表(自动生成)
"""


TEMPLATE_TAIL = """
---

## 三、等待官方发布的文件(出现即入库)

| 城市 | 待发布文件 | 线索 | 预期 |
| --- | --- | --- | --- |
| 大理州 | 《"十五五"低空经济发展规划》 | 州政府官网工作动态确认已编制 | 印发后大理双全 |
| 深圳市宝安区 | 《促进低空经济产业发展的若干措施》新版 | 2023版有效期3年已届满(2026-07-29),发改局2026年度仍按旧版公示资助 | 新版印发后宝安双全 |
| 西安市 | 《加快推进低空经济发展实施方案(2024-2026)》原文 | 2024年制定,媒体报道确认存在,官方URL未公开 | 找到官方URL即入库 |
| 福建省 | 《低空经济高质量发展行动方案(2024-2027)》正式版 | 2024-11 征求意见,正式印发版未检索到 | 印发后入库 |
| 北京市 | 原版《无人驾驶航空器管理规定》(2026-05-01施行)全文 | 11-15被修订版替代,失效后公报可查 | 获取URL后入库标"已失效" |

("公布未施行"窗口期文件见 2.4 节,由工具自动生成)

---

## 四、复查操作规程

1. **每月一次**:运行 `python tools/status_tracker.py --check`,核对异常;到 2.1 有条目立即改标并重建库。
2. **每季一次**:按"三"表搜索一次待发布文件(`site:gov.cn 城市 + 文件名 + 印发`)。
3. **出题前**:所有涉及"现在/目前/还能吗"字样的题,先对当前日期核对 ground_truth;跨状态变化日的题写明基准日期。
4. **修正留痕**:覆盖表修改都更新 `note` 字段记录修正原因与日期(参见宝安区 2026-10-27 勘误)。
"""


def main():
    today = date.today()
    entries = collect_entries()
    check_only = "--check" in sys.argv

    problems = check_anomalies(entries, today)
    table, expired = build_table(entries, today)

    if check_only:
        print(f"[check] 状态记录总数: {len(entries)}")
        print(f"[check] 异常(已到期仍标现行): {len(problems)}")
        for tag, e in problems:
            print(f"  - {e['expiry']} | {e['region']} | {(e['name'] or e['url'])[:50]}")
        return

    content = TEMPLATE_HEAD.format(total=len(entries), today=today.isoformat()) + "\n" + table + "\n" + TEMPLATE_TAIL
    TRACKING.write_text(content, encoding="utf-8")
    print(f"[write] {TRACKING}")
    print(f"[stats] 总记录 {len(entries)} | 已到期未改标 {len(problems)} | 12个月内到期 {sum(1 for e in entries if e['status'] in VALID_OK and e['expiry'] and today <= e['expiry'] <= date(today.year+1, today.month, today.day))}")


if __name__ == "__main__":
    main()
