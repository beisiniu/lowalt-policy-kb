"""规则版抽取器（不联网）。

两个用途：
  1. `--offline` 跑通全链路（没配 API Key 时也能验证数据结构）
  2. 大模型调用失败时的兜底，保证入库链路不断

能力有限：只能处理"自 XXXX 年 XX 月 XX 日起施行""有效期 X 年"这类标准表述。
被新文件取代、政策冲突这类判断必须靠大模型（phase 2 主路径）。
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Optional

from .utils import parse_date_to_iso

_FIELDS = ("policy_name", "issue_date", "effective_date", "expiry_date", "apply_region", "status", "summary")

_DATE_RE = r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"
_TITLE_RE = re.compile(r"《([^》]{4,80})》")
_DOC_KIND = ("条例", "办法", "规定", "通知", "意见", "细则", "方案", "决定", "公告", "规则", "标准", "指南")


def _iso(y: int, m: int, d: int) -> str:
    try:
        return datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def _add_years(iso: str, years: int) -> str:
    d = datetime.strptime(iso, "%Y-%m-%d").date()
    try:
        return d.replace(year=d.year + years).isoformat()
    except ValueError:  # 2/29
        return (d.replace(year=d.year + years, day=28)).isoformat()


def parse_header(text: str) -> dict:
    """爬虫落盘的 txt 头部是 `键：值` + 分隔线，先把它读出来当强先验。"""
    meta: dict[str, str] = {}
    for line in text.splitlines()[:10]:
        if line.startswith("---"):
            break
        m = re.match(r"^(政策名称|发布日期|来源链接|抓取时间|正文哈希|版本)[：:]\s*(.+)$", line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()
    return meta


def _find_policy_name(text: str, header: dict) -> str:
    name = header.get("政策名称", "").strip(" 《》")
    if len(name) >= 4:
        return name
    for m in _TITLE_RE.finditer(text[:1500]):
        cand = m.group(1)
        if any(k in cand for k in _DOC_KIND):
            return cand
    # 没有书名号：取第一行长度合适的标题
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return first[:80]


def _find_issue_date(text: str, header: dict) -> str:
    # 爬虫已经把列表页/详情页的发布日期写进头部，这是最可靠的来源
    if header.get("发布日期"):
        iso = parse_date_to_iso(header["发布日期"])
        if iso:
            return iso
    # 兜底：从正文里找"××××年××月××日……公布/印发"这类句式
    for kw in ("公布", "发布", "印发", "通过"):
        m = re.search(_DATE_RE + r"[^。]{0,20}?" + kw, text[:2500])
        if m:
            return _iso(*m.groups())
    m = re.search(_DATE_RE, text[:2500])
    return _iso(*m.groups()) if m else ""


def _find_effective_date(text: str, issue_date: str) -> str:
    m = re.search(r"自" + _DATE_RE + r"起(?:施行|生效|执行)", text)
    if m:
        return _iso(*m.groups())
    if re.search(r"自(?:公布|发布|印发)之日起(?:施行|生效)", text):
        return issue_date  # "自公布之日起施行" → 生效日 = 发布日
    return ""


def _find_expiry_date(text: str, effective_date: str) -> str:
    m = re.search(r"有效期(?:至|到)" + _DATE_RE, text)
    if m:
        return _iso(*m.groups())
    m = re.search(r"有效期\s*(\d+)\s*年", text)
    if m and effective_date:
        # 有效期 5 年 = 施行日起满 5 年，最后一天要减 1 天
        end = _add_years(effective_date, int(m.group(1)))
        return (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    return ""


# 与 kb/schema.py 的 REGIONS 保持一致（这里独立一份，避免爬虫层反向依赖 kb）
KNOWN_PLACES = (
    "北京市", "上海市", "天津市", "重庆市", "深圳市", "广州市", "珠海市", "东莞市", "佛山市",
    "杭州市", "南京市", "苏州市", "成都市", "武汉市", "西安市", "合肥市", "长沙市",
    "广东省", "浙江省", "江苏省", "四川省", "湖北省", "陕西省", "安徽省", "湖南省",
)
_NATIONAL_RE = re.compile(r"中华人民共和国境内|全国范围|中国民用航空局|国务院|民航局|全国")


def _find_region(text: str, name: str) -> str:
    """判适用地域。

    顺序很重要：**具体地名优先于"全国"判据**。
    反例：一份北京市的文件正文里提到"国务院"，旧逻辑先命中"国务院"就返回全国，
    结果北京的地方文件被当成全国性文件，检索时会给上海的问题当依据。
    """
    # 1) 标题里的地名最可靠 —— 发文机关一定写在标题上
    for place in sorted(KNOWN_PLACES, key=len, reverse=True):
        if place in name:
            return place
    # 2) 正文开头出现的具体地名
    head = text[:1200]
    for place in sorted(KNOWN_PLACES, key=len, reverse=True):
        if place in head:
            return place
    # 3) 通用地名正则（覆盖列表外的城市）
    for m in re.finditer(r"([\u4e00-\u9fa5]{2,6}(?:市|省|自治区))", head):
        cand = m.group(1)
        if not any(bad in cand for bad in ("本市", "该市", "各市", "省市", "区市")):
            return cand
    # 4) 最后才判全国
    if _NATIONAL_RE.search(text[:1500]) or _NATIONAL_RE.search(name):
        return "全国"
    return "全国"


def _find_status(text: str, expiry: str, today: Optional[date] = None) -> str:
    today = today or date.today()
    if expiry:
        try:
            if datetime.strptime(expiry, "%Y-%m-%d").date() < today:
                return "已废止"
        except ValueError:
            pass
    # 明确声明自身废止 / 被取代（"本××自...废止"、"停止执行"）
    if re.search(r"(?:本|该)(?:办法|条例|规定|通知|意见|细则|规则)[^。]{0,30}(?:废止|停止执行)", text):
        return "已废止"
    return "现行有效"


def _find_summary(text: str) -> str:
    """挑一句最能体现"审批/合规口径"的条文。

    规则版的 summary 只是兜底产物：优先选同时含「飞行」+「申请/审批/无需」的句子，
    并去掉条号前缀。精准一句话总结交给大模型。
    """
    body = text.split("-----")[-1] if "-----" in text else text
    sentences = [s.strip() for s in re.split(r"[。；\n]", body) if s.strip()]
    strong, weak = [], []
    for s in sentences:
        if len(s) < 12 or s.startswith(("政策名称", "发布日期", "来源链接", "抓取时间", "正文哈希", "版本")):
            continue
        # 「三、提高线上线下贷款审批效率」这类是小节标题，不是要求本身
        if re.match(r"^[一二三四五六七八九十]+、", s) and len(s) < 32:
            continue
        s = re.sub(r"^第[一二三四五六七八九十百零]+条\s*", "", s)
        if "飞行" in s and re.search(r"申请|审批|许可|无需|免于|备案", s):
            strong.append(s)
        elif re.search(r"申请|审批|许可|无需|免于|备案", s):
            weak.append(s)
    pick = (strong or weak or sentences or [""])[0]
    pick = re.sub(r"\s+", "", pick)
    return (pick[:110] + "…") if len(pick) > 110 else pick


def extract_by_rules(text: str, today: Optional[date] = None) -> dict:
    header = parse_header(text)
    name = _find_policy_name(text, header)
    issue = _find_issue_date(text, header)
    effective = _find_effective_date(text, issue) or issue
    expiry = _find_expiry_date(text, effective)
    region = _find_region(text, name)
    status = _find_status(text, expiry, today)

    return {
        "policy_name": name,
        "issue_date": issue or "无",
        "effective_date": effective or "无",
        "expiry_date": expiry or "无",
        "apply_region": region,
        "status": status,
        "summary": _find_summary(text),
        "_source": "rules",
    }
