"""时间元数据规范化、有效性判断与条款身份匹配。

这一层是整个实验的“客观事实来源”：
  - 构建 benchmark 时用它判断旧条款是否真的过期
  - 评估时用它判断 Top-5 是否包含旧条款
  - Temporal_Check 节点只用它过滤，**不读取 benchmark 的 old_clause_id**

如果这里跟着评测样本走（例如把 old_clause_id 传给过滤节点），实验就泄漏了答案。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date, datetime
from typing import Any, Optional

from .models import Clause

_NULLISH = {"", "无", "none", "null", "nat", "n/a", "na", "-", "--", "至今", "长期"}
_DATE_RE = re.compile(r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})")


def parse_date(value: Any) -> Optional[date]:
    """尽量把 Chroma / CSV / 中文日期解析成 date，失败返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    s = str(value).strip()
    if s.lower() in _NULLISH:
        return None
    # Chroma 里可能存 ISO 时间戳（2026-01-01T00:00:00Z）
    m = _DATE_RE.search(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def iso_or_none(value: Any) -> Optional[str]:
    d = parse_date(value)
    return d.isoformat() if d else None


def clause_valid_from(clause: Clause) -> Optional[date]:
    md = clause.metadata
    for key in ("valid_from", "effective_date", "effective_at", "start_date", "issue_date"):
        d = parse_date(md.get(key))
        if d:
            return d
    return None


def clause_valid_to(clause: Clause) -> Optional[date]:
    md = clause.metadata
    # 兼容 Chroma 的 null；也兼容把空值写成字符串的导出工具
    for key in ("valid_to", "expiry_date", "expires_at", "end_date", "invalid_at"):
        if key not in md:
            continue
        raw = md.get(key)
        d = parse_date(raw)
        if d:
            return d
        if raw is None or str(raw).strip().lower() in _NULLISH:
            return None
    return None


def clause_status(clause: Clause) -> str:
    return str(clause.metadata.get("status") or "").strip()


def effective_topic(clause: Clause) -> str:
    """metadata.topic 优先；没有时按低空政策主题词典兜底。

    大规模库里 topic 往往缺失，不能直接拿 policy_name 当主题，
    否则同一政策内的所有条款会互相配对。词典兜底可解释、可复现、零 API。
    """
    if clause.topic:
        return clause.topic
    # 顺序有讲究：一条“对小型起降点按投资额补贴”同时含「小型」「起降」「补贴」，
    # 应按业务主题归到补贴或基础设施，而不是因为出现“小型”就归到重量分类。
    rules = (
        ("补贴与奖励", ("补贴", "奖励", "资助", "补助")),
        ("起降基础设施", ("起降点", "起降场", "机库", "通用机场")),
        ("实名登记", ("实名登记", "登记备案", "信息核验")),
        ("操控员资质", ("操控员", "驾驶员", "执照")),
        ("运营合格证", ("运营合格证", "经营许可", "经营资质")),
        ("飞行计划审批", ("飞行计划", "飞行活动申请", "提前申请", "审批", "批准")),
        ("飞行服务保障", ("飞行服务站", "服务平台", "管理平台", "保障平台")),
        ("重量与机型分类", ("最大起飞重量", "微型", "轻型", "小型", "中型", "大型")),
        ("适飞空域与高度", ("适飞空域", "真高", "飞行高度", "空域")),
    )
    text = clause.text
    for topic, kws in rules:
        if any(k in text for k in kws):
            return topic
    return "未分类"


def effective_region(clause: Clause) -> str:
    r = clause.region
    if not r:
        return "未标注"
    if r in ("国家", "国家级", "全国范围", "中华人民共和国"):
        return "全国"
    return r


def usable(clause: Clause, as_of: date) -> tuple[bool, str]:
    """返回 (是否可作为 as_of 时点的有效依据, 原因)。

    规则：
      1. 明确“已废止/失效” → 不可用
      2. valid_from 晚于 as_of → 尚未生效
      3. valid_to 早于 as_of → 已过有效期
      4. valid_to 为空 → 长期有效
    注意 valid_to 在未来时仍可用；不能只判断“非空”就当成过期。
    """
    status = clause_status(clause)
    if status and any(k in status for k in ("废止", "失效", "无效", "停止执行")):
        return False, f"状态为「{status}」"

    vf = clause_valid_from(clause)
    if vf and vf > as_of:
        return False, f"尚未生效（valid_from={vf.isoformat()}）"

    vt = clause_valid_to(clause)
    if vt and vt < as_of:
        return False, f"已过有效期（valid_to={vt.isoformat()}）"
    return True, ""


def normalize_for_match(text: str) -> str:
    """用于“Top-5 是否包含旧条款”的文本比对。

    去掉空白和所有标点，避免 Chroma 导出时多一个空格就漏判。
    不用模糊相似度做唯一判据：数值条款改动一个数字就是不同政策，
    模糊匹配会把“15万元”和“50万元”误认成同一条。
    """
    s = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[\W_]+", "", s, flags=re.UNICODE)


def stable_clause_id(text: str, metadata: dict | None = None) -> str:
    """没有向量库主键时（本地索引）生成稳定 ID。"""
    md = metadata or {}
    key = "|".join((
        str(md.get("policy_id") or md.get("policy_name") or ""),
        str(md.get("article") or ""),
        normalize_for_match(text),
    ))
    return "local_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def clause_matches(
    candidate: Clause,
    *,
    expected_id: str = "",
    expected_text: str = "",
    min_ratio: float = 0.92,
) -> bool:
    """判断召回条款是否就是 benchmark 里的旧条款。

    优先用库主键；没有主键或主键跨库不一致时，用规范化文本比对。
    文本只做“互为包含 + 长度比”判断，避免短句包含造成的误报。
    """
    if expected_id and candidate.id == expected_id:
        return True
    a = normalize_for_match(candidate.text)
    b = normalize_for_match(expected_text)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) / max(1, len(long)) >= min_ratio and short in long


def bigram_jaccard(a: str, b: str) -> float:
    """词面相似度的轻量代理，用于挑候选对，不用于最终判定。"""
    x = normalize_for_match(a)
    y = normalize_for_match(b)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    gx = {x[i:i + 2] for i in range(len(x) - 1)}
    gy = {y[i:i + 2] for i in range(len(y) - 1)}
    return len(gx & gy) / max(1, len(gx | gy))
