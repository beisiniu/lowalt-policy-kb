"""条款时效的三态判定：valid / invalid / uncertain。

与 temporal.usable() 的区别：
  usable() 是二值的，且把缺失字段当作可用的默认值；
  本模块把“无法核实”显式建模为 uncertain，并记录判断依据与置信度。

TemporalRAGGraph 的旧 filter/downweight 仍使用 usable()，行为不变；
只有 uncertainty 模式调用本模块。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from .models import Clause
from .temporal import clause_status, clause_valid_from, clause_valid_to

VALID = "valid"
INVALID = "invalid"
UNCERTAIN = "uncertain"
STATES = (VALID, INVALID, UNCERTAIN)

POSITIVE_STATUS = ("现行有效", "有效", "生效", "执行中")
NEGATIVE_STATUS = ("废止", "失效", "无效", "停止执行", "暂停", "撤销")


@dataclass(frozen=True)
class Validity:
    state: str
    reason: str
    confidence: float
    basis: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "confidence": self.confidence,
            "basis": list(self.basis),
        }


def _status_kind(status: str) -> str:
    if not status:
        return "missing"
    if any(k in status for k in NEGATIVE_STATUS):
        return "negative"
    if any(k in status for k in POSITIVE_STATUS):
        return "positive"
    return "unknown"


def assess_validity(clause: Clause, as_of: date, family: Any = None) -> Validity:
    """判断条款在 as_of 时点的时效状态。

    family 是可选的政策族谱对象，需提供 node(policy_id) -> PolicyNode|None。
    只有在 chunk 自身字段不足时才查族谱；族谱是独立政策注册表，不能用来覆盖
    chunk 明确写出的相反事实。
    """
    status = clause_status(clause)
    vf = clause_valid_from(clause)
    vt = clause_valid_to(clause)
    kind = _status_kind(status)
    basis: list[str] = []

    if kind == "negative":
        return Validity(INVALID, f"状态为「{status}」", 0.99, ("chunk_status",))

    if vf and vf > as_of:
        return Validity(INVALID, f"尚未生效（valid_from={vf.isoformat()}）",
                        0.95, ("chunk_valid_from",))

    if vt and vt < as_of:
        return Validity(INVALID, f"已过有效期（valid_to={vt.isoformat()}）",
                        0.95, ("chunk_valid_to",))

    if kind == "positive":
        basis = ["chunk_status"]
        if vf:
            basis.append("chunk_valid_from")
        if vt:
            basis.append("chunk_valid_to")
        else:
            basis.append("no_expiry_declared")
        return Validity(VALID, f"状态为「{status}」且日期不冲突", 0.90, tuple(basis))

    # status 缺失或未知：日期能给出部分证据
    pid = str(clause.metadata.get("policy_id") or "")
    node = family.node(pid) if family is not None and pid else None
    if node is not None:
        node_status = str(getattr(node, "status", "") or "")
        node_kind = _status_kind(node_status)
        node_vf = getattr(node, "valid_from", None)
        node_vt = getattr(node, "valid_to", None)
        if node_kind == "negative":
            return Validity(INVALID, f"政策级状态为「{node_status}」",
                            0.85, ("policy_registry", "chunk_missing_status"))
        if node_vt and node_vt < as_of:
            return Validity(INVALID, f"政策级有效期已过（valid_to={node_vt.isoformat()}）",
                            0.85, ("policy_registry", "chunk_missing_status"))
        if node_vf and node_vf > as_of:
            return Validity(INVALID, f"政策级尚未生效（valid_from={node_vf.isoformat()}）",
                            0.85, ("policy_registry", "chunk_missing_status"))
        if node_kind == "positive":
            return Validity(VALID, f"政策级状态为「{node_status}」，条文级状态缺失",
                            0.65, ("policy_registry", "chunk_missing_status"))

    if kind == "unknown":
        return Validity(UNCERTAIN, f"状态无法识别「{status}」", 0.30,
                        ("chunk_status", "unknown_status"))

    # status 完全缺失。若日期明确已过期，仍可判 invalid；否则不能当 valid。
    if vt and vt < as_of:
        return Validity(INVALID, f"状态缺失但 valid_to={vt.isoformat()} 已过",
                        0.80, ("chunk_valid_to", "chunk_missing_status"))
    if vf and vf > as_of:
        return Validity(INVALID, f"状态缺失但 valid_from={vf.isoformat()} 尚未生效",
                        0.80, ("chunk_valid_from", "chunk_missing_status"))
    if vt:
        return Validity(UNCERTAIN, "状态缺失，日期未过期但无法确认是否现行",
                        0.50, ("chunk_valid_to", "chunk_missing_status"))
    return Validity(UNCERTAIN, "status 与失效日期均缺失，无法核实时效",
                    0.00, ("no_validity_evidence",))


def assess_all(clauses: list[Clause], as_of: date, family: Any = None) -> list[dict[str, Any]]:
    return [{"id": c.id, **assess_validity(c, as_of, family).as_dict()} for c in clauses]
