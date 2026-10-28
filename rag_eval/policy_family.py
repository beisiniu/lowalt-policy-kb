"""政策版本族谱与同源扩展。

用途：
  1. 给出旧政策 -> 现行继任政策的显式关系；
  2. 当检索命中旧政策时，按政策身份把继任政策中的对应条款补进候选；
  3. 为不确定性门禁提供政策级 status/valid_to 证据。

不依赖向量相似度来决定新旧关系；相似度只用于在继任政策内部挑选与问题最相关的条款。
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

from .models import Clause
from .temporal import (bigram_jaccard, clause_status, clause_valid_from, clause_valid_to,
                       normalize_for_match)

MANUAL_EDGES = (
    ("广州开发区（黄埔区）促进低空经济高质量发展的若干措施实施细则",
     "广州开发区（黄埔区）促进低空经济与航空航天高质量发展的若干措施"),
    ("广州开发区（黄埔区）促进低空经济高质量发展的若干措施",
     "广州开发区（黄埔区）促进低空经济与航空航天高质量发展的若干措施"),
    ("龙岗区关于促进低空经济产业发展的若干措施",
     "深圳市龙岗区关于支持低空经济产业高质量发展的若干措施（修订）"),
    ("张家界市加快通用航空产业发展推进方案",
     "张家界市人民政府关于废止张政办发〔2023〕4号文件部分内容的通告"),
)
NEGATIVE = ("废止", "失效", "无效", "停止执行", "暂停", "撤销")


def _norm_name(value: str) -> str:
    s = unicodedata.normalize("NFKC", value or "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[（(][^）)]*[）)]", "", s)
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", s)
    return s


def _policy_id(clause: Clause) -> str:
    return str(clause.metadata.get("policy_id") or "")


def _overlap(a: str, b: str) -> float:
    """a 的二元词覆盖率；比 Jaccard 更适合“旧条款的要点是否出现在新条款里”。"""
    x = normalize_for_match(a)
    y = normalize_for_match(b)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    gx = {x[i:i + 2] for i in range(len(x) - 1)}
    gy = {y[i:i + 2] for i in range(len(y) - 1)}
    return len(gx & gy) / max(1, len(gx))


@dataclass
class PolicyNode:
    policy_id: str
    name: str
    region: str
    status: str
    valid_from: Optional[date]
    valid_to: Optional[date]
    source_url: str
    clauses: list[Clause] = field(default_factory=list)


@dataclass(frozen=True)
class FamilyEdge:
    old_id: str
    new_id: str
    relation: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class Expansion:
    clause: Clause
    old_policy_id: str
    new_policy_id: str
    relation: str
    score: float


class PolicyFamily:
    def __init__(self, nodes: dict[str, PolicyNode], edges: Iterable[FamilyEdge]):
        self.nodes = nodes
        self.edges: list[FamilyEdge] = []
        self._succ: dict[str, list[FamilyEdge]] = {}
        seen = set()
        for edge in edges:
            key = (edge.old_id, edge.new_id)
            if key in seen or edge.old_id not in nodes or edge.new_id not in nodes:
                continue
            seen.add(key)
            self.edges.append(edge)
            self._succ.setdefault(edge.old_id, []).append(edge)

    def node(self, policy_id: str) -> Optional[PolicyNode]:
        return self.nodes.get(policy_id)

    def successors(self, policy_id: str) -> list[FamilyEdge]:
        return list(self._succ.get(policy_id, []))

    def clauses_of(self, policy_id: str) -> list[Clause]:
        node = self.nodes.get(policy_id)
        return list(node.clauses) if node else []

    def expand(self, docs: list[Clause], question: str, as_of: date,
               max_extra: int = 2) -> list[Expansion]:
        """只对命中的旧政策扩展；返回按问题相关性排序的继任条款。"""
        if max_extra <= 0:
            return []
        base_ids = {d.id for d in docs}
        old_ids = {
            _policy_id(d) for d in docs
            if _policy_id(d) and self.is_old(_policy_id(d), as_of)
        }
        old_topics: dict[str, set[str]] = {}
        old_texts: dict[str, list[str]] = {}
        for d in docs:
            pid = _policy_id(d)
            if pid in old_ids:
                if d.topic:
                    old_topics.setdefault(pid, set()).add(d.topic)
                old_texts.setdefault(pid, []).append(d.text)
        candidates: list[Expansion] = []
        for old_id in sorted(old_ids):
            for edge in self.successors(old_id):
                for c in self.clauses_of(edge.new_id):
                    if c.id in base_ids:
                        continue
                    # 旧条款到继任条款的文本对应关系比“问题到新条款”更可靠：
                    # 新旧版本常只改数字/条件，字面重合能定位到对应条款。
                    old_score = max((_overlap(t, c.text) for t in old_texts.get(old_id, [])),
                                    default=0.0)
                    q_score = _overlap(question, c.text)
                    score = 0.5 * old_score + 0.5 * q_score
                    if c.topic and c.topic in old_topics.get(old_id, set()):
                        score += 0.02
                    candidates.append(Expansion(c, old_id, edge.new_id, edge.relation, round(score, 4)))
        candidates.sort(key=lambda x: (x.score, len(x.clause.text)), reverse=True)
        picked: list[Expansion] = []
        used: set[str] = set()
        for x in candidates:
            if x.clause.id in used:
                continue
            used.add(x.clause.id)
            picked.append(x)
            if len(picked) >= max_extra:
                break
        return picked

    def is_old(self, policy_id: str, as_of: date) -> bool:
        node = self.nodes.get(policy_id)
        if node is None:
            return False
        if any(k in (node.status or "") for k in NEGATIVE):
            return True
        return bool(node.valid_to and node.valid_to < as_of)

    def summary(self) -> dict[str, Any]:
        return {
            "policies": len(self.nodes),
            "edges": len(self.edges),
            "edge_list": [
                {"old": self.nodes[e.old_id].name, "new": self.nodes[e.new_id].name,
                 "relation": e.relation, "confidence": e.confidence}
                for e in self.edges
            ],
        }


def _match_name(name: str, nodes: dict[str, PolicyNode], exclude: str = "") -> Optional[str]:
    target = _norm_name(name)
    if not target:
        return None
    exact = [pid for pid, n in nodes.items() if pid != exclude and _norm_name(n.name) == target]
    if len(exact) == 1:
        return exact[0]
    scored: list[tuple[float, str]] = []
    for pid, n in nodes.items():
        if pid == exclude:
            continue
        norm = _norm_name(n.name)
        if not norm:
            continue
        if norm in target or target in norm:
            score = 1.0
        else:
            score = bigram_jaccard(norm, target)
        if score >= 0.35:
            scored.append((score, pid))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def build_policy_family(clauses: list[Clause],
                        overrides: Optional[dict[str, Any]] = None,
                        *, manual: bool = True) -> PolicyFamily:
    groups: dict[str, list[Clause]] = {}
    for c in clauses:
        pid = _policy_id(c)
        if pid:
            groups.setdefault(pid, []).append(c)
    nodes: dict[str, PolicyNode] = {}
    for pid, group in groups.items():
        c = group[0]
        nodes[pid] = PolicyNode(
            policy_id=pid,
            name=c.policy_name,
            region=c.region,
            status=clause_status(c),
            valid_from=clause_valid_from(c),
            valid_to=clause_valid_to(c),
            source_url=str(c.metadata.get("source_url") or ""),
            clauses=group,
        )
    edges: list[FamilyEdge] = []
    if manual:
        for old_name, new_name in MANUAL_EDGES:
            old_id = _match_name(old_name, nodes)
            new_id = _match_name(new_name, nodes, exclude=old_id or "")
            if old_id and new_id:
                edges.append(FamilyEdge(old_id, new_id, "manual", "项目内已知的新旧对照", 0.99))
    if overrides:
        by_url = {n.source_url: pid for pid, n in nodes.items() if n.source_url}
        for url, item in overrides.items():
            old_id = by_url.get(url)
            if not old_id:
                continue
            text = str(item.get("superseded_by") or "").strip()
            if not text or "无后续文件" in text or "修订程序启动中" in text:
                continue
            new_id = _match_name(text, nodes, exclude=old_id)
            if new_id:
                edges.append(FamilyEdge(
                    old_id, new_id, "superseded_by",
                    str(item.get("evidence") or text)[:160], 0.90,
                ))
    return PolicyFamily(nodes, edges)


def load_overrides(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
