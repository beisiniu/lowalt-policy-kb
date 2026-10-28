"""实验组：入口即 Temporal_Check 的 LangGraph。

关键点（也是把外部 Top-5 强行塞给图的做法）：
    graph.invoke({
        "question": question,
        "as_of": "2026-10-26",
        "retrieved_docs": top5,     # 外部检索结果直接作为初始 State
        "steps": [],
    })

图里**没有 retrieve 节点**，因此不会在评估时重新检索，保证两组用完全相同的 Top-5。
Temporal_Check 只读每条条款的 valid_from / valid_to / status，不读 benchmark 标签。
"""
from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from .models import Clause
from .temporal import clause_valid_from, clause_valid_to, iso_or_none, parse_date, usable
from .validity import INVALID, UNCERTAIN, VALID, assess_validity

ANSWER_SYSTEM = """你是低空政策问答助手。只能依据给定的资料回答，不得使用资料之外的知识。
要求：
1. 直接回答所问事实或审批结论，数字、单位、适用条件必须完整。
2. 引用时写明《政策名》和条款号（没有条号则只写政策名）。
3. 若资料不足以回答，明确说“根据现有政策资料无法确定”，不要猜。
"""


class TemporalState(TypedDict, total=False):
    question: str
    as_of: str
    retrieved_docs: list[Clause]
    filtered_docs: list[Clause]
    dropped: list[dict[str, Any]]
    uncertain_docs: list[dict[str, Any]]
    route: str
    answer: str
    steps: Annotated[list[str], operator.add]


def format_context(docs: list[Clause]) -> str:
    """两个分支共用的上下文格式：只给来源与正文，不提示条款是否过期。

    如果在实验组 prompt 里加“请忽略过期政策”，就无法区分提升来自
    Temporal_Check 架构还是来自一句更强的提示词。
    """
    lines: list[str] = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d.label}\n{d.text}")
    return "\n\n".join(lines)


def build_answer_user_prompt(question: str, docs: list[Clause]) -> str:
    return f"【资料】\n{format_context(docs)}\n\n【问题】\n{question}"


def generate_answer(client: Any, question: str, docs: list[Clause],
                    *, thinking: bool = False, max_tokens: int = 0) -> str:
    if not docs:
        return "根据现有政策资料无法确定。"
    # thinking 会占用 completion token；预算太小时会出现 content=""，
    # 这种基础设施失败不能被裁判当成“拒答”计入正确/毒化率。
    budget = max_tokens or (2400 if thinking else 1200)
    ans = client.chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": build_answer_user_prompt(question, docs)},
        ],
        temperature=0.0,
        max_tokens=budget,
        json_mode=False,
        thinking=thinking,
    )
    if not (ans or "").strip():
        raise RuntimeError("模型返回空答案（可能是 thinking token 预算耗尽），请调大 max_tokens")
    return ans


class TemporalRAGGraph:
    """只做过滤与生成的时序 RAG 图。"""

    def __init__(self, client: Any = None, *, thinking: bool = False,
                 mode: str = "filter", family: Any = None):
        """
        mode="filter"       过期条款直接剔除（原主实验，行为不变）
        mode="downweight"   过期条款不剔除，只排到上下文末尾（原消融）
        mode="uncertainty"  三态判定：明确失效剔除，无法核实单独标记并拒答
        family              可选的政策族谱，为缺失字段提供政策级证据
        """
        if mode not in ("filter", "downweight", "uncertainty"):
            raise ValueError(f"未知 Temporal 模式：{mode}")
        self.client = client
        self.thinking = thinking
        self.mode = mode
        self.family = family
        self.graph = self._build()

    def _build(self):
        g = StateGraph(TemporalState)
        g.add_node("temporal_check", self._temporal_check)
        g.add_node("generate", self._generate)
        g.add_node("refuse", self._refuse)
        # 入口不是 retrieve，而是 temporal_check：外部检索结果已放在 State 里
        g.set_entry_point("temporal_check")
        g.add_conditional_edges(
            "temporal_check",
            lambda s: s.get("route", "refuse"),
            {"generate": "generate", "refuse": "refuse"},
        )
        g.add_edge("generate", END)
        g.add_edge("refuse", END)
        return g.compile()

    def _temporal_check(self, state: TemporalState) -> dict:
        as_of = parse_date(state.get("as_of")) or date.today()
        keep: list[Clause] = []
        stale: list[Clause] = []
        uncertain: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        for d in state.get("retrieved_docs") or []:
            if self.mode == "uncertainty":
                v = assess_validity(d, as_of, self.family)
                if v.state == VALID:
                    keep.append(d)
                    continue
                record = {
                    "id": d.id,
                    "policy_name": d.policy_name,
                    "article": d.article,
                    "reason": v.reason,
                    "state": v.state,
                    "confidence": v.confidence,
                    "basis": list(v.basis),
                    "valid_from": iso_or_none(clause_valid_from(d)),
                    "valid_to": iso_or_none(clause_valid_to(d)),
                }
                dropped.append(record)
                if v.state == UNCERTAIN:
                    uncertain.append(record)
                else:
                    stale.append(d)
                continue

            ok, reason = usable(d, as_of)
            if ok:
                keep.append(d)
            else:
                stale.append(d)
                dropped.append({
                    "id": d.id,
                    "policy_name": d.policy_name,
                    "article": d.article,
                    "reason": reason,
                    "valid_from": iso_or_none(clause_valid_from(d)),
                    "valid_to": iso_or_none(clause_valid_to(d)),
                })
        if self.mode == "downweight":
            # 消融：旧条款保留在上下文里（排在有效条款之后），不剔除
            keep = keep + stale
        route = "generate" if keep else "refuse"
        tag = {"filter": "filter", "downweight": "downweight",
               "uncertainty": "uncertainty"}[self.mode]
        return {
            "filtered_docs": keep,
            "dropped": dropped,
            "uncertain_docs": uncertain,
            "route": route,
            "steps": [f"temporal_check[{tag}]:keep={len(keep)},drop={len(dropped)},"
                      f"uncertain={len(uncertain)}"],
        }

    def _generate(self, state: TemporalState) -> dict:
        if self.client is None:
            raise RuntimeError("TemporalRAGGraph 需要 ChatClient 才能生成答案")
        ans = generate_answer(
            self.client,
            state["question"],
            state.get("filtered_docs") or [],
            thinking=self.thinking,
        )
        return {"answer": ans, "steps": ["generate"]}

    def _refuse(self, state: TemporalState) -> dict:
        n = len(state.get("retrieved_docs") or [])
        dropped = len(state.get("dropped") or [])
        uncertain = len(state.get("uncertain_docs") or [])
        if uncertain:
            ans = (
                f"检索到的 {n} 条政策依据中，有 {uncertain} 条的时效无法核实"
                f"（评估时点 {state.get('as_of')}）；这些条款不能作为确定依据，"
                "知识库中也没有可用的确定性现行条款，无法给出确定结论。"
                "建议以主管部门最新公布的文件为准。"
            )
        elif dropped:
            ans = (
                f"检索到的 {n} 条政策依据在当前时点（{state.get('as_of')}）均已失效，"
                "不能作为现行依据；知识库中没有可用的现行条款，无法给出确定结论。"
            )
        else:
            ans = "知识库中没有检索到与该问题相关的政策条款，无法给出确定结论。"
        return {"answer": ans, "steps": ["refuse"]}

    def ask(self, question: str, retrieved_docs: list[Clause],
            as_of: str | date) -> dict[str, Any]:
        """用外部 Top-5 作为初始 State 运行图。"""
        as_of_s = as_of.isoformat() if isinstance(as_of, date) else str(as_of)
        out = self.graph.invoke({
            "question": question,
            "as_of": as_of_s,
            "retrieved_docs": retrieved_docs,
            "steps": [],
        })
        return {
            "answer": out.get("answer", ""),
            "route": out.get("route", ""),
            "steps": out.get("steps", []),
            "filtered_docs": out.get("filtered_docs") or [],
            "dropped": out.get("dropped") or [],
            "uncertain_docs": out.get("uncertain_docs") or [],
        }
