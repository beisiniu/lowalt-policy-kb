"""LangGraph 编排：把"检索 → 时效门禁 → 生成/拒答"写成一张状态图。

为什么这里值得用 LangGraph，而不是一个 if-else 函数？

    因为这条链路有**分支**和**回路**，而且分支条件本身是研究对象：

        parse ──> retrieve ──> gate ──┬─(命中且有效)──> answer ──> END
                                       ├─(全失效)──────> refuse  ──> END
                                       └─(没命中)──> expand ──> retrieve(第二次)

    - `gate` 是"时效门禁"：拦掉"字面最像但已经废止"的文件，
      拦截次数本身就是论文里可以量化的指标（stale_blocked）。
    - `expand → retrieve` 是一次真实的重试回路（放宽地域过滤再检一次）。
    - 每个节点只改自己那一块 state，加一个节点不用动其他代码。

    纯人工 if-else 也能写，但状态会散在十几个局部变量里，图和实验记录对不上。
"""
from __future__ import annotations

import operator
from typing import Annotated, Callable, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from .answer import extractive_answer, llm_answer, refusal_answer
from .retriever import PolicyRetriever
from .schema import REGIONS, detect_region


class QAState(TypedDict, total=False):
    question: str
    region: str
    query: str
    attempts: int
    hits: list[Document]
    stale_hits: list[Document]
    route: str
    answer: str
    steps: Annotated[list[str], operator.add]


class PolicyQA:
    """低空政策问答图。

    mode:
        extractive —— 不调模型，直接摘条文（无 Key 可跑，用作检索质量对照组）
        llm        —— 调大模型生成（需要 DEEPSEEK_API_KEY）
    """

    def __init__(
        self,
        retriever: PolicyRetriever,
        mode: str = "extractive",
        client=None,
        top_k: int = 4,
        max_expand: int = 1,
    ):
        if mode == "llm" and client is None:
            raise ValueError("mode='llm' 需要传入 ChatClient")
        self.retriever = retriever
        self.mode = mode
        self.client = client
        self.top_k = top_k
        self.max_expand = max_expand
        # 统一成 (question, hits, region) 三参签名，避免和 extractive_answer 的 max_cite 混掉
        self._answer_fn: Callable[[str, list[Document], str], str] = (
            (lambda q, h, r: extractive_answer(q, h))
            if mode == "extractive"
            else (lambda q, h, r: llm_answer(client, q, h, r))
        )
        self.graph = self._build()

    # ------------------------------------------------------------------
    def _build(self):
        g = StateGraph(QAState)

        g.add_node("parse", self._parse)
        g.add_node("retrieve", self._retrieve)
        g.add_node("gate", self._gate)
        g.add_node("expand", self._expand)
        g.add_node("answer", self._answer)
        g.add_node("refuse", self._refuse)

        g.set_entry_point("parse")
        g.add_edge("parse", "retrieve")
        g.add_edge("retrieve", "gate")
        g.add_conditional_edges(
            "gate",
            lambda s: s.get("route", "answer"),
            {"answer": "answer", "refuse": "refuse", "refuse_empty": "refuse", "expand": "expand"},
        )
        g.add_edge("expand", "retrieve")
        g.add_edge("answer", END)
        g.add_edge("refuse", END)
        return g.compile()

    # ------------------------------------------------------------------
    def _parse(self, state: QAState) -> dict:
        q = state["question"]
        return {"region": detect_region(q), "query": q, "attempts": 0, "steps": ["parse"]}

    def _retrieve(self, state: QAState) -> dict:
        hits = self.retriever.retrieve(state.get("query") or state["question"], k=self.top_k,
                                       region=state.get("region", ""))
        return {"hits": hits, "steps": [f"retrieve:{len(hits)}"]}

    def _gate(self, state: QAState) -> dict:
        """时效门禁：有可用条文 → 答；只有失效条文 → 明确拒答；全无 → 放宽再检一次。"""
        hits = state.get("hits") or []
        if hits:
            return {"route": "answer", "steps": ["gate>answer"]}

        stale = self.retriever.stale_probe(state.get("query") or state["question"],
                                           k=self.top_k, region=state.get("region", ""))
        # 只有"确实召回到了但不合规"才算时效拦截；
        # 纯相关度不足（召回为空或全部可用）应走放宽容错，不能谎报"政策已失效"
        blocked = [d for d in stale if not d.metadata.get("_usable", True)]
        if blocked:
            return {"stale_hits": blocked, "route": "refuse", "steps": ["gate>refuse(stale)"]}

        if state.get("attempts", 0) < self.max_expand:
            return {"route": "expand", "steps": ["gate>expand"]}

        return {"route": "refuse_empty", "steps": ["gate>refuse(empty)"]}

    def _expand(self, state: QAState) -> dict:
        """放宽**词面**约束：删掉问题中的地域词，但保留地域元数据过滤。

        地域词会占 BM25 权重，偶尔让真正相关的条款排不上来；这时可以删掉它再检一次。
        不能连 region 过滤也删掉，否则问东莞补贴时可能拿珠海条款回答。
        简称（中山/惠州）也要一并删除，不能只删「中山市/惠州市」。
        """
        q = state.get("query") or state["question"]
        for r in sorted(REGIONS, key=len, reverse=True):
            q = q.replace(r, "")
            short = r[:-1] if r.endswith(("市", "省")) else ""
            if len(short) >= 2:
                q = q.replace(short, "")
        return {"query": q, "attempts": state.get("attempts", 0) + 1, "steps": ["expand"]}

    def _answer(self, state: QAState) -> dict:
        ans = self._answer_fn(state["question"], state.get("hits") or [], state.get("region", ""))
        return {"answer": ans, "steps": ["answer"]}

    def _refuse(self, state: QAState) -> dict:
        ans = refusal_answer(state["question"], state.get("stale_hits") or [], state.get("region", ""))
        return {"answer": ans, "steps": ["refuse"]}

    # ------------------------------------------------------------------
    def ask(self, question: str) -> dict:
        """跑一遍图，返回答案 + 可解释的中间状态。"""
        out = self.graph.invoke({"question": question, "steps": []})
        hits = out.get("hits") or []
        return {
            "answer": out.get("answer", ""),
            "route": out.get("route", ""),
            "steps": out.get("steps", []),
            "region": out.get("region", ""),
            "cited": [
                {
                    "policy_name": d.metadata.get("policy_name"),
                    "article": d.metadata.get("article"),
                    "effective_date": d.metadata.get("effective_date"),
                    "expiry_date": d.metadata.get("expiry_date"),
                    "status": d.metadata.get("status"),
                    "apply_region": d.metadata.get("apply_region"),
                    "score": d.metadata.get("_score"),
                }
                for d in hits
            ],
            "blocked_stale": [d.metadata.get("policy_name") for d in (out.get("stale_hits") or [])],
        }
