"""检索与评估之间传递的最小数据对象。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Clause:
    """一条可检索的政策条款。

    不同向量库的 metadata 字段并不统一：
      - 你的 Chroma 使用 valid_from / valid_to / topic / region
      - 本项目旧索引使用 effective_date / expiry_date / apply_region
    因此这里不假设字段名，统一由 temporal.py 做规范化读取。
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata or {})
        self.text = (self.text or "").strip()

    @property
    def policy_name(self) -> str:
        return str(self.metadata.get("policy_name") or self.metadata.get("title") or "").strip()

    @property
    def article(self) -> str:
        return str(self.metadata.get("article") or self.metadata.get("article_no") or "").strip()

    @property
    def region(self) -> str:
        return str(
            self.metadata.get("region")
            or self.metadata.get("apply_region")
            or self.metadata.get("jurisdiction")
            or ""
        ).strip()

    @property
    def topic(self) -> str:
        return str(
            self.metadata.get("topic")
            or self.metadata.get("category")
            or self.metadata.get("section")
            or ""
        ).strip()

    @property
    def label(self) -> str:
        """给 prompt 用的可读来源名，不包含有效期，避免两个分支提示不对称。"""
        name = self.policy_name or "未命名政策"
        return f"《{name}》{self.article}".strip()

    def preview(self, n: int = 180) -> str:
        return " ".join(self.text.split())[:n]
