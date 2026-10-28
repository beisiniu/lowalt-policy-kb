"""时序 RAG 对抗基准与双分支评估。

这个包把“检索后端”和“实验逻辑”解耦：
  - 检索后端只负责返回带时间元数据的条款（Chroma / 本地索引可替换）
  - 时序图只负责按 valid_from / valid_to 过滤，不读取 benchmark 标签
  - 评估脚本负责公平对比与统计，不参与检索或过滤
"""
from .models import Clause
from .temporal_graph import TemporalRAGGraph

__all__ = ["Clause", "TemporalRAGGraph"]
