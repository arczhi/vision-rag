"""
高光标注 (Highlight Annotation) 模块 - P0 实现

数据模型:
  - KB (Knowledge Base): 每客户一个独立知识库, kb_id 是逻辑隔离单位
  - HighlightSample: KB 内的样例片段, 带 label/向量/时间戳
  - Annotation: 对一个视频跑出来的标注结果 (label, start, end, score, 触发样例)

知识库样例写入 Qdrant collection: highlight_example_v1
"""
from .knowledge_base import HighlightSample, KnowledgeBase, KBRetriever
from .annotator import Annotation, Annotator

__all__ = [
    "HighlightSample", "KnowledgeBase", "KBRetriever",
    "Annotation", "Annotator",
]
