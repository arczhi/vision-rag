"""
检索 Pipeline: 文本/图片 query → CLIP 编码 → Qdrant 粗排 → Rerank → 聚合结果
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from config import cfg
from ingest.embedding import CLIPEncoder

from .reranker import ClipResult, Reranker
from .qdrant_retriever import QdrantRetriever

logger = logging.getLogger(__name__)


@dataclass
class SearchTimings:
    """各阶段耗时 (毫秒)。"""
    embed_ms: float = 0.0
    coarse_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    coarse_count: int = 0  # 粗排候选数

    def as_dict(self) -> dict:
        return {
            "embed_ms": round(self.embed_ms, 2),
            "coarse_ms": round(self.coarse_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "coarse_count": self.coarse_count,
        }


@dataclass
class SearchOutcome:
    results: list[ClipResult] = field(default_factory=list)
    timings: SearchTimings = field(default_factory=SearchTimings)


def _now_ms() -> float:
    return time.perf_counter() * 1000


class SearchPipeline:
    def __init__(
        self,
        encoder: CLIPEncoder | None = None,
        retriever: QdrantRetriever | None = None,
        reranker: Reranker | None = None,
    ):
        self.encoder = encoder or CLIPEncoder()
        # 维度对齐: encoder 加载后用真实 dim 初始化 retriever
        if retriever is None:
            self.encoder._ensure_loaded()
            self.retriever = QdrantRetriever(embedding_dim=self.encoder.dim)
        else:
            self.retriever = retriever
        self.reranker = reranker or Reranker(encoder=self.encoder)

    def _default_coarse_k(self) -> int:
        return int(getattr(cfg.qdrant, "top_k", 60))

    # ---------- text → video ----------
    def search_text(
        self,
        query: str,
        top_k: int | None = None,
        coarse_k: int | None = None,
        rerank: bool = True,
        rerank_mode: str | None = None,
    ) -> SearchOutcome:
        """rerank_mode: 'merge' (默认, 仅片段合并) / 'internvl' (LLM 精排) / 'clip' (兼容)"""
        coarse_k = coarse_k or self._default_coarse_k()
        top_k = top_k or cfg.reranker.top_k
        t = SearchTimings()
        t0 = _now_ms()

        ts = _now_ms()
        q_vec = self.encoder.encode_texts([query])[0]
        t.embed_ms = _now_ms() - ts

        ts = _now_ms()
        if hasattr(self.retriever, "search_multi") and hasattr(self.retriever, "vector_names"):
            names = self.retriever.vector_names
            weights = {
                names.visual: float(getattr(cfg.qdrant, "visual_weight", 0.30)),
                names.caption: float(getattr(cfg.qdrant, "caption_weight", 0.45)),
                names.transcript: float(getattr(cfg.qdrant, "transcript_weight", 0.25)),
            }
            hits = self.retriever.search_multi(
                {
                    names.visual: q_vec,
                    names.caption: q_vec,
                    names.transcript: q_vec,
                },
                weights=weights,
                top_k=coarse_k,
                per_vector_k=coarse_k,
                query_text=query,
            )
        else:
            hits = self.retriever.search(q_vec, top_k=coarse_k)
        t.coarse_ms = _now_ms() - ts
        t.coarse_count = len(hits)

        if not hits:
            t.total_ms = _now_ms() - t0
            return SearchOutcome(results=[], timings=t)

        ts = _now_ms()
        if not rerank:
            results = [Reranker._hit_to_result(h, h.score) for h in hits[:top_k]]
        else:
            results = self.reranker.rerank(query, hits, top_k=top_k, mode=rerank_mode)
        t.rerank_ms = _now_ms() - ts
        t.total_ms = _now_ms() - t0
        return SearchOutcome(results=results, timings=t)

    # ---------- image → video ----------
    def search_image(
        self,
        image: str | Path | Image.Image | np.ndarray,
        top_k: int | None = None,
        coarse_k: int | None = None,
        rerank: bool = True,
        rerank_mode: str | None = None,
    ) -> SearchOutcome:
        coarse_k = coarse_k or self._default_coarse_k()
        top_k = top_k or cfg.reranker.top_k
        t = SearchTimings()
        t0 = _now_ms()

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        ts = _now_ms()
        q_vec = self.encoder.encode_images([image])[0]
        t.embed_ms = _now_ms() - ts

        ts = _now_ms()
        if hasattr(self.retriever, "vector_names"):
            hits = self.retriever.search(q_vec, top_k=coarse_k, vector_name=self.retriever.vector_names.visual)
        else:
            hits = self.retriever.search(q_vec, top_k=coarse_k)
        t.coarse_ms = _now_ms() - ts
        t.coarse_count = len(hits)

        if not hits:
            t.total_ms = _now_ms() - t0
            return SearchOutcome(results=[], timings=t)

        ts = _now_ms()
        if not rerank:
            results = [Reranker._hit_to_result(h, h.score) for h in hits[:top_k]]
        else:
            # 图片 query 不支持 InternVL (它需要文本), 自动降级到 merge
            mode_for_img = rerank_mode if rerank_mode != "internvl" else "merge"
            results = self.reranker.rerank(q_vec, hits, top_k=top_k, mode=mode_for_img)
        t.rerank_ms = _now_ms() - ts
        t.total_ms = _now_ms() - t0
        return SearchOutcome(results=results, timings=t)
