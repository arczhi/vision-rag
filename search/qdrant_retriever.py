"""
Qdrant 多模态视频片段主索引.

一个 point = 一个 clip 实体:
  payload: video_id / clip_index / 时间戳 / 缩略图 / Qwen2-VL 剧情描述 / ASR 台词
  vectors:
    highlight_visual_*      : Chinese-CLIP 图像向量
    highlight_caption_*     : Qwen2-VL 剧情描述文本向量
    highlight_transcript_*  : ASR/字幕文本向量

这里是视频片段主索引实现。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from config import cfg
from .types import ClipHit

logger = logging.getLogger(__name__)

_LEXICAL_LANE = "__lexical__"
_HIGHLIGHT_POSITIVE_TAGS = {
    "反转", "冲突", "打脸", "危机", "悬念", "强情绪", "攻击", "打斗", "爆炸", "追杀", "觉醒",
}
_HIGHLIGHT_NEGATIVE_HINTS = (
    "不是高光", "不适合作为高光", "普通画面", "普通的画面", "普通场景", "普通对话",
    "没有明显", "没有明确", "没有提供足够", "信息不足",
)
_ORDINARY_TAG_HINTS = ("普通", "无")


def _slug(value: str, max_len: int = 96) -> str:
    raw = str(value or "unknown")
    out = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    out = re.sub(r"_+", "_", out) or "unknown"
    if len(out) <= max_len:
        return out
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{out[:max_len - 9].rstrip('_')}_{digest}"


def _distance(name: str):
    from qdrant_client import models

    t = str(name or "COSINE").upper()
    if t in {"IP", "DOT", "INNER_PRODUCT"}:
        return models.Distance.DOT
    if t == "L2":
        return models.Distance.EUCLID
    return models.Distance.COSINE


@dataclass(frozen=True)
class QdrantVectorNames:
    visual: str
    caption: str
    transcript: str

    def as_dict(self) -> dict[str, str]:
        return {
            "visual": self.visual,
            "caption": self.caption,
            "transcript": self.transcript,
        }

    def all(self) -> list[str]:
        return [self.visual, self.caption, self.transcript]


def build_qdrant_vector_names(
    embedding_dim: int,
    *,
    embedding_cfg=cfg.embedding,
    qdrant_cfg=cfg.qdrant,
) -> QdrantVectorNames:
    """用业务 + 模态 + 模型名 + 参数构造 named-vector 名称.

    例:
      highlight_visual_cn_clip_OFA_Sys_chinese_clip_vit_large_patch14_336px_fp32_norm_768d
    后续换 BGE/SigLIP/更大 CN-CLIP 时会自然生成新名字, 旧向量仍留在 collection 中.
    """
    business = _slug(qdrant_cfg.business, max_len=32)
    backend = _slug(str(getattr(embedding_cfg, "backend", "embedding")), max_len=32)
    model = _slug(str(getattr(embedding_cfg, "model_name", "model")), max_len=80)
    precision = _slug(str(getattr(embedding_cfg, "precision", "fp32")), max_len=24)
    norm = "norm" if bool(getattr(embedding_cfg, "normalize", True)) else "raw"
    suffix = f"{backend}_{model}_{precision}_{norm}_{int(embedding_dim)}d"
    return QdrantVectorNames(
        visual=f"{business}_visual_{suffix}",
        caption=f"{business}_caption_{suffix}",
        transcript=f"{business}_transcript_{suffix}",
    )


@dataclass
class QdrantClipRecord:
    """待写入 Qdrant 的一个 clip 多模态实体."""
    visual_vec: np.ndarray
    caption_vec: np.ndarray
    transcript_vec: np.ndarray
    video_id: str
    video_path: str
    clip_index: int
    start_time: float
    end_time: float
    frame_index: int = 0
    timestamp: float = 0.0
    thumbnail: str = ""
    caption_text: str = ""
    transcript_text: str = ""
    transcript_source: str = ""
    understanding: dict = field(default_factory=dict)


class QdrantRetriever:
    supports_multimodal_vectors = True

    def __init__(
        self,
        qdrant_cfg=cfg.qdrant,
        embedding_dim: int = cfg.embedding.embedding_dim,
        vector_names: QdrantVectorNames | None = None,
    ):
        self.cfg = qdrant_cfg
        self.dim = int(embedding_dim)
        self.vector_names = vector_names or build_qdrant_vector_names(self.dim)
        self.vector_sizes = {name: self.dim for name in self.vector_names.all()}
        self._client = None
        self._ensured = False

    # ---------- connection / schema ----------
    def connect(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient

        kwargs = {
            "host": self.cfg.host,
            "port": self.cfg.port,
            "grpc_port": self.cfg.grpc_port,
            "prefer_grpc": self.cfg.prefer_grpc,
            "https": self.cfg.https,
            "timeout": self.cfg.timeout,
        }
        if self.cfg.api_key:
            kwargs["api_key"] = self.cfg.api_key
        self._client = QdrantClient(**kwargs)
        logger.info("Connected to Qdrant at %s:%s", self.cfg.host, self.cfg.port)
        return self._client

    def close(self):
        client = self._client
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass
        self._client = None
        self._ensured = False

    def ensure_collection(self):
        if self._ensured:
            return self
        from qdrant_client import models

        client = self.connect()
        vectors_config = {
            name: models.VectorParams(size=size, distance=_distance(self.cfg.metric_type))
            for name, size in self.vector_sizes.items()
        }
        if not client.collection_exists(self.cfg.collection_name):
            client.create_collection(
                collection_name=self.cfg.collection_name,
                vectors_config=vectors_config,
                on_disk_payload=True,
                timeout=int(self.cfg.timeout),
            )
            logger.info("Created Qdrant collection %s with vectors=%s", self.cfg.collection_name, list(vectors_config))
        else:
            self._ensure_named_vectors(client)
        self._ensure_payload_indexes(client)
        self._ensured = True
        return self

    def _ensure_named_vectors(self, client):
        from qdrant_client import models

        info = client.get_collection(self.cfg.collection_name)
        current = getattr(info.config.params, "vectors", None)
        if not isinstance(current, dict):
            raise RuntimeError(
                f"Qdrant collection '{self.cfg.collection_name}' is not a named-vector collection. "
                "Please use a separate QDRANT_COLLECTION or migrate the old collection."
            )
        for name, size in self.vector_sizes.items():
            if name in current:
                got = int(getattr(current[name], "size", 0) or 0)
                if got and got != int(size):
                    raise RuntimeError(
                        f"Qdrant vector '{name}' exists with dim={got}, expected dim={size}. "
                        "Vector names include model parameters; change QDRANT_VECTOR_BUSINESS or model config "
                        "if you intentionally want a new vector lane."
                    )
                continue
            logger.info("Adding Qdrant named vector %s dim=%s", name, size)
            client.create_vector_name(
                collection_name=self.cfg.collection_name,
                vector_name=name,
                vector_name_config=models.DenseVectorNameConfig(
                    dense=models.DenseVectorConfig(size=int(size), distance=_distance(self.cfg.metric_type))
                ),
                wait=True,
                timeout=int(self.cfg.timeout),
            )

    def _ensure_payload_indexes(self, client):
        from qdrant_client import models

        for field_name, schema in [
            ("video_id", models.PayloadSchemaType.KEYWORD),
            ("clip_index", models.PayloadSchemaType.INTEGER),
        ]:
            try:
                client.create_payload_index(
                    collection_name=self.cfg.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                    timeout=int(self.cfg.timeout),
                )
            except Exception as e:
                # Already exists is fine; Qdrant returns an error for duplicates on some versions.
                logger.debug("create payload index skipped for %s: %s", field_name, e)

    def drop_collection(self):
        client = self.connect()
        if client.collection_exists(self.cfg.collection_name):
            client.delete_collection(self.cfg.collection_name, timeout=int(self.cfg.timeout))
        self._ensured = False

    # ---------- payload / ids ----------
    def _point_id(self, video_id: str, clip_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vision-rag:{self.cfg.collection_name}:{video_id}:{int(clip_index)}"))

    @staticmethod
    def _float_list(vec: np.ndarray) -> list[float]:
        arr = np.asarray(vec, dtype=np.float32)
        return arr.astype(np.float32).tolist()

    def _filter_video(self, video_id: str):
        from qdrant_client import models

        return models.Filter(
            must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=str(video_id)))]
        )

    def _filter_videos(self, video_ids: Iterable[str] | None):
        from qdrant_client import models

        ids = [str(v) for v in (video_ids or []) if str(v)]
        if not ids:
            return None
        if len(ids) == 1:
            return self._filter_video(ids[0])
        return models.Filter(
            should=[
                models.FieldCondition(key="video_id", match=models.MatchValue(value=v))
                for v in ids
            ]
        )

    @staticmethod
    def _clean_candidate_text(text: str) -> str:
        """清理自动 KB 候选描述里的否定模板, 尽量只保留画面主体。"""
        cleaned_parts: list[str] = []
        for raw in re.split(r"[\n\r]+", str(text or "")):
            part = raw.strip()
            if not part or part.startswith("标签:"):
                continue
            for marker in ("，没有", "。没有", ",没有", ".没有", "没有明显", "没有明确", "没有提供足够"):
                if marker in part:
                    part = part.split(marker, 1)[0].strip(" ，。,.")
            if any(h in part for h in _HIGHLIGHT_NEGATIVE_HINTS):
                continue
            if part:
                cleaned_parts.append(part)
        return "\n".join(cleaned_parts)

    # ---------- insert ----------
    def insert(self, records: list[QdrantClipRecord]) -> list[str]:
        if not records:
            return []
        from qdrant_client import models

        self.ensure_collection()
        points = []
        ids: list[str] = []
        now = time.time()
        names = self.vector_names
        for r in records:
            pid = self._point_id(r.video_id, r.clip_index)
            ids.append(pid)
            understanding = dict(r.understanding or {})
            payload = {
                "point_id": pid,
                "video_id": r.video_id,
                "video_path": r.video_path,
                "video_name": r.video_path.split("/")[-1],
                "clip_index": int(r.clip_index),
                "start_time": float(r.start_time),
                "end_time": float(r.end_time),
                "frame_index": int(r.frame_index),
                "timestamp": float(r.timestamp),
                "thumbnail": r.thumbnail or "",
                "caption_text": r.caption_text or "",
                "transcript_text": r.transcript_text or "",
                "transcript_source": r.transcript_source or "",
                "has_transcript": bool((r.transcript_text or "").strip()),
                "understanding": understanding,
                "qwen_is_highlight": understanding.get("is_highlight"),
                "qwen_score": understanding.get("score"),
                "qwen_tags": understanding.get("tags") or [],
                "qwen_reason": understanding.get("reason") or "",
                "qwen_cut_advice": understanding.get("cut_advice") or "",
                "qwen_caption": understanding.get("caption") or r.caption_text or "",
                "vector_names": names.as_dict(),
                "created_at": now,
            }
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        names.visual: self._float_list(r.visual_vec),
                        names.caption: self._float_list(r.caption_vec),
                        names.transcript: self._float_list(r.transcript_vec),
                    },
                    payload=payload,
                )
            )
        self.connect().upsert(self.cfg.collection_name, points=points, wait=True, timeout=int(self.cfg.timeout))
        return ids

    # ---------- metadata ----------
    def has_video(self, video_id: str) -> bool:
        self.ensure_collection()
        rows, _ = self.connect().scroll(
            collection_name=self.cfg.collection_name,
            scroll_filter=self._filter_video(video_id),
            limit=1,
            with_payload=["video_id"],
            with_vectors=False,
        )
        return bool(rows)

    def delete_video(self, video_id: str):
        from qdrant_client import models

        self.ensure_collection()
        self.connect().delete(
            collection_name=self.cfg.collection_name,
            points_selector=models.FilterSelector(filter=self._filter_video(video_id)),
            wait=True,
            timeout=int(self.cfg.timeout),
        )

    def list_videos(self) -> list[dict]:
        self.ensure_collection()
        client = self.connect()
        offset = None
        seen: dict[str, str] = {}
        while True:
            rows, offset = client.scroll(
                collection_name=self.cfg.collection_name,
                limit=512,
                offset=offset,
                with_payload=["video_id", "video_path"],
                with_vectors=False,
            )
            for row in rows:
                p = row.payload or {}
                vid = str(p.get("video_id") or "")
                if vid:
                    seen.setdefault(vid, str(p.get("video_path") or ""))
            if offset is None:
                break
        return [{"video_id": vid, "video_path": p} for vid, p in seen.items()]

    def list_clips(self, video_id: str) -> list[dict]:
        self.ensure_collection()
        client = self.connect()
        rows_out: list[dict] = []
        offset = None
        fields = [
            "point_id", "video_id", "video_path", "video_name", "clip_index", "start_time", "end_time",
            "frame_index", "timestamp", "thumbnail", "caption_text", "transcript_text",
            "transcript_source", "has_transcript", "understanding", "qwen_caption",
            "qwen_reason", "qwen_tags", "qwen_cut_advice", "qwen_score", "qwen_is_highlight",
        ]
        while True:
            rows, offset = client.scroll(
                collection_name=self.cfg.collection_name,
                scroll_filter=self._filter_video(video_id),
                limit=512,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            for row in rows:
                payload = dict(row.payload or {})
                payload["point_id"] = str(row.id)
                rows_out.append(payload)
            if offset is None:
                break
        rows_out.sort(key=lambda r: int(r.get("clip_index") or 0))
        return rows_out

    def get_video_path(self, video_id: str) -> str | None:
        self.ensure_collection()
        rows, _ = self.connect().scroll(
            collection_name=self.cfg.collection_name,
            scroll_filter=self._filter_video(video_id),
            limit=1,
            with_payload=["video_path"],
            with_vectors=False,
        )
        if not rows:
            return None
        return str((rows[0].payload or {}).get("video_path") or "") or None

    def get_clip_visual_rows(self, video_id: str) -> list[dict]:
        """返回某视频所有 clip 的视觉向量, 供 auto-kb 从主索引读取样例向量."""
        self.ensure_collection()
        client = self.connect()
        out: list[dict] = []
        offset = None
        fields = ["clip_index", "start_time", "end_time", "thumbnail", "video_id"]
        while True:
            rows, offset = client.scroll(
                collection_name=self.cfg.collection_name,
                scroll_filter=self._filter_video(video_id),
                limit=512,
                offset=offset,
                with_payload=fields,
                with_vectors=[self.vector_names.visual],
            )
            for row in rows:
                payload = dict(row.payload or {})
                vecs = row.vector or {}
                vec = vecs.get(self.vector_names.visual) if isinstance(vecs, dict) else vecs
                if vec is None:
                    continue
                payload["embedding"] = vec
                out.append(payload)
            if offset is None:
                break
        out.sort(key=lambda r: int(r.get("clip_index") or 0))
        return out

    def get_highlight_candidate_rows(
        self,
        *,
        limit: int = 4096,
        min_score: float = 0.5,
        candidate_mode: str = "strict",
        broad_min_score: float = 0.35,
        target_min: int = 80,
        per_video_limit: int = 4,
    ) -> list[dict]:
        """从主 Qdrant clip collection 直接读取可归纳为高光 KB 的候选片段.

        strict: 只收 qwen_is_highlight=true 或 qwen_score > min_score.
        balanced/broad: strict + 正向标签; 数量不足时加入 qwen_score>=broad_min_score 的多样性兜底。
        """
        self.ensure_collection()
        client = self.connect()
        mode = str(candidate_mode or "strict").lower()
        strict_rows: list[dict] = []
        positive_rows: list[dict] = []
        fallback_rows: list[dict] = []
        seen: set[str] = set()
        offset = None
        fields = [
            "video_id", "video_path", "video_name", "clip_index", "start_time", "end_time",
            "thumbnail", "caption_text", "transcript_text", "understanding", "qwen_caption",
            "qwen_reason", "qwen_tags", "qwen_cut_advice", "qwen_score", "qwen_is_highlight",
        ]
        while True:
            rows, offset = client.scroll(
                collection_name=self.cfg.collection_name,
                limit=512,
                offset=offset,
                with_payload=fields,
                with_vectors=[self.vector_names.visual],
            )
            for row in rows:
                payload = dict(row.payload or {})
                pid = str(row.id)
                if pid in seen:
                    continue
                score = payload.get("qwen_score")
                try:
                    score_f = float(score) if score is not None else None
                except Exception:
                    score_f = None
                tags = payload.get("qwen_tags")
                if isinstance(tags, list):
                    clean_tags = [
                        str(t) for t in tags
                        if str(t) and not any(h in str(t) for h in _ORDINARY_TAG_HINTS)
                    ]
                else:
                    clean_tags = []
                tag_text = " ".join(clean_tags)
                has_positive_tag = any(t in tag_text for t in _HIGHLIGHT_POSITIVE_TAGS)
                is_strict = payload.get("qwen_is_highlight") is True or (
                    score_f is not None and score_f > float(min_score)
                )
                is_fallback = score_f is not None and score_f >= float(broad_min_score)
                if mode == "strict" and not is_strict:
                    continue
                if mode != "strict" and not (is_strict or has_positive_tag or is_fallback):
                    continue
                vecs = row.vector or {}
                vec = vecs.get(self.vector_names.visual) if isinstance(vecs, dict) else vecs
                if vec is None:
                    continue
                caption = self._clean_candidate_text(
                    str(payload.get("qwen_caption") or payload.get("caption_text") or "")
                )
                reason = self._clean_candidate_text(str(payload.get("qwen_reason") or ""))
                cut_advice = str(payload.get("qwen_cut_advice") or "").strip()
                negative_reason = any(h in reason or h in cut_advice for h in _HIGHLIGHT_NEGATIVE_HINTS)
                parts = [
                    caption,
                    tag_text.strip(),
                    str(payload.get("transcript_text") or "").strip(),
                ]
                if is_strict or has_positive_tag or not negative_reason:
                    parts.insert(1, reason)
                payload["point_id"] = pid
                payload["embedding"] = vec
                payload["description"] = "\n".join(p for p in parts if p)
                if is_strict:
                    payload["candidate_reason"] = "strict_qwen"
                    payload["_candidate_rank"] = 3.0 + float(score_f or 0.0)
                    strict_rows.append(payload)
                elif has_positive_tag:
                    payload["candidate_reason"] = "positive_tag"
                    payload["_candidate_rank"] = 2.0 + float(score_f or 0.0)
                    positive_rows.append(payload)
                else:
                    payload["candidate_reason"] = "uncertain_score"
                    payload["_candidate_rank"] = 1.0 + float(score_f or 0.0)
                    fallback_rows.append(payload)
                seen.add(pid)
            if offset is None:
                break

        def sort_rows(items: list[dict]) -> list[dict]:
            return sorted(
                items,
                key=lambda r: (
                    -float(r.get("_candidate_rank") or 0.0),
                    str(r.get("video_id") or ""),
                    int(r.get("clip_index") or 0),
                ),
            )

        selected = sort_rows(strict_rows + positive_rows)
        if mode != "strict" and len(selected) < int(target_min):
            per_video: dict[str, int] = {}
            for row in sort_rows(fallback_rows):
                vid = str(row.get("video_id") or "")
                n = per_video.get(vid, 0)
                if n >= int(per_video_limit):
                    continue
                selected.append(row)
                per_video[vid] = n + 1
                if len(selected) >= int(target_min):
                    break
        selected = sort_rows(selected)[:int(limit)]
        for row in selected:
            row.pop("_candidate_rank", None)
        selected.sort(key=lambda r: (str(r.get("video_id") or ""), int(r.get("clip_index") or 0)))
        return selected

    # ---------- search ----------
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int | None = None,
        video_ids: Iterable[str] | None = None,
        vector_name: str | None = None,
    ) -> list[ClipHit]:
        vector_name = vector_name or self.vector_names.visual
        return self.search_multi(
            {vector_name: query_vec},
            weights={vector_name: 1.0},
            top_k=top_k,
            per_vector_k=top_k,
            video_ids=video_ids,
        )

    def search_multi(
        self,
        query_vectors: dict[str, np.ndarray],
        *,
        weights: dict[str, float] | None = None,
        top_k: int | None = None,
        per_vector_k: int | None = None,
        video_ids: Iterable[str] | None = None,
        query_text: str | None = None,
    ) -> list[ClipHit]:
        self.ensure_collection()
        client = self.connect()
        top_k = int(top_k or self.cfg.top_k)
        per_vector_k = int(per_vector_k or max(top_k, self.cfg.top_k))
        weights = weights or {name: 1.0 for name in query_vectors}
        qfilter = self._filter_videos(video_ids)

        payloads: dict[str, dict] = {}
        lane_scores: dict[str, dict[str, float]] = {}
        active_weights: dict[str, float] = {}
        for name, vec in query_vectors.items():
            if vec is None:
                continue
            w = float(weights.get(name, 0.0))
            if w <= 0:
                continue
            active_weights[name] = w
            resp = client.query_points(
                collection_name=self.cfg.collection_name,
                query=self._float_list(vec),
                using=name,
                query_filter=qfilter,
                limit=per_vector_k,
                with_payload=True,
                with_vectors=False,
                timeout=int(self.cfg.timeout),
            )
            for point in resp.points:
                pid = str(point.id)
                payloads.setdefault(pid, dict(point.payload or {}))
                lane_scores.setdefault(pid, {})[name] = float(point.score)

        lexical_hit_ids: set[str] = set()
        if query_text:
            for pid, payload, score in self._lexical_candidates(query_text, limit=max(per_vector_k, top_k)):
                lexical_hit_ids.add(pid)
                payloads.setdefault(pid, payload)
                lane_scores.setdefault(pid, {})[_LEXICAL_LANE] = score

        hits: list[ClipHit] = []
        if self._prefer_lexical_candidates(query_text, lexical_hit_ids):
            score_items = [(pid, scores) for pid, scores in lane_scores.items() if pid in lexical_hit_ids]
        else:
            score_items = list(lane_scores.items())
        for pid, scores in score_items:
            p = payloads.get(pid) or {}
            vector_scores = {name: score for name, score in scores.items() if name != _LEXICAL_LANE}
            lexical_score = float(scores.get(_LEXICAL_LANE, 0.0) or 0.0)
            denom = sum(active_weights.values()) or 1.0
            combined = sum(
                active_weights[name] * float(vector_scores.get(name, 0.0))
                for name in active_weights
            ) / denom
            if lexical_score > 0:
                # Chinese-CLIP 对短中文实体/动作 query 的分数很密集, 纯向量容易把明确命中的
                # caption/transcript 压在后面。给文本命中的 clip 一个可排序的分数地板。
                lexical_floor = min(0.985, 0.72 + 0.24 * lexical_score)
                combined = max(combined, lexical_floor)
                if vector_scores:
                    combined = min(0.995, combined + 0.035 * lexical_score)
            # 多路同时命中时给很小的覆盖奖励, 让画面/字幕/描述一致的片段更靠前.
            combined *= 1.0 + min(0.06, 0.03 * max(0, len(vector_scores) - 1))
            combined = min(0.999, combined)
            hits.append(self._payload_to_hit(pid, p, combined, scores))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _prefer_lexical_candidates(query_text: str | None, lexical_hit_ids: set[str]) -> bool:
        if len(lexical_hit_ids) < 3:
            return False
        compact = re.sub(r"[\s,，。；;:：/|]+", "", str(query_text or "").strip().lower())
        if not compact:
            return False
        return len(compact) <= 12

    def _lexical_candidates(self, query_text: str, *, limit: int) -> list[tuple[str, dict, float]]:
        terms = self._lexical_terms(query_text)
        if not terms:
            return []
        client = self.connect()
        fields = [
            "video_id", "video_path", "video_name", "clip_index", "start_time", "end_time",
            "frame_index", "timestamp", "thumbnail", "caption_text", "transcript_text",
            "transcript_source", "has_transcript", "understanding", "qwen_caption",
            "qwen_reason", "qwen_tags", "qwen_cut_advice", "qwen_score",
        ]
        out: list[tuple[str, dict, float]] = []
        offset = None
        while True:
            rows, offset = client.scroll(
                collection_name=self.cfg.collection_name,
                limit=512,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            for row in rows:
                payload = dict(row.payload or {})
                score = self._lexical_score(payload, terms)
                if score > 0:
                    out.append((str(row.id), payload, score))
            if offset is None:
                break
        out.sort(key=lambda x: x[2], reverse=True)
        return out[:limit]

    @staticmethod
    def _lexical_terms(query_text: str) -> list[tuple[str, float]]:
        q = re.sub(r"\s+", " ", str(query_text or "").strip().lower())
        if not q:
            return []
        terms: dict[str, float] = {q: 1.0}
        for token in re.split(r"[\s,，。；;:：/|]+", q):
            token = token.strip()
            if len(token) >= 2:
                terms[token] = max(terms.get(token, 0.0), 0.92)
        compact = re.sub(r"[\s,，。；;:：/|]+", "", q)
        if 2 <= len(compact) <= 4 and re.search(r"[\u4e00-\u9fff]", compact):
            # 通用中文短 query 兜底: caption 常用近义动作词, 但至少会共享一个关键汉字。
            # 这里不维护特殊同义词表, 只给单字低权重词面信号, 让 Qdrant 多模态分数继续参与排序。
            for ch in compact:
                if re.match(r"[\u4e00-\u9fff]", ch):
                    terms[ch] = max(terms.get(ch, 0.0), 0.68)
        return sorted(terms.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _lexical_texts(payload: dict) -> list[tuple[str, float]]:
        tag_text = ""
        tags = payload.get("qwen_tags")
        if isinstance(tags, list):
            tag_text = " ".join(str(t) for t in tags)
        understanding = payload.get("understanding")
        understanding_text = ""
        if isinstance(understanding, dict):
            understanding_text = json.dumps(understanding, ensure_ascii=False)
        return [
            (str(payload.get("video_name") or "").lower(), 1.0),
            (str(payload.get("caption_text") or "").lower(), 0.96),
            (str(payload.get("qwen_caption") or "").lower(), 0.96),
            (str(payload.get("transcript_text") or "").lower(), 0.92),
            (str(payload.get("qwen_reason") or "").lower(), 0.72),
            (str(payload.get("qwen_cut_advice") or "").lower(), 0.62),
            (tag_text.lower(), 0.58),
            (understanding_text.lower(), 0.55),
        ]

    @classmethod
    def _lexical_score(cls, payload: dict, terms: list[tuple[str, float]]) -> float:
        best = 0.0
        matched = 0
        for text, field_weight in cls._lexical_texts(payload):
            if not text:
                continue
            for term, term_weight in terms:
                if term and term in text:
                    matched += 1
                    best = max(best, float(term_weight) * float(field_weight))
        if matched > 1:
            best = min(1.0, best + min(0.08, 0.02 * (matched - 1)))
        return best

    @staticmethod
    def _payload_to_hit(pid: str, p: dict, score: float, scores: dict[str, float]) -> ClipHit:
        return ClipHit(
            score=float(score),
            video_id=str(p.get("video_id") or ""),
            video_path=str(p.get("video_path") or ""),
            clip_index=int(p.get("clip_index") or 0),
            start_time=float(p.get("start_time") or 0.0),
            end_time=float(p.get("end_time") or 0.0),
            frame_index=int(p.get("frame_index") or 0),
            timestamp=float(p.get("timestamp") or 0.0),
            thumbnail=str(p.get("thumbnail") or ""),
            pk=pid,
            score_breakdown={k: round(float(v), 6) for k, v in scores.items()},
            payload=p,
        )
