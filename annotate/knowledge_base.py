"""
高光样例表存储：
  Qdrant collection: highlight_example_v1
  payload: kb_id / label / sample_id / source_video_id / start_time / end_time / thumbnail / note
  vectors:
    highlight_visual_*      : 高光 clip 视觉向量
    highlight_caption_*     : Qwen/聚合描述文本向量
    highlight_transcript_*  : ASR/字幕文本向量
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from config import cfg
from search.qdrant_retriever import QdrantVectorNames, build_qdrant_vector_names

logger = logging.getLogger(__name__)

KB_COLLECTION = "highlight_example_v1"


def _distance(name: str):
    from qdrant_client import models

    t = str(name or "COSINE").upper()
    if t in {"IP", "DOT", "INNER_PRODUCT"}:
        return models.Distance.DOT
    if t == "L2":
        return models.Distance.EUCLID
    return models.Distance.COSINE


@dataclass
class HighlightSample:
    """高光样例: 一段已知高光视频片段的向量表示。"""
    embedding: np.ndarray  # visual embedding; keep the field name for existing callers.
    kb_id: str
    label: str
    sample_id: str
    caption_embedding: np.ndarray | None = None
    transcript_embedding: np.ndarray | None = None
    source_video_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    thumbnail: str = ""
    note: str = ""
    caption_text: str = ""
    transcript_text: str = ""
    transcript_source: str = ""
    created_at: float = 0.0


@dataclass
class KBStats:
    kb_id: str
    sample_count: int
    label_counts: dict[str, int] = field(default_factory=dict)


class KBRetriever:
    """高光样例的 Qdrant 存取。"""

    def __init__(self, embedding_dim: int, collection_name: str = KB_COLLECTION):
        self.dim = int(embedding_dim)
        self.collection_name = collection_name
        self.vector_names: QdrantVectorNames = build_qdrant_vector_names(self.dim)
        self._client = None
        self._ensured = False

    # ---------- connection / schema ----------
    def connect(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient

        self._client = QdrantClient(
            host=cfg.qdrant.host,
            port=int(cfg.qdrant.port),
            grpc_port=int(cfg.qdrant.grpc_port),
            api_key=cfg.qdrant.api_key or None,
            https=bool(cfg.qdrant.https),
            prefer_grpc=bool(cfg.qdrant.prefer_grpc),
            timeout=float(cfg.qdrant.timeout),
        )
        return self._client

    def _vectors_config(self):
        from qdrant_client import models

        return {
            name: models.VectorParams(size=self.dim, distance=_distance(cfg.qdrant.metric_type))
            for name in self.vector_names.all()
        }

    def _create_collection(self, client) -> None:
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._vectors_config(),
            on_disk_payload=True,
            timeout=int(cfg.qdrant.timeout),
        )
        logger.info(
            "Created Qdrant KB collection %s with vectors=%s",
            self.collection_name,
            self.vector_names.as_dict(),
        )

    def _collection_count(self, client) -> int:
        try:
            return int(client.count(
                collection_name=self.collection_name,
                exact=True,
                timeout=int(cfg.qdrant.timeout),
            ).count or 0)
        except Exception:
            return -1

    def _recreate_empty_collection(self, client, reason: str) -> None:
        count = self._collection_count(client)
        if count != 0:
            raise RuntimeError(
                f"Qdrant KB collection {self.collection_name} is incompatible with named-vector schema "
                f"({reason}) and contains {count} point(s). 请先从 highlight_clips 全量重建高光库。"
            )
        client.delete_collection(self.collection_name, timeout=int(cfg.qdrant.timeout))
        self._create_collection(client)

    def ensure_collection(self):
        if self._ensured:
            return self.connect()
        client = self.connect()
        if not client.collection_exists(self.collection_name):
            self._create_collection(client)
        else:
            info = client.get_collection(self.collection_name)
            vectors = getattr(info.config.params, "vectors", None)
            if isinstance(vectors, dict):
                missing: list[str] = []
                bad_dims: list[str] = []
                for name in self.vector_names.all():
                    if name not in vectors:
                        missing.append(name)
                        continue
                    got = int(getattr(vectors[name], "size", 0) or 0)
                    if got and got != self.dim:
                        bad_dims.append(f"{name}:{got}")
                if missing or bad_dims:
                    self._recreate_empty_collection(
                        client,
                        f"missing={missing or []}, bad_dims={bad_dims or []}",
                    )
            else:
                existing = int(getattr(vectors, "size", 0) or 0)
                self._recreate_empty_collection(client, f"single_vector_dim={existing or 'unknown'}")
        self._ensure_payload_indexes(client)
        self._ensured = True
        return client

    def _ensure_payload_indexes(self, client):
        from qdrant_client import models

        for field_name, schema in [
            ("kb_id", models.PayloadSchemaType.KEYWORD),
            ("sample_id", models.PayloadSchemaType.KEYWORD),
            ("label", models.PayloadSchemaType.KEYWORD),
            ("source_video_id", models.PayloadSchemaType.KEYWORD),
        ]:
            try:
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                    timeout=int(cfg.qdrant.timeout),
                )
            except Exception as e:
                logger.debug("create KB payload index skipped for %s: %s", field_name, e)

    # ---------- helpers ----------
    @staticmethod
    def _point_id(kb_id: str, sample_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vision-rag:highlight-kb:{kb_id}:{sample_id}"))

    @staticmethod
    def _float_list(vec: np.ndarray) -> list[float]:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        return [float(x) for x in arr.tolist()]

    @staticmethod
    def _vec_or_fallback(vec: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
        if vec is None:
            return np.asarray(fallback, dtype=np.float32).reshape(-1)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    @staticmethod
    def _kb_filter(kb_id: str):
        from qdrant_client import models

        return models.Filter(
            must=[models.FieldCondition(key="kb_id", match=models.MatchValue(value=str(kb_id)))]
        )

    @staticmethod
    def _sample_filter(kb_id: str, sample_id: str):
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(key="kb_id", match=models.MatchValue(value=str(kb_id))),
                models.FieldCondition(key="sample_id", match=models.MatchValue(value=str(sample_id))),
            ]
        )

    @staticmethod
    def _label_filter(kb_id: str, label: str):
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(key="kb_id", match=models.MatchValue(value=str(kb_id))),
                models.FieldCondition(key="label", match=models.MatchValue(value=str(label))),
            ]
        )

    def _scroll(self, *, scroll_filter=None, with_vectors=False, fields=None, limit: int = 512):
        client = self.ensure_collection()
        offset = None
        while True:
            rows, offset = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=fields or True,
                with_vectors=with_vectors,
                timeout=int(cfg.qdrant.timeout),
            )
            for row in rows:
                yield row
            if offset is None:
                break

    # ---------- insert / delete / query ----------
    def insert_samples(self, samples: list[HighlightSample]) -> list[str]:
        if not samples:
            return []
        from qdrant_client import models

        client = self.ensure_collection()
        now = time.time()
        points = []
        ids: list[str] = []
        for s in samples:
            if not s.created_at:
                s.created_at = now
            pid = self._point_id(s.kb_id, s.sample_id)
            ids.append(pid)
            visual_vec = self._vec_or_fallback(s.embedding, s.embedding)
            caption_vec = self._vec_or_fallback(s.caption_embedding, visual_vec)
            transcript_vec = self._vec_or_fallback(s.transcript_embedding, caption_vec)
            points.append(
                models.PointStruct(
                    id=pid,
                    vector={
                        self.vector_names.visual: self._float_list(visual_vec),
                        self.vector_names.caption: self._float_list(caption_vec),
                        self.vector_names.transcript: self._float_list(transcript_vec),
                    },
                    payload={
                        "point_id": pid,
                        "kb_id": s.kb_id,
                        "label": s.label,
                        "sample_id": s.sample_id,
                        "source_video_id": s.source_video_id,
                        "start_time": float(s.start_time),
                        "end_time": float(s.end_time),
                        "thumbnail": s.thumbnail or "",
                        "note": s.note or "",
                        "caption_text": s.caption_text or s.note or "",
                        "transcript_text": s.transcript_text or "",
                        "transcript_source": s.transcript_source or "",
                        "vector_names": self.vector_names.as_dict(),
                        "created_at": float(s.created_at),
                    },
                )
            )
        client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
            timeout=int(cfg.qdrant.timeout),
        )
        return ids

    def delete_kb(self, kb_id: str, wait_for_drain: bool = True, timeout_s: float = 10.0):
        from qdrant_client import models

        client = self.ensure_collection()
        client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=self._kb_filter(kb_id)),
            wait=True,
            timeout=int(cfg.qdrant.timeout),
        )
        if wait_for_drain:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                count = client.count(
                    collection_name=self.collection_name,
                    count_filter=self._kb_filter(kb_id),
                    exact=True,
                    timeout=int(cfg.qdrant.timeout),
                ).count
                if int(count or 0) == 0:
                    return
                time.sleep(0.2)
            logger.warning("delete_kb(%s) drain timeout after %.1fs", kb_id, timeout_s)

    def delete_sample(self, kb_id: str, sample_id: str):
        from qdrant_client import models

        self.ensure_collection().delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=self._sample_filter(kb_id, sample_id)),
            wait=True,
            timeout=int(cfg.qdrant.timeout),
        )

    def rename_labels(self, kb_id: str, label_map: dict[str, str]) -> dict[str, int]:
        """批量更新当前 KB 内的 label payload, 不改向量和样本内容。"""
        client = self.ensure_collection()
        updated: dict[str, int] = {}
        for old_label, new_label in label_map.items():
            old = str(old_label or "").strip()
            new = str(new_label or "").strip()
            if not old or not new or old == new:
                continue
            ids = [
                row.id
                for row in self._scroll(
                    scroll_filter=self._label_filter(kb_id, old),
                    fields=["label"],
                    limit=512,
                )
            ]
            if not ids:
                updated[old] = 0
                continue
            client.set_payload(
                collection_name=self.collection_name,
                payload={"label": new},
                points=ids,
                wait=True,
                timeout=int(cfg.qdrant.timeout),
            )
            updated[old] = len(ids)
        return updated

    def list_kbs(self) -> list[KBStats]:
        fields = ["kb_id", "label"]
        agg: dict[str, dict[str, int]] = {}
        for row in self._scroll(fields=fields):
            payload = row.payload or {}
            kb = str(payload.get("kb_id") or "")
            label = str(payload.get("label") or "")
            if not kb:
                continue
            agg.setdefault(kb, {})
            agg[kb][label] = agg[kb].get(label, 0) + 1
        return [
            KBStats(kb_id=kb, sample_count=sum(labels.values()), label_counts=labels)
            for kb, labels in sorted(agg.items())
        ]

    def list_samples(self, kb_id: str) -> list[dict]:
        fields = [
            "sample_id", "label", "source_video_id", "start_time", "end_time",
            "thumbnail", "note", "caption_text", "transcript_text", "transcript_source", "created_at",
        ]
        rows = []
        for row in self._scroll(scroll_filter=self._kb_filter(kb_id), fields=fields):
            payload = dict(row.payload or {})
            payload["point_id"] = str(row.id)
            rows.append(payload)
        rows.sort(key=lambda r: (str(r.get("label") or ""), float(r.get("start_time") or 0.0)))
        return rows

    def search_kb(
        self,
        kb_id: str,
        query_vecs: np.ndarray,
        top_k: int,
        query_caption_vecs: np.ndarray | None = None,
        query_transcript_vecs: np.ndarray | None = None,
        query_caption_mask: list[bool] | None = None,
        query_transcript_mask: list[bool] | None = None,
        weights: dict[str, float] | None = None,
        per_vector_k: int | None = None,
    ) -> list[list[dict]]:
        """
        给定一批 query 向量 (N, D)，对 kb_id 范围内的 Qdrant 高光样例做多模态 ANN。
        返回: List[N] of List[hit_dict]
        """
        client = self.ensure_collection()
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)
        if query_caption_vecs is not None and query_caption_vecs.ndim == 1:
            query_caption_vecs = query_caption_vecs.reshape(1, -1)
        if query_transcript_vecs is not None and query_transcript_vecs.ndim == 1:
            query_transcript_vecs = query_transcript_vecs.reshape(1, -1)

        out: list[list[dict]] = []
        fields = [
            "sample_id", "label", "source_video_id", "start_time", "end_time",
            "thumbnail", "note", "caption_text", "transcript_text", "transcript_source",
        ]
        names = self.vector_names
        aliases = {
            names.visual: "visual",
            names.caption: "caption",
            names.transcript: "transcript",
        }
        weights = weights or {
            names.visual: float(cfg.qdrant.visual_weight),
            names.caption: float(cfg.qdrant.caption_weight),
            names.transcript: float(cfg.qdrant.transcript_weight),
        }
        per_vector_k = int(per_vector_k or max(int(top_k), int(top_k) * 2))
        for i, visual_vec in enumerate(query_vecs):
            query_by_lane: dict[str, np.ndarray] = {names.visual: visual_vec}
            if (
                query_caption_vecs is not None
                and i < len(query_caption_vecs)
                and (query_caption_mask is None or bool(query_caption_mask[i]))
            ):
                query_by_lane[names.caption] = query_caption_vecs[i]
            if (
                query_transcript_vecs is not None
                and i < len(query_transcript_vecs)
                and (query_transcript_mask is None or bool(query_transcript_mask[i]))
            ):
                query_by_lane[names.transcript] = query_transcript_vecs[i]

            payloads: dict[str, dict] = {}
            lane_scores: dict[str, dict[str, float]] = {}
            active_weights: dict[str, float] = {}
            for lane, vec in query_by_lane.items():
                w = float(weights.get(lane, 0.0))
                if vec is None or w <= 0:
                    continue
                active_weights[lane] = w
                resp = client.query_points(
                    collection_name=self.collection_name,
                    query=self._float_list(vec),
                    using=lane,
                    query_filter=self._kb_filter(kb_id),
                    limit=per_vector_k,
                    with_payload=fields,
                    with_vectors=False,
                    timeout=int(cfg.qdrant.timeout),
                )
                for point in resp.points:
                    pid = str(point.id)
                    payloads.setdefault(pid, dict(point.payload or {}))
                    lane_scores.setdefault(pid, {})[lane] = float(point.score)

            row = []
            denom = sum(active_weights.values()) or 1.0
            for pid, scores in lane_scores.items():
                payload = payloads.get(pid) or {}
                combined = sum(
                    active_weights[lane] * float(scores.get(lane, 0.0))
                    for lane in active_weights
                ) / denom
                combined *= 1.0 + min(0.06, 0.03 * max(0, len(scores) - 1))
                combined = min(0.999, combined)
                row.append({
                    "score": float(combined),
                    "sample_id": payload.get("sample_id"),
                    "label": payload.get("label"),
                    "source_video_id": payload.get("source_video_id"),
                    "start_time": float(payload.get("start_time") or 0),
                    "end_time": float(payload.get("end_time") or 0),
                    "thumbnail": payload.get("thumbnail") or "",
                    "note": payload.get("note") or "",
                    "caption_text": payload.get("caption_text") or "",
                    "transcript_text": payload.get("transcript_text") or "",
                    "transcript_source": payload.get("transcript_source") or "",
                    "score_breakdown": {
                        aliases.get(lane, lane): round(float(score), 4)
                        for lane, score in scores.items()
                    },
                    "modalities": [aliases.get(lane, lane) for lane in scores],
                })
            row.sort(key=lambda h: h.get("score", 0.0), reverse=True)
            out.append(row[:int(top_k)])
        return out


class KnowledgeBase:
    """
    高层 API: 把视频片段或单图作为高光样例加进 KB。

    底层用 CLIPEncoder 编码，KBRetriever 写入 Qdrant。
    样例可以是: 1) 已入库视频的某个 clip (start/end) -> 解码取帧 -> encode
                2) 上传的图片 -> encode
                3) 直接文本描述 -> encode_texts (作为"零样本"标签)
    """

    def __init__(self, encoder, processor, retriever: KBRetriever | None = None):
        self.encoder = encoder
        self.processor = processor
        self.retriever = retriever or KBRetriever(embedding_dim=encoder.dim)

    def add_clip_sample(
        self,
        kb_id: str,
        label: str,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_id: str | None = None,
        source_video_id: str = "",
        thumbnail: str = "",
        note: str = "",
    ) -> HighlightSample:
        meta = self.processor.probe(video_path)
        n = self.processor.cfg.max_frames_per_clip
        if n <= 1:
            times = [(start_time + end_time) / 2]
        else:
            times = np.linspace(start_time, max(end_time - 1e-3, start_time), n).tolist()
        idxs = [min(int(t * meta.fps), meta.num_frames - 1) for t in times]

        from ingest.video_processor import _HAS_DECORD  # type: ignore
        if _HAS_DECORD:
            from decord import VideoReader, cpu
            vr = VideoReader(meta.path, ctx=cpu(0))
            frames = vr.get_batch(idxs).asnumpy()
        else:
            import cv2
            cap = cv2.VideoCapture(meta.path)
            frames = []
            try:
                for fi in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ok, f = cap.read()
                    if ok:
                        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
            frames = np.stack(frames) if frames else np.zeros((0, meta.height, meta.width, 3), dtype=np.uint8)

        vec = self.encoder.encode_clip_frames(frames, pooling="mean")
        text = (note or label or "").strip() or "高光样例"
        text_vec = self.encoder.encode_texts([text])[0]
        sample = HighlightSample(
            embedding=vec,
            caption_embedding=text_vec,
            transcript_embedding=text_vec,
            kb_id=kb_id,
            label=label,
            sample_id=sample_id or f"{int(time.time() * 1000)}",
            source_video_id=source_video_id,
            start_time=float(start_time),
            end_time=float(end_time),
            thumbnail=thumbnail,
            note=note,
            caption_text=text,
            transcript_text="",
        )
        self.retriever.insert_samples([sample])
        return sample

    def add_text_sample(self, kb_id: str, label: str, text: str, sample_id: str | None = None, note: str = "") -> HighlightSample:
        vec = self.encoder.encode_texts([text])[0]
        sample = HighlightSample(
            embedding=vec,
            caption_embedding=vec,
            transcript_embedding=vec,
            kb_id=kb_id,
            label=label,
            sample_id=sample_id or f"text_{int(time.time() * 1000)}",
            note=note or text,
            caption_text=text,
            transcript_text=text,
        )
        self.retriever.insert_samples([sample])
        return sample

    def add_image_sample(self, kb_id: str, label: str, image_path: str, sample_id: str | None = None, note: str = "") -> HighlightSample:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        vec = self.encoder.encode_images([img])[0]
        text = (note or label or "").strip() or "高光样例"
        text_vec = self.encoder.encode_texts([text])[0]
        sample = HighlightSample(
            embedding=vec,
            caption_embedding=text_vec,
            transcript_embedding=text_vec,
            kb_id=kb_id,
            label=label,
            sample_id=sample_id or f"img_{int(time.time() * 1000)}",
            note=note,
            caption_text=text,
            transcript_text="",
        )
        self.retriever.insert_samples([sample])
        return sample
