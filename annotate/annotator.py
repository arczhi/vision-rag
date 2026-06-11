"""
高光标注核心算法:

输入: video_path + kb_id + 阈值
流程:
  1. 对视频做与 ingest 相同的滑窗切片 (复用 VideoProcessor)
  2. 对每个 clip 编码得到向量 (复用 CLIPEncoder)
  3. 对每个 clip 向量查 KB top_k_per_clip 个高光样例
  4. 按 label 聚合: label_score = mean(top-k 同 label 的相似度)
  5. label_score >= threshold 的 clip → 候选
  6. 同 label 相邻 clip 合并 (gap_tol)
  7. 长度过滤: min_duration / max_duration

输出: List[Annotation] = [{label, start, end, score, evidence_samples}]
"""
from __future__ import annotations

import logging
import time
import hashlib
import json
from collections.abc import Callable
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import cfg

from .knowledge_base import KBRetriever
from .transcript import Transcript, TranscriptProvider

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    label: str
    start_time: float
    end_time: float
    score: float
    thumbnail: str = ""
    clip_indices: list[int] = field(default_factory=list)
    evidence_samples: list[dict] = field(default_factory=list)  # 触发该标注的样例(去重)
    understanding: dict | None = None                           # 本地 VLM 高光理解结果

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.end_time - self.start_time, 3),
            "score": round(self.score, 4),
            "thumbnail": self.thumbnail,
            "clip_indices": self.clip_indices,
            "evidence_samples": self.evidence_samples,
            "understanding": self.understanding,
            "dialogue": self.__dict__.get("dialogue", ""),
            "transcript_source": self.__dict__.get("transcript_source", ""),
            "base_score": round(float(self.__dict__.get("base_score")), 4) if self.__dict__.get("base_score") is not None else None,
            "vl_score": round(float(self.__dict__.get("vl_score")), 4) if self.__dict__.get("vl_score") is not None else None,
            "vl_rank_signal": round(float(self.__dict__.get("vl_rank_signal")), 4) if self.__dict__.get("vl_rank_signal") is not None else None,
            "vl_rank_score": round(float(self.__dict__.get("vl_rank_score")), 4) if self.__dict__.get("vl_rank_score") is not None else None,
            "vl_rank_reason": self.__dict__.get("vl_rank_reason", ""),
            "vl_rejected": bool(self.__dict__.get("vl_rejected", False)),
            "vl_unchecked": bool(self.__dict__.get("vl_unchecked", False)),
        }


@dataclass
class AnnotateTimings:
    decode_ms: float = 0.0
    embed_ms: float = 0.0
    kb_search_ms: float = 0.0
    aggregate_ms: float = 0.0
    transcript_ms: float = 0.0
    vl_rerank_ms: float = 0.0
    total_ms: float = 0.0
    num_clips: int = 0
    num_transcript_segments: int = 0
    num_vl_calls: int = 0

    def as_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class AnnotateOutcome:
    annotations: list[Annotation] = field(default_factory=list)
    rejected_annotations: list[Annotation] = field(default_factory=list)
    timings: AnnotateTimings = field(default_factory=AnnotateTimings)
    labels_in_kb: list[str] = field(default_factory=list)  # 该 KB 里所有 label
    kb_sample_count: int = 0


def _now_ms() -> float:
    return time.perf_counter() * 1000


class Annotator:
    """
    高光标注器。复用 ingest 的 VideoProcessor + CLIPEncoder, 不重复加载模型。
    """

    def __init__(
        self,
        encoder,
        processor,
        kb_retriever: KBRetriever | None = None,
        vl_reranker=None,
        clip_retriever=None,
    ):
        self.encoder = encoder
        self.processor = processor
        self.kb = kb_retriever or KBRetriever(embedding_dim=encoder.dim)
        self.vl_reranker = vl_reranker  # search.reranker.Reranker 实例, 复用其 VL 模型
        self.clip_retriever = clip_retriever
        self.transcripts = TranscriptProvider()
        self._clip_embedding_cache: OrderedDict[str, tuple[list, np.ndarray]] = OrderedDict()
        self._clip_embedding_cache_limit = 2

    def annotate(
        self,
        video_path: str | Path,
        kb_id: str,
        threshold: float = 0.25,
        top_k_per_clip: int = 5,
        merge_gap: float = 1.5,
        min_duration: float = 1.0,
        max_duration: float = 60.0,
        labels: list[str] | None = None,
        normalize: bool = True,
        z_threshold: float = 0.75,
        vl_rerank: bool = False,
        vl_top_k: int = 8,
        vl_weight: float = 0.35,
        vl_max_frames: int = 1,
        custom_segments: list[tuple[float, float]] | None = None,
        extra_segments: list[tuple[float, float]] | None = None,
        progress_callback: Callable[[str, float, dict | None], None] | None = None,
    ) -> AnnotateOutcome:
        """
        normalize=True 时启用 KB 内 z-score 归一化:
          相对阈值 = z_threshold (默认 0.75σ), 跨 KB 通用
          绝对阈值 = threshold (兜底)

        vl_rerank=True 时, 对最终标注命中段全部送代表帧到本地 Qwen2-VL 做高光理解,
        并把 VLM 高光分与 CLIP/KB 分数融合重排; 明确判 false 的候选会单独返回。

        custom_segments 不为空时沿用旧语义: 用给定时间段替代默认切片。
        extra_segments 不为空时追加到默认/自定义切片之后, 用作额外召回候选。
        """
        timings = AnnotateTimings()
        t0 = _now_ms()

        def report(message: str, progress: float, extra: dict | None = None) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(message, progress, extra)
            except Exception:
                logger.debug("annotation progress callback failed", exc_info=True)

        # ---- 0. KB sanity ----
        report("读取高光知识库", 0.12)
        kbs = {k.kb_id: k for k in self.kb.list_kbs()}
        kb_stats = kbs.get(kb_id)
        if kb_stats is None or kb_stats.sample_count == 0:
            timings.total_ms = _now_ms() - t0
            return AnnotateOutcome(timings=timings)
        report(
            f"读取高光样例表: {kb_stats.sample_count} 个样例 / {len(kb_stats.label_counts)} 类",
            0.15,
            {"kb_sample_count": kb_stats.sample_count, "labels_in_kb": list(kb_stats.label_counts.keys())},
        )

        cache_key = self._video_cache_key(video_path, custom_segments, extra_segments)
        cached = self._get_clip_embedding_cache(cache_key)

        # ---- 1. decode + slice ----
        ts = _now_ms()
        meta = self.processor.probe(video_path)
        report(
            f"场景检测 + 切片中: {meta.duration:.1f}s / {meta.num_frames} 帧",
            0.2,
            {
                "video_duration": round(float(meta.duration), 3),
                "video_frames": int(meta.num_frames),
                "video_fps": round(float(meta.fps), 3),
            },
        )
        tts = _now_ms()
        report("加载字幕/对白", 0.19)
        transcript = self.transcripts.load(video_path)
        timings.transcript_ms = _now_ms() - tts
        timings.num_transcript_segments = len(transcript.segments)

        if cached:
            clips, clip_vecs = cached
            timings.decode_ms = 0.0
            timings.embed_ms = 0.0
            timings.num_clips = len(clips)
            report(
                f"命中视频缓存: {len(clips)} 段",
                0.60,
                {
                    "cache_hit": True,
                    "clips_decoded": len(clips),
                    "num_clips": len(clips),
                    "embed_done": len(clips),
                },
            )
        else:
            clips = []
            primary_count = 0
            for c in self.processor.iter_clips(meta, save_thumbnails=False, custom_segments=custom_segments):
                c.clip_index = len(clips)
                c.__dict__["segment_source"] = "custom" if custom_segments else "hybrid"
                clips.append(c)
                primary_count += 1
                if len(clips) == 1 or len(clips) % 10 == 0:
                    report(
                        f"切片生成 {len(clips)} 段",
                        min(0.21, 0.2 + len(clips) * 0.0005),
                        {"clips_decoded": len(clips), "cache_hit": False},
                    )
            extra_count = 0
            if extra_segments:
                for c in self.processor.iter_clips(meta, save_thumbnails=False, custom_segments=extra_segments):
                    c.clip_index = len(clips)
                    c.__dict__["segment_source"] = "extra"
                    clips.append(c)
                    extra_count += 1
                report(
                    f"追加本地理解切分候选: {extra_count} 段",
                    0.215,
                    {
                        "clips_decoded": len(clips),
                        "primary_clips": primary_count,
                        "extra_clips": extra_count,
                        "cache_hit": False,
                    },
                )
            timings.decode_ms = _now_ms() - ts
            timings.num_clips = len(clips)
            report(
                f"完成切片: {len(clips)} 段",
                0.22,
                {
                    "num_clips": len(clips),
                    "primary_clips": primary_count,
                    "extra_clips": extra_count,
                    "cache_hit": False,
                },
            )
            if not clips:
                timings.total_ms = _now_ms() - t0
                return AnnotateOutcome(timings=timings, labels_in_kb=list(kb_stats.label_counts.keys()), kb_sample_count=kb_stats.sample_count)

            # ---- 2. embed all clips (mean pooling) ----
            ts = _now_ms()
            clip_vecs = np.zeros((len(clips), self.encoder.dim), dtype=np.float32)
            step = max(1, len(clips) // 20)
            for i, c in enumerate(clips):
                clip_vecs[i] = self.encoder.encode_clip_frames(c.frames, pooling="mean")
                done = i + 1
                if done == 1 or done == len(clips) or done % step == 0:
                    frac = done / max(len(clips), 1)
                    report(
                        f"embedding {done}/{len(clips)}",
                        0.22 + 0.38 * frac,
                        {"embed_done": done, "num_clips": len(clips), "cache_hit": False},
                    )
            timings.embed_ms = _now_ms() - ts
            self._put_clip_embedding_cache(cache_key, clips, clip_vecs)

        if not clips:
            timings.total_ms = _now_ms() - t0
            return AnnotateOutcome(timings=timings, labels_in_kb=list(kb_stats.label_counts.keys()), kb_sample_count=kb_stats.sample_count)

        if cached:
            report("embedding 命中缓存", 0.60, {"embed_done": len(clips), "num_clips": len(clips), "cache_hit": True})
        else:
            # 已在 embedding 循环内上报进度。
            pass

        target_qdrant_by_clip = self._load_target_qdrant_clip_contexts(meta.video_id)
        ts_text_query = _now_ms()
        query_caption_vecs, query_caption_mask, query_transcript_vecs, query_transcript_mask = self._build_kb_text_query_vectors(
            clips,
            transcript,
            target_qdrant_by_clip,
        )
        timings.embed_ms += _now_ms() - ts_text_query

        # ---- 3. highlight_example_v1 multi-vector search per clip ----
        ts = _now_ms()
        active_modalities = ["visual"]
        if any(query_caption_mask or []):
            active_modalities.append("caption")
        if any(query_transcript_mask or []):
            active_modalities.append("transcript")
        report(
            "highlight_example_v1 多模态高光样例召回",
            0.62,
            {"kb_query_modalities": active_modalities},
        )
        all_hits = self.kb.search_kb(
            kb_id,
            clip_vecs,
            top_k=top_k_per_clip,
            query_caption_vecs=query_caption_vecs,
            query_transcript_vecs=query_transcript_vecs,
            query_caption_mask=query_caption_mask,
            query_transcript_mask=query_transcript_mask,
        )
        timings.kb_search_ms = _now_ms() - ts
        report(
            "highlight_example_v1 多模态高光样例召回完成",
            0.68,
            {"kb_search_source": "highlight_example_v1", "kb_query_modalities": active_modalities},
        )

        # ---- 4. aggregate per (clip, label) ----
        ts = _now_ms()
        report("聚合高光评分", 0.7)
        label_filter = set(labels) if labels else None

        # 先收集所有 (clip_i, label, mean_score) 用来算 KB 内的归一化基线
        raw_pairs: list[tuple[int, str, float, list]] = []
        for i, hits in enumerate(all_hits):
            by_label: dict[str, list] = {}
            evidence_by_label: dict[str, list] = {}
            for h in hits:
                lb = h["label"]
                if label_filter and lb not in label_filter:
                    continue
                by_label.setdefault(lb, []).append(h["score"])
                evidence_by_label.setdefault(lb, []).append(h)
            for lb, scores in by_label.items():
                avg = float(np.mean(scores))
                raw_pairs.append((i, lb, avg, evidence_by_label[lb]))

        # z-score 归一化基线 (按 label 分别算)
        label_baseline: dict[str, tuple[float, float, int]] = {}  # label -> (mean, std, count)
        if normalize:
            by_lb: dict[str, list[float]] = {}
            for _, lb, sc, _ in raw_pairs:
                by_lb.setdefault(lb, []).append(sc)
            for lb, vals in by_lb.items():
                arr = np.asarray(vals)
                m = float(arr.mean())
                s = float(arr.std()) if len(arr) > 1 else 1.0
                if s < 1e-6:
                    s = 1.0
                label_baseline[lb] = (m, s, len(vals))

        # 过滤: 双门 (绝对 + 相对)
        clip_label_score: list[dict[str, dict]] = [dict() for _ in clips]
        for i, lb, avg, evidence in raw_pairs:
            keep_abs = avg >= threshold
            if normalize and lb in label_baseline:
                m, s, n = label_baseline[lb]
                z = (avg - m) / s
                # z-score needs a real comparison population. For tiny custom/extra
                # segment sets, use the absolute gate instead of inventing a fake baseline.
                keep_rel = True if n < 3 else z >= z_threshold
            else:
                z = 0.0
                keep_rel = True
            # 双门: 必须同时满足 (相对阈值更关键)
            if keep_abs and keep_rel:
                clip_label_score[i][lb] = {
                    "score": avg,
                    "z": z,
                    "evidence": evidence,
                }

        # ---- 5. 时间维度合并: 改造后的"重叠不强合并"算法 ----
        annotations: list[Annotation] = []
        per_label_clips: dict[str, list] = {}
        for i, ls in enumerate(clip_label_score):
            for lb, info in ls.items():
                per_label_clips.setdefault(lb, []).append((i, info["score"], info["evidence"], info.get("z", 0.0)))

        for lb, items in per_label_clips.items():
            items.sort(key=lambda x: clips[x[0]].start_time)
            cur: Annotation | None = None
            for idx, score, evidence, z in items:
                c = clips[idx]
                if cur is None:
                    cur = self._mk_ann(lb, c, score, evidence, z)
                    continue

                # 改造点: 重叠/相邻 clip 合并条件更严
                overlap_or_touch = c.start_time <= cur.end_time + merge_gap
                # 当时间真正重叠 (非简单相邻) 时, 要求新 clip 分数也"够好"才合并
                actually_overlap = c.start_time < cur.end_time - 0.01
                if overlap_or_touch:
                    if actually_overlap:
                        # 重叠: 取分数高的扩展, 低分 clip 不强行合入 (避免被几何强迫拉成整段)
                        if score >= cur.score * 0.9:
                            cur.end_time = max(cur.end_time, c.end_time)
                            cur.score = max(cur.score, score)
                            cur.clip_indices.append(c.clip_index)
                            self._merge_evidence(cur, evidence)
                            if not cur.thumbnail and c.thumbnail_path:
                                cur.thumbnail = c.thumbnail_path
                        else:
                            # 分数低于当前 90%: 不合并, 当前段封口
                            annotations.append(cur)
                            cur = self._mk_ann(lb, c, score, evidence, z)
                    else:
                        # 仅相邻 (gap_tol 内): 正常合并
                        cur.end_time = max(cur.end_time, c.end_time)
                        cur.score = max(cur.score, score)
                        cur.clip_indices.append(c.clip_index)
                        self._merge_evidence(cur, evidence)
                else:
                    annotations.append(cur)
                    cur = self._mk_ann(lb, c, score, evidence, z)
            if cur is not None:
                annotations.append(cur)

        # ---- 6. 长度过滤 + 排序 + evidence 截断 ----
        annotations = [
            a for a in annotations
            if min_duration <= (a.end_time - a.start_time) <= max_duration
        ]
        annotations.sort(key=lambda x: x.score, reverse=True)
        for a in annotations:
            a.evidence_samples = sorted(a.evidence_samples, key=lambda e: e.get("score", 0), reverse=True)[:5]

        timings.aggregate_ms = _now_ms() - ts
        report(f"聚合完成: {len(annotations)} 个候选片段", 0.76, {"candidate_annotations": len(annotations)})

        # ---- 7. 本地 Qwen2-VL 高光理解 + 重排序: 检查全部最终召回候选 ----
        rejected_annotations: list[Annotation] = []
        if vl_rerank and annotations and self.vl_reranker is not None:
            ts = _now_ms()
            num_vl_candidates = len(annotations)

            def report_vl(done: int, total: int, phase: str = "") -> None:
                frac = float(done) / max(float(total), 1.0)
                suffix = f" {phase}" if phase else ""
                report(
                    f"本地 Qwen2-VL 重排序 {done}/{total}{suffix}",
                    0.91 + 0.05 * frac,
                    {"vl_done": done, "vl_candidates": total},
                )

            report_vl(0, num_vl_candidates)
            annotations, rejected_annotations = self._rerank_with_local_vlm(
                meta=meta,
                annotations=annotations,
                top_k=num_vl_candidates,
                weight=vl_weight,
                max_frames=vl_max_frames,
                transcript=transcript,
                target_qdrant_by_clip=target_qdrant_by_clip,
                progress_callback=report_vl,
            )
            timings.vl_rerank_ms = _now_ms() - ts
            timings.num_vl_calls = num_vl_candidates
            report(
                "本地 Qwen2-VL 重排序完成",
                0.96,
                {"vl_done": num_vl_candidates, "vl_rejected": len(rejected_annotations)},
            )

        all_for_thumbnail = annotations + rejected_annotations
        if all_for_thumbnail:
            report("生成标注缩略图", 0.98)
            self._assign_first_frame_thumbnails(meta, all_for_thumbnail)

        timings.total_ms = _now_ms() - t0
        report(
            f"标注完成: {len(annotations)} 个高光片段 / {len(rejected_annotations)} 个被拒候选",
            0.99,
            {"annotations_count": len(annotations), "rejected_annotations_count": len(rejected_annotations)},
        )

        return AnnotateOutcome(
            annotations=annotations,
            rejected_annotations=rejected_annotations,
            timings=timings,
            labels_in_kb=sorted(kb_stats.label_counts.keys()),
            kb_sample_count=kb_stats.sample_count,
        )

    # ---------- helpers ----------
    def _video_cache_key(
        self,
        video_path: str | Path,
        custom_segments: list[tuple[float, float]] | None,
        extra_segments: list[tuple[float, float]] | None = None,
    ) -> str:
        path = Path(video_path)
        st = path.stat()
        h = hashlib.sha1()
        h.update(str(st.st_size).encode())
        sample = 4 * 1024 * 1024
        with path.open("rb") as f:
            h.update(f.read(sample))
            if st.st_size > sample:
                f.seek(max(0, st.st_size - sample))
                h.update(f.read(sample))

        video_cfg = getattr(self.processor, "cfg", None)
        payload = {
            "file": h.hexdigest(),
            "segments": custom_segments or [],
            "extra_segments": extra_segments or [],
            "encoder_dim": int(getattr(self.encoder, "dim", 0) or 0),
            "clip_duration": getattr(video_cfg, "clip_duration", None),
            "clip_stride": getattr(video_cfg, "clip_stride", None),
            "max_frames_per_clip": getattr(video_cfg, "max_frames_per_clip", None),
            "resize": getattr(video_cfg, "resize", None),
            "slicing_strategy": getattr(video_cfg, "slicing_strategy", None),
            "frame_sampling": getattr(video_cfg, "frame_sampling", None),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def _get_clip_embedding_cache(self, key: str) -> tuple[list, np.ndarray] | None:
        cached = self._clip_embedding_cache.get(key)
        if cached is None:
            return None
        self._clip_embedding_cache.move_to_end(key)
        clips, clip_vecs = cached
        return clips, clip_vecs.copy()

    def _put_clip_embedding_cache(self, key: str, clips: list, clip_vecs: np.ndarray) -> None:
        self._clip_embedding_cache[key] = (clips, clip_vecs.copy())
        self._clip_embedding_cache.move_to_end(key)
        while len(self._clip_embedding_cache) > self._clip_embedding_cache_limit:
            self._clip_embedding_cache.popitem(last=False)

    @staticmethod
    def _usable_dialogue_for_vector(text: str) -> bool:
        t = str(text or "").strip()
        return bool(t and t not in {"无", "无台词", "无台词字幕", "none", "None"})

    @staticmethod
    def _join_context_texts(rows: list[dict], *keys: str, limit: int = 1200) -> str:
        seen: set[str] = set()
        parts: list[str] = []
        for row in rows:
            for key in keys:
                text = str(row.get(key) or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                parts.append(text)
        return "\n".join(parts)[:limit]

    @staticmethod
    def _slim_qdrant_clip_context(row: dict) -> dict:
        fields = [
            "point_id", "video_id", "video_name", "video_path", "clip_index",
            "start_time", "end_time", "caption_text", "transcript_text",
            "transcript_source", "understanding", "qwen_caption", "qwen_reason",
            "qwen_tags", "qwen_cut_advice", "qwen_score", "qwen_is_highlight",
        ]
        return {k: row.get(k) for k in fields if row.get(k) not in (None, "", [])}

    def _load_target_qdrant_clip_contexts(self, video_id: str) -> dict[int, dict]:
        if not video_id or self.clip_retriever is None or not hasattr(self.clip_retriever, "list_clips"):
            return {}
        try:
            rows = self.clip_retriever.list_clips(video_id)
        except Exception as e:
            logger.warning("load target qdrant clip context failed: %s", e)
            return {}

        out: dict[int, dict] = {}
        for row in rows:
            try:
                idx = int(row.get("clip_index"))
            except Exception:
                continue
            out[idx] = self._slim_qdrant_clip_context(row)
        return out

    def _build_kb_text_query_vectors(
        self,
        clips: list,
        transcript: Transcript,
        target_qdrant_by_clip: dict[int, dict],
    ) -> tuple[np.ndarray | None, list[bool], np.ndarray | None, list[bool]]:
        caption_texts: list[str] = []
        transcript_texts: list[str] = []
        caption_mask: list[bool] = []
        transcript_mask: list[bool] = []

        for c in clips:
            row = target_qdrant_by_clip.get(int(getattr(c, "clip_index", -1))) or {}
            caption_text = self._join_context_texts([row], "caption_text", "qwen_caption", "qwen_reason", limit=900) if row else ""
            dialogue = transcript.text_for_range(c.start_time, c.end_time, max_chars=900) if transcript else ""
            transcript_text = dialogue.strip() or str(row.get("transcript_text") or "").strip()

            has_caption = self._usable_dialogue_for_vector(caption_text)
            has_transcript = self._usable_dialogue_for_vector(transcript_text)
            caption_mask.append(has_caption)
            transcript_mask.append(has_transcript)
            caption_texts.append(caption_text if has_caption else "无描述")
            transcript_texts.append(transcript_text if has_transcript else "无台词字幕")

        caption_vecs = self.encoder.encode_texts(caption_texts) if any(caption_mask) else None
        transcript_vecs = self.encoder.encode_texts(transcript_texts) if any(transcript_mask) else None
        return caption_vecs, caption_mask, transcript_vecs, transcript_mask

    def _target_qdrant_context_for_annotation(
        self,
        annotation: Annotation,
        target_qdrant_by_clip: dict[int, dict],
        dialogue_text: str,
        transcript_source: str,
    ) -> dict | None:
        rows: list[dict] = []
        for clip_index in annotation.clip_indices or []:
            try:
                row = target_qdrant_by_clip.get(int(clip_index))
            except Exception:
                row = None
            if row:
                rows.append(row)

        target: dict = {
            "candidate_start_time": round(float(annotation.start_time), 3),
            "candidate_end_time": round(float(annotation.end_time), 3),
            "candidate_label": annotation.label,
            "clip_indices": list(annotation.clip_indices or []),
        }
        if rows:
            target["caption_text"] = self._join_context_texts(rows, "caption_text", "qwen_caption")
            target["transcript_text"] = dialogue_text.strip() or self._join_context_texts(rows, "transcript_text", limit=900)
            target["transcript_source"] = transcript_source or rows[0].get("transcript_source") or ""
            target["qwen_caption"] = self._join_context_texts(rows, "qwen_caption")
            target["qwen_reason"] = self._join_context_texts(rows, "qwen_reason")
            target["qwen_cut_advice"] = self._join_context_texts(rows, "qwen_cut_advice", limit=700)
            tags: list[str] = []
            for row in rows:
                value = row.get("qwen_tags")
                if isinstance(value, list):
                    tags.extend(str(x) for x in value if str(x).strip())
            if tags:
                target["qwen_tags"] = list(dict.fromkeys(tags))[:12]
            understandings = [row.get("understanding") for row in rows if row.get("understanding")]
            if understandings:
                target["understanding"] = understandings[:3]
            target["qdrant_clip_rows"] = rows[:5]
        elif dialogue_text.strip():
            target["transcript_text"] = dialogue_text.strip()
            target["transcript_source"] = transcript_source
        else:
            return None
        return target

    @staticmethod
    def _mk_ann(lb, c, score, evidence, z) -> Annotation:
        a = Annotation(
            label=lb,
            start_time=c.start_time,
            end_time=c.end_time,
            score=score,
            thumbnail=c.thumbnail_path or "",
            clip_indices=[c.clip_index],
            evidence_samples=list(evidence),
        )
        # 把 z-score 也带到 evidence_samples 头部以便 UI 展示 (放 a 上扩展字段)
        a.__dict__["z_score"] = z
        return a

    @staticmethod
    def _merge_evidence(cur: Annotation, evidence: list[dict]):
        seen = {(e.get("sample_id"), e.get("label")) for e in cur.evidence_samples}
        for e in evidence:
            key = (e.get("sample_id"), e.get("label"))
            if key not in seen:
                cur.evidence_samples.append(e)
                seen.add(key)

    def _assign_first_frame_thumbnails(self, meta, annotations: list[Annotation]) -> None:
        """为最终高光段保存起始帧缩略图, 供结果卡片直接展示。"""
        if not annotations or getattr(meta, "num_frames", 0) <= 0:
            return
        idxs = [
            max(0, min(int(float(a.start_time) * float(meta.fps)), int(meta.num_frames) - 1))
            for a in annotations
        ]
        try:
            from ingest.video_processor import _HAS_DECORD  # type: ignore

            pairs: list[tuple[Annotation, np.ndarray]] = []
            if _HAS_DECORD:
                from decord import VideoReader, cpu

                vr = VideoReader(meta.path, ctx=cpu(0))
                frames = vr.get_batch(idxs).asnumpy()
                pairs = list(zip(annotations, frames))
            else:
                import cv2

                frame_cache = self.processor._decode_frames_sequential(meta, idxs)
                pairs = [
                    (a, frame_cache[fi])
                    for a, fi in zip(annotations, idxs)
                    if fi in frame_cache
                ]

            for i, (a, frame_rgb) in enumerate(pairs):
                a.thumbnail = self.processor._save_thumbnail(
                    meta.video_id,
                    i,
                    frame_rgb,
                    namespace="highlight",
                    start_time=a.start_time,
                )
        except Exception as e:
            logger.warning("save annotation thumbnails failed: %s", e)

    # ---------- 本地 VLM 高光理解 ----------
    @staticmethod
    def _qdrant_context_from_evidence(
        evidence_samples: list[dict],
        target_context: dict | None = None,
    ) -> dict | None:
        matched: list[dict] = []
        for evidence in evidence_samples or []:
            ctx = evidence.get("qdrant_context")
            if not isinstance(ctx, dict):
                continue
            matched.append({
                "kb_label": evidence.get("label"),
                "kb_sample_id": evidence.get("sample_id"),
                "kb_score": evidence.get("score"),
                "source_video_id": evidence.get("source_video_id") or ctx.get("video_id"),
                "source_clip_index": ctx.get("clip_index"),
                "source_start_time": ctx.get("start_time"),
                "source_end_time": ctx.get("end_time"),
                "score_breakdown": ctx.get("score_breakdown") or evidence.get("score_breakdown"),
                "candidate_reason": ctx.get("candidate_reason") or evidence.get("candidate_reason"),
                "qwen_is_highlight": ctx.get("qwen_is_highlight"),
                "qwen_score": ctx.get("qwen_score"),
            })
        if not matched and not target_context:
            return None
        out: dict = {}
        if target_context:
            out["current_clip"] = target_context
        if matched:
            out["reference_samples"] = matched[:5]
            # Keep the old key for callers/debug tooling, but render it as reference-only.
            out["matched_samples"] = matched[:5]
            out["score_breakdown"] = matched[0].get("score_breakdown")
            out["candidate_reason"] = matched[0].get("candidate_reason")
        return out

    def _rerank_with_local_vlm(
        self,
        *,
        meta,
        annotations: list[Annotation],
        top_k: int,
        weight: float,
        max_frames: int,
        transcript: Transcript | None = None,
        target_qdrant_by_clip: dict[int, dict] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[list[Annotation], list[Annotation]]:
        """用本地 Qwen2-VL 对召回候选做结构化理解, 并显式重排序。

        融合策略:
          - base_score 是 Qdrant/CLIP/KB 召回聚合分
          - vl_score 是 Qwen2-VL 输出的原始高光分 (0-1)
          - vl_rank_signal 是把 Qwen2-VL 结论转换成排序信号后的分数
          - vl_rank_score = base_score 与 vl_rank_signal 按 weight 融合后的最终排序分
          - Qwen2-VL 明确拒绝的候选进入 rejected_annotations, 不再混入高光结果
        """
        if top_k <= 0:
            return annotations, []
        w = min(1.0, max(0.0, float(weight)))
        head = annotations[:top_k]
        tail = annotations[top_k:]
        total = len(head)
        accepted: list[Annotation] = []
        rejected: list[Annotation] = []

        prefetched_frames: dict[int, list] = {}
        try:
            if progress_callback:
                progress_callback(0, total, "预取帧")
            from ingest.video_processor import _HAS_DECORD  # type: ignore

            frame_indices_by_pos: dict[int, list[int]] = {}
            all_indices: list[int] = []
            for pos, a in enumerate(head):
                idxs = self.vl_reranker.segment_frame_indices(meta, a.start_time, a.end_time, max_frames)
                frame_indices_by_pos[pos] = idxs
                all_indices.extend(idxs)

            if all_indices:
                unique_indices = sorted(set(all_indices))
                if _HAS_DECORD:
                    from decord import VideoReader, cpu

                    vr = VideoReader(meta.path, ctx=cpu(0))
                    batch = vr.get_batch(unique_indices).asnumpy()
                    frame_cache = {fi: batch[i] for i, fi in enumerate(unique_indices)}
                else:
                    frame_cache = self.processor._decode_frames_sequential(meta, unique_indices)

                for pos, idxs in frame_indices_by_pos.items():
                    frames = [frame_cache[fi] for fi in idxs if fi in frame_cache]
                    prefetched_frames[pos] = self.vl_reranker.frames_to_images(frames)
        except Exception as e:
            logger.warning("local VLM rerank frame prefetch failed, fallback per segment: %s", e)
            prefetched_frames = {}

        for pos, a in enumerate(head):
            try:
                dialogue_text = transcript.text_for_range(a.start_time, a.end_time) if transcript else ""
                a.__dict__["dialogue"] = dialogue_text
                a.__dict__["transcript_source"] = transcript.source if transcript else ""
                target_context = self._target_qdrant_context_for_annotation(
                    a,
                    target_qdrant_by_clip or {},
                    dialogue_text,
                    transcript.source if transcript else "",
                )
                u = self.vl_reranker.analyze_segment(
                    meta,
                    a.start_time,
                    a.end_time,
                    label=a.label,
                    evidence_samples=a.evidence_samples,
                    max_frames=max_frames,
                    dialogue_text=dialogue_text,
                    qdrant_context=self._qdrant_context_from_evidence(
                        a.evidence_samples,
                        target_context=target_context,
                    ),
                    frames=prefetched_frames.get(pos),
                )
                a.understanding = u.as_dict()
                original = float(a.score)
                a.__dict__["base_score"] = original
                vl_score = float(u.score) if u.score is not None else None
                vl_rank_signal: float | None
                vl_rank_reason: str
                if u.is_highlight is False:
                    vl_rank_signal = min(vl_score if vl_score is not None else 0.0, 0.35)
                    vl_rank_reason = "qwen_rejected"
                elif u.is_highlight is True:
                    vl_rank_signal = max(vl_score if vl_score is not None else 0.0, 0.55)
                    vl_rank_reason = "qwen_confirmed_highlight"
                elif vl_score is not None:
                    vl_rank_signal = vl_score
                    vl_rank_reason = "qwen_scored_uncertain"
                else:
                    vl_rank_signal = None
                    vl_rank_reason = "qwen_uncertain"
                vl_rank_score = original if vl_rank_signal is None else (1.0 - w) * original + w * vl_rank_signal
                a.__dict__["vl_score"] = vl_score
                a.__dict__["vl_rank_signal"] = vl_rank_signal
                a.__dict__["vl_rank_score"] = vl_rank_score
                a.__dict__["vl_rank_reason"] = vl_rank_reason
                a.score = vl_rank_score
                if u.is_highlight is False:
                    a.__dict__["vl_rejected"] = True
                    rejected.append(a)
                else:
                    accepted.append(a)
            except Exception as e:
                logger.warning("local VLM rerank failed for %.2f-%.2f: %s", a.start_time, a.end_time, e)
                original = float(a.score)
                a.__dict__["base_score"] = original
                a.__dict__["vl_score"] = None
                a.__dict__["vl_rank_signal"] = None
                a.__dict__["vl_rank_score"] = original
                a.__dict__["vl_rank_reason"] = "qwen_error"
                a.understanding = {"error": f"{type(e).__name__}: {e}"}
                accepted.append(a)
            finally:
                if progress_callback:
                    progress_callback(pos + 1, total, "")

        for a in tail:
            a.__dict__["vl_unchecked"] = True
            a.__dict__.setdefault("base_score", float(a.score))
            a.__dict__.setdefault("vl_rank_score", float(a.score))
            a.__dict__.setdefault("vl_rank_reason", "qwen_unchecked")
        accepted.sort(key=lambda x: x.score, reverse=True)
        tail.sort(key=lambda x: x.score, reverse=True)
        rejected.sort(key=lambda x: x.score, reverse=True)
        return accepted + tail, rejected
