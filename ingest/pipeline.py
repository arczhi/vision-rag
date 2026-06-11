"""
入库 Pipeline:
   视频文件 → VideoProcessor → 切片帧 → CLIPEncoder/Qwen2-VL/ASR → Qdrant
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from config import cfg
from search.qdrant_retriever import QdrantClipRecord, QdrantRetriever

from annotate.highlight_understander import LocalHighlightUnderstander
from annotate.transcript import TranscriptProvider
from .embedding import CLIPEncoder
from .video_processor import VideoMeta, VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    video_id: str
    path: str
    num_clips: int
    skipped: bool = False
    error: str | None = None


class IngestPipeline:
    def __init__(
        self,
        processor: VideoProcessor | None = None,
        encoder: CLIPEncoder | None = None,
        retriever: QdrantRetriever | None = None,
        highlight_understander: LocalHighlightUnderstander | None = None,
        transcript_provider: TranscriptProvider | None = None,
    ):
        self.processor = processor or VideoProcessor()
        self.encoder = encoder or CLIPEncoder()
        # retriever 维度必须和 encoder 实际维度对齐——先触发 encoder 加载探测真实 dim
        if retriever is None:
            self.encoder._ensure_loaded()
            self.retriever = QdrantRetriever(embedding_dim=self.encoder.dim)
        else:
            self.retriever = retriever
        self.highlight_understander = highlight_understander
        self.transcripts = transcript_provider or TranscriptProvider()

    def ingest_video(
        self,
        video_path: str | Path,
        skip_existing: bool = True,
        custom_segments: list[tuple[float, float]] | None = None,
        should_cancel=None,
        on_progress=None,
    ) -> IngestStats:
        path = Path(video_path)
        try:
            meta = self.processor.probe(path)
        except Exception as e:
            logger.exception(f"probe failed: {path}")
            return IngestStats(video_id="", path=str(path), num_clips=0, error=str(e))

        if skip_existing and self.retriever.has_video(meta.video_id):
            logger.info(f"skip existing: {meta.video_id} ({path.name})")
            return IngestStats(video_id=meta.video_id, path=str(path), num_clips=0, skipped=True)

        records: list[QdrantClipRecord] = []
        try:
            transcript = self.transcripts.load(path)
            clip_done = 0
            for clip in self.processor.iter_clips(meta, custom_segments=custom_segments):
                if should_cancel and should_cancel():
                    return IngestStats(video_id=meta.video_id, path=str(path), num_clips=len(records), error="canceled")
                clip_done += 1
                if on_progress:
                    try:
                        on_progress(clip_done, clip.start_time, clip.end_time)
                    except Exception:
                        pass
                vec = self.encoder.encode_clip_frames(clip.frames, pooling="mean")
                # 选 clip 中间帧作为代表帧的时间戳
                mid = len(clip.frame_timestamps) // 2 if clip.frame_timestamps else 0
                ts = clip.frame_timestamps[mid] if clip.frame_timestamps else clip.start_time
                dialogue = transcript.text_for_range(clip.start_time, clip.end_time, max_chars=900) if transcript else ""
                understanding = self._understand_clip(meta, clip.start_time, clip.end_time, dialogue)
                caption_text = self._caption_text(meta, clip.start_time, clip.end_time, understanding, dialogue)
                transcript_text = dialogue.strip() or "无台词字幕"
                caption_vec = self.encoder.encode_texts([caption_text])[0]
                transcript_vec = self.encoder.encode_texts([transcript_text])[0]
                records.append(
                    QdrantClipRecord(
                        visual_vec=vec,
                        caption_vec=caption_vec,
                        transcript_vec=transcript_vec,
                        video_id=meta.video_id,
                        video_path=meta.path,
                        clip_index=clip.clip_index,
                        start_time=clip.start_time,
                        end_time=clip.end_time,
                        frame_index=mid,
                        timestamp=float(ts),
                        thumbnail=clip.thumbnail_path or "",
                        caption_text=caption_text,
                        transcript_text=dialogue.strip(),
                        transcript_source=transcript.source if transcript else "",
                        understanding=understanding,
                    )
                )
        except Exception as e:
            logger.exception(f"clip/encode failed: {path}")
            return IngestStats(video_id=meta.video_id, path=str(path), num_clips=0, error=str(e))

        if records:
            self.retriever.insert(records)
        return IngestStats(video_id=meta.video_id, path=str(path), num_clips=len(records))

    def _understand_clip(
        self,
        meta: VideoMeta,
        start_time: float,
        end_time: float,
        dialogue_text: str,
    ) -> dict:
        if not bool(getattr(cfg.qdrant, "enable_vlm_caption", True)):
            return {}
        if self.highlight_understander is None:
            self.highlight_understander = LocalHighlightUnderstander(processor=self.processor)
        try:
            u = self.highlight_understander.analyze_segment(
                meta,
                start_time,
                end_time,
                label="AI漫剧片段剧情描述与高光理解",
                evidence_samples=None,
                max_frames=max(1, int(getattr(cfg.qdrant, "vlm_caption_max_frames", 2))),
                max_tokens=160,
                dialogue_text=dialogue_text,
                qdrant_context={
                    "transcript_text": dialogue_text,
                    "transcript_source": "ingest_transcript",
                },
            )
            return u.as_dict()
        except Exception as e:
            logger.warning("Qwen2-VL clip understanding failed %.2f-%.2f: %s", start_time, end_time, e)
            return {"error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _caption_text(meta: VideoMeta, start_time: float, end_time: float, understanding: dict, dialogue_text: str) -> str:
        caption = str((understanding or {}).get("caption") or "").strip()
        reason = str((understanding or {}).get("reason") or "").strip()
        tags = understanding.get("tags") if isinstance(understanding, dict) else []
        tag_text = "、".join(str(t) for t in tags if str(t).strip()) if isinstance(tags, list) else ""
        bits: list[str] = []
        if caption:
            bits.append(caption)
        if reason:
            bits.append(reason)
        if tag_text:
            bits.append(f"标签: {tag_text}")
        if dialogue_text.strip():
            bits.append(f"台词: {dialogue_text.strip()[:500]}")
        if not bits:
            bits.append(f"视频片段 {start_time:.1f}-{end_time:.1f}s, 暂无明显剧情描述")
        return "\n".join(bits)[:1200]

    def ingest_dir(
        self,
        directory: str | Path,
        patterns: Iterable[str] = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"),
        skip_existing: bool = True,
    ) -> list[IngestStats]:
        directory = Path(directory)
        files: list[Path] = []
        for pat in patterns:
            files.extend(sorted(directory.rglob(pat)))
        # 去重
        files = sorted(set(files))

        stats: list[IngestStats] = []
        for f in tqdm(files, desc="ingest"):
            stats.append(self.ingest_video(f, skip_existing=skip_existing))
        return stats
