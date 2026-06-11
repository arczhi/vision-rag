"""
视频处理: decord/PyAV/OpenCV 解码 → 滑动窗口切片 → 均匀抽帧 → 缩略图
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import decord
    from decord import VideoReader, cpu
    decord.bridge.set_bridge("native")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

try:
    import av
    _HAS_PYAV = True
except ImportError:
    av = None
    _HAS_PYAV = False

import cv2

from config import cfg


@dataclass
class VideoMeta:
    video_id: str
    path: str
    duration: float
    fps: float
    width: int
    height: int
    num_frames: int


@dataclass
class Clip:
    video_id: str
    clip_index: int
    start_time: float
    end_time: float
    frame_timestamps: list[float] = field(default_factory=list)
    frames: np.ndarray | None = None  # (N, H, W, 3) uint8 RGB
    thumbnail_path: str | None = None


def _video_id(path: Path) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


class VideoProcessor:
    def __init__(self, video_cfg=cfg.video, thumbnail_dir: Path = cfg.thumbnail_dir):
        self.cfg = video_cfg
        self.thumb_dir = Path(thumbnail_dir)
        self.thumb_dir.mkdir(parents=True, exist_ok=True)

    def probe(self, video_path: str | Path) -> VideoMeta:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        if _HAS_DECORD:
            vr = VideoReader(str(path), ctx=cpu(0))
            num_frames = len(vr)
            fps = float(vr.get_avg_fps()) or 25.0
            h, w, _ = vr[0].asnumpy().shape
            duration = num_frames / fps
        else:
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open video: {path}")
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = num_frames / fps
            finally:
                cap.release()

        return VideoMeta(
            video_id=_video_id(path),
            path=str(path.resolve()),
            duration=duration,
            fps=fps,
            width=w,
            height=h,
            num_frames=num_frames,
        )

    def iter_clips(
        self,
        meta: VideoMeta,
        save_thumbnails: bool = True,
        custom_segments: list[tuple[float, float]] | None = None,
    ) -> Iterator[Clip]:
        """切片 + 抽帧入口. 按 cfg.slicing_strategy 路由到三档算法.

        三档:
          sliding (原算法): 滑动窗口 5s/2.5s, 短视频自适应
          scene            : PySceneDetect 场景切换, clip 不跨场景
          hybrid (默认)    : scene 切硬边界 + 长场景内 sliding 细分

        custom_segments 不为空时, 跳过自动切片, 直接用用户给的 [(start, end), ...] 切.
            适用: 用户已有 LLM 推理出的高光时间戳 / 自定义 JSON 时间段.
            会按 [start, end] 在视频范围内截断, 过滤无效区间.
        """
        # 1) 自定义 segments 优先级最高
        if custom_segments:
            valid_segs: list[tuple[float, float]] = []
            for s, e in custom_segments:
                s = max(0.0, float(s))
                e = min(meta.duration, float(e))
                if e - s >= 0.1:
                    valid_segs.append((s, e))
            if valid_segs:
                logger.info(f"using {len(valid_segs)} custom segments for {meta.path}")
                yield from self._iter_clips_from_segments(meta, valid_segs, save_thumbnails)
                return
            logger.warning(f"custom_segments empty after validation; fallback to {self.cfg.slicing_strategy}")

        # 2) 走原有策略路由
        strategy = getattr(self.cfg, "slicing_strategy", "sliding")

        if strategy in ("scene", "hybrid"):
            try:
                segments = self._detect_scenes(meta)  # list[(start, end)]
            except Exception as e:
                logger.warning(f"scene detect failed ({e}); fallback to sliding")
                segments = None
        else:
            segments = None

        if segments is None:
            # sliding 兜底
            yield from self._iter_clips_sliding(meta, save_thumbnails)
            return

        if strategy == "hybrid":
            # 长场景内再 sliding
            sub_segs: list[tuple[float, float]] = []
            for s, e in segments:
                if e - s > self.cfg.scene_max_len:
                    # 在 [s, e] 内做 sliding
                    sub_segs.extend(self._sliding_within(s, e))
                else:
                    sub_segs.append((s, e))
            segments = sub_segs

        yield from self._iter_clips_from_segments(meta, segments, save_thumbnails)

    # ---------- 场景切换检测 (PySceneDetect ContentDetector) ----------
    def _detect_scenes(self, meta: VideoMeta) -> list[tuple[float, float]]:
        """用 PySceneDetect 探测场景边界, 返回 [(start_sec, end_sec), ...]."""
        from scenedetect import SceneManager, open_video, ContentDetector
        video = open_video(meta.path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=self.cfg.scene_threshold))
        sm.detect_scenes(video)
        scenes = sm.get_scene_list()
        # FrameTimecode → seconds
        segs = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
        if not segs:
            # 没检出任何切换, 整段当一个场景
            segs = [(0.0, meta.duration)]
        # 过短场景合并
        merged: list[tuple[float, float]] = []
        for s, e in segs:
            if merged and (e - s) < self.cfg.scene_min_len:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        logger.info(f"scenes detected: {len(merged)} (raw {len(segs)}) for {meta.path}")
        return merged

    def _sliding_within(self, start: float, end: float) -> list[tuple[float, float]]:
        clip_dur = self.cfg.clip_duration
        stride = self.cfg.clip_stride
        out = []
        t = start
        while t < end:
            seg_end = min(t + clip_dur, end)
            out.append((t, seg_end))
            if seg_end >= end:
                break
            t += stride
        return out or [(start, end)]

    # ---------- 滑动窗口 (原算法, 抽出来保留) ----------
    def _iter_clips_sliding(self, meta: VideoMeta, save_thumbnails: bool) -> Iterator[Clip]:
        cfg_dur = self.cfg.clip_duration
        cfg_stride = self.cfg.clip_stride

        default_seg_count = max(1, int(meta.duration / cfg_stride))
        if default_seg_count < 6:
            stride = max(0.5, meta.duration / 6.0)
            clip_dur = min(cfg_dur, max(1.5, stride * 2))
        else:
            clip_dur = cfg_dur
            stride = cfg_stride

        if meta.duration < 0.1:
            return

        starts: list[float] = []
        t = 0.0
        while t < meta.duration:
            starts.append(t)
            if t + clip_dur >= meta.duration:
                break
            t += stride
        if not starts:
            starts = [0.0]

        segments = [(s, min(s + clip_dur, meta.duration)) for s in starts]
        yield from self._iter_clips_from_segments(meta, segments, save_thumbnails)

    # ---------- 共用: 从 segment 列表生成 Clip (含抽帧) ----------
    def _iter_clips_from_segments(
        self, meta: VideoMeta, segments: list[tuple[float, float]], save_thumbnails: bool,
    ) -> Iterator[Clip]:
        max_frames = self.cfg.max_frames_per_clip
        resize = self.cfg.resize

        if _HAS_DECORD:
            vr = VideoReader(str(meta.path), ctx=cpu(0))
            frame_cache = None
        else:
            vr = None
            sampled_segments: list[tuple[int, float, float, list[int], list[float]]] = []
            all_frame_indices: list[int] = []
            if getattr(self.cfg, "frame_sampling", "uniform") == "keyframe":
                logger.warning("decord unavailable; fallback decoder uses uniform sampling to avoid slow random seek")
            for idx, (start, end) in enumerate(segments):
                frame_indices, times = self._sample_uniform(meta, start, end, max_frames)
                sampled_segments.append((idx, start, end, frame_indices, times))
                all_frame_indices.extend(frame_indices)
            frame_cache = self._decode_frames_sequential(meta, all_frame_indices)

        try:
            if _HAS_DECORD:
                iterator = []
                for idx, (start, end) in enumerate(segments):
                    if getattr(self.cfg, "frame_sampling", "uniform") == "keyframe":
                        frame_indices, times = self._sample_keyframes(meta, start, end, max_frames, vr, None)
                    else:
                        frame_indices, times = self._sample_uniform(meta, start, end, max_frames)
                    iterator.append((idx, start, end, frame_indices, times))
            else:
                iterator = sampled_segments

            for idx, start, end, frame_indices, times in iterator:
                # ---- 抽帧 ----
                if not frame_indices:
                    continue

                if _HAS_DECORD:
                    batch = vr.get_batch(frame_indices).asnumpy()
                else:
                    batch = [frame_cache[fi] for fi in frame_indices if fi in frame_cache]
                    batch = np.stack(batch) if batch else np.zeros((0, meta.height, meta.width, 3), dtype=np.uint8)

                resized = np.stack([
                    np.array(Image.fromarray(f).resize(resize, Image.BILINEAR))
                    for f in batch
                ]) if len(batch) else batch

                clip = Clip(
                    video_id=meta.video_id,
                    clip_index=idx,
                    start_time=float(start),
                    end_time=float(end),
                    frame_timestamps=times,
                    frames=resized,
                )

                if save_thumbnails and len(batch):
                    clip.thumbnail_path = self._save_thumbnail(meta.video_id, idx, batch[0])

                yield clip
        finally:
            pass

    def _decode_frames_sequential(self, meta: VideoMeta, frame_indices: list[int]) -> dict[int, np.ndarray]:
        """Decode requested RGB frames with one forward pass.

        Random seeking with CAP_PROP_POS_FRAMES can hang for some H264 MP4s on macOS.
        Prefer PyAV when available, then fall back to OpenCV.
        """
        wanted = sorted({int(fi) for fi in frame_indices if 0 <= int(fi) < meta.num_frames})
        if not wanted:
            return {}
        if _HAS_PYAV:
            try:
                return self._decode_pyav_frames_sequential(meta, wanted)
            except Exception as e:
                logger.warning("pyav sequential decode failed, fallback cv2: %s", e)
        return self._decode_cv2_frames_sequential(meta, wanted)

    def _decode_pyav_frames_sequential(self, meta: VideoMeta, frame_indices: list[int]) -> dict[int, np.ndarray]:
        wanted = sorted({int(fi) for fi in frame_indices if 0 <= int(fi) < meta.num_frames})
        if not wanted:
            return {}
        if not _HAS_PYAV or av is None:
            raise RuntimeError("PyAV is not available")

        out: dict[int, np.ndarray] = {}
        target_pos = 0
        last_target = wanted[-1]
        with av.open(str(meta.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            current = 0
            for frame in container.decode(stream):
                while target_pos < len(wanted) and wanted[target_pos] < current:
                    target_pos += 1
                if target_pos >= len(wanted) or current > last_target:
                    break
                if current == wanted[target_pos]:
                    out[current] = frame.to_ndarray(format="rgb24")
                    while target_pos < len(wanted) and wanted[target_pos] == current:
                        target_pos += 1
                current += 1
        if len(out) < len(wanted):
            logger.warning(
                "pyav sequential decode got %s/%s requested frames for %s",
                len(out), len(wanted), meta.path,
            )
        return out

    def _decode_cv2_frames_sequential(self, meta: VideoMeta, frame_indices: list[int]) -> dict[int, np.ndarray]:
        """Decode requested RGB frames with one forward OpenCV pass."""
        wanted = sorted({int(fi) for fi in frame_indices if 0 <= int(fi) < meta.num_frames})
        if not wanted:
            return {}

        cap = cv2.VideoCapture(str(meta.path))
        out: dict[int, np.ndarray] = {}
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {meta.path}")
            target_pos = 0
            current = 0
            last_target = wanted[-1]
            while target_pos < len(wanted) and current <= last_target:
                ok, frame = cap.read()
                if not ok:
                    break
                while target_pos < len(wanted) and wanted[target_pos] < current:
                    target_pos += 1
                if target_pos < len(wanted) and current == wanted[target_pos]:
                    out[current] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    while target_pos < len(wanted) and wanted[target_pos] == current:
                        target_pos += 1
                current += 1
        finally:
            cap.release()
        if len(out) < len(wanted):
            logger.warning(
                "cv2 sequential decode got %s/%s requested frames for %s",
                len(out), len(wanted), meta.path,
            )
        return out

    @staticmethod
    def _sample_uniform(meta: VideoMeta, start: float, end: float, max_frames: int):
        if max_frames == 1:
            times = [start + (end - start) / 2]
        else:
            times = np.linspace(start, max(end - 1e-3, start), max_frames).tolist()
        idxs = [min(int(t * meta.fps), meta.num_frames - 1) for t in times]
        return idxs, times

    def _sample_keyframes(self, meta: VideoMeta, start: float, end: float, max_frames: int, vr, cap):
        """基于帧间差异选关键帧 (motion diff).

        策略:
          1. 在 [start, end] 内以 2x max_frames 密度做候选采样
          2. 对相邻候选帧算 abs diff (灰度 + downsample 8x), 得到 motion 序列
          3. 起始帧 + 累计差异跨越阈值的帧 → 关键帧, 至少 3 帧
          4. 若关键帧 < max_frames, 用 uniform 补足
        """
        seg_dur = max(end - start, 1e-3)
        # 候选采样: 4x 密度, 但不少于 8 个候选
        n_candidates = max(8, max_frames * 4)
        cand_times = np.linspace(start, max(end - 1e-3, start), n_candidates).tolist()
        cand_idxs = [min(int(t * meta.fps), meta.num_frames - 1) for t in cand_times]

        # 解码候选帧用于算 motion (用 84x84 灰度 减 numpy 计算量)
        try:
            if vr is not None:
                cand_batch = vr.get_batch(cand_idxs).asnumpy()  # (N, H, W, 3)
            else:
                cand_batch = []
                for fi in cand_idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ok, f = cap.read()
                    if ok:
                        cand_batch.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                if not cand_batch:
                    return self._sample_uniform(meta, start, end, max_frames)
                cand_batch = np.stack(cand_batch)
        except Exception:
            return self._sample_uniform(meta, start, end, max_frames)

        # 灰度 + 缩到 84x84
        small = np.empty((len(cand_batch), 84, 84), dtype=np.uint8)
        for i, f in enumerate(cand_batch):
            g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
            small[i] = cv2.resize(g, (84, 84))

        # 帧间差异
        diffs = np.zeros(len(small), dtype=np.float32)
        for i in range(1, len(small)):
            d = np.abs(small[i].astype(np.int16) - small[i - 1].astype(np.int16)).mean() / 255.0
            diffs[i] = float(d)

        # 选关键帧: 累计差异超阈值
        thr = self.cfg.keyframe_diff_thr
        selected: list[int] = [0]  # 必含起始帧
        cumulative = 0.0
        for i in range(1, len(diffs)):
            cumulative += diffs[i]
            if cumulative >= thr:
                selected.append(i)
                cumulative = 0.0
        # 末帧也加入
        if (len(small) - 1) not in selected:
            selected.append(len(small) - 1)

        # 控制数量: 多了均匀降采样
        if len(selected) > max_frames:
            sub = np.linspace(0, len(selected) - 1, max_frames).astype(int)
            selected = [selected[i] for i in sub]
        elif len(selected) < max_frames:
            # 不够就 uniform 补
            extra_n = max_frames - len(selected)
            extra_idx = np.linspace(0, len(diffs) - 1, extra_n + 2).astype(int)[1:-1]
            selected = sorted(set(list(selected) + extra_idx.tolist()))

        # 转成全局帧号 + 时间
        out_idxs = [cand_idxs[i] for i in selected]
        out_times = [cand_times[i] for i in selected]
        return out_idxs, out_times

    def _save_thumbnail(
        self,
        video_id: str,
        clip_index: int,
        frame_rgb: np.ndarray,
        *,
        namespace: str = "",
        start_time: float | None = None,
    ) -> str:
        if namespace:
            safe_ns = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in namespace)
            if start_time is None:
                out = self.thumb_dir / f"{video_id}_{safe_ns}_{clip_index:05d}.jpg"
            else:
                start_ms = max(0, int(round(float(start_time) * 1000)))
                out = self.thumb_dir / f"{video_id}_{safe_ns}_{clip_index:05d}_{start_ms:08d}.jpg"
        else:
            out = self.thumb_dir / f"{video_id}_{clip_index:05d}.jpg"
        img = Image.fromarray(frame_rgb).resize(self.cfg.thumbnail_size, Image.BILINEAR)
        img.save(out, "JPEG", quality=85)
        return str(out)
