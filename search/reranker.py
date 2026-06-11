"""
多模态 Reranker：

实现两种打分策略：
1. CLIP rerank (默认、轻量) ─ 对候选片段重新加载多帧, 用 CLIP 编码所有帧求平均相似度,
   再做相邻片段聚合，得到更稳定、可解释的 top-K。
2. InternVL rerank (可选、重) ─ 加载 InternVL2 多模态模型，
   用图文 likelihood 打分。开启 use_internvl=True 时启用。

设计为"score 函数可替换"，便于后续无痛切到更强 reranker。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import cfg
from ingest.embedding import CLIPEncoder
from ingest.video_processor import VideoMeta, VideoProcessor

from .types import ClipHit

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """聚合后的片段结果。"""
    video_id: str
    video_path: str
    start_time: float
    end_time: float
    score: float
    thumbnail: str
    clip_indices: list[int] = field(default_factory=list)
    hits: list[ClipHit] = field(default_factory=list)


ScoreFn = Callable[[str | np.ndarray, list[np.ndarray]], list[float]]


class Reranker:
    def __init__(
        self,
        encoder: CLIPEncoder | None = None,
        processor: VideoProcessor | None = None,
        use_internvl: bool = False,
        rerank_cfg=cfg.reranker,
    ):
        self.encoder = encoder
        self.processor = processor or VideoProcessor()
        self.use_internvl = use_internvl
        self.cfg = rerank_cfg
        self._internvl = None
        self._internvl_tokenizer = None

    # ---------- frame loading ----------
    def _load_clip_frames(self, hit: ClipHit, video_meta_cache: dict[str, VideoMeta]) -> np.ndarray:
        """按 hit 的时间区间从原视频重新抽几帧用于精排。"""
        meta = video_meta_cache.get(hit.video_id)
        if meta is None:
            meta = self.processor.probe(hit.video_path)
            video_meta_cache[hit.video_id] = meta

        # 不复用 ingest 的滑窗，单独抽 max_frames_per_clip 帧
        n = self.processor.cfg.max_frames_per_clip
        if n <= 1:
            times = [(hit.start_time + hit.end_time) / 2]
        else:
            times = np.linspace(hit.start_time, max(hit.end_time - 1e-3, hit.start_time), n).tolist()
        idxs = [min(int(t * meta.fps), meta.num_frames - 1) for t in times]

        # 走 video_processor 内部相同的解码路径
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

        return frames

    # ---------- scoring ----------
    def _clip_score(self, query: str | np.ndarray, hits: list[ClipHit]) -> list[float]:
        """CLIP 二次打分：query × 每个 clip 的多帧均值向量。"""
        if self.encoder is None:
            self.encoder = CLIPEncoder()

        # query embedding
        if isinstance(query, str):
            q_vec = self.encoder.encode_texts([query])[0]
        else:
            q_vec = np.asarray(query, dtype=np.float32)
            n = np.linalg.norm(q_vec)
            if n > 0:
                q_vec = q_vec / n

        meta_cache: dict[str, VideoMeta] = {}
        scores: list[float] = []
        for h in hits:
            try:
                frames = self._load_clip_frames(h, meta_cache)
                v = self.encoder.encode_clip_frames(frames, pooling="mean")
                scores.append(float(np.dot(q_vec, v)))
            except Exception as e:
                logger.warning(f"rerank score failed for {h.video_id}#{h.clip_index}: {e}")
                scores.append(h.score)  # 兜底回退到粗排分
        return scores

    def _internvl_score(self, query: str, hits: list[ClipHit]) -> list[float]:
        """多模态 LLM 精排. 按 cfg.reranker.backend 路由到 transformers 或 mlx."""
        backend = getattr(self.cfg, "backend", "transformers")
        if backend == "mlx":
            return self._mlx_vl_score(query, hits)
        return self._transformers_vl_score(query, hits)

    # ---------- MLX 后端 (Apple Silicon 原生, 量化模型, 速度快) ----------
    def _mlx_vl_score(self, query: str, hits: list[ClipHit]) -> list[float]:
        """用 mlx-vlm 跑 Qwen2-VL 量化模型 (8bit/4bit/bf16)."""
        if self._internvl is None:
            try:
                from mlx_vlm import load as mlx_load, generate as mlx_generate
                from mlx_vlm.prompt_utils import apply_chat_template
            except ImportError as e:
                logger.error(f"mlx-vlm not installed: {e}; falling back to mid-frame CLIP")
                self._internvl = "DISABLED"
                return self._clip_midframe_score(query, hits)

            # MLX 模型路径优先用 modelscope 本地缓存
            from pathlib import Path
            model_id = self.cfg.model_name
            project_root = Path(__file__).resolve().parent.parent
            local_candidates = [
                project_root / "models" / "modelscope" / model_id,
                Path("/Users/alex/coding/vision-rag/models/modelscope") / model_id,
            ]
            local_path = next((str(p) for p in local_candidates if p.exists()), None)
            ref = local_path or model_id

            try:
                logger.info(f"Loading MLX VL model: {ref}")
                model, processor = mlx_load(ref)
                self._internvl = model
                self._internvl_tokenizer = processor
                self._mlx_apply_chat = apply_chat_template
                self._mlx_generate = mlx_generate
                logger.info("MLX VL ready")
            except Exception as e:
                logger.error(f"MLX VL load failed: {e}; falling back to mid-frame CLIP")
                self._internvl = "DISABLED"
                return self._clip_midframe_score(query, hits)

        if self._internvl == "DISABLED":
            return self._clip_midframe_score(query, hits)

        meta_cache: dict[str, VideoMeta] = {}
        scores: list[float] = []
        for h in hits:
            try:
                frames = self._load_clip_frames(h, meta_cache)
                if len(frames) == 0:
                    scores.append(float(h.score))
                    continue
                from PIL import Image as PILImage
                mid = PILImage.fromarray(frames[len(frames) // 2])

                # mlx_vlm prompt 组装 (含图像占位符)
                prompt_text = f"这一帧画面是否符合下述描述: \"{query}\"? 只回答 是 或 否."
                formatted = self._mlx_apply_chat(
                    self._internvl_tokenizer, self._internvl.config,
                    prompt_text, num_images=1,
                )
                # mlx_vlm.generate 高阶 API
                resp = self._mlx_generate(
                    self._internvl, self._internvl_tokenizer,
                    formatted, image=[mid], max_tokens=4, verbose=False,
                )
                resp_text = resp.text if hasattr(resp, "text") else str(resp)
                resp_lower = (resp_text or "").lower().strip()

                hit_yes = any(k in resp_lower for k in ("yes", "是", "符合"))
                hit_no = any(k in resp_lower for k in ("no", "否", "不符"))
                if hit_yes and not hit_no:
                    s = 1.0
                elif hit_no and not hit_yes:
                    s = 0.0
                else:
                    s = float(h.score)
                scores.append(0.5 * s + 0.5 * float(h.score))
            except Exception as e:
                logger.warning(f"MLX VL score failed for {h.video_id}#{h.clip_index}: {e}")
                scores.append(float(h.score))
        return scores

    # ---------- transformers 后端 (通用, 但 InternVL2 不兼容 v5.x) ----------
    def _transformers_vl_score(self, query: str, hits: list[ClipHit]) -> list[float]:
        """多模态 LLM 精排 (默认 Qwen2-VL-2B-Instruct, transformers 5.x 原生).

        实现: 取每个候选片段的中间帧 + 文本 query 进 VL 模型,
              问"这帧是否符合描述", yes/no 折成 0/1 分数,
              再与粗排分数 0.5/0.5 融合.
        """
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
        except ImportError as e:
            raise RuntimeError(f"VL reranker requires transformers/torch: {e}")

        if self._internvl is None:
            import os
            from ingest.embedding import _pick_device
            logger.info(f"Loading VL reranker: {self.cfg.model_name}")
            device = _pick_device(self.cfg.device)
            if device == "cuda":
                dtype = torch.bfloat16 if self.cfg.precision == "bf16" else torch.float16
            elif device == "mps":
                # MPS 上 Qwen2-VL bf16 偶尔有 NaN, fp16 比较稳
                dtype = torch.float16
            else:
                dtype = torch.float32

            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            kwargs = {"dtype": dtype}
            if offline:
                kwargs["local_files_only"] = True

            try:
                model = AutoModelForVision2Seq.from_pretrained(self.cfg.model_name, **kwargs)
                model = model.eval().to(device)
                proc_kwargs = {}
                if offline:
                    proc_kwargs["local_files_only"] = True
                processor = AutoProcessor.from_pretrained(self.cfg.model_name, **proc_kwargs)
                self._internvl = model
                self._internvl_tokenizer = processor  # 复用字段名
                self._internvl_device = device
                self._internvl_dtype = dtype
                logger.info(f"VL reranker ready on {device} ({dtype})")
            except Exception as e:
                logger.error(f"VL reranker load failed: {e}; falling back to mid-frame CLIP")
                self._internvl = "DISABLED"

        if self._internvl == "DISABLED":
            return self._clip_midframe_score(query, hits)

        meta_cache: dict[str, VideoMeta] = {}
        scores: list[float] = []
        for h in hits:
            try:
                frames = self._load_clip_frames(h, meta_cache)
                if len(frames) == 0:
                    scores.append(float(h.score))
                    continue
                from PIL import Image as PILImage
                mid = PILImage.fromarray(frames[len(frames) // 2])

                # Qwen2-VL chat template
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": mid},
                        {"type": "text",
                         "text": f"这一帧画面是否符合下述描述：\"{query}\"？只回答 是 或 否。"},
                    ],
                }]
                text = self._internvl_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._internvl_tokenizer(
                    text=[text], images=[mid], return_tensors="pt", padding=True,
                )
                inputs = {k: (v.to(self._internvl_device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                import torch as _torch
                with _torch.inference_mode():
                    out_ids = self._internvl.generate(
                        **inputs, max_new_tokens=8, do_sample=False,
                    )
                input_len = inputs["input_ids"].shape[1]
                gen_ids = out_ids[:, input_len:]
                resp = self._internvl_tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )[0].lower().strip()

                hit_yes = any(k in resp for k in ("yes", "是", "符合"))
                hit_no = any(k in resp for k in ("no", "否", "不符"))
                if hit_yes and not hit_no:
                    s = 1.0
                elif hit_no and not hit_yes:
                    s = 0.0
                else:
                    s = float(h.score)
                scores.append(0.5 * s + 0.5 * float(h.score))
            except Exception as e:
                logger.warning(f"VL score failed for {h.video_id}#{h.clip_index}: {e}")
                scores.append(float(h.score))
        return scores

    def _clip_midframe_score(self, query: str | np.ndarray, hits: list[ClipHit]) -> list[float]:
        """退化版精排: 用 CLIP 对每个候选的 *中间帧* 打分 (与 ingest 时多帧均值不同, 引入新信息).

        颗粒度比纯 merge 高 (引入中间帧维度), 但比 InternVL 弱 (没有 LLM reasoning).
        """
        if self.encoder is None:
            self.encoder = CLIPEncoder()

        if isinstance(query, str):
            q_vec = self.encoder.encode_texts([query])[0]
        else:
            q_vec = np.asarray(query, dtype=np.float32)
            n = np.linalg.norm(q_vec)
            if n > 0:
                q_vec = q_vec / n

        meta_cache: dict[str, VideoMeta] = {}
        scores: list[float] = []
        for h in hits:
            try:
                frames = self._load_clip_frames(h, meta_cache)
                if len(frames) == 0:
                    scores.append(float(h.score))
                    continue
                # 只编码中间一帧 (vs ingest 多帧均值, 引入"中间瞬间"信息)
                mid = frames[len(frames) // 2]
                v = self.encoder.encode_images([mid])[0]
                s = float(np.dot(q_vec, v))
                # 与粗排分数融合, 防止单帧抖动
                scores.append(0.5 * s + 0.5 * float(h.score))
            except Exception as e:
                logger.warning(f"midframe score failed for {h.video_id}#{h.clip_index}: {e}")
                scores.append(float(h.score))
        return scores

    # ---------- public API ----------
    def rerank(
        self,
        query: str | np.ndarray,
        hits: list[ClipHit],
        top_k: int | None = None,
        merge_adjacent: bool = True,
        mode: str | None = None,
    ) -> list[ClipResult]:
        """
        rerank 三档模式:
          mode='merge'    - 仅相邻片段合并 (P0, 默认, <1s, 不引入新信息但去碎片)
          mode='internvl' - VL 多模态 LLM 精排 (P1, 真精排, 慢 + 引入新信息)
          mode='clip'     - 旧的 CLIP 二次打分 (历史兼容, 冗余 + 慢, 一般不用)

        参数 mode 优先, 没传时按 self.use_internvl 推断 (兼容旧调用).

        重要: VL 这种慢模型只能给"决赛圈"少量候选打分, 不能 50 个全打。
              内部对 internvl 模式裁剪到 max(top_k * 3, 10), 不超过 hits 长度。
        """
        top_k = top_k or self.cfg.top_k
        if not hits:
            return []

        # 决定模式
        if mode is None:
            mode = "internvl" if self.use_internvl else "merge"

        # ✅ VL/CLIP 重型 rerank 只送决赛圈, 避免 N 倍延迟
        if mode in ("internvl", "clip"):
            heavy_n = min(len(hits), max(top_k * 3, 10))
            heavy_hits = hits[:heavy_n]
            tail_hits = hits[heavy_n:]
        else:
            heavy_hits, tail_hits = hits, []

        if mode == "internvl" and isinstance(query, str):
            scores = self._internvl_score(query, heavy_hits)
        elif mode == "clip":
            scores = self._clip_score(query, heavy_hits)
        else:
            # mode == "merge" 或 fallback: 直接用粗排分数, 不重新打分
            scores = [h.score for h in heavy_hits]

        scored = list(zip(heavy_hits, scores))
        # 没参与精排的尾部 hits 沿用粗排分数 (确保排序稳定)
        scored.extend((h, h.score) for h in tail_hits)
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [self._hit_to_result(h, s) for h, s in scored]

        if merge_adjacent:
            results = self._merge_adjacent(results)

        return results[:top_k]

    # ---------- merging ----------
    @staticmethod
    def _hit_to_result(h: ClipHit, score: float) -> ClipResult:
        return ClipResult(
            video_id=h.video_id,
            video_path=h.video_path,
            start_time=h.start_time,
            end_time=h.end_time,
            score=score,
            thumbnail=h.thumbnail,
            clip_indices=[h.clip_index],
            hits=[h],
        )

    @staticmethod
    def _merge_adjacent(results: list[ClipResult], gap_tol: float = 0.5) -> list[ClipResult]:
        """同一视频中相邻/重叠的片段合并为一个，score 取最大值。"""
        # 按 video_id + start_time 分桶
        by_video: dict[str, list[ClipResult]] = {}
        for r in results:
            by_video.setdefault(r.video_id, []).append(r)

        merged: list[ClipResult] = []
        for vid, items in by_video.items():
            items.sort(key=lambda x: x.start_time)
            cur = None
            for r in items:
                if cur is None:
                    cur = r
                    continue
                if r.start_time <= cur.end_time + gap_tol:
                    cur.end_time = max(cur.end_time, r.end_time)
                    cur.score = max(cur.score, r.score)
                    cur.clip_indices.extend(r.clip_indices)
                    cur.hits.extend(r.hits)
                    if not cur.thumbnail:
                        cur.thumbnail = r.thumbnail
                else:
                    merged.append(cur)
                    cur = r
            if cur is not None:
                merged.append(cur)

        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
