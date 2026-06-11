"""
EVA-CLIP / SigLIP / OpenCLIP 等 CLIP 系列编码器封装：
- 图像 → D 维向量 (D 由 model 决定, 默认 demo 模型 ViT-B/32 = 512)
- 文本 → D 维向量
- 自动 device 选择 (cuda/mps/cpu)
- 多帧聚合 (mean pooling)

镜像支持: 设置环境变量 HF_ENDPOINT=https://hf-mirror.com 走国内镜像。
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from config import cfg

logger = logging.getLogger(__name__)


def _pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(precision: str, device: str) -> torch.dtype:
    if device in ("cpu", "mps"):
        # MPS 对 fp16 部分算子不稳, 安全起见走 fp32
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


class CLIPEncoder:
    """
    用 open_clip 加载 HuggingFace 上的 CLIP 系列模型。

    支持的命名格式:
      "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"   → hf-hub:laion/...
      "hf-hub:laion/CLIP-ViT-B-32-..."          → 原样
      "ViT-B-32"                                → open_clip 内置名 (需配合 pretrained)
    """

    def __init__(self, emb_cfg=cfg.embedding, model_cache_dir=cfg.model_cache_dir):
        self.cfg = emb_cfg
        self.device = _pick_device(emb_cfg.device)
        self.dtype = _pick_dtype(emb_cfg.precision, self.device)
        self.cache_dir = str(model_cache_dir)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._actual_dim: int | None = None

    # ------- lazy load -------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import open_clip

        # HF_ENDPOINT 同步给 huggingface_hub (open_clip 用 hf_hub_download)
        endpoint = os.environ.get("HF_ENDPOINT")
        if endpoint:
            os.environ.setdefault("HF_HUB_ENDPOINT", endpoint)

        name = self.cfg.model_name
        if "/" in name and not name.startswith("hf-hub:"):
            ref = f"hf-hub:{name}"
        else:
            ref = name

        logger.info(f"Loading CLIP model: {ref} on {self.device} ({self.dtype})")
        if endpoint:
            logger.info(f"Using HF mirror: {endpoint}")

        model, _, preprocess = open_clip.create_model_and_transforms(
            ref, cache_dir=self.cache_dir
        )
        tokenizer = open_clip.get_tokenizer(ref)

        model = model.to(self.device)
        if self.dtype != torch.float32:
            model = model.to(self.dtype)
        model.eval()

        # 用一次 dummy 推理探测真实输出维度，校验 config.embedding_dim
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.dtype)
            try:
                feat = model.encode_image(dummy)
                self._actual_dim = int(feat.shape[-1])
                if self._actual_dim != self.cfg.embedding_dim:
                    logger.warning(
                        f"embedding_dim mismatch: config={self.cfg.embedding_dim} "
                        f"actual={self._actual_dim}; will use actual"
                    )
            except Exception as e:
                logger.warning(f"dim probe failed: {e}; falling back to config={self.cfg.embedding_dim}")
                self._actual_dim = self.cfg.embedding_dim

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer

    # ------- public API -------
    @property
    def dim(self) -> int:
        if self._actual_dim is not None:
            return self._actual_dim
        return self.cfg.embedding_dim

    @torch.inference_mode()
    def encode_images(self, images: Iterable[Image.Image | np.ndarray]) -> np.ndarray:
        """编码 N 张图像 → (N, D) float32."""
        self._ensure_loaded()
        pil_imgs: list[Image.Image] = []
        for im in images:
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im)
            pil_imgs.append(im.convert("RGB"))
        if not pil_imgs:
            return np.zeros((0, self.dim), dtype=np.float32)

        bs = self.cfg.batch_size
        outs = []
        for i in range(0, len(pil_imgs), bs):
            chunk = pil_imgs[i : i + bs]
            tensors = torch.stack([self._preprocess(x) for x in chunk]).to(self.device)
            if self.dtype != torch.float32:
                tensors = tensors.to(self.dtype)
            feats = self._model.encode_image(tensors)
            if self.cfg.normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outs.append(feats.float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    @torch.inference_mode()
    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        self._ensure_loaded()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        tokens = self._tokenizer(texts).to(self.device)
        feats = self._model.encode_text(tokens)
        if self.cfg.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.float().cpu().numpy()

    def encode_clip_frames(self, frames: np.ndarray, pooling: str = "mean") -> np.ndarray:
        """
        多帧 → 单向量。
        frames: (N, H, W, 3) uint8 RGB
        return: (D,) float32
        """
        if frames is None or len(frames) == 0:
            return np.zeros((self.dim,), dtype=np.float32)
        feats = self.encode_images([f for f in frames])
        if pooling == "mean":
            v = feats.mean(axis=0)
        elif pooling == "max":
            v = feats.max(axis=0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        if self.cfg.normalize:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v.astype(np.float32)
