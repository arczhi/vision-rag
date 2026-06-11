"""
Chinese CLIP 编码器适配 (OFA-Sys 系列, 来自达摩院):
  - chinese-clip-vit-base-patch16  (~718 MB, dim=512)
  - chinese-clip-vit-large-patch14-336px  (~1.51 GB, dim=768)
  - chinese-clip-vit-huge-patch14  (~3.57 GB, dim=1024)

模型架构与 OpenAI CLIP 兼容, 但 tokenizer 用 BertTokenizer (中文友好), 必须走 transformers 加载,
不能用 open_clip。

API 与 ingest.embedding.CLIPEncoder 保持兼容: encode_texts / encode_images / encode_clip_frames / dim
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from config import cfg
from ingest.embedding import _pick_device, _pick_dtype

logger = logging.getLogger(__name__)


CN_CLIP_DIMS = {
    "OFA-Sys/chinese-clip-vit-base-patch16": 512,
    "OFA-Sys/chinese-clip-vit-large-patch14": 768,
    "OFA-Sys/chinese-clip-vit-large-patch14-336px": 768,
    "OFA-Sys/chinese-clip-vit-huge-patch14": 1024,
}


def _as_tensor(out) -> torch.Tensor:
    """transformers 新版可能返回 BaseModelOutput; 取 pooler_output / last_hidden_state / image_embeds 兜底。"""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None and isinstance(v, torch.Tensor):
            # last_hidden_state 是 (B, T, D), 需要做 CLS pool
            if attr == "last_hidden_state" and v.ndim == 3:
                return v[:, 0]
            return v
    raise RuntimeError(f"Cannot extract tensor from {type(out).__name__}")


class ChineseCLIPEncoder:
    """中文 CLIP 编码器, 用 transformers 加载, 接口同 ingest.embedding.CLIPEncoder。"""

    def __init__(self, emb_cfg=cfg.embedding, model_cache_dir=cfg.model_cache_dir):
        self.cfg = emb_cfg
        self.device = _pick_device(emb_cfg.device)
        self.dtype = _pick_dtype(emb_cfg.precision, self.device)
        self.cache_dir = str(model_cache_dir)
        self._model = None
        self._processor = None
        self._actual_dim: int | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import os
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        name = self.cfg.model_name
        logger.info(f"Loading Chinese CLIP: {name} on {self.device} ({self.dtype})")
        # 走 HF cache (HF_ENDPOINT 已由环境变量控制)
        endpoint = os.environ.get("HF_ENDPOINT")
        if endpoint:
            logger.info(f"HF mirror: {endpoint}")

        # local_files_only: 模型已下载就别再连网验证 (transformers 5.x 会偷偷连 hub.co)
        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        kwargs = {"cache_dir": self.cache_dir}
        if offline:
            kwargs["local_files_only"] = True
            logger.info("OFFLINE mode: skip remote version check")

        model = ChineseCLIPModel.from_pretrained(name, **kwargs)
        processor = ChineseCLIPProcessor.from_pretrained(name, **kwargs)

        model = model.to(self.device)
        if self.dtype != torch.float32:
            model = model.to(self.dtype)
        model.eval()

        self._actual_dim = int(model.config.projection_dim)
        self._model = model
        self._processor = processor

    @property
    def dim(self) -> int:
        if self._actual_dim is not None:
            return self._actual_dim
        return CN_CLIP_DIMS.get(self.cfg.model_name, self.cfg.embedding_dim)

    @torch.inference_mode()
    def encode_images(self, images: Iterable[Image.Image | np.ndarray]) -> np.ndarray:
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
            batch = self._processor(images=chunk, return_tensors="pt")
            pixel = batch["pixel_values"].to(self.device)
            if self.dtype != torch.float32:
                pixel = pixel.to(self.dtype)
            feats = self._model.get_image_features(pixel_values=pixel)
            feats = _as_tensor(feats)
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
        batch = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        feats = self._model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        feats = _as_tensor(feats)
        if self.cfg.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.float().cpu().numpy()

    def encode_clip_frames(self, frames: np.ndarray, pooling: str = "mean") -> np.ndarray:
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
