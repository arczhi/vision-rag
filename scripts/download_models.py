"""
预下载 demo 模型到本地缓存，避免首次启动卡在 lifespan 阶段。

用法:
    # 国内镜像 (推荐)
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py

    # 官方源
    python scripts/download_models.py

    # 同时下载 InternVL reranker
    python scripts/download_models.py --with-reranker

下载量 (镜像实测 2026-05-25):
    embedding (ViT-L/14, 默认):  ~1.59 GB
    + InternVL2-2B reranker: ~4.1 GB
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg


def download_clip(model_name: str, cache_dir: str):
    import open_clip
    ref = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
    print(f"==> embedding: {ref}")
    open_clip.create_model_and_transforms(ref, cache_dir=cache_dir)
    print(f"    done.")


def download_internvl(model_name: str, cache_dir: str):
    from huggingface_hub import snapshot_download
    print(f"==> reranker: {model_name}")
    snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    print(f"    done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-reranker", action="store_true", help="同时下载 InternVL reranker")
    p.add_argument("--clip-model", default=cfg.embedding.model_name)
    p.add_argument("--reranker-model", default=cfg.reranker.model_name)
    p.add_argument("--cache-dir", default=str(cfg.model_cache_dir))
    args = p.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co (default)")
    print(f"HF endpoint: {endpoint}")
    print(f"Cache dir  : {args.cache_dir}\n")

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    download_clip(args.clip_model, args.cache_dir)

    if args.with_reranker:
        download_internvl(args.reranker_model, args.cache_dir)

    print("\n✅ All models downloaded.")


if __name__ == "__main__":
    main()
