"""
Vision RAG 全局配置
Qdrant 多模态视频检索 + 高光知识库

支持环境变量覆盖 (容器/CI 友好):
  QDRANT_HOST  / QDRANT_PORT
  API_HOST     / API_PORT
  HF_ENDPOINT  (镜像源)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def _env(key: str, default, cast=str):
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class VideoConfig:
    """视频处理配置"""
    # 抽帧策略
    fps: float = 1.0  # 每秒抽帧数
    clip_duration: float = 5.0  # 视频片段长度（秒）
    clip_stride: float = 2.5  # 滑动窗口步长（秒）
    max_frames_per_clip: int = 8  # 每个 clip 最大帧数
    resize: tuple = (224, 224)  # 帧缩放尺寸
    thumbnail_size: tuple = (320, 180)  # 缩略图尺寸

    # ---- 切片策略 ----
    # sliding (默认)  : 原滑窗算法, 简单稳定
    # scene           : PySceneDetect 场景切换检测, clip 不跨场景
    # hybrid          : 场景切换为硬边界, 长场景内再滑窗细分
    slicing_strategy: str = field(default_factory=lambda: _env("SLICING_STRATEGY", "hybrid"))
    scene_threshold: float = 27.0  # PySceneDetect ContentDetector 阈值 (越小切越细, 默认 27)
    scene_min_len: float = 1.0     # 场景最短秒数, 防过碎
    scene_max_len: float = 8.0     # 场景最长秒数, 超长再 sliding 拆

    # ---- 抽帧策略 ----
    # uniform (默认): 时间线性等距
    # keyframe       : 基于帧间差异 (motion diff), 动作多的地方多抽, 静态少抽
    frame_sampling: str = field(default_factory=lambda: _env("FRAME_SAMPLING", "keyframe"))
    keyframe_diff_thr: float = 0.08  # motion diff 阈值 (0~1, 大于此值视为变化)


@dataclass
class EmbeddingConfig:
    """EVA-CLIP / open_clip / Chinese-CLIP 配置"""
    # 通过环境变量 EMBEDDING_BACKEND / EMBEDDING_MODEL 切换:
    #   EMBEDDING_BACKEND=open_clip (默认 demo, 容器内跑)
    #   EMBEDDING_BACKEND=cn_clip   (native venv 中文模式, MPS 加速)
    backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "open_clip"))
    model_name: str = field(default_factory=lambda: _env(
        "EMBEDDING_MODEL", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    ))
    embedding_dim: int = field(default_factory=lambda: _env("EMBEDDING_DIM", 512, int))
    # --- 备选模型 (镜像实测大小) ---
    # 英文 open_clip 系:
    #   ★ 容器 demo: laion/CLIP-ViT-B-32-laion2B-s34B-b79K        ~577 MB  dim=512
    #   有 GPU:      laion/CLIP-ViT-L-14-laion2B-s32B-b82K        ~1.59 GB dim=768
    #   质量更好:    google/siglip-so400m-patch14-384             ~3.27 GB dim=1152
    # 中文 cn_clip 系 (走 transformers, MPS 友好):
    #   ★ Mac venv: OFA-Sys/chinese-clip-vit-large-patch14-336px  ~1.51 GB dim=768
    #   小型:       OFA-Sys/chinese-clip-vit-base-patch16          ~718 MB  dim=512
    #   最强:       OFA-Sys/chinese-clip-vit-huge-patch14          ~3.57 GB dim=1024
    device: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "auto"))
    batch_size: int = 16
    normalize: bool = True
    precision: str = field(default_factory=lambda: _env("EMBEDDING_PRECISION", "fp16"))


@dataclass
class QdrantConfig:
    """Qdrant 多向量视频片段主索引配置."""
    host: str = field(default_factory=lambda: _env("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env("QDRANT_PORT", 6333, int))
    grpc_port: int = field(default_factory=lambda: _env("QDRANT_GRPC_PORT", 6334, int))
    collection_name: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "highlight_clips"))
    api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))
    https: bool = field(default_factory=lambda: _env_bool("QDRANT_HTTPS", False))
    prefer_grpc: bool = field(default_factory=lambda: _env_bool("QDRANT_PREFER_GRPC", False))
    timeout: float = field(default_factory=lambda: _env("QDRANT_TIMEOUT", 30.0, float))
    metric_type: str = field(default_factory=lambda: _env("QDRANT_METRIC", "COSINE"))
    business: str = field(default_factory=lambda: _env("QDRANT_VECTOR_BUSINESS", "highlight"))
    top_k: int = field(default_factory=lambda: _env("QDRANT_TOP_K", 60, int))
    # 文本 query 的多路召回融合权重. 当前三路都用同一个 Chinese-CLIP 空间;
    # 后续换 BGE/SigLIP 时新增 named vector 即可.
    visual_weight: float = field(default_factory=lambda: _env("QDRANT_VISUAL_WEIGHT", 0.30, float))
    caption_weight: float = field(default_factory=lambda: _env("QDRANT_CAPTION_WEIGHT", 0.45, float))
    transcript_weight: float = field(default_factory=lambda: _env("QDRANT_TRANSCRIPT_WEIGHT", 0.25, float))
    enable_vlm_caption: bool = field(default_factory=lambda: _env_bool("QDRANT_ENABLE_VLM_CAPTION", True))
    vlm_caption_max_frames: int = field(default_factory=lambda: _env("QDRANT_VLM_CAPTION_MAX_FRAMES", 2, int))


@dataclass
class RerankerConfig:
    """Reranker 配置 (默认走 CLIP 二次打分, 不额外下载模型)"""
    # Demo 默认: 关闭 InternVL, 用 CLIP encoder 二次打分 + 相邻片段聚合 (0 GB 额外下载)
    enabled: bool = True
    use_internvl: bool = False
    # 后端选择:
    #   transformers - 通用, 但 transformers 5.x 与 InternVL2 trust_remote_code 不兼容
    #   mlx          - Apple Silicon 原生量化, 速度快 + 内存低 (M 系芯片首选)
    backend: str = field(default_factory=lambda: _env("RERANKER_BACKEND", "mlx"))
    # 多模态精排模型 (use_internvl=True 时才加载):
    #   ★ MLX 后端默认: mlx-community/Qwen2-VL-2B-Instruct-8bit  ~2.5 GB  (M5 推荐, 8bit 量化)
    #     备选 4bit:    mlx-community/Qwen2-VL-2B-Instruct-4bit  ~1.5 GB
    #     bf16:         mlx-community/Qwen2-VL-2B-Instruct-bf16  ~4.5 GB
    #   transformers:   Qwen/Qwen2-VL-2B-Instruct                ~4.4 GB
    model_name: str = field(default_factory=lambda: _env(
        "RERANKER_MODEL", "mlx-community/Qwen2-VL-2B-Instruct-8bit"
    ))
    device: str = field(default_factory=lambda: _env("RERANKER_DEVICE", "auto"))
    top_k: int = 10  # rerank 后返回的结果数
    batch_size: int = 4
    precision: str = field(default_factory=lambda: _env("RERANKER_PRECISION", "bf16"))
    max_new_tokens: int = 512


@dataclass
class APIConfig:
    """API 服务配置"""
    host: str = field(default_factory=lambda: _env("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env("API_PORT", 28765, int))
    workers: int = 1
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    """主配置"""
    # 路径
    base_dir: Path = Path(__file__).parent
    data_dir: Path = Path(__file__).parent / "data"
    video_dir: Path = Path(__file__).parent / "data" / "videos"
    frame_dir: Path = Path(__file__).parent / "data" / "frames"
    thumbnail_dir: Path = Path(__file__).parent / "data" / "thumbnails"
    model_cache_dir: Path = Path(__file__).parent / "models"

    # 子配置
    video: VideoConfig = field(default_factory=VideoConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    api: APIConfig = field(default_factory=APIConfig)


# 全局单例
cfg = Config()
