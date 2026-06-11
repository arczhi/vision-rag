from .video_processor import VideoProcessor, VideoMeta, Clip
from .embedding import CLIPEncoder


def make_encoder(emb_cfg=None):
    """根据 cfg.embedding.backend 返回 open_clip 或 cn_clip 编码器, 接口一致。"""
    from config import cfg
    c = emb_cfg or cfg.embedding
    if c.backend == "cn_clip":
        from .embedding_cn import ChineseCLIPEncoder
        return ChineseCLIPEncoder(c)
    return CLIPEncoder(c)


def __getattr__(name: str):
    if name == "IngestPipeline":
        from .pipeline import IngestPipeline
        return IngestPipeline
    raise AttributeError(name)


__all__ = ["VideoProcessor", "VideoMeta", "Clip", "CLIPEncoder", "IngestPipeline", "make_encoder"]
