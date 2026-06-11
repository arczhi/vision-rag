from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClipHit:
    """检索结果。"""
    score: float
    video_id: str
    video_path: str
    clip_index: int
    start_time: float
    end_time: float
    frame_index: int
    timestamp: float
    thumbnail: str
    pk: int | str = 0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
