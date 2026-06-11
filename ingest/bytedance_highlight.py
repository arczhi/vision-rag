"""
字节火山引擎 (Volcengine MediaKit) 高光提取 API 客户端.

接口: POST https://mediakit.cn-beijing.volces.com/api/v1/tools/analyze-video-highlights
异步: 提交得 task_id, 轮询 GET /api/v1/tools/get-task-info?task_id=...

返回的 highlight_info 与 vision-rag 内部 LLMSegment 对齐:
  start_time, end_time → seconds
  description           → description
  score                 → 分数
  ocr                   → 字幕
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://mediakit.cn-beijing.volces.com/api/v1/tools"


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key) or default


def _require_token(api_token: str | None) -> str:
    token = api_token or _env("BYTEDANCE_MEDIAKIT_TOKEN")
    if not token:
        raise RuntimeError(
            "缺失 BYTEDANCE_MEDIAKIT_TOKEN; 请在 .env 中配置 (参考 .env.example)"
        )
    return token


@dataclass
class BytedanceSegment:
    start_time: float
    end_time: float
    score: float
    description: str = ""
    ocr: str = ""
    source_video_index: int = 0

    def as_dict(self) -> dict:
        return {
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration": round(self.end_time - self.start_time, 3),
            "score": self.score,
            "description": self.description,
            "ocr": self.ocr,
            "source_video_index": self.source_video_index,
        }


@dataclass
class BytedanceResult:
    task_id: str = ""
    request_id: str = ""
    duration: float = 0.0
    segments: list[BytedanceSegment] = field(default_factory=list)
    raw_result: dict = field(default_factory=dict)
    submit_ms: int = 0
    poll_ms: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "request_id": self.request_id,
            "duration": self.duration,
            "segments": [s.as_dict() for s in self.segments],
            "submit_ms": self.submit_ms,
            "poll_ms": self.poll_ms,
            "error": self.error,
        }


def submit_task(video_url: str, model: str = "Miniseries", api_token: str | None = None,
                api_base: str | None = None, timeout: int = 60) -> tuple[str, str]:
    """提交字节高光提取任务. 返回 (task_id, request_id)."""
    import requests
    token = _require_token(api_token)
    base = (api_base or _env("BYTEDANCE_MEDIAKIT_BASE", DEFAULT_API_BASE)).rstrip("/")
    mode = "StorylineCuts" if model == "Miniseries" else "HighlightExtract"
    body = {
        "video_urls": [video_url],
        "model": model,
        "mode": mode,
    }
    logger.info(f"[Bytedance] submit model={model} mode={mode} url={video_url}")
    r = requests.post(
        f"{base}/analyze-video-highlights",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    j = r.json()
    if not j.get("success"):
        err = j.get("error", {})
        raise RuntimeError(f"submit failed: {err.get('code')}: {err.get('message')}")
    return j["task_id"], j.get("request_id", "")


def query_task(task_id: str, api_token: str | None = None,
               api_base: str | None = None, timeout: int = 30) -> dict:
    """查询字节任务状态. 返回原始 JSON."""
    import requests
    token = _require_token(api_token)
    base = (api_base or _env("BYTEDANCE_MEDIAKIT_BASE", DEFAULT_API_BASE)).rstrip("/")
    r = requests.get(
        f"{base}/get-task-info",
        params={"task_id": task_id},
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def poll_until_done(task_id: str, *, interval: float = 5.0, timeout_s: float = 1500.0,
                    on_progress=None) -> dict:
    """轮询字节任务直到 done. 返回 result 字段."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        j = query_task(task_id)
        # 火山引擎返回里 status 字段: 任务状态字段名 status / state, 不同 API 不同
        # 通过 result 是否有内容判断
        status = j.get("status") or j.get("state") or ""
        result = j.get("result")
        if on_progress:
            try:
                on_progress(j)
            except Exception:
                pass
        if result and isinstance(result, dict) and result.get("highlight_info") is not None:
            return j
        # 也兼容 status 显式字符串
        if str(status).lower() in ("failed", "error"):
            raise RuntimeError(f"bytedance task failed: {j}")
        time.sleep(interval)
    raise RuntimeError(f"bytedance task {task_id} timeout (>{timeout_s}s)")


def infer_highlights_from_url(video_url: str, *, model: str = "Miniseries",
                              poll_interval: float = 5.0, poll_timeout_s: float = 1500.0,
                              on_progress=None) -> BytedanceResult:
    """端到端: 提交 + 轮询 + 解析 segments."""
    result = BytedanceResult()
    t0 = time.time()
    try:
        task_id, req_id = submit_task(video_url, model=model)
        result.task_id = task_id
        result.request_id = req_id
        result.submit_ms = int((time.time() - t0) * 1000)
        logger.info(f"[Bytedance] task submitted: {task_id}")

        t1 = time.time()
        full = poll_until_done(task_id, interval=poll_interval,
                                timeout_s=poll_timeout_s, on_progress=on_progress)
        result.poll_ms = int((time.time() - t1) * 1000)

        r = full.get("result") or {}
        result.raw_result = r
        result.duration = float(r.get("duration") or 0.0)
        for h in r.get("highlight_info") or []:
            try:
                result.segments.append(BytedanceSegment(
                    start_time=float(h.get("start_time") or 0),
                    end_time=float(h.get("end_time") or 0),
                    score=float(h.get("score") or 0),
                    description=h.get("description") or "",
                    ocr=h.get("ocr") or "",
                    source_video_index=int(h.get("source_video_index") or 0),
                ))
            except Exception as e:
                logger.warning(f"parse highlight failed: {e}")
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.exception("[Bytedance] infer failed")
    return result
