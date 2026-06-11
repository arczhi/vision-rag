"""
FastAPI 服务入口
启动: python -m api.server  或  uvicorn api.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import io
import os
import secrets
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import cfg
from ingest import make_encoder
from ingest.embedding import CLIPEncoder
from ingest.pipeline import IngestPipeline
from ingest.video_processor import VideoProcessor
from search.pipeline import SearchPipeline
from search.reranker import ClipResult, Reranker
from search.qdrant_retriever import QdrantRetriever
from annotate.knowledge_base import KBRetriever, KnowledgeBase
from annotate.annotator import Annotation, Annotator
from annotate.highlight_understander import LocalHighlightUnderstander
from .tasks import TaskManager

logger = logging.getLogger(__name__)


class State:
    encoder: CLIPEncoder
    retriever: QdrantRetriever
    processor: VideoProcessor
    ingest: IngestPipeline
    search: SearchPipeline
    kb: KnowledgeBase
    annotator: Annotator
    highlight_understander: LocalHighlightUnderstander
    tasks: TaskManager
    embed_lock: threading.Lock
    temp_videos: dict[str, Path]


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.encoder = make_encoder()
    # 触发 encoder 加载, 用真实 dim 初始化 retriever
    state.encoder._ensure_loaded()
    state.retriever = QdrantRetriever(embedding_dim=state.encoder.dim)
    state.processor = VideoProcessor()
    state.highlight_understander = LocalHighlightUnderstander(processor=state.processor)
    state.ingest = IngestPipeline(
        processor=state.processor,
        encoder=state.encoder,
        retriever=state.retriever,
        highlight_understander=state.highlight_understander,
    )
    reranker = Reranker(encoder=state.encoder, processor=state.processor)
    state.search = SearchPipeline(
        encoder=state.encoder, retriever=state.retriever, reranker=reranker
    )
    # 高光知识库 + 标注器 (复用 encoder/processor, 不重复加载模型)
    kb_retriever = KBRetriever(embedding_dim=state.encoder.dim)
    state.kb = KnowledgeBase(
        encoder=state.encoder, processor=state.processor, retriever=kb_retriever
    )
    state.annotator = Annotator(
        encoder=state.encoder,
        processor=state.processor,
        kb_retriever=kb_retriever,
        vl_reranker=state.highlight_understander,
        clip_retriever=state.retriever,
    )
    state.tasks = TaskManager(max_workers=8)
    # 两阶段锁: LLM 阶段 (TOS+qwen) 并发跑, embedding/Qdrant 阶段串行 (MPS GPU 独占)
    state.embed_lock = threading.Lock()
    state.temp_videos = {}
    # 初始化 SQLite (LLM 高光持久化)
    try:
        from annotate.llm_highlight_store import init_db
        init_db()
    except Exception as e:
        logger.warning(f"init llm_highlight_store failed: {e}")
    try:
        from annotate.feedback_store import init_db as init_feedback_db
        init_feedback_db()
    except Exception as e:
        logger.warning(f"init feedback_store failed: {e}")
    # 提前连接 Qdrant，让启动失败立即可见
    try:
        state.retriever.ensure_collection()
        state.kb.retriever.ensure_collection()
    except Exception as e:
        logger.exception(f"Qdrant init failed: {e}")
    yield
    state.retriever.close()


app = FastAPI(title="Vision RAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源：web UI 与缩略图
_web_dir = cfg.base_dir / "web"
if _web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(_web_dir), html=True), name="web")

cfg.thumbnail_dir.mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(cfg.thumbnail_dir)), name="thumbnails")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/web/")


# ----------------- schemas -----------------
class SearchTextRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = None
    coarse_k: int | None = None
    rerank: bool = True
    rerank_mode: str | None = None  # 'merge' | 'internvl' | 'clip'; None=按 cfg 默认


class SearchHit(BaseModel):
    video_id: str
    video_path: str
    start_time: float
    end_time: float
    score: float
    thumbnail: str
    clip_indices: list[int]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    timings: dict[str, float] = Field(default_factory=dict)


def _to_hit(r: ClipResult) -> SearchHit:
    return SearchHit(
        video_id=r.video_id,
        video_path=r.video_path,
        start_time=r.start_time,
        end_time=r.end_time,
        score=r.score,
        thumbnail=r.thumbnail,
        clip_indices=r.clip_indices,
    )


# ----------------- endpoints -----------------
@app.get("/health")
def health() -> dict[str, Any]:
    info: dict[str, Any] = {"status": "ok"}
    try:
        state.retriever.ensure_collection()
        info["qdrant"] = "connected"
        info["qdrant_collection"] = cfg.qdrant.collection_name
        info["qdrant_vectors"] = state.retriever.vector_names.as_dict()
    except Exception as e:
        info["status"] = "degraded"
        info["qdrant"] = f"error: {e}"
    return info


@app.get("/videos")
def list_videos() -> dict[str, Any]:
    try:
        return {"videos": state.retriever.list_videos()}
    except Exception as e:
        raise HTTPException(500, f"list failed: {e}")


@app.delete("/videos/{video_id}")
def delete_video(video_id: str) -> dict[str, Any]:
    try:
        state.retriever.delete_video(video_id)
        return {"deleted": video_id}
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")


@app.get("/videos/{video_id}/clips")
def list_video_clips(video_id: str) -> dict[str, Any]:
    """列出某视频在 Qdrant 里的所有 clip (供高光标注页直接选中导入样例用)."""
    try:
        rows = state.retriever.list_clips(video_id)
        # thumbnail 是绝对路径, 转成 web 可访问的相对路径
        for r in rows:
            tp = r.get("thumbnail") or ""
            if tp:
                r["thumbnail_url"] = f"/thumbnails/{Path(tp).name}"
        return {"video_id": video_id, "clips": rows}
    except Exception as e:
        raise HTTPException(500, f"list clips failed: {e}")


_MIME = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".webm": "video/webm", ".avi": "video/x-msvideo",
}


def _resolve_video_path(video_id: str) -> Path:
    """根据 video_id 查 Qdrant 拿到 video_path, 校验落在 video_dir 里防路径穿越。"""
    video_path = state.retriever.get_video_path(video_id)
    if not video_path:
        raise HTTPException(404, f"video_id not found: {video_id}")
    raw = Path(video_path).resolve()
    # 路径安全: 必须在 video_dir 下 (或挂载的等价路径)
    allowed_roots = [cfg.video_dir.resolve()]
    if not any(str(raw).startswith(str(r)) for r in allowed_roots):
        # 容器内 host path 可能不一致, 尝试用文件名 fallback
        candidate = (cfg.video_dir / raw.name).resolve()
        if candidate.exists():
            raw = candidate
        else:
            raise HTTPException(403, f"video path outside allowed dirs")
    if not raw.exists():
        raise HTTPException(404, f"video file missing on disk: {raw.name}")
    return raw


def _resolve_temp_video_path(temp_id: str) -> Path:
    """Resolve a temporary uploaded video even after an API restart."""
    path = state.temp_videos.get(temp_id)
    if path is not None and path.exists():
        return path

    tmp_dir = cfg.video_dir / "_tmp_annotate"
    for ext in [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"]:
        candidate = (tmp_dir / f"{temp_id}{ext}").resolve()
        if candidate.exists():
            state.temp_videos[temp_id] = candidate
            return candidate

    matches = sorted(tmp_dir.glob(f"{temp_id}.*")) if tmp_dir.exists() else []
    for candidate in matches:
        if candidate.is_file():
            resolved = candidate.resolve()
            state.temp_videos[temp_id] = resolved
            return resolved

    raise HTTPException(404, f"temp video not found: {temp_id}")


@app.get("/videos/{video_id}/stream")
def stream_video(video_id: str, request: Request):
    """带 Range 支持的视频流, 让 <video> 可以 seek 到 start_time。"""
    path = _resolve_video_path(video_id)
    return _stream_video_path(path, request)


@app.get("/videos/{video_id}/thumbnail")
def video_thumbnail(video_id: str, t: float = 0.0):
    """返回指定时间点的视频帧缩略图, 用作高光卡片封面兜底。"""
    path = _resolve_video_path(video_id)
    return _frame_thumbnail_response(path, t)


@app.get("/temp-videos/{temp_id}/stream")
def stream_temp_video(temp_id: str, request: Request):
    """临时上传视频流, 仅用于本次标注结果预览, 不进入主视频库。"""
    path = _resolve_temp_video_path(temp_id)
    return _stream_video_path(path, request)


@app.get("/temp-videos/{temp_id}/thumbnail")
def temp_video_thumbnail(temp_id: str, t: float = 0.0):
    """返回临时视频指定时间点的缩略图。"""
    path = _resolve_temp_video_path(temp_id)
    return _frame_thumbnail_response(path, t)


def _frame_thumbnail_response(path: Path, t: float):
    try:
        from PIL import Image
        from ingest.video_processor import _HAS_DECORD

        if _HAS_DECORD:
            from decord import VideoReader, cpu

            vr = VideoReader(str(path), ctx=cpu(0))
            fps = float(vr.get_avg_fps()) or 25.0
            frame_idx = max(0, min(int(float(t) * fps), len(vr) - 1))
            frame_rgb = vr[frame_idx].asnumpy()
        else:
            import cv2

            cap = cv2.VideoCapture(str(path))
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                frame_idx = max(0, min(int(float(t) * fps), total - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("frame decode failed")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()

        img = Image.fromarray(frame_rgb).resize(cfg.video.thumbnail_size, Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except Exception as e:
        raise HTTPException(500, f"thumbnail decode failed: {e}")


def _stream_video_path(path: Path, request: Request):
    file_size = path.stat().st_size
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    range_header = request.headers.get("range") or request.headers.get("Range")

    if not range_header:
        # 全量返回 (浏览器首次拉 metadata 也走这里)
        def iter_full():
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(
            iter_full(),
            media_type=mime,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "public, max-age=3600",
            },
        )

    # 解析 Range: bytes=START-END
    try:
        units, _, rng = range_header.strip().partition("=")
        if units.lower() != "bytes":
            raise ValueError("only bytes ranges")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start < 0:
            raise ValueError("invalid range")
    except Exception:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    length = end - start + 1

    def iter_range():
        with path.open("rb") as f:
            f.seek(start)
            remain = length
            while remain > 0:
                chunk = f.read(min(1024 * 1024, remain))
                if not chunk:
                    break
                remain -= len(chunk)
                yield chunk

    return StreamingResponse(
        iter_range(),
        status_code=206,
        media_type=mime,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/ingest")
async def ingest_video(
    file: UploadFile = File(...),
    skip_existing: bool = Form(True),
    segments_json: str | None = Form(None),
    use_llm_segments: bool = Form(False),
    use_local_vlm_segments: bool = Form(False),
) -> dict[str, Any]:
    """上传视频并入库 (异步). 立即返回 task_id, 前端轮询 GET /tasks/{id}.

    segments_json: 可选, JSON 字符串. 两种格式都接受:
      - [{"start": 0.0, "end": 5.0}, ...]  时间戳为秒
      - [{"start": "00:00:00", "end": "00:00:05"}, ...]  HH:MM:SS
      - {"highlights": [{"timestamp": "00:00:05", "duration": 3, "description": "..."}, ...]}
        (兼容 LLM 推理出的 highlights 数组, 每条用 timestamp + duration 或与下一条之差)
    传了就跳过自动 hybrid 切片, 严格按时间戳切.

    use_llm_segments=true 时, 后端会自动跑: TOS 上传 → qwen3.5-plus 推理 → 解析时间戳 → 用作 segments.
        与 segments_json 互斥, 优先 use_llm_segments.
    use_local_vlm_segments=true 时, 后端用本地 Qwen2-VL 滑窗推理高光时间戳, 再按时间戳切片入库.
        与 use_llm_segments 同时开启时本地优先, 不调用云端.
    """
    cfg.video_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name
    target = cfg.video_dir / safe_name

    # 防同名覆盖
    if target.exists():
        with tempfile.NamedTemporaryFile(
            prefix=target.stem + "_", suffix=target.suffix, dir=cfg.video_dir, delete=False
        ) as tmp:
            target = Path(tmp.name)

    # 上传仍然 sync (HTTP body 必须读完); 入库本身放后台
    try:
        with target.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    file_size = target.stat().st_size
    file_name = target.name

    # 解析自定义 segments
    custom_segments: list[tuple[float, float]] | None = None
    if segments_json:
        try:
            custom_segments = _parse_segments_json(segments_json)
        except Exception as e:
            raise HTTPException(400, f"invalid segments_json: {e}")

    effective_use_llm_segments = bool(use_llm_segments and not use_local_vlm_segments)

    def _run(task, video_path_str: str, skip: bool, segs, llm_mode: bool, local_vlm_mode: bool):
        # 1) LLM 推理路径 (优先级高于 segs)
        if llm_mode and not segs:
            from ingest.llm_segments import infer_segments_for_local_video
            task.message = "上传 TOS + qwen3.5-plus 推理"
            task.progress = 0.05
            try:
                meta = state.processor.probe(video_path_str)
                duration = meta.duration
            except Exception:
                duration = None
            llm_result = infer_segments_for_local_video(
                video_path_str, fps=2, video_duration=duration,
            )
            task.extra["llm_video_url"] = llm_result.video_url
            task.extra["llm_timings"] = llm_result.timings
            task.extra["llm_error"] = llm_result.error
            task.extra["llm_segments_count"] = len(llm_result.segments)
            task.extra["llm_segment_stats"] = llm_result.segment_stats
            # 持久化 LLM 段 (供后续 auto_kb 归纳读取)
            task.extra["llm_segments"] = [s.as_dict() for s in llm_result.segments]
            # 可观测性: 把 raw_response 头尾各 800 字符 + parsed JSON 摘要写进 extra
            raw = llm_result.raw_response or ""
            if len(raw) > 1600:
                task.extra["llm_raw_preview"] = raw[:800] + "\n...[truncated]...\n" + raw[-800:]
            else:
                task.extra["llm_raw_preview"] = raw
            # 把段写进 SQLite (持久化, 供 /auto-kb/run 跨任务归纳)
            # 注意: video_id 此时还没确定 (要等 ingest 算 hash), 先暂存到闭包变量
            _llm_segments_for_persist = llm_result.segments
            _llm_video_url_for_persist = llm_result.video_url
            if llm_result.parsed_json:
                task.extra["llm_parsed_summary"] = {
                    "highlights": len(llm_result.parsed_json.get("highlights") or []),
                    "hook": len(llm_result.parsed_json.get("hook") or []),
                }
            if llm_result.error:
                task.message = f"LLM 失败, 走默认切片: {llm_result.error[:80]}"
                segs = None
            elif not llm_result.segments:
                task.message = "LLM 无 segments, 走默认切片"
                segs = None
            else:
                segs = [(s.start, s.end) for s in llm_result.segments]
                task.message = f"LLM 推理出 {len(segs)} 段"
                task.progress = 0.4

        task.message = (task.message or "") + " · 等待 GPU 锁"
        task.progress = max(task.progress, 0.45)
        task.extra["filename"] = file_name
        task.extra["size_bytes"] = file_size
        if segs:
            task.extra["custom_segments"] = len(segs)

        # ★ 两阶段锁: TOS+qwen 已在锁外并发完成, 这里串行做 decode+embedding+Qdrant 写入
        with state.embed_lock:
            if local_vlm_mode and not segs:
                segs = _local_vlm_custom_segments(task, video_path_str, enabled=True, merge_gap=1.5)
                if task.status == "canceled":
                    return None
            task.message = (task.message or "").replace("等待 GPU 锁", "解码 + embedding")
            task.progress = max(task.progress, 0.55)
            stats = state.ingest.ingest_video(video_path_str, skip_existing=skip, custom_segments=segs)
        task.progress = 1.0
        task.message = "完成" if not stats.error else f"失败: {stats.error}"
        if stats.error:
            raise RuntimeError(stats.error)
        # 写 SQLite: LLM 推理段在此持久化 (拿到 stats.video_id)
        if llm_mode and segs:
            try:
                from annotate.llm_highlight_store import upsert_highlight
                # llm_result 闭包里仍可访问
                for i, s in enumerate(llm_result.segments):
                    upsert_highlight(
                        video_id=stats.video_id,
                        clip_index=i,
                        start_time=s.start, end_time=s.end,
                        description=s.description, label=s.label,
                        thumbnail="",
                        source_video_url=llm_result.video_url,
                    )
                task.extra["llm_persisted"] = len(llm_result.segments)
            except Exception as e:
                logger.warning(f"persist llm segments failed: {e}")
        return {
            "video_id": stats.video_id,
            "path": stats.path,
            "num_clips": stats.num_clips,
            "skipped": stats.skipped,
            "error": stats.error,
            "filename": file_name,
            "used_custom_segments": bool(segs),
            "used_llm_segments": bool(llm_mode and segs),
            "used_local_vlm_segments": bool(task.extra.get("local_vlm_segments_count")),
            "local_vlm_segments_count": task.extra.get("local_vlm_segments_count", 0),
            "local_vlm_segment_ms": task.extra.get("local_vlm_segment_ms", 0),
            "llm_segments_count": task.extra.get("llm_segments_count", 0),
        }

    task = state.tasks.submit(
        "ingest", _run, str(target), skip_existing, custom_segments, effective_use_llm_segments, use_local_vlm_segments,
    )
    return {
        "task_id": task.task_id,
        "status": task.status,
        "kind": "ingest",
        "filename": file_name,
        "size_bytes": file_size,
        "custom_segments_count": len(custom_segments) if custom_segments else 0,
        "use_llm_segments": effective_use_llm_segments,
        "use_local_vlm_segments": use_local_vlm_segments,
    }


def _parse_segments_json(s: str) -> list[tuple[float, float]]:
    """解析两种格式的 segments JSON.

    A) [{"start": 0, "end": 5}, ...]
    B) {"highlights": [{"timestamp": "00:00:05", "duration": 3, ...}, ...]}
       duration 缺省时用与下一条 timestamp 之差; 最后一条缺省给 5s.
    时间戳支持 float(秒) 或 "HH:MM:SS" / "MM:SS".
    """
    import json as _json
    raw = _json.loads(s)

    def _to_sec(v) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if not isinstance(v, str):
            raise ValueError(f"unsupported time type: {type(v).__name__}")
        v = v.strip()
        parts = v.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
        return float(v)

    # 格式 B: 含 highlights / hook 的 LLM 风格
    if isinstance(raw, dict) and "highlights" in raw:
        items = raw["highlights"]
        out: list[tuple[float, float]] = []
        starts = [_to_sec(it["timestamp"]) for it in items]
        for i, it in enumerate(items):
            s_t = starts[i]
            if "duration" in it:
                e_t = s_t + float(it["duration"])
            elif "end" in it:
                e_t = _to_sec(it["end"])
            elif i + 1 < len(starts):
                e_t = min(starts[i + 1], s_t + 8.0)
            else:
                e_t = s_t + 5.0
            if e_t > s_t:
                out.append((s_t, e_t))
        return out

    # 格式 A: 直接的 segment 数组
    if isinstance(raw, list):
        out: list[tuple[float, float]] = []
        for it in raw:
            s_t = _to_sec(it.get("start", it.get("start_time", 0)))
            e_t = _to_sec(it.get("end", it.get("end_time", s_t + 5.0)))
            if e_t > s_t:
                out.append((s_t, e_t))
        return out

    raise ValueError("segments_json 必须是 array 或带 highlights key 的 object")


@app.post("/search/text", response_model=None)
def search_text(req: SearchTextRequest) -> dict[str, Any]:
    """文本搜索 (异步). 立即返回 task_id, 前端轮询 GET /tasks/{id}."""
    def _run(task, req_obj: SearchTextRequest):
        task.message = "encode + 检索"
        task.progress = 0.1
        outcome = state.search.search_text(
            req_obj.query, top_k=req_obj.top_k, coarse_k=req_obj.coarse_k,
            rerank=req_obj.rerank, rerank_mode=req_obj.rerank_mode,
        )
        task.progress = 1.0
        task.message = "完成"
        return {
            "query": req_obj.query,
            "results": [_to_hit(r).model_dump() for r in outcome.results],
            "timings": outcome.timings.as_dict(),
        }

    task = state.tasks.submit("search_text", _run, req)
    task.extra["query"] = req.query
    task.extra["rerank_mode"] = req.rerank_mode or ("internvl" if req.rerank else "merge")
    return {"task_id": task.task_id, "status": task.status, "kind": "search_text"}


@app.post("/search/image", response_model=None)
async def search_image(
    file: UploadFile = File(...),
    top_k: int | None = Form(None),
    coarse_k: int | None = Form(None),
    rerank: bool = Form(True),
    rerank_mode: str | None = Form(None),
) -> dict[str, Any]:
    """图片搜索 (异步). 立即返回 task_id, 前端轮询."""
    # 上传同步 (HTTP body 必须读完), 检索本身放后台
    suffix = Path(file.filename or "img").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    await file.close()
    fname = file.filename or "image"

    def _run(task, image_path: str, top_k_, coarse_k_, rerank_, rerank_mode_):
        task.message = "encode + 检索"
        task.progress = 0.1
        try:
            outcome = state.search.search_image(
                image_path, top_k=top_k_, coarse_k=coarse_k_,
                rerank=rerank_, rerank_mode=rerank_mode_,
            )
            task.progress = 1.0
            task.message = "完成"
            return {
                "query": f"<image:{fname}>",
                "results": [_to_hit(r).model_dump() for r in outcome.results],
                "timings": outcome.timings.as_dict(),
            }
        finally:
            try:
                Path(image_path).unlink(missing_ok=True)
            except Exception:
                pass

    task = state.tasks.submit("search_image", _run, tmp_path, top_k, coarse_k, rerank, rerank_mode)
    task.extra["filename"] = fname
    return {"task_id": task.task_id, "status": task.status, "kind": "search_image"}


# ----------------- highlight annotation (P0) -----------------

class AddTextSampleRequest(BaseModel):
    kb_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    note: str = ""
    sample_id: str | None = None


class AddClipSampleRequest(BaseModel):
    kb_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    video_id: str = Field(..., min_length=1)
    start_time: float
    end_time: float
    note: str = ""
    sample_id: str | None = None


class AnnotateRequest(BaseModel):
    kb_id: str = Field(..., min_length=1)
    video_id: str = Field(..., min_length=1)
    threshold: float = 0.05         # 绝对最低分 (兜底, 配合 z_threshold)
    top_k_per_clip: int = 5
    merge_gap: float = 1.5
    min_duration: float = 1.0
    max_duration: float = 60.0
    labels: list[str] | None = None
    normalize: bool = True          # 是否启用 z-score 归一化
    z_threshold: float = 0.75       # 相对阈值: 超过 KB 内均值多少 σ
    vl_rerank: bool = False         # 是否启用本地 Qwen2-VL 高光理解与重排序
    vl_top_k: int = 8               # 兼容旧前端字段；当前开启 VLM 时会检查全部候选
    vl_weight: float = 0.35         # VLM 分数融合权重
    vl_max_frames: int = 1          # 每个候选片段送入 VLM 的代表帧数
    use_llm_segments: bool = False  # 是否显式调用云端 qwen3.5-plus 做切分
    use_local_vlm_segments: bool = False  # 是否显式调用本地 Qwen2-VL 全片理解切分


class HighlightFeedbackRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    video_name: str = ""
    kb_id: str = ""
    source: str = "manual"
    original_label: str = ""
    final_label: str = ""
    start_time: float
    end_time: float
    corrected_start: float | None = None
    corrected_end: float | None = None
    model_score: float | None = None
    user_score: float | None = None
    accepted: bool
    reason: str = ""
    tags: list[str] = Field(default_factory=list)
    understanding: dict[str, Any] = Field(default_factory=dict)


class RebuildLocalVlmIngestRequest(BaseModel):
    include_tmp: bool = False
    limit: int | None = None
    clear_video_embeddings: bool = False
    clear_llm_highlights: bool = False
    strict_local_vlm: bool = True


class RebuildQdrantIngestRequest(BaseModel):
    include_tmp: bool = False
    limit: int | None = None
    clear_video_embeddings: bool = True
    skip_existing: bool = False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _local_vlm_custom_segments(
    task,
    video_path_str: str,
    req_obj: AnnotateRequest | None = None,
    *,
    enabled: bool | None = None,
    merge_gap: float | None = None,
    progress_start: float = 0.08,
    progress_end: float = 0.65,
    message_prefix: str = "",
) -> list[tuple[float, float]] | None:
    """开启本地 Qwen2-VL 理解切分时, 先用滑窗 VLM 生成候选切分段。"""
    if enabled is None:
        enabled = bool(req_obj and req_obj.use_local_vlm_segments)
    if not enabled:
        return None

    started = time.perf_counter()
    prefix = f"{message_prefix} · " if message_prefix else ""
    task.message = f"{prefix}本地 Qwen2-VL 理解切分"
    progress_start = min(0.99, max(0.0, float(progress_start)))
    progress_end = min(0.99, max(progress_start, float(progress_end)))
    task.progress = max(float(getattr(task, "progress", 0.0) or 0.0), progress_start)
    try:
        meta = state.processor.probe(video_path_str)
        transcript = state.annotator.transcripts.load(video_path_str)
        task.extra["transcript_source"] = transcript.source
        task.extra["transcript_segments_count"] = len(transcript.segments)

        window_s = max(2.0, _env_float("LOCAL_VLM_SEGMENT_WINDOW_S", 6.0))
        stride_s = max(1.0, _env_float("LOCAL_VLM_SEGMENT_STRIDE_S", 3.0))
        max_windows = _env_int("LOCAL_VLM_SEGMENT_MAX_WINDOWS", 0)  # 0 = 不限制, 不再做最多 8 窗采样
        min_score = min(1.0, max(0.0, _env_float("LOCAL_VLM_SEGMENT_MIN_SCORE", 0.50)))
        max_frames = max(1, min(_env_int("LOCAL_VLM_SEGMENT_MAX_FRAMES", 1), 2))
        gap = float(merge_gap if merge_gap is not None else (req_obj.merge_gap if req_obj else 1.5))
        estimated_windows = 1 if meta.duration <= window_s else int(max(1, ((meta.duration - window_s) // stride_s) + 2))
        task.extra["local_vlm_window_s"] = window_s
        task.extra["local_vlm_stride_s"] = stride_s
        task.extra["local_vlm_max_windows"] = max_windows
        task.extra["local_vlm_min_score"] = min_score
        task.extra["local_vlm_estimated_windows"] = estimated_windows

        def _on_progress(i: int, total: int, start: float, end: float):
            task.extra["local_vlm_windows_done"] = i
            task.extra["local_vlm_windows_total"] = total
            task.message = f"{prefix}本地 Qwen2-VL 理解切分 {i}/{total} ({start:.1f}-{end:.1f}s)"
            frac = i / max(total, 1)
            task.progress = max(
                float(getattr(task, "progress", 0.0) or 0.0),
                min(progress_end, progress_start + (progress_end - progress_start) * frac),
            )

        segments = state.highlight_understander.segment_video(
            meta,
            window_s=window_s,
            stride_s=stride_s,
            max_windows=max_windows,
            max_frames=max_frames,
            min_score=min_score,
            merge_gap=gap,
            should_cancel=lambda: task.status == "canceled",
            dialogue_lookup=transcript.text_for_range,
            on_progress=_on_progress,
        )
        if task.status == "canceled":
            return None
        elapsed_ms = (time.perf_counter() - started) * 1000
        task.extra["local_vlm_segment_ms"] = round(elapsed_ms, 2)
        task.extra["local_vlm_segments_count"] = len(segments)
        task.extra["local_vlm_segments"] = segments
        if not segments:
            task.message = f"{prefix}本地 Qwen2-VL 理解切分未切出高光, 回退 hybrid 切分"
            return None
        custom_segments = [
            (float(s["start"]), float(s["end"]))
            for s in segments
            if float(s["end"]) > float(s["start"])
        ]
        task.message = f"{prefix}本地 Qwen2-VL 理解切分出 {len(custom_segments)} 段 · 嵌入"
        task.progress = max(float(getattr(task, "progress", 0.0) or 0.0), progress_end)
        return custom_segments or None
    except Exception as e:
        elapsed_ms = (time.perf_counter() - started) * 1000
        task.extra["local_vlm_segment_ms"] = round(elapsed_ms, 2)
        task.extra["local_vlm_segment_error"] = f"{type(e).__name__}: {e}"
        task.message = f"{prefix}本地 Qwen2-VL 理解切分失败, 回退 hybrid 切分"
        logger.warning("local VLM segmentation failed: %s", e)
        return None


def _local_vlm_label(understanding: dict[str, Any]) -> str:
    tags = understanding.get("tags") if isinstance(understanding, dict) else []
    if not isinstance(tags, list):
        tags = []
    text = " ".join([str(t) for t in tags] + [
        str(understanding.get("reason") or ""),
        str(understanding.get("cut_advice") or ""),
    ])
    if "亲情" in text or "催泪" in text or "感动" in text:
        return "亲情催泪型"
    if "危机" in text or "危险" in text or "紧迫" in text:
        return "危机紧迫型"
    if "反击" in text or "逆袭" in text:
        return "绝境反击"
    if "悬念" in text or "谜" in text:
        return "悬念型"
    if "脑洞" in text or "猎奇" in text or "设定" in text:
        return "脑洞猎奇型"
    if "冲突" in text or "争吵" in text or "对抗" in text:
        return "冲突顶点型"
    if "萌娃" in text or "治愈" in text:
        return "萌娃治愈型"
    return "本地Qwen高光"


def _local_vlm_fallback_annotations(task, video_path_str: str) -> list[Annotation]:
    """KB/z-score 没命中时, 保留本地 Qwen2-VL 已确认的高光段。"""
    segments = task.extra.get("local_vlm_segments") or []
    if not segments:
        return []
    try:
        meta = state.processor.probe(video_path_str)
        transcript = state.annotator.transcripts.load(video_path_str)
    except Exception as e:
        logger.warning("local VLM fallback probe/transcript failed: %s", e)
        return []

    annotations: list[Annotation] = []
    for i, seg in enumerate(segments):
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            if end <= start:
                continue
            understanding = seg.get("understanding") or {}
            if isinstance(understanding, dict) and understanding.get("is_highlight") is False:
                continue
            score = float(seg.get("score") or (understanding.get("score") if isinstance(understanding, dict) else 0.6) or 0.6)
            score = min(1.0, max(0.0, score))
            reason = understanding.get("reason", "") if isinstance(understanding, dict) else ""
            ann = Annotation(
                label=_local_vlm_label(understanding if isinstance(understanding, dict) else {}),
                start_time=start,
                end_time=end,
                score=score,
                clip_indices=[i],
                evidence_samples=[{
                    "label": "本地 Qwen2-VL",
                    "score": score,
                    "note": str(reason)[:120],
                }],
                understanding=understanding if isinstance(understanding, dict) else {},
            )
            ann.__dict__["dialogue"] = transcript.text_for_range(start, end)
            ann.__dict__["transcript_source"] = transcript.source
            annotations.append(ann)
        except Exception as e:
            logger.warning("skip local VLM fallback segment: %s", e)

    if annotations:
        state.annotator._assign_first_frame_thumbnails(meta, annotations)
    return annotations


def _annotation_overlap_ratio(a: Annotation, b: Annotation) -> float:
    start = max(float(a.start_time), float(b.start_time))
    end = min(float(a.end_time), float(b.end_time))
    overlap = max(0.0, end - start)
    if overlap <= 0:
        return 0.0
    a_dur = max(1e-6, float(a.end_time) - float(a.start_time))
    b_dur = max(1e-6, float(b.end_time) - float(b.start_time))
    return overlap / max(1e-6, min(a_dur, b_dur))


def _annotation_dicts_with_local_vlm_fallback(outcome, task, video_path_str: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    annotations = list(outcome.annotations)
    rejected = list(getattr(outcome, "rejected_annotations", []) or [])
    used_fallback = False
    if task.extra.get("local_vlm_segments_count", 0):
        local_annotations = _local_vlm_fallback_annotations(task, video_path_str)
        added: list[Annotation] = []
        for ann in local_annotations:
            if any(_annotation_overlap_ratio(ann, existing) >= 0.5 for existing in annotations):
                continue
            added.append(ann)
        if added:
            annotations.extend(added)
            annotations.sort(key=lambda a: float(a.score), reverse=True)
            used_fallback = True
        task.extra["local_vlm_fallback_candidates"] = len(local_annotations)
        task.extra["local_vlm_fallback_added"] = len(added)
    return [a.as_dict() for a in annotations], [a.as_dict() for a in rejected], used_fallback


@app.get("/kb")
def kb_list() -> dict[str, Any]:
    """列出所有知识库 + 每个 KB 内的样例数 / label 分布。"""
    try:
        kbs = state.kb.retriever.list_kbs()
        return {
            "kbs": [
                {"kb_id": k.kb_id, "sample_count": k.sample_count, "label_counts": k.label_counts}
                for k in kbs
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"kb list failed: {e}")


@app.get("/kb/{kb_id}/samples")
def kb_samples(kb_id: str) -> dict[str, Any]:
    try:
        rows = state.kb.retriever.list_samples(kb_id)
        return {"kb_id": kb_id, "samples": rows}
    except Exception as e:
        raise HTTPException(500, f"kb samples failed: {e}")


@app.delete("/kb/{kb_id}")
def kb_delete(kb_id: str) -> dict[str, Any]:
    try:
        state.kb.retriever.delete_kb(kb_id)
        return {"deleted_kb": kb_id}
    except Exception as e:
        raise HTTPException(500, f"kb delete failed: {e}")


@app.delete("/kb/{kb_id}/samples/{sample_id}")
def kb_delete_sample(kb_id: str, sample_id: str) -> dict[str, Any]:
    try:
        state.kb.retriever.delete_sample(kb_id, sample_id)
        return {"deleted": sample_id}
    except Exception as e:
        raise HTTPException(500, f"sample delete failed: {e}")


@app.post("/kb/sample/text")
def kb_add_text(req: AddTextSampleRequest) -> dict[str, Any]:
    """用一段文字描述当作零样本高光定义 (CLIP 文本编码)。"""
    try:
        s = state.kb.add_text_sample(
            req.kb_id, req.label, req.text, sample_id=req.sample_id, note=req.note
        )
        return {"ok": True, "sample_id": s.sample_id, "label": s.label}
    except Exception as e:
        raise HTTPException(500, f"add text sample failed: {e}")


@app.post("/kb/sample/clip")
def kb_add_clip(req: AddClipSampleRequest) -> dict[str, Any]:
    """把一个已入库视频的指定时间段作为高光样例 (异步)."""
    try:
        video_path = _resolve_video_path(req.video_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"resolve video failed: {e}")

    # 复用现有缩略图: 找最接近的 clip 缩略图 (这步轻量, 同步算就行)
    thumb = ""
    try:
        rows = state.retriever.list_clips(req.video_id)
        mid = (req.start_time + req.end_time) / 2
        rows = [r for r in rows if r.get("thumbnail")]
        if rows:
            rows.sort(key=lambda r: abs(((r["start_time"] + r["end_time"]) / 2) - mid))
            thumb = rows[0]["thumbnail"]
    except Exception:
        pass

    def _run(task, video_path_str: str, req_obj: AddClipSampleRequest, thumb_path: str):
        task.message = "解码 + encode"
        task.progress = 0.1
        s = state.kb.add_clip_sample(
            kb_id=req_obj.kb_id,
            label=req_obj.label,
            video_path=video_path_str,
            start_time=req_obj.start_time,
            end_time=req_obj.end_time,
            sample_id=req_obj.sample_id,
            source_video_id=req_obj.video_id,
            thumbnail=thumb_path,
            note=req_obj.note,
        )
        task.progress = 1.0
        task.message = "完成"
        return {
            "ok": True,
            "sample_id": s.sample_id,
            "label": s.label,
            "source_video_id": req_obj.video_id,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "thumbnail": s.thumbnail,
        }

    task = state.tasks.submit("kb_clip_sample", _run, str(video_path), req, thumb)
    task.extra["label"] = req.label
    task.extra["video_id"] = req.video_id
    task.extra["start_time"] = req.start_time
    task.extra["end_time"] = req.end_time
    return {"task_id": task.task_id, "status": task.status, "kind": "kb_clip_sample"}


@app.post("/kb/sample/image")
async def kb_add_image(
    kb_id: str = Form(...),
    label: str = Form(...),
    note: str = Form(""),
    sample_id: str | None = Form(None),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "img").suffix or ".jpg", delete=False
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        await file.close()
        s = state.kb.add_image_sample(kb_id, label, tmp_path, sample_id=sample_id, note=note)
        return {"ok": True, "sample_id": s.sample_id, "label": s.label}
    except Exception as e:
        raise HTTPException(500, f"add image sample failed: {e}")
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/annotate")
def annotate_video(req: AnnotateRequest) -> dict[str, Any]:
    """对视频跑高光标注 (异步). 立即返回 task_id, 前端轮询 GET /tasks/{id}."""
    try:
        video_path = _resolve_video_path(req.video_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"resolve video failed: {e}")

    def _run(task, video_path_str: str, req_obj: AnnotateRequest):
        task.message = "等待 GPU 锁"
        task.progress = 0.05
        custom_segments: list[tuple[float, float]] | None = None
        extra_segments: list[tuple[float, float]] | None = None
        cloud_segments_used = False
        if req_obj.use_llm_segments:
            from ingest.llm_segments import infer_segments_for_local_video
            task.message = "上传 TOS + qwen3.5-plus 推理高光"
            task.progress = 0.05
            try:
                meta = state.processor.probe(video_path_str)
                duration = meta.duration
            except Exception:
                duration = None
            llm_result = infer_segments_for_local_video(
                video_path_str, fps=2, video_duration=duration,
            )
            task.extra["llm_video_url"] = llm_result.video_url
            task.extra["llm_timings"] = llm_result.timings
            task.extra["llm_error"] = llm_result.error
            task.extra["llm_segments_count"] = len(llm_result.segments)
            task.extra["llm_segment_stats"] = llm_result.segment_stats
            task.extra["llm_segments"] = [s.as_dict() for s in llm_result.segments]
            raw = llm_result.raw_response or ""
            task.extra["llm_raw_preview"] = (raw[:800] + "\n...[truncated]...\n" + raw[-800:]) if len(raw) > 1600 else raw
            if llm_result.error:
                task.message = f"云端 LLM 失败, 回退本地/默认切分: {llm_result.error[:80]}"
            elif not llm_result.segments:
                task.message = "云端 LLM 无 segments, 回退本地/默认切分"
            else:
                custom_segments = [(s.start, s.end) for s in llm_result.segments]
                cloud_segments_used = True
                task.message = f"云端 LLM 推理出 {len(custom_segments)} 段 · 等待 GPU 锁"
                task.progress = 0.35
        with state.embed_lock:
            if req_obj.use_local_vlm_segments and not custom_segments:
                extra_segments = _local_vlm_custom_segments(task, video_path_str, req_obj)
                if extra_segments:
                    task.extra["local_vlm_segments_mode"] = "extra_candidates"
            if task.status == "canceled":
                return None
            def _progress(message: str, progress: float, extra: dict | None = None):
                task.message = message
                task.progress = max(float(task.progress or 0.0), min(float(progress), 0.99))
                if extra:
                    task.extra.update(extra)

            task.message = "解码 + embedding"
            task.progress = max(float(task.progress or 0.0), 0.1)
            outcome = state.annotator.annotate(
                video_path=video_path_str,
                kb_id=req_obj.kb_id,
                threshold=req_obj.threshold,
                top_k_per_clip=req_obj.top_k_per_clip,
                merge_gap=req_obj.merge_gap,
                min_duration=req_obj.min_duration,
                max_duration=req_obj.max_duration,
                labels=req_obj.labels,
                normalize=req_obj.normalize,
                z_threshold=req_obj.z_threshold,
                vl_rerank=req_obj.vl_rerank,
                vl_top_k=req_obj.vl_top_k,
                vl_weight=req_obj.vl_weight,
                vl_max_frames=req_obj.vl_max_frames,
                custom_segments=custom_segments,
                extra_segments=extra_segments,
                progress_callback=_progress,
            )
        annotations, rejected_annotations, used_local_vlm_fallback = _annotation_dicts_with_local_vlm_fallback(outcome, task, video_path_str)
        task.extra["rejected_annotations_count"] = len(rejected_annotations)
        task.progress = 1.0
        task.message = "完成"
        return {
            "video_id": req_obj.video_id,
            "kb_id": req_obj.kb_id,
            "kb_sample_count": outcome.kb_sample_count,
            "labels_in_kb": outcome.labels_in_kb,
            "annotations": annotations,
            "rejected_annotations": rejected_annotations,
            "timings": outcome.timings.as_dict(),
            "used_local_vlm_segments": bool(task.extra.get("local_vlm_segments_count")),
            "local_vlm_segments_count": task.extra.get("local_vlm_segments_count", 0),
            "local_vlm_segment_ms": task.extra.get("local_vlm_segment_ms", 0),
            "local_vlm_segments_mode": task.extra.get("local_vlm_segments_mode", ""),
            "used_local_vlm_fallback": used_local_vlm_fallback,
            "local_vlm_fallback_added": task.extra.get("local_vlm_fallback_added", 0),
            "used_llm_segments": cloud_segments_used,
            "llm_segments_count": task.extra.get("llm_segments_count", 0),
        }

    task = state.tasks.submit("annotate", _run, str(video_path), req)
    return {"task_id": task.task_id, "status": task.status, "kind": "annotate"}


@app.post("/annotate/upload")
async def annotate_uploaded_video(
    kb_id: str = Form(...),
    threshold: float = Form(0.05),
    top_k_per_clip: int = Form(5),
    merge_gap: float = Form(1.5),
    min_duration: float = Form(1.0),
    max_duration: float = Form(60.0),
    normalize: bool = Form(True),
    z_threshold: float = Form(0.75),
    vl_rerank: bool = Form(False),
    vl_top_k: int = Form(8),
    vl_weight: float = Form(0.35),
    vl_max_frames: int = Form(1),
    use_llm_segments: bool = Form(False),  # 显式开启时才调用 qwen3.5-plus 云端切分
    use_local_vlm_segments: bool = Form(False),  # 显式开启时才跑本地 Qwen2-VL 全片理解切分
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """上传一个临时视频并直接跑高光标注；不写入主视频库。

    use_llm_segments=True 时, 临时视频先走 qwen3.5-plus 推理高光时间戳, 再用这些段做标注语义匹配.
    vl_rerank=True 时, 对 Qdrant/CLIP 召回后的最终候选做本地 Qwen2-VL 理解与重排序.
    use_local_vlm_segments=True 时, 才额外先跑本地 Qwen2-VL 全片理解切分.
    LLM 失败 / 无 segments 时自动回退到 hybrid 默认切片.
    """
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    temp_id = secrets.token_hex(8)
    tmp_dir = cfg.video_dir / "_tmp_annotate"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / f"{temp_id}{suffix}").resolve()
    try:
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(500, f"save uploaded video failed: {e}")
    finally:
        await file.close()

    state.temp_videos[temp_id] = tmp_path
    original_name = file.filename or tmp_path.name
    cloud_llm_requested = bool(use_llm_segments)
    effective_use_llm_segments = bool(use_llm_segments)

    req = AnnotateRequest(
        kb_id=kb_id,
        video_id=temp_id,
        threshold=threshold,
        top_k_per_clip=top_k_per_clip,
        merge_gap=merge_gap,
        min_duration=min_duration,
        max_duration=max_duration,
        normalize=normalize,
        z_threshold=z_threshold,
        vl_rerank=vl_rerank,
        vl_top_k=vl_top_k,
        vl_weight=vl_weight,
        vl_max_frames=vl_max_frames,
        use_llm_segments=effective_use_llm_segments,
        use_local_vlm_segments=use_local_vlm_segments,
    )

    def _run(task, video_path_str: str, req_obj: AnnotateRequest, filename: str, llm_mode: bool):
        # 1) LLM 推理切片 (仅用户显式开启)
        custom_segments: list[tuple[float, float]] | None = None
        extra_segments: list[tuple[float, float]] | None = None
        cloud_segments_used = False
        if llm_mode:
            from ingest.llm_segments import infer_segments_for_local_video
            task.message = "上传 TOS + qwen3.5-plus 推理高光"
            task.progress = 0.05
            try:
                meta = state.processor.probe(video_path_str)
                duration = meta.duration
            except Exception:
                duration = None
            llm_result = infer_segments_for_local_video(
                video_path_str, fps=2, video_duration=duration,
            )
            task.extra["llm_video_url"] = llm_result.video_url
            task.extra["llm_timings"] = llm_result.timings
            task.extra["llm_error"] = llm_result.error
            task.extra["llm_segments_count"] = len(llm_result.segments)
            task.extra["llm_segment_stats"] = llm_result.segment_stats
            task.extra["llm_segments"] = [s.as_dict() for s in llm_result.segments]
            raw = llm_result.raw_response or ""
            task.extra["llm_raw_preview"] = (raw[:800] + "\n...[truncated]...\n" + raw[-800:]) if len(raw) > 1600 else raw
            if llm_result.error:
                task.message = f"LLM 失败, 走默认切片: {llm_result.error[:80]}"
            elif not llm_result.segments:
                task.message = "LLM 无 segments, 走默认切片"
            else:
                custom_segments = [(s.start, s.end) for s in llm_result.segments]
                cloud_segments_used = True
                task.message = f"LLM 推理出 {len(custom_segments)} 段 · 嵌入 + 标注匹配"
                task.progress = 0.4

        if not (llm_mode and custom_segments):
            if req_obj.use_local_vlm_segments:
                task.message = "临时视频本地 Qwen2-VL 理解切分 + 默认召回"
            elif req_obj.vl_rerank:
                task.message = "临时视频 Qdrant 召回 + 本地 Qwen2-VL 重排序"
            else:
                task.message = "临时视频默认 hybrid 切分 + 嵌入"

        # ★ embedding 阶段串行 (MPS GPU 独占)
        with state.embed_lock:
            if req_obj.use_local_vlm_segments and not custom_segments:
                extra_segments = _local_vlm_custom_segments(task, video_path_str, req_obj)
                if extra_segments:
                    task.extra["local_vlm_segments_mode"] = "extra_candidates"
            if task.status == "canceled":
                return None
            def _progress(message: str, progress: float, extra: dict | None = None):
                task.message = message
                task.progress = max(float(task.progress or 0.0), min(float(progress), 0.99))
                if extra:
                    task.extra.update(extra)

            outcome = state.annotator.annotate(
                video_path=video_path_str,
                kb_id=req_obj.kb_id,
                threshold=req_obj.threshold,
                top_k_per_clip=req_obj.top_k_per_clip,
                merge_gap=req_obj.merge_gap,
                min_duration=req_obj.min_duration,
                max_duration=req_obj.max_duration,
                labels=req_obj.labels,
                normalize=req_obj.normalize,
                z_threshold=req_obj.z_threshold,
                vl_rerank=req_obj.vl_rerank,
                vl_top_k=req_obj.vl_top_k,
                vl_weight=req_obj.vl_weight,
                vl_max_frames=req_obj.vl_max_frames,
                custom_segments=custom_segments,
                extra_segments=extra_segments,
                progress_callback=_progress,
            )
        annotations, rejected_annotations, used_local_vlm_fallback = _annotation_dicts_with_local_vlm_fallback(outcome, task, video_path_str)
        task.extra["rejected_annotations_count"] = len(rejected_annotations)
        task.progress = 1.0
        task.message = "完成"
        return {
            "video_id": req_obj.video_id,
            "video_name": filename,
            "stream_url": f"/temp-videos/{req_obj.video_id}/stream",
            "temporary": True,
            "kb_id": req_obj.kb_id,
            "kb_sample_count": outcome.kb_sample_count,
            "labels_in_kb": outcome.labels_in_kb,
            "annotations": annotations,
            "rejected_annotations": rejected_annotations,
            "timings": outcome.timings.as_dict(),
            "used_llm_segments": cloud_segments_used,
            "llm_segments_count": task.extra.get("llm_segments_count", 0),
            "cloud_llm_disabled_by_local_vlm": False,
            "used_local_vlm_segments": bool(task.extra.get("local_vlm_segments_count")),
            "local_vlm_segments_count": task.extra.get("local_vlm_segments_count", 0),
            "local_vlm_segment_ms": task.extra.get("local_vlm_segment_ms", 0),
            "local_vlm_segments_mode": task.extra.get("local_vlm_segments_mode", ""),
            "used_local_vlm_fallback": used_local_vlm_fallback,
            "local_vlm_fallback_added": task.extra.get("local_vlm_fallback_added", 0),
            "requested_local_vlm_segments": bool(req_obj.use_local_vlm_segments),
        }

    task = state.tasks.submit(
        "annotate_upload", _run, str(tmp_path), req, original_name, effective_use_llm_segments,
    )
    task.extra["video_id"] = temp_id
    task.extra["video_name"] = original_name
    task.extra["use_llm_segments_requested"] = cloud_llm_requested
    task.extra["use_llm_segments_effective"] = effective_use_llm_segments
    task.extra["cloud_llm_disabled_by_local_vlm"] = False
    return {
        "task_id": task.task_id,
        "status": task.status,
        "kind": "annotate_upload",
        "video_id": temp_id,
        "video_name": original_name,
        "use_llm_segments": effective_use_llm_segments,
        "cloud_llm_disabled_by_local_vlm": False,
        "use_local_vlm_segments": use_local_vlm_segments,
    }


@app.post("/annotate/upload-bytedance")
async def annotate_uploaded_video_bytedance(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """字节火山引擎高光标注 (简化链路): 上传 TOS → 调火山 MediaKit → 直接返回 segments.

    不走语义匹配/向量库, 直接展示火山返回的高光片段 (含 score/description/ocr).
    model 固定 Miniseries (漫剧), mode=StorylineCuts.
    """
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    temp_id = secrets.token_hex(8)
    tmp_dir = cfg.video_dir / "_tmp_annotate"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / f"{temp_id}{suffix}").resolve()
    try:
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(500, f"save uploaded video failed: {e}")
    finally:
        await file.close()

    state.temp_videos[temp_id] = tmp_path
    original_name = file.filename or tmp_path.name

    def _run(task, video_path_str: str, filename: str):
        from ingest.llm_segments import upload_to_tos
        from ingest.bytedance_highlight import infer_highlights_from_url

        # 1) TOS 上传
        task.message = "上传 TOS"
        task.progress = 0.05
        try:
            video_url = upload_to_tos(video_path_str, prefix="vision-rag/bytedance-input")
            task.extra["video_url"] = video_url
        except Exception as e:
            raise RuntimeError(f"TOS upload failed: {e}")

        # 2) 调火山 MediaKit (Miniseries 漫剧模式)
        task.message = "字节 MediaKit 分析中 (漫剧模式)"
        task.progress = 0.15
        bt_result = infer_highlights_from_url(
            video_url, model="Miniseries", poll_interval=5.0, poll_timeout_s=1500,
            on_progress=lambda j: setattr(task, 'message',
                f"字节分析中… (task={bt_result.task_id[:8] if bt_result.task_id else '?'})"),
        )
        task.extra["bytedance_task_id"] = bt_result.task_id
        task.extra["bytedance_segments_count"] = len(bt_result.segments)
        task.extra["bytedance_timings"] = {
            "submit_ms": bt_result.submit_ms,
            "poll_ms": bt_result.poll_ms,
        }
        if bt_result.error:
            raise RuntimeError(f"字节 API 失败: {bt_result.error}")

        task.progress = 1.0
        task.message = "完成"
        return {
            "video_id": temp_id,
            "video_name": filename,
            "stream_url": f"/temp-videos/{temp_id}/stream",
            "source": "bytedance",
            "bytedance_task_id": bt_result.task_id,
            "duration": bt_result.duration,
            "segments": [s.as_dict() for s in bt_result.segments],
            "timings": {
                "submit_ms": bt_result.submit_ms,
                "poll_ms": bt_result.poll_ms,
                "total_ms": bt_result.submit_ms + bt_result.poll_ms,
            },
        }

    task = state.tasks.submit("annotate_bytedance", _run, str(tmp_path), original_name)
    task.extra["video_id"] = temp_id
    task.extra["video_name"] = original_name
    return {
        "task_id": task.task_id,
        "status": task.status,
        "kind": "annotate_bytedance",
        "video_id": temp_id,
        "video_name": original_name,
    }


# ----------------- highlight feedback -----------------

@app.post("/feedback/highlight")
def add_highlight_feedback(req: HighlightFeedbackRequest) -> dict[str, Any]:
    """保存人工高光反馈, 用于后续偏好排序器 / LoRA 数据集。"""
    from annotate.feedback_store import add_feedback

    if req.end_time <= req.start_time:
        raise HTTPException(400, "end_time must be greater than start_time")
    if req.corrected_start is not None and req.corrected_end is not None:
        if req.corrected_end <= req.corrected_start:
            raise HTTPException(400, "corrected_end must be greater than corrected_start")

    try:
        feedback_id = add_feedback(
            video_id=req.video_id,
            video_name=req.video_name,
            kb_id=req.kb_id,
            source=req.source,
            original_label=req.original_label,
            final_label=req.final_label,
            start_time=req.start_time,
            end_time=req.end_time,
            corrected_start=req.corrected_start,
            corrected_end=req.corrected_end,
            model_score=req.model_score,
            user_score=req.user_score,
            accepted=req.accepted,
            reason=req.reason,
            tags_json=json.dumps(req.tags, ensure_ascii=False),
            understanding_json=json.dumps(req.understanding, ensure_ascii=False),
        )
        return {"ok": True, "feedback_id": feedback_id}
    except Exception as e:
        raise HTTPException(500, f"save feedback failed: {e}")


@app.get("/feedback/highlights")
def list_highlight_feedback(
    video_id: str | None = None,
    kb_id: str | None = None,
    accepted: bool | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    from annotate.feedback_store import list_feedback
    rows = list_feedback(video_id=video_id, kb_id=kb_id, accepted=accepted, limit=limit)
    return {"feedback": rows, "count": len(rows)}


@app.get("/feedback/stats")
def highlight_feedback_stats() -> dict[str, Any]:
    from annotate.feedback_store import stats
    return stats()


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    t = state.tasks.get(task_id)
    if t is None:
        raise HTTPException(404, f"task not found: {task_id}")
    return t.as_dict()


@app.get("/tasks")
def list_tasks(kind: str | None = None, limit: int = 30) -> dict[str, Any]:
    return {"tasks": [t.as_dict() for t in state.tasks.list(kind=kind, limit=limit)]}


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str) -> dict[str, Any]:
    ok = state.tasks.cancel(task_id)
    if not ok:
        raise HTTPException(409, "task not cancelable (not found or already finished)")
    return {"canceled": task_id}


# ----------------- maintenance -----------------

@app.post("/maintenance/rebuild-qdrant-ingest")
def rebuild_qdrant_ingest(req: RebuildQdrantIngestRequest | None = None) -> dict[str, Any]:
    """Qdrant 主方案: 按 clip 全量重建 data/videos 多模态 named-vector 索引。

    这个接口不会用 Qwen2-VL 决定"是否入库"; 每个 clip 都会入库。
    Qwen2-VL 只负责生成 caption/reason/tags/cut_advice payload。
    """
    req = req or RebuildQdrantIngestRequest()

    def _video_files() -> list[Path]:
        exts = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
        files = [
            p for p in sorted(cfg.video_dir.iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
        if req.include_tmp:
            tmp_dir = cfg.video_dir / "_tmp_annotate"
            if tmp_dir.exists():
                files.extend(
                    p for p in sorted(tmp_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in exts
                )
        if req.limit and req.limit > 0:
            files = files[: int(req.limit)]
        return files

    def _run(task, req_obj: RebuildQdrantIngestRequest):
        task.message = "准备重建 Qdrant 多模态主索引"
        task.progress = 0.01
        with state.embed_lock:
            if req_obj.clear_video_embeddings:
                task.message = "清空 Qdrant 主索引"
                state.retriever.drop_collection()
                state.retriever.ensure_collection()
                task.extra["qdrant_cleared"] = True

            files = _video_files()
            total = len(files)
            task.extra["rebuild_total"] = total
            task.extra["rebuild_done"] = 0
            task.extra["qdrant_vectors"] = state.retriever.vector_names.as_dict()
            results: list[dict[str, Any]] = []
            if not total:
                task.message = "没有找到可重建的视频文件"
                return {"total": 0, "results": []}

            for idx, path in enumerate(files, start=1):
                if task.status == "canceled":
                    return {"total": total, "results": results, "canceled": True}
                prefix = f"[{idx}/{total}] {path.name}"
                task.extra["rebuild_current"] = path.name
                task.message = f"{prefix} · Qdrant 多模态 clip 入库"
                task.progress = max(float(task.progress or 0.0), min(0.99, (idx - 1) / total))

                def _clip_progress(clip_done: int, start: float, end: float):
                    task.extra["clip_done_in_video"] = clip_done
                    task.message = f"{prefix} · clip {clip_done} ({start:.1f}-{end:.1f}s) 写入 Qdrant"
                    task.progress = max(float(task.progress or 0.0), min(0.99, (idx - 1 + 0.5) / total))

                stats = state.ingest.ingest_video(
                    path,
                    skip_existing=bool(req_obj.skip_existing),
                    custom_segments=None,
                    should_cancel=lambda: task.status == "canceled",
                    on_progress=_clip_progress,
                )
                row = {
                    "video_id": stats.video_id,
                    "path": stats.path,
                    "num_clips": stats.num_clips,
                    "skipped": stats.skipped,
                    "error": stats.error,
                }
                results.append(row)
                task.extra["rebuild_done"] = idx
                task.extra["rebuild_last"] = row
                task.progress = max(float(task.progress or 0.0), min(0.99, idx / total))
                if stats.error == "canceled":
                    return {"total": total, "results": results, "canceled": True}
                if stats.error:
                    task.extra.setdefault("rebuild_errors", []).append(row)

            task.message = "Qdrant 重建完成"
            return {
                "total": total,
                "ok": sum(1 for r in results if not r.get("error")),
                "failed": sum(1 for r in results if r.get("error")),
                "clips": sum(int(r.get("num_clips") or 0) for r in results),
                "results": results,
            }

    task = state.tasks.submit("rebuild_qdrant_ingest", _run, req)
    return {
        "task_id": task.task_id,
        "status": task.status,
        "kind": "rebuild_qdrant_ingest",
        "include_tmp": req.include_tmp,
        "limit": req.limit,
        "clear_video_embeddings": req.clear_video_embeddings,
        "skip_existing": req.skip_existing,
    }


@app.post("/maintenance/rebuild-local-vlm-ingest")
def rebuild_local_vlm_ingest(req: RebuildLocalVlmIngestRequest | None = None) -> dict[str, Any]:
    """用本地 Qwen2-VL 重新切分 data/videos 下的视频, 并写入视频 embedding 表。"""
    req = req or RebuildLocalVlmIngestRequest()

    def _video_files() -> list[Path]:
        exts = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
        files = [
            p for p in sorted(cfg.video_dir.iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
        if req.include_tmp:
            tmp_dir = cfg.video_dir / "_tmp_annotate"
            if tmp_dir.exists():
                files.extend(
                    p for p in sorted(tmp_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in exts
                )
        if req.limit and req.limit > 0:
            files = files[: int(req.limit)]
        return files

    def _run(task, req_obj: RebuildLocalVlmIngestRequest):
        task.message = "准备重建视频片段索引"
        task.progress = 0.01

        if req_obj.clear_llm_highlights:
            import sqlite3

            with sqlite3.connect(cfg.data_dir / "llm_highlights.db") as conn:
                conn.execute("DELETE FROM llm_highlights")
                conn.commit()
            task.extra["llm_highlights_cleared"] = True

        with state.embed_lock:
            if req_obj.clear_video_embeddings:
                task.message = "清空视频 embedding 表"
                state.retriever.drop_collection()
                state.retriever.ensure_collection()
                task.extra["video_embeddings_cleared"] = True

            files = _video_files()
            total = len(files)
            task.extra["rebuild_total"] = total
            task.extra["rebuild_done"] = 0
            results: list[dict[str, Any]] = []
            if not total:
                task.message = "没有找到可重建的视频文件"
                return {"total": 0, "results": []}

            for idx, path in enumerate(files, start=1):
                if task.status == "canceled":
                    return {"total": total, "results": results, "canceled": True}
                base = (idx - 1) / total
                seg_start = min(0.98, base + 0.02 / total)
                seg_end = min(0.98, base + 0.78 / total)
                prefix = f"[{idx}/{total}] {path.name}"
                task.extra["rebuild_current"] = path.name
                segs = _local_vlm_custom_segments(
                    task,
                    str(path),
                    enabled=True,
                    merge_gap=1.5,
                    progress_start=seg_start,
                    progress_end=seg_end,
                    message_prefix=prefix,
                )
                if task.status == "canceled":
                    return {"total": total, "results": results, "canceled": True}
                if req_obj.strict_local_vlm and not segs:
                    row = {
                        "video_id": "",
                        "path": str(path),
                        "num_clips": 0,
                        "skipped": True,
                        "error": None,
                        "used_local_vlm_segments": False,
                        "local_vlm_segments_count": 0,
                        "skip_reason": "local_vlm_no_segments",
                    }
                    results.append(row)
                    task.extra["rebuild_done"] = idx
                    task.extra["rebuild_last"] = row
                    task.message = f"{prefix} · 本地 Qwen2-VL 未切出高光, 跳过入库"
                    task.progress = max(float(task.progress or 0.0), min(0.99, idx / total))
                    continue
                task.message = f"{prefix} · embedding 入库"
                task.progress = max(float(task.progress or 0.0), min(0.99, base + 0.9 / total))
                stats = state.ingest.ingest_video(path, skip_existing=False, custom_segments=segs)
                row = {
                    "video_id": stats.video_id,
                    "path": stats.path,
                    "num_clips": stats.num_clips,
                    "skipped": stats.skipped,
                    "error": stats.error,
                    "used_local_vlm_segments": bool(segs),
                    "local_vlm_segments_count": len(segs or []),
                }
                results.append(row)
                task.extra["rebuild_done"] = idx
                task.extra["rebuild_last"] = row
                if stats.error:
                    task.extra.setdefault("rebuild_errors", []).append(row)

            task.message = "重建完成"
            return {
                "total": total,
                "ok": sum(1 for r in results if not r.get("error")),
                "failed": sum(1 for r in results if r.get("error")),
                "results": results,
            }

    task = state.tasks.submit("rebuild_local_vlm_ingest", _run, req)
    return {
        "task_id": task.task_id,
        "status": task.status,
        "kind": "rebuild_local_vlm_ingest",
        "include_tmp": req.include_tmp,
        "limit": req.limit,
        "clear_video_embeddings": req.clear_video_embeddings,
        "clear_llm_highlights": req.clear_llm_highlights,
        "strict_local_vlm": req.strict_local_vlm,
    }


# ----------------- auto kb: 从 Qdrant 主片段自动归纳高光到 KB -----------------

@app.get("/auto-kb/stats")
def auto_kb_stats() -> dict[str, Any]:
    """Qdrant 主片段中的高光候选与当前 Qdrant KB 统计。"""
    from qdrant_client import models

    client = state.retriever.connect()
    collection = cfg.qdrant.collection_name
    total = client.count(collection_name=collection, exact=True).count
    qwen_true = client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="qwen_is_highlight", match=models.MatchValue(value=True))]
        ),
        exact=True,
    ).count
    qwen_score = client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="qwen_score", range=models.Range(gt=0.5))]
        ),
        exact=True,
    ).count
    qwen_score_gte_05 = client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="qwen_score", range=models.Range(gte=0.5))]
        ),
        exact=True,
    ).count
    qwen_score_gte_035 = client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="qwen_score", range=models.Range(gte=0.35))]
        ),
        exact=True,
    ).count
    qwen_positive_tags = client.count(
        collection_name=collection,
        count_filter=models.Filter(
            should=[
                models.FieldCondition(key="qwen_tags", match=models.MatchValue(value="反转")),
                models.FieldCondition(key="qwen_tags", match=models.MatchValue(value="冲突")),
            ]
        ),
        exact=True,
    ).count
    return {
        "source": "qdrant",
        "collection": collection,
        "total_clips": total,
        "qwen_is_highlight": qwen_true,
        "qwen_score_gt_0_5": qwen_score,
        "qwen_score_gte_0_5": qwen_score_gte_05,
        "qwen_score_gte_0_35": qwen_score_gte_035,
        "qwen_positive_tags": qwen_positive_tags,
        "kbs": [k.__dict__ for k in state.kb.retriever.list_kbs()],
    }


@app.get("/auto-kb/highlights")
def auto_kb_highlights(
    limit: int = 200,
    min_qwen_score: float = 0.5,
    candidate_mode: str = "balanced",
    broad_min_qwen_score: float = 0.35,
    target_candidates: int = 80,
    per_video_limit: int = 4,
) -> dict[str, Any]:
    """列出 Qdrant 主片段中可用于自动分类的高光候选。"""
    rows = state.retriever.get_highlight_candidate_rows(
        limit=limit,
        min_score=min_qwen_score,
        candidate_mode=candidate_mode,
        broad_min_score=broad_min_qwen_score,
        target_min=target_candidates,
        per_video_limit=per_video_limit,
    )
    public_rows = []
    for row in rows:
        item = dict(row)
        item.pop("embedding", None)
        public_rows.append(item)
    reasons: dict[str, int] = {}
    for row in public_rows:
        key = str(row.get("candidate_reason") or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "source": "qdrant",
        "candidate_mode": candidate_mode,
        "highlights": public_rows,
        "count": len(public_rows),
        "reason_counts": reasons,
    }


class AutoKbRunRequest(BaseModel):
    text_weight: float = 0.7
    min_cluster_size: int = 3
    kb_id: str = "auto_llm"
    only_uncategorized: bool = True   # 只跑还没写入当前 Qdrant KB 的片段
    name_with_llm: bool = True
    force_full: bool = False          # True 时清掉 kb_id 全部样例后全量重聚类
    limit: int = 4096
    min_qwen_score: float = 0.5
    candidate_mode: str = "balanced"
    broad_min_qwen_score: float = 0.35
    target_candidates: int = 80
    per_video_limit: int = 4
    vlm_refine: bool = True
    vlm_refine_limit: int = 120
    vlm_min_score: float = 0.45
    vlm_max_frames: int = 2
    vlm_max_tokens: int = 220


class RenameKbLabelsRequest(BaseModel):
    label_map: dict[str, str] = Field(default_factory=dict)


@app.post("/kb/{kb_id}/labels/rename")
def kb_rename_labels(kb_id: str, req: RenameKbLabelsRequest) -> dict[str, Any]:
    try:
        updated = state.kb.retriever.rename_labels(kb_id, req.label_map)
        return {"kb_id": kb_id, "updated": updated}
    except Exception as e:
        raise HTTPException(500, f"kb rename labels failed: {e}")


@app.post("/auto-kb/run")
def auto_kb_run(req: AutoKbRunRequest) -> dict[str, Any]:
    """异步任务: 从 Qdrant 主视频片段读取高光候选, 聚类 + 命名 + 写 Qdrant KB."""
    def _run(task, req_obj: AutoKbRunRequest):
        import numpy as np
        from annotate.auto_kb import HighlightCandidate, auto_categorize_and_upsert

        task.message = "扫描 Qdrant 高光候选片段"
        task.progress = 0.1
        if req_obj.force_full:
            try:
                state.kb.retriever.delete_kb(req_obj.kb_id)
                task.extra["full_rebuild"] = True
                logger.info(f"[auto_kb] force_full: dropped all samples in kb={req_obj.kb_id}")
            except Exception as e:
                logger.warning(f"force_full delete_kb failed: {e}")

        rows = state.retriever.get_highlight_candidate_rows(
            limit=req_obj.limit,
            min_score=req_obj.min_qwen_score,
            candidate_mode=req_obj.candidate_mode,
            broad_min_score=req_obj.broad_min_qwen_score,
            target_min=req_obj.target_candidates,
            per_video_limit=req_obj.per_video_limit,
        )
        if not rows:
            return {"inserted": 0, "skipped": 0, "label_count": {},
                    "cluster_count": 0, "kb_id": req_obj.kb_id,
                    "note": "no qdrant highlight candidates"}

        existing_ids = set()
        if req_obj.only_uncategorized and not req_obj.force_full:
            try:
                existing_ids = {str(r.get("sample_id") or "") for r in state.kb.retriever.list_samples(req_obj.kb_id)}
            except Exception as e:
                logger.warning(f"list existing qdrant kb samples failed: {e}")

        rows_for_candidates = []
        for r in rows:
            video_id = str(r.get("video_id") or "")
            clip_index = int(r.get("clip_index") or 0)
            sample_id = f"{video_id}:{clip_index}"
            if sample_id in existing_ids:
                continue
            if not video_id or r.get("embedding") is None:
                continue
            rows_for_candidates.append(r)

        vlm_stats = {"checked": 0, "kept": 0, "rejected": 0, "errors": 0, "unavailable": False}
        if req_obj.vlm_refine and rows_for_candidates:
            task.message = f"Qwen2-VL 综合 Qdrant 多模态精判 0/{min(len(rows_for_candidates), req_obj.vlm_refine_limit)}"
            task.progress = 0.18
            refined_rows: list[dict[str, Any]] = []
            meta_cache: dict[str, Any] = {}
            max_check = max(0, int(req_obj.vlm_refine_limit))
            rows_to_check = rows_for_candidates[:max_check] if max_check else rows_for_candidates
            with state.embed_lock:
                for i, r in enumerate(rows_to_check, start=1):
                    if task.status == "canceled":
                        return {"canceled": True, "stage": "vlm_refine", "checked": i - 1}
                    task.message = f"Qwen2-VL 综合 Qdrant 多模态精判 {i}/{len(rows_to_check)}"
                    task.progress = min(0.55, 0.18 + 0.35 * (i / max(len(rows_to_check), 1)))
                    video_path = str(r.get("video_path") or "")
                    try:
                        if video_path not in meta_cache:
                            p = Path(video_path)
                            if not p.exists():
                                raise FileNotFoundError(video_path)
                            meta_cache[video_path] = state.processor.probe(p)
                        u = state.highlight_understander.analyze_segment(
                            meta_cache[video_path],
                            float(r.get("start_time") or 0.0),
                            float(r.get("end_time") or 0.0),
                            label="自动高光知识库候选",
                            evidence_samples=None,
                            max_frames=max(1, int(req_obj.vlm_max_frames)),
                            max_tokens=max(64, int(req_obj.vlm_max_tokens)),
                            dialogue_text=str(r.get("transcript_text") or "")[:900],
                            qdrant_context=r,
                        )
                        vlm_stats["checked"] += 1
                        if u.error:
                            vlm_stats["errors"] += 1
                            if "unavailable" in u.error or "disabled" in u.error:
                                vlm_stats["unavailable"] = True
                            continue
                        score = float(u.score) if u.score is not None else 0.0
                        if u.is_highlight is True or score >= float(req_obj.vlm_min_score):
                            nr = dict(r)
                            nr["candidate_reason"] = f"{r.get('candidate_reason') or 'qdrant'}+vlm"
                            nr["vlm_understanding"] = u.as_dict()
                            tags = " ".join(u.tags or [])
                            refined_caption = "\n".join(
                                x for x in [
                                    u.caption.strip(),
                                    u.reason.strip(),
                                    tags.strip(),
                                ] if x
                            )
                            nr["caption_text"] = refined_caption or str(r.get("caption_text") or r.get("qwen_caption") or "")
                            nr["description"] = "\n".join(
                                x for x in [
                                    nr["caption_text"].strip(),
                                    str(r.get("transcript_text") or "").strip(),
                                ] if x
                            )
                            refined_rows.append(nr)
                            vlm_stats["kept"] += 1
                        else:
                            vlm_stats["rejected"] += 1
                    except Exception as e:
                        logger.warning("auto-kb qwen2-vl refine failed: %s", e)
                        vlm_stats["errors"] += 1
            if len(rows_for_candidates) > len(rows_to_check):
                refined_rows.extend(rows_for_candidates[len(rows_to_check):])
            if refined_rows or not vlm_stats["unavailable"]:
                rows_for_candidates = refined_rows
            task.extra["vlm_refine"] = vlm_stats

        candidates: list[HighlightCandidate] = []
        reason_counts: dict[str, int] = {}
        for r in rows_for_candidates:
            if task.status == "canceled":
                return {"canceled": True, "stage": "build_candidates", "candidates": len(candidates)}
            video_id = str(r.get("video_id") or "")
            clip_index = int(r.get("clip_index") or 0)
            reason = str(r.get("candidate_reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            caption_text = str(
                r.get("caption_text")
                or r.get("qwen_caption")
                or r.get("description")
                or ""
            ).strip()
            transcript_text = str(r.get("transcript_text") or "").strip()
            candidates.append(HighlightCandidate(
                video_id=video_id,
                clip_index=clip_index,
                visual_vec=np.asarray(r["embedding"], dtype=np.float32),
                description=r.get("description") or "",
                start_time=float(r.get("start_time") or 0.0),
                end_time=float(r.get("end_time") or 0.0),
                thumbnail=r.get("thumbnail") or "",
                caption_text=caption_text,
                transcript_text=transcript_text,
                transcript_source=str(r.get("transcript_source") or ""),
            ))

        task.extra["candidates"] = len(candidates)
        task.extra["qdrant_rows"] = len(rows)
        task.extra["skipped_existing"] = max(0, len(rows) - len(candidates))
        task.extra["candidate_mode"] = req_obj.candidate_mode
        task.extra["candidate_reason_counts"] = reason_counts
        task.extra["vlm_refine"] = vlm_stats
        if not candidates:
            return {"inserted": 0, "skipped": 0, "label_count": {},
                    "cluster_count": 0, "kb_id": req_obj.kb_id,
                    "note": "no new qdrant highlight candidates",
                    "qdrant_rows": len(rows),
                    "skipped_existing": len(rows)}
        if task.status == "canceled":
            return {"canceled": True, "stage": "before_cluster", "candidates": len(candidates)}

        task.message = f"对 {len(candidates)} 个高光做融合+聚类+命名"
        task.progress = 0.3
        result = auto_categorize_and_upsert(
            encoder=state.encoder,
            kb_retriever=state.kb.retriever,
            candidates=candidates,
            text_weight=req_obj.text_weight,
            min_cluster_size=req_obj.min_cluster_size,
            kb_id=req_obj.kb_id,
            name_with_llm=req_obj.name_with_llm,
            cluster_namer=state.highlight_understander,
            force=req_obj.force_full,
        )
        task.progress = 1.0
        task.message = "完成"
        out = result.as_dict()
        out["source"] = "qdrant"
        out["qdrant_rows"] = len(rows)
        out["candidates"] = len(candidates)
        out["skipped_existing"] = task.extra["skipped_existing"]
        out["candidate_mode"] = req_obj.candidate_mode
        out["candidate_reason_counts"] = reason_counts
        out["vlm_refine"] = vlm_stats
        return out

    task = state.tasks.submit("auto_kb", _run, req)
    return {"task_id": task.task_id, "status": task.status, "kind": "auto_kb"}


def main():
    import uvicorn
    if int(cfg.api.workers) <= 1:
        uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
    else:
        uvicorn.run(
            "api.server:app",
            host=cfg.api.host,
            port=cfg.api.port,
            workers=cfg.api.workers,
        )


if __name__ == "__main__":
    main()
