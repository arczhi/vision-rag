"""
轻量异步任务管理器 (in-memory, 单进程).

适用 demo 场景: API 收到长耗时请求立即返回 task_id, 后台线程跑, 前端轮询.
若要分布式 / 持久化, 改用 Celery / RQ + Redis.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    kind: str                      # "annotate" / "ingest" / ...
    status: str = "pending"        # pending / running / done / failed / canceled
    progress: float = 0.0          # 0.0 ~ 1.0
    message: str = ""              # 当前阶段提示
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": round(
                ((self.finished_at or time.time()) - (self.started_at or self.created_at)) * 1000, 1
            ),
            "extra": self.extra,
        }


class TaskManager:
    def __init__(self, max_workers: int = 2):
        # 默认 2 个 worker: 同时跑两个标注就把 MPS GPU 撑满了, 再多没用
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vrag-task")
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[..., Any], *args, **kwargs) -> Task:
        """提交后台任务. fn 第一个参数会被注入 task 对象 (供进度回调)."""
        tid = uuid.uuid4().hex[:12]
        task = Task(task_id=tid, kind=kind)
        with self._lock:
            self._tasks[tid] = task
        self._executor.submit(self._wrap, task, fn, args, kwargs)
        return task

    def _wrap(self, task: Task, fn: Callable, args: tuple, kwargs: dict):
        task.status = "running"
        task.started_at = time.time()
        try:
            task.result = fn(task, *args, **kwargs)
            if task.status != "canceled":
                task.status = "done"
                task.progress = 1.0
        except Exception as e:
            if task.status == "canceled":
                return
            logger.exception(f"task {task.task_id} failed: {e}")
            task.error = f"{type(e).__name__}: {e}"
            task.status = "failed"
            task.extra["traceback"] = traceback.format_exc().splitlines()[-10:]
        finally:
            task.finished_at = time.time()

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self, kind: str | None = None, limit: int = 50) -> list[Task]:
        with self._lock:
            items = list(self._tasks.values())
        if kind:
            items = [t for t in items if t.kind == kind]
        items.sort(key=lambda t: t.created_at, reverse=True)
        return items[:limit]

    def cancel(self, task_id: str) -> bool:
        """合作式取消: 把 status 标为 canceled, 业务代码定期检查 task.status."""
        t = self.get(task_id)
        if not t or t.status not in ("pending", "running"):
            return False
        t.status = "canceled"
        return True

    def cleanup(self, older_than_sec: float = 3600) -> int:
        """删除超过 older_than_sec 完成的任务."""
        now = time.time()
        cutoff = now - older_than_sec
        deleted = 0
        with self._lock:
            for tid in list(self._tasks.keys()):
                t = self._tasks[tid]
                if t.status in ("done", "failed", "canceled") and (t.finished_at or 0) < cutoff:
                    del self._tasks[tid]
                    deleted += 1
        return deleted
