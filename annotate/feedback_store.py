"""
人工高光反馈存储。

用途:
  - 记录候选片段被接受/拒绝/修正后的结果
  - 为后续训练轻量排序器或 LoRA 准备干净样本
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from config import cfg

DB_PATH = cfg.data_dir / "highlight_feedback.db"
_lock = threading.Lock()


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS highlight_feedback (
          id                 INTEGER PRIMARY KEY AUTOINCREMENT,
          video_id           TEXT NOT NULL,
          video_name         TEXT,
          kb_id              TEXT,
          source             TEXT,
          original_label     TEXT,
          final_label        TEXT,
          start_time         REAL NOT NULL,
          end_time           REAL NOT NULL,
          corrected_start    REAL,
          corrected_end      REAL,
          model_score        REAL,
          user_score         REAL,
          accepted           INTEGER NOT NULL,
          reason             TEXT,
          tags_json          TEXT,
          understanding_json TEXT,
          created_at         REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_video ON highlight_feedback(video_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_kb ON highlight_feedback(kb_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_accept ON highlight_feedback(accepted);
        """)
        c.commit()


def add_feedback(
    *,
    video_id: str,
    start_time: float,
    end_time: float,
    accepted: bool,
    video_name: str = "",
    kb_id: str = "",
    source: str = "",
    original_label: str = "",
    final_label: str = "",
    corrected_start: float | None = None,
    corrected_end: float | None = None,
    model_score: float | None = None,
    user_score: float | None = None,
    reason: str = "",
    tags_json: str = "[]",
    understanding_json: str = "{}",
) -> int:
    init_db()
    with _lock, _conn() as c:
        cur = c.execute("""
            INSERT INTO highlight_feedback
              (video_id, video_name, kb_id, source, original_label, final_label,
               start_time, end_time, corrected_start, corrected_end,
               model_score, user_score, accepted, reason, tags_json,
               understanding_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id, video_name, kb_id, source, original_label, final_label,
            float(start_time), float(end_time),
            corrected_start if corrected_start is None else float(corrected_start),
            corrected_end if corrected_end is None else float(corrected_end),
            model_score if model_score is None else float(model_score),
            user_score if user_score is None else float(user_score),
            1 if accepted else 0,
            reason,
            tags_json,
            understanding_json,
            time.time(),
        ))
        c.commit()
        return int(cur.lastrowid)


def list_feedback(
    *,
    video_id: str | None = None,
    kb_id: str | None = None,
    accepted: bool | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    init_db()
    where = []
    args: list[Any] = []
    if video_id:
        where.append("video_id = ?")
        args.append(video_id)
    if kb_id:
        where.append("kb_id = ?")
        args.append(kb_id)
    if accepted is not None:
        where.append("accepted = ?")
        args.append(1 if accepted else 0)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM highlight_feedback{clause} ORDER BY created_at DESC LIMIT ?"
    args.append(int(limit))
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def stats() -> dict[str, Any]:
    init_db()
    with _lock, _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM highlight_feedback").fetchone()[0]
        accepted = c.execute("SELECT COUNT(*) FROM highlight_feedback WHERE accepted=1").fetchone()[0]
        rejected = c.execute("SELECT COUNT(*) FROM highlight_feedback WHERE accepted=0").fetchone()[0]
        videos = c.execute("SELECT COUNT(DISTINCT video_id) FROM highlight_feedback").fetchone()[0]
        labels = c.execute("""
            SELECT COALESCE(NULLIF(final_label, ''), original_label) AS label, COUNT(*) c
            FROM highlight_feedback
            WHERE accepted=1
            GROUP BY label
            ORDER BY c DESC
            LIMIT 50
        """).fetchall()
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "videos": videos,
        "accepted_labels": {r["label"]: r["c"] for r in labels if r["label"]},
    }
