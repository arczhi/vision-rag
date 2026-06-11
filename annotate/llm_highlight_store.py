"""
LLM 高光段持久化 (SQLite, 单进程足够 demo).

表结构 llm_highlights:
  sample_id        TEXT PRIMARY KEY     # video_id:clip_index
  video_id         TEXT
  clip_index       INTEGER
  start_time       REAL
  end_time         REAL
  description      TEXT
  label            TEXT                 # qwen 输出的 highlight/hook
  thumbnail        TEXT
  source_video_url TEXT
  created_at       REAL
  in_kb            INTEGER DEFAULT 0    # 是否已写入 highlight_example_v1
  kb_label         TEXT                 # 自动归纳后的簇标签
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from config import cfg

DB_PATH = cfg.data_dir / "llm_highlights.db"

_lock = threading.Lock()


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS llm_highlights (
          sample_id        TEXT PRIMARY KEY,
          video_id         TEXT NOT NULL,
          clip_index       INTEGER NOT NULL,
          start_time       REAL,
          end_time         REAL,
          description      TEXT,
          label            TEXT,
          thumbnail        TEXT,
          source_video_url TEXT,
          created_at       REAL,
          in_kb            INTEGER DEFAULT 0,
          kb_label         TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_llm_video ON llm_highlights(video_id);
        CREATE INDEX IF NOT EXISTS idx_llm_inkb  ON llm_highlights(in_kb);
        """)
        c.commit()


def upsert_highlight(
    *,
    video_id: str,
    clip_index: int,
    start_time: float,
    end_time: float,
    description: str = "",
    label: str = "",
    thumbnail: str = "",
    source_video_url: str = "",
):
    """幂等插入: sample_id 重复时只 UPDATE 描述/缩略图等元数据 (不动 in_kb/kb_label)."""
    sid = f"{video_id}:{clip_index}"
    now = time.time()
    with _lock, _conn() as c:
        c.execute("""
            INSERT INTO llm_highlights
              (sample_id, video_id, clip_index, start_time, end_time,
               description, label, thumbnail, source_video_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sample_id) DO UPDATE SET
              start_time=excluded.start_time,
              end_time=excluded.end_time,
              description=excluded.description,
              label=excluded.label,
              thumbnail=excluded.thumbnail,
              source_video_url=excluded.source_video_url
        """, (sid, video_id, clip_index, float(start_time), float(end_time),
              description, label, thumbnail, source_video_url, now))
        c.commit()
    return sid


def list_highlights(only_uncategorized: bool = False, limit: int = 4096) -> list[dict]:
    with _lock, _conn() as c:
        if only_uncategorized:
            cur = c.execute(
                "SELECT * FROM llm_highlights WHERE in_kb=0 ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            cur = c.execute(
                "SELECT * FROM llm_highlights ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


def list_by_video(video_id: str) -> list[dict]:
    with _lock, _conn() as c:
        cur = c.execute(
            "SELECT * FROM llm_highlights WHERE video_id=? ORDER BY clip_index", (video_id,))
        return [dict(r) for r in cur.fetchall()]


def mark_in_kb(sample_ids: list[str], kb_label: str):
    """把一批 sample 标记为已入 KB, 同时记录最终的 kb_label."""
    if not sample_ids:
        return
    with _lock, _conn() as c:
        c.executemany(
            "UPDATE llm_highlights SET in_kb=1, kb_label=? WHERE sample_id=?",
            [(kb_label, sid) for sid in sample_ids],
        )
        c.commit()


def delete_by_video(video_id: str):
    with _lock, _conn() as c:
        c.execute("DELETE FROM llm_highlights WHERE video_id=?", (video_id,))
        c.commit()


def stats() -> dict:
    with _lock, _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM llm_highlights").fetchone()[0]
        in_kb = c.execute("SELECT COUNT(*) FROM llm_highlights WHERE in_kb=1").fetchone()[0]
        videos = c.execute("SELECT COUNT(DISTINCT video_id) FROM llm_highlights").fetchone()[0]
        labels = c.execute(
            "SELECT kb_label, COUNT(*) c FROM llm_highlights WHERE in_kb=1 GROUP BY kb_label ORDER BY c DESC"
        ).fetchall()
    return {
        "total": total,
        "in_kb": in_kb,
        "videos": videos,
        "kb_labels": {r["kb_label"]: r["c"] for r in labels},
    }
