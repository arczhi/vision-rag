#!/usr/bin/env python3
"""重放此前 LLM 切片失败的入库任务.

针对两类失败:
  1) LLM 时间戳幻觉导致段全部越界被钳零 (kept=0)
  2) TOS 大文件 write timeout, 回退默认切片 (拿不到 LLM 高光段)

用法:
  # 默认 dry-run, 只打印将要重放的清单, 不发请求
  python scripts/replay_failed_ingest.py

  # 真正执行重放
  python scripts/replay_failed_ingest.py --run

  # 自定义后端地址
  python scripts/replay_failed_ingest.py --run --base http://10.41.7.102:28765
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

# 待重放的视频文件名 -> 旧 video_id (先删后入)
# 本次只重放"时间戳幻觉"的 3 个视频 (TOS 超时那个另算)
REPLAY = [
    ("1779175721591血染茉莉花ep1.mp4", "c6becb280a334117"),   # 越界幻觉
    ("17701993739072月4日 (1)(1).mp4", "db65abded44ab390"),    # 越界幻觉
    ("17701938684882月4日(7).mp4", "6e3590cfbec35aed"),        # 越界幻觉
]

VIDEO_DIR = Path(__file__).resolve().parent.parent / "data" / "videos"


def delete_video(base: str, video_id: str) -> dict:
    """删除旧 clip 数据."""
    r = requests.delete(f"{base}/videos/{video_id}", timeout=60)
    r.raise_for_status()
    return r.json()


def submit_ingest(base: str, video_path: Path) -> str:
    """重新上传并入库, use_llm_segments=true. 返回 task_id."""
    with video_path.open("rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        data = {"skip_existing": "false", "use_llm_segments": "true"}
        r = requests.post(f"{base}/ingest", files=files, data=data, timeout=900)
    r.raise_for_status()
    return r.json()["task_id"]


def poll(base: str, task_id: str, interval: int = 10, max_wait: int = 1800) -> dict:
    """轮询任务直到 done/failed."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        r = requests.get(f"{base}/tasks/{task_id}", timeout=15)
        r.raise_for_status()
        t = r.json()
        st = t.get("status")
        ex = t.get("extra", {})
        seg_stats = ex.get("llm_segment_stats") or {}
        print(
            f"  [{task_id}] {st:8} p={t.get('progress',0):.2f} "
            f"llm_cnt={ex.get('llm_segments_count','-')} "
            f"kept={seg_stats.get('kept','-')} oob={seg_stats.get('dropped_oob','-')} "
            f"| {t.get('message','')[:50]}"
        )
        if st in ("done", "failed", "canceled"):
            return t
        time.sleep(interval)
    return {"status": "timeout", "task_id": task_id}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:28765")
    ap.add_argument("--run", action="store_true", help="真正执行 (默认 dry-run)")
    ap.add_argument("--interval", type=int, default=10)
    args = ap.parse_args()

    files = [f for f, _ in REPLAY]
    missing = [f for f in files if not (VIDEO_DIR / f).exists()]
    if missing:
        print(f"✗ 缺失视频文件, 无法重放:\n  " + "\n  ".join(missing))
        return 1

    print(f"将「先删旧数据 → 重新入库(use_llm_segments=true)」{len(REPLAY)} 个视频:")
    for f, vid in REPLAY:
        size_mb = (VIDEO_DIR / f).stat().st_size / 1024 / 1024
        print(f"  - {f}  ({size_mb:.1f} MB)  旧 video_id={vid}")

    if not args.run:
        print("\n[dry-run] 加 --run 真正执行.")
        return 0

    print(f"\n开始执行 → {args.base}\n")
    results = []
    for f, old_vid in REPLAY:
        path = VIDEO_DIR / f
        print(f"▶ {f}")
        # 1) 删旧数据
        try:
            dr = delete_video(args.base, old_vid)
            print(f"  ✓ 删除旧数据 video_id={old_vid}: {dr}")
        except Exception as e:
            print(f"  ⚠ 删除失败(可能已不存在,继续): {e}")
        # 2) 重新入库
        try:
            tid = submit_ingest(args.base, path)
            print(f"  ✓ 已提交 task_id={tid}, 轮询中...")
            res = poll(args.base, tid, interval=args.interval)
            results.append((f, res))
        except Exception as e:
            print(f"  ✗ 入库提交失败: {e}")
            results.append((f, {"status": "submit_error", "error": str(e)}))
        print()

    print("=" * 70)
    print("重放结果汇总")
    print("=" * 70)
    for f, res in results:
        ex = (res or {}).get("extra", {})
        seg_stats = ex.get("llm_segment_stats") or {}
        rj = (res or {}).get("result") or {}
        print(
            f"{res.get('status','?'):10} "
            f"clips={rj.get('num_clips','-')} "
            f"llm_cnt={ex.get('llm_segments_count','-')} "
            f"kept={seg_stats.get('kept','-')} oob={seg_stats.get('dropped_oob','-')} "
            f"llm_err={(ex.get('llm_error') or '-')[:36]}  {f[:24]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
