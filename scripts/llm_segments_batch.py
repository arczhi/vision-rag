#!/usr/bin/env python3
"""
离线批量: qwen3.5-plus 对一个目录内所有视频做高光推理.

输出 JSON 落在视频同目录, 文件名 <video>.llm.json.
跳过已经存在 .llm.json 的 (除非 --force).

用法:
    .venv/bin/python scripts/llm_segments_batch.py "/Users/alex/Downloads/漫剧素材"
    .venv/bin/python scripts/llm_segments_batch.py "/path" --force --fps 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llm-batch")


def main():
    ap = argparse.ArgumentParser(description="批量跑 qwen3.5-plus 高光推理")
    ap.add_argument("directory", help="包含视频文件的目录")
    ap.add_argument("--force", action="store_true", help="即使已有 .llm.json 也重新推理")
    ap.add_argument("--fps", type=int, default=2, help="qwen video_url fps 参数")
    ap.add_argument("--ext", nargs="+", default=[".mp4", ".mov", ".mkv", ".webm", ".avi"])
    ap.add_argument("--max-files", type=int, default=0, help="最多处理多少个 (0=全部)")
    args = ap.parse_args()

    from ingest.llm_segments import infer_segments_for_local_video

    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        log.error(f"directory not found: {root}")
        sys.exit(1)

    videos: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in {e.lower() for e in args.ext}:
            videos.append(p)
    if args.max_files:
        videos = videos[: args.max_files]

    log.info(f"found {len(videos)} videos in {root}")
    summary: list[dict] = []
    for i, vp in enumerate(videos, 1):
        sidecar = vp.with_suffix(".llm.json")
        log.info(f"[{i}/{len(videos)}] {vp.name}  ({vp.stat().st_size/1024/1024:.1f} MB)")
        if sidecar.exists() and not args.force:
            log.info(f"  ↳ sidecar exists, skip ({sidecar.name}). 用 --force 覆盖")
            try:
                with sidecar.open("r", encoding="utf-8") as f:
                    summary.append({
                        "video": vp.name, "skipped": True,
                        "segments": len(json.load(f).get("segments", []))
                    })
            except Exception:
                summary.append({"video": vp.name, "skipped": True})
            continue

        # video duration (尽量补齐, 避免 segment 越界)
        duration = None
        try:
            from ingest.video_processor import VideoProcessor
            proc = VideoProcessor()
            meta = proc.probe(str(vp))
            duration = meta.duration
        except Exception as e:
            log.warning(f"  probe duration failed: {e}")

        t0 = time.time()
        result = infer_segments_for_local_video(
            str(vp), fps=args.fps, video_duration=duration,
        )
        elapsed = time.time() - t0

        try:
            sidecar.write_text(
                json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(f"  ↳ wrote {sidecar.name}")
        except Exception as e:
            log.error(f"  write sidecar failed: {e}")

        if result.error:
            log.error(f"  ✗ {result.error}")
            summary.append({"video": vp.name, "error": result.error, "elapsed_s": round(elapsed, 1)})
        else:
            log.info(
                f"  ✓ segments={len(result.segments)} "
                f"highlights={len(result.parsed_json.get('highlights') or []) if result.parsed_json else 0} "
                f"hooks={len(result.parsed_json.get('hook') or []) if result.parsed_json else 0} "
                f"({elapsed:.1f}s)"
            )
            summary.append({
                "video": vp.name,
                "segments": len(result.segments),
                "elapsed_s": round(elapsed, 1),
                "video_url": result.video_url,
            })

    log.info("=== SUMMARY ===")
    for s in summary:
        log.info(f"  {s}")


if __name__ == "__main__":
    main()
