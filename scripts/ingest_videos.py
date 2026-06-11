"""
批量入库脚本
用法:
  python scripts/ingest_videos.py --video-dir ./data/videos
  python scripts/ingest_videos.py --video-dir ./data/videos --no-skip-existing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 让 scripts/ 下也能 import 项目根模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.pipeline import IngestPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Batch ingest videos into Vision RAG")
    p.add_argument("--video-dir", type=Path, required=True, help="视频目录")
    p.add_argument(
        "--patterns",
        nargs="+",
        default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"],
        help="文件 glob 列表",
    )
    p.add_argument("--no-skip-existing", action="store_true", help="不跳过已入库视频，强制重入")
    p.add_argument("--report", type=Path, default=None, help="输出 JSON 报告路径")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.video_dir.exists():
        raise SystemExit(f"video dir not found: {args.video_dir}")

    pipeline = IngestPipeline()
    stats = pipeline.ingest_dir(
        args.video_dir,
        patterns=args.patterns,
        skip_existing=not args.no_skip_existing,
    )

    total = len(stats)
    ok = sum(1 for s in stats if s.error is None and not s.skipped)
    skipped = sum(1 for s in stats if s.skipped)
    failed = sum(1 for s in stats if s.error is not None)
    clips = sum(s.num_clips for s in stats)

    print(f"\n=== Ingest Summary ===")
    print(f"  total : {total}")
    print(f"  ok    : {ok}  (clips={clips})")
    print(f"  skip  : {skipped}")
    print(f"  fail  : {failed}")
    if failed:
        print("  failures:")
        for s in stats:
            if s.error:
                print(f"    - {s.path}: {s.error}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w") as f:
            json.dump([s.__dict__ for s in stats], f, indent=2, default=str)
        print(f"  report: {args.report}")


if __name__ == "__main__":
    main()
