#!/usr/bin/env python3
"""收敛验证: 对 3 个曾经"时间戳幻觉"的视频, 用新 prompt(整数秒 start/end)
重跑 qwen LLM 推理, 对比修复前后的 dropped_oob / kept.

不入库, 只调 infer_segments_for_local_video. 直接按文件加载模块绕开 cv2.
"""
from __future__ import annotations
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- 加载 .env 到环境变量 ----
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# ---- 直接按文件加载 llm_segments (绕开 ingest/__init__ 的 cv2) ----
spec = importlib.util.spec_from_file_location(
    "llm_segments_mod", str(ROOT / "ingest" / "llm_segments.py")
)
m = importlib.util.module_from_spec(spec)
sys.modules["llm_segments_mod"] = m
spec.loader.exec_module(m)

VIDEO_DIR = ROOT / "data" / "videos"
TARGETS = [
    "1779175721591血染茉莉花ep1.mp4",
    "17701993739072月4日 (1)(1).mp4",
    "17701938684882月4日(7).mp4",
]


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def main() -> int:
    print("=" * 78)
    print("收敛验证: 新 prompt(整数秒 start/end) 对 3 个幻觉视频重跑 LLM 推理")
    print("=" * 78)
    results = []
    for fn in TARGETS:
        path = VIDEO_DIR / fn
        if not path.exists():
            print(f"✗ 缺失: {fn}")
            continue
        dur = probe_duration(path)
        print(f"\n▶ {fn}  (真实时长 {dur:.0f}s)")
        try:
            res = m.infer_segments_for_local_video(path, video_duration=dur)
        except Exception as e:
            print(f"  ✗ 推理异常: {type(e).__name__}: {e}")
            results.append((fn, None))
            continue
        ss = res.segment_stats or {}
        print(f"  parsed_total={ss.get('parsed_total','-')} "
              f"kept={ss.get('kept','-')} "
              f"dropped_oob={ss.get('dropped_oob','-')} "
              f"max_ts={ss.get('max_timestamp','-')}s  vs dur={dur:.0f}s")
        if res.error:
            print(f"  error: {res.error}")
        # 打印前几段看看
        for seg in res.segments[:4]:
            print(f"    [{seg.label}] {seg.start:.0f}-{seg.end:.0f}s  {seg.description[:24]}")
        results.append((fn, ss))

    print("\n" + "=" * 78)
    print("收敛判定")
    print("=" * 78)
    ok = True
    for fn, ss in results:
        if not ss:
            print(f"  ✗ {fn[:28]}: 推理失败"); ok = False; continue
        kept = ss.get("kept", 0)
        oob = ss.get("dropped_oob", 0)
        verdict = "✓ 收敛" if kept > 0 else "✗ 仍幻觉"
        if kept == 0:
            ok = False
        print(f"  {verdict}  kept={kept} oob={oob}  {fn[:28]}")
    print("\n" + ("✅ 全部收敛: 修复有效" if ok else "⚠ 仍有视频 kept=0, 需进一步分析"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
