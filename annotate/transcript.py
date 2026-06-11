"""
Transcript and local ASR support for highlight understanding.

Default behavior is conservative:
  - read same-name sidecar subtitles first (.srt/.vtt/.ass/.json)
  - cache parsed transcripts under data/transcripts/
  - run faster-whisper only when ENABLE_LOCAL_ASR=1 and the package is installed
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class Transcript:
    source: str = "none"
    segments: list[TranscriptSegment] = field(default_factory=list)
    error: str = ""

    def text_for_range(self, start: float, end: float, max_chars: int = 900) -> str:
        rows: list[str] = []
        for seg in self.segments:
            if seg.end < start or seg.start > end:
                continue
            text = " ".join(seg.text.split())
            if text:
                rows.append(f"[{seg.start:.1f}-{seg.end:.1f}] {text}")
            if sum(len(x) for x in rows) >= max_chars:
                break
        out = "\n".join(rows)
        return out[:max_chars]

    def as_cache(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "segments": [s.as_dict() for s in self.segments],
            "error": self.error,
        }


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _parse_ts(text: str) -> float:
    text = text.strip().replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(text)
    except Exception:
        return 0.0


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\\N", " ").replace("\\n", " ")
    return " ".join(text.split())


def _parse_srt_or_vtt(path: Path) -> list[TranscriptSegment]:
    raw = path.read_text("utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw)
    out: list[TranscriptSegment] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        time_idx = -1
        for i, line in enumerate(lines):
            if "-->" in line:
                time_idx = i
                break
        if time_idx < 0:
            continue
        m = re.match(r"(.+?)\s*-->\s*(.+?)(?:\s|$)", lines[time_idx])
        if not m:
            continue
        start = _parse_ts(m.group(1))
        end = _parse_ts(m.group(2))
        text = _clean_text(" ".join(lines[time_idx + 1:]))
        if text and end > start:
            out.append(TranscriptSegment(start, end, text))
    return out


def _parse_ass(path: Path) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    for line in path.read_text("utf-8", errors="ignore").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        payload = line.split(":", 1)[1].strip()
        parts = payload.split(",", 9)
        if len(parts) < 10:
            continue
        start = _parse_ts(parts[1])
        end = _parse_ts(parts[2])
        text = _clean_text(parts[9])
        if text and end > start:
            out.append(TranscriptSegment(start, end, text))
    return out


def _parse_json(path: Path) -> list[TranscriptSegment]:
    data = json.loads(path.read_text("utf-8", errors="ignore"))
    if isinstance(data, dict):
        items = data.get("segments") or data.get("subtitles") or data.get("items") or []
    else:
        items = data
    out: list[TranscriptSegment] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = item.get("start", item.get("start_time", item.get("from", 0)))
        end = item.get("end", item.get("end_time", item.get("to", 0)))
        text = _clean_text(str(item.get("text", item.get("content", item.get("sentence", "")))))
        try:
            start_f = float(start)
            end_f = float(end)
        except Exception:
            continue
        if text and end_f > start_f:
            out.append(TranscriptSegment(start_f, end_f, text))
    return out


class TranscriptProvider:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (cfg.data_dir / "transcripts")
        self._whisper_model = None

    def load(self, video_path: str | Path) -> Transcript:
        path = Path(video_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        sidecar = self._load_sidecar(path)
        if sidecar.segments:
            self._write_cache(path, sidecar)
            return sidecar

        cached = self._read_cache(path)
        if cached.segments:
            return cached

        if _env_bool("ENABLE_LOCAL_ASR", False):
            asr = self._run_asr(path)
            if asr.segments:
                self._write_cache(path, asr)
            return asr

        return Transcript(source="none")

    def _sidecar_paths(self, video_path: Path) -> list[Path]:
        exts = [".srt", ".vtt", ".ass", ".json"]
        return [video_path.with_suffix(ext) for ext in exts]

    def _load_sidecar(self, video_path: Path) -> Transcript:
        for p in self._sidecar_paths(video_path):
            if not p.exists():
                continue
            try:
                if p.suffix == ".ass":
                    segs = _parse_ass(p)
                elif p.suffix == ".json":
                    segs = _parse_json(p)
                else:
                    segs = _parse_srt_or_vtt(p)
                return Transcript(source=f"sidecar:{p.name}", segments=segs)
            except Exception as e:
                logger.warning("parse subtitle sidecar failed: %s", e)
                return Transcript(source=f"sidecar:{p.name}", error=f"{type(e).__name__}: {e}")
        return Transcript(source="none")

    def _cache_path(self, video_path: Path) -> Path:
        try:
            st = video_path.stat()
            asr_key = ":".join([
                os.environ.get("ENABLE_LOCAL_ASR", ""),
                os.environ.get("LOCAL_ASR_MODEL", ""),
                os.environ.get("LOCAL_ASR_DEVICE", ""),
                os.environ.get("LOCAL_ASR_COMPUTE_TYPE", ""),
                os.environ.get("LOCAL_ASR_LANGUAGE", ""),
            ])
            raw = f"{video_path.resolve()}:{st.st_size}:{st.st_mtime}:{asr_key}".encode("utf-8")
        except Exception:
            raw = str(video_path).encode("utf-8")
        key = hashlib.sha1(raw).hexdigest()[:20]
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, video_path: Path) -> Transcript:
        p = self._cache_path(video_path)
        if not p.exists():
            return Transcript(source="none")
        try:
            data = json.loads(p.read_text("utf-8"))
            segs = [
                TranscriptSegment(float(x["start"]), float(x["end"]), str(x["text"]))
                for x in data.get("segments", [])
            ]
            return Transcript(source=data.get("source", "cache"), segments=segs, error=data.get("error", ""))
        except Exception as e:
            logger.warning("read transcript cache failed: %s", e)
            return Transcript(source="cache", error=f"{type(e).__name__}: {e}")

    def _write_cache(self, video_path: Path, transcript: Transcript):
        try:
            self._cache_path(video_path).write_text(
                json.dumps(transcript.as_cache(), ensure_ascii=False, indent=2),
                "utf-8",
            )
        except Exception as e:
            logger.warning("write transcript cache failed: %s", e)

    def _run_asr(self, video_path: Path) -> Transcript:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            return Transcript(source="asr:faster-whisper", error=f"faster-whisper not installed: {e}")

        try:
            if self._whisper_model is None:
                model_name = os.environ.get("LOCAL_ASR_MODEL", "base")
                device = os.environ.get("LOCAL_ASR_DEVICE", "cpu")
                compute_type = os.environ.get("LOCAL_ASR_COMPUTE_TYPE", "int8")
                self._whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
            language = os.environ.get("LOCAL_ASR_LANGUAGE", "zh") or None
            segments, _info = self._whisper_model.transcribe(
                str(video_path),
                language=language,
                vad_filter=True,
                beam_size=3,
            )
            out = [
                TranscriptSegment(float(s.start), float(s.end), _clean_text(s.text))
                for s in segments
                if _clean_text(s.text)
            ]
            return Transcript(source="asr:faster-whisper", segments=out)
        except Exception as e:
            logger.warning("local ASR failed: %s", e)
            return Transcript(source="asr:faster-whisper", error=f"{type(e).__name__}: {e}")
