"""
LLM 视频高光推理: TOS 上传 → qwen3.5-plus video_url → 解析时间戳

参考: /Users/alex/chuangliang/codebase/mobgi_ai_backend/internal/llmdriver/qianwen_chat.go
      /Users/alex/chuangliang/codebase/mobgi_ai_backend/config/config.dev.yaml

环境变量 (有默认, 但建议生产通过 env 覆盖):
  TOS_ENDPOINT, TOS_REGION, TOS_BUCKET, TOS_ACCESS_KEY, TOS_SECRET_KEY,
  TOS_PUBLIC_DOMAIN, TOS_MAIN_PATH
  DASHSCOPE_API_KEY, DASHSCOPE_BASE_URI
  LLM_SEGMENT_MODEL (默认 qwen3.5-plus)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ============= 默认配置 (敏感凭证留空，从 .env / 环境变量读取) =============
# 非敏感的 endpoint / 模型名保留默认；敏感字段留空，缺失时显式报错。
DEFAULTS = {
    "TOS_ENDPOINT":      "tos-cn-beijing.volces.com",
    "TOS_REGION":        "cn-beijing",
    "TOS_BUCKET":        "",
    "TOS_ACCESS_KEY":    "",
    "TOS_SECRET_KEY":    "",
    "TOS_PUBLIC_DOMAIN": "",
    "TOS_MAIN_PATH":     "",
    "DASHSCOPE_API_KEY": "",
    "DASHSCOPE_BASE_URI": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "LLM_SEGMENT_MODEL":  "qwen3.5-plus",
}

_REQUIRED_SECRETS = {
    "DASHSCOPE_API_KEY",
    "TOS_BUCKET", "TOS_ACCESS_KEY", "TOS_SECRET_KEY",
    "TOS_PUBLIC_DOMAIN", "TOS_MAIN_PATH",
}


def _env(key: str) -> str:
    val = os.environ.get(key) or DEFAULTS.get(key, "")
    if not val and key in _REQUIRED_SECRETS:
        raise RuntimeError(
            f"缺失环境变量 {key}; 请在项目根目录创建 .env (参考 .env.example)"
        )
    return val


# ============= TOS 上传 =============
def upload_to_tos(
    local_path: str | Path,
    prefix: str = "vision-rag/llm-input",
    max_attempts: int = 3,
) -> str:
    """上传本地视频到 TOS, 返回公网 https URL.

    底层逻辑: 走 volcengine TOS Python SDK, S3 兼容. 凭证从 env / DEFAULTS 取.

    超时与重试 (修复大文件 write timeout 导致 LLM 推理整体失败的 bug):
      - SDK 默认 request_timeout/socket_timeout=30s, 对 100MB+ 视频必然 write timeout.
        按文件大小动态放宽到 max(120, 30s/10MB), 上限 600s.
      - 应用层重试 max_attempts 次 (指数退避), 覆盖 SDK 对连接中断不重试的缺口.
    """
    try:
        import tos
    except ImportError as e:
        raise RuntimeError("tos sdk not installed (pip install tos)") from e

    local_path = Path(local_path).resolve()
    if not local_path.exists():
        raise FileNotFoundError(local_path)

    endpoint = _env("TOS_ENDPOINT")
    region = _env("TOS_REGION")
    bucket = _env("TOS_BUCKET")
    ak = _env("TOS_ACCESS_KEY")
    sk = _env("TOS_SECRET_KEY")
    public_domain = _env("TOS_PUBLIC_DOMAIN").rstrip("/") + "/"
    main_path = _env("TOS_MAIN_PATH").strip("/")

    # path 形如 tos_beijing/magic_cut/mobgi_ai/local/vision-rag/llm-input/<uuid>_filename
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", local_path.name)
    obj_key = f"{main_path}/{prefix.strip('/')}/{uuid.uuid4().hex[:8]}_{safe_name}"

    size_mb = local_path.stat().st_size / 1024 / 1024
    # 动态超时: 假设最差 ~10MB/s, 给 size_mb/10 * 1.5 倍余量, 钳到 [120, 600]
    timeout = int(min(600, max(120, size_mb / 10 * 1.5)))

    client = tos.TosClientV2(
        ak, sk, endpoint, region,
        request_timeout=timeout,
        socket_timeout=timeout,
        max_retry_count=3,
    )
    logger.info(
        f"[TOS] uploading {local_path.name} ({size_mb:.1f} MB) "
        f"timeout={timeout}s -> {obj_key}"
    )
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            client.put_object_from_file(bucket, obj_key, str(local_path))
            elapsed = time.time() - t0
            url = public_domain + obj_key
            logger.info(f"[TOS] uploaded in {elapsed:.1f}s (attempt {attempt}) -> {url}")
            return url
        except Exception as e:
            last_err = e
            elapsed = time.time() - t0
            logger.warning(
                f"[TOS] upload attempt {attempt}/{max_attempts} failed after {elapsed:.1f}s: {e}"
            )
            if attempt < max_attempts:
                backoff = 2 ** (attempt - 1)  # 1s, 2s, 4s
                time.sleep(backoff)
    raise RuntimeError(f"TOS upload failed after {max_attempts} attempts: {last_err}")


# ============= qwen3.5-plus 视频推理 =============
DEFAULT_SYSTEM_PROMPT = "你是一名专业的视频内容分析师。"

DEFAULT_USER_PROMPT_TEMPLATE = """高光标识4-21

# 漫剧内容分析师

## Profile

- **language**：中文
- **description**：你是一位专业的漫剧内容分析师，擅长深度解构视频内容，精准识别可用于短视频引流的高光时刻（吸引观众停留的开头素材）和精准时间点悬念钩子（促使观众看完并点击下一集的关键中断点），并尽可能形成完整组合，实现引流-转化闭环。

## Skills

1. 逐帧拆解漫剧剧情，精准定位所有关键场景的时间戳
2. 区分台词驱动型高光与画面驱动型高光，严格匹配预设分类标准
3. 识别强情绪、强冲突、强反转的核心高光点，适配短视频3秒留存逻辑
4. 识别全片所有戛然而止的高转化悬念钩子，适配短视频完播和追更逻辑
5. 精准构建高光-钩子最佳组合，确保高光有对应的钩子承接
10. 将剧情转化为符合抖音/快手平台调性的短平快吸睛文案

## Goals

1. 分析指定漫剧的完整内容
2. 在全片所有时间段无限制识别所有符合标准的高光点，最终输出3-30个
3. 在全片所有时间段无限制识别所有符合标准的悬念钩子，最终输出2-10个
4. 输出完整可解析的JSON格式结果，无符合条件内容则对应字段返回空数组

## Output Format
{
  "highlights": [
    {
      "start": 10,
      "end": 16,
      "description": "3秒抓眼球的短视频文案，爽点前置、节奏紧凑"
    }
  ],
  "hook": [
    {
      "start": 88,
      "end": 95,
      "description": "制造强烈悬念的文案，结尾留问号或未完待续感"
    }
  ]
}

### Rules
 所有内容必须基于实际漫剧剧情，严禁虚构不存在的场景和台词
 start / end 为【整数秒】，表示从视频开头(第0秒)起算的时间点，禁止输出帧号、百分比或 HH:MM:SS 字符串
 start / end 必须落在 [0, 视频总时长] 范围内，且 end > start；单个片段时长建议 2-8 秒
 高光点数量严格控制在3-30个，钩子点数量严格控制在2-10个
 所有描述文案控制在15-30字之间
 输出必须为完整、可直接解析的JSON对象，不得包含任何额外文字说明
"""


@dataclass
class LLMSegment:
    start: float          # seconds
    end: float            # seconds
    label: str            # "highlight" / "hook"
    description: str = ""
    timestamp_str: str = ""

    def as_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.end - self.start, 2),
            "label": self.label,
            "description": self.description,
            "timestamp": self.timestamp_str,
        }


@dataclass
class LLMInferenceResult:
    raw_response: str
    parsed_json: dict | None
    segments: list[LLMSegment]
    timings: dict = field(default_factory=dict)
    video_url: str = ""
    model: str = ""
    error: str | None = None
    segment_stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "video_url": self.video_url,
            "model": self.model,
            "segments": [s.as_dict() for s in self.segments],
            "highlights": (self.parsed_json or {}).get("highlights") or [],
            "hooks": (self.parsed_json or {}).get("hook") or [],
            "timings": self.timings,
            "error": self.error,
            "segment_stats": self.segment_stats,
        }


def _to_seconds(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        raise ValueError(f"unsupported time type: {type(s).__name__}")
    parts = s.strip().split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    return float(s)


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出里抽出 JSON. qwen 有时会用 ```json fenced 或多余说明.

    增强: 当 JSON 因 max_tokens 被截断时, 尝试: (a) 修补结尾 (b) 单独提取
          highlights / hook 数组, 任意一个解析成功就返回.
    """
    if not text:
        return None
    # ① fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else None
    # ② 第一个 { 到最后一个 }
    if not candidate:
        s = text.find("{"); e = text.rfind("}")
        if s >= 0 and e > s:
            candidate = text[s:e + 1]
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # ③ 截断兜底: 单独提取 highlights / hook 数组 (允许末尾不闭合)
    out: dict = {}
    for key in ("highlights", "hook"):
        # 找 "key": [   ... 后面尽量多的 } 配对
        pat = re.compile(rf'"{key}"\s*:\s*\[', re.S)
        m = pat.search(text)
        if not m:
            continue
        start = m.end()
        # 扫描 balanced [], 但允许中途截断 (找到完整的 ]) 不到时取已配齐的最后一个 })
        items: list[dict] = []
        # 简单: 在 start 之后逐对象抓 { ... }
        depth = 0
        obj_start = -1
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    try:
                        items.append(json.loads(text[obj_start:i + 1]))
                    except Exception:
                        pass
                    obj_start = -1
            elif c == "]" and depth == 0:
                break
        if items:
            out[key] = items
    return out or None


def _extract_start_end(it: dict) -> tuple[float, float | None]:
    """从单条 item 取 (start, end).

    优先读新格式整数秒 start/end; 回退老格式 timestamp(HH:MM:SS).
    end 缺失返回 None, 由上层用 next_start / 兜底推断.
    """
    if "start" in it and it.get("start") is not None:
        s = _to_seconds(it.get("start"))
        e = _to_seconds(it.get("end")) if it.get("end") is not None else None
        return s, e
    # 老格式兼容
    ts = it.get("timestamp")
    if ts:
        return _to_seconds(ts), None
    raise ValueError("item has neither start nor timestamp")


def _segments_from_json(
    parsed: dict,
    video_duration: float | None = None,
    stats: dict | None = None,
) -> list[LLMSegment]:
    """把 highlights/hook 数组转成 (start, end) 段.

    新格式: 每条直接给 {start:int, end:int}, 直接采用 (end 仍做 clamp 防御).
    老格式: 只有 timestamp, end = next_start - 0.1 或 +5s 兜底; 最多 8s.

    越界处理 (修复 LLM 时间戳幻觉导致整批段被钳零的 bug):
      - start >= video_duration 的段属于 LLM 幻觉 (把短视频时间戳编成几十分钟),
        直接丢弃, 并在 stats 里累计 dropped_oob, 供上层告警.
      - start < video_duration 但 end 被钳到 <= start 的, 给一个最小 0.5s 段长兜底
        (clamp 到 video_duration 内), 避免有效起点被误删.

    stats (可选 dict) 回填:
      parsed_total / kept / dropped_oob / dropped_zero / max_timestamp
    """
    out: list[LLMSegment] = []
    parsed_total = 0
    dropped_oob = 0      # start 越过视频时长 → LLM 幻觉
    dropped_zero = 0     # 钳制后仍无法构成有效段
    max_ts = 0.0
    for label, key in (("highlight", "highlights"), ("hook", "hook")):
        items = parsed.get(key) or []
        # 解析出 (start, end_or_None, item), end 可能来自新格式
        parsed_items: list[tuple[float, float | None, dict]] = []
        for it in items:
            try:
                s, e = _extract_start_end(it)
                max_ts = max(max_ts, s, e or 0.0)
                parsed_items.append((s, e, it))
            except Exception:
                continue
        parsed_items.sort(key=lambda x: x[0])
        parsed_total += len(parsed_items)
        for i, (s, e_explicit, it) in enumerate(parsed_items):
            # start 越界: LLM 把时间戳幻觉成了超出视频时长的值, 丢弃
            if video_duration is not None and s >= video_duration:
                dropped_oob += 1
                continue
            # end: 新格式直接用; 老格式用 next_start / 兜底推断
            if e_explicit is not None and e_explicit > s:
                e = min(e_explicit, s + 8.0)  # 仍限制单段最长 8s
            elif i + 1 < len(parsed_items):
                e = min(parsed_items[i + 1][0] - 0.1, s + 8.0)
            else:
                e = s + 5.0
            if video_duration is not None:
                e = min(e, video_duration)
                # start 在时长内但 end 被钳到 <= start: 给最小段长兜底
                if e <= s + 0.1:
                    e = min(s + 0.5, video_duration)
            if e > s + 0.1:
                out.append(LLMSegment(
                    start=s, end=e, label=label,
                    description=it.get("description") or "",
                    timestamp_str=str(it.get("timestamp") or f"{int(s)}-{int(e)}s"),
                ))
            else:
                dropped_zero += 1
    if stats is not None:
        stats.update({
            "parsed_total": parsed_total,
            "kept": len(out),
            "dropped_oob": dropped_oob,
            "dropped_zero": dropped_zero,
            "max_timestamp": round(max_ts, 1),
            "video_duration": round(video_duration, 1) if video_duration else None,
        })
    if dropped_oob:
        logger.warning(
            f"[LLM] {dropped_oob}/{parsed_total} 段时间戳越界 "
            f"(max_ts={max_ts:.0f}s > video_duration={video_duration}s), 已丢弃; "
            f"疑似 LLM 时间戳幻觉"
        )
    return out


def call_qwen_video(
    video_url: str,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    fps: int = 2,
    model: str | None = None,
    timeout: int = 1800,
    max_tokens: int = 40960,
    video_duration: float | None = None,
) -> tuple[str, dict]:
    """调 qwen3.5-plus 视频推理. 返回 (text, usage_dict).

    URL 必须 http(s). 走 dashscope OpenAI 兼容 /chat/completions, fps=2.

    video_duration: 已知视频时长(秒)时, 在 prompt 里强约束 LLM 输出的 start/end
        整数秒不得超过该时长 — 修复 qwen 对短视频把时间戳幻觉成几十分钟的问题.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests not installed") from e

    api_key = _env("DASHSCOPE_API_KEY")
    base = _env("DASHSCOPE_BASE_URI").rstrip("/")
    model = model or _env("LLM_SEGMENT_MODEL")

    text_prompt = user_prompt or DEFAULT_USER_PROMPT_TEMPLATE
    if video_duration and video_duration > 0:
        dur = int(round(video_duration))
        mm = dur // 60
        ss = dur % 60
        text_prompt += (
            f"\n\n## 重要时长约束\n"
            f"本视频总时长仅为 {dur} 秒（约 {mm:02d}:{ss:02d}）。"
            f"所有 start 和 end 都是【整数秒】，必须满足 0 <= start < end <= {dur}。"
            f"严禁输出大于 {dur} 的数值，严禁把短视频的时间点编造成几十分钟。"
        )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url, "fps": fps},
                    },
                    {"type": "text", "text": text_prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    logger.info(f"[LLM] calling {model} with fps={fps}, video_url={video_url}")
    t0 = time.time()
    r = requests.post(
        f"{base}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-DataInspection": '{"input": "disable", "output": "disable"}',
        },
        json=body,
        timeout=timeout,
    )
    elapsed = time.time() - t0
    logger.info(f"[LLM] response status={r.status_code} elapsed={elapsed:.1f}s")
    if r.status_code != 200:
        raise RuntimeError(f"qwen api {r.status_code}: {r.text[:500]}")
    data = r.json()
    if not data.get("choices"):
        raise RuntimeError(f"qwen empty choices: {data}")
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    return text, {
        "elapsed_ms": int(elapsed * 1000),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "video_tokens": (usage.get("prompt_tokens_details") or {}).get("video_tokens", 0),
    }


# ============= 端到端: 本地视频 → segments =============
def infer_segments_for_local_video(
    local_video_path: str | Path,
    fps: int = 2,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    video_duration: float | None = None,
    upload_prefix: str = "vision-rag/llm-input",
) -> LLMInferenceResult:
    """端到端: 本地视频 → TOS → qwen3.5-plus → JSON → segments.

    失败时 result.error 会有错误信息, segments 可能为空.
    """
    timings: dict = {}
    result = LLMInferenceResult(
        raw_response="", parsed_json=None, segments=[],
        timings=timings, model=_env("LLM_SEGMENT_MODEL"),
    )
    try:
        t0 = time.time()
        url = upload_to_tos(local_video_path, prefix=upload_prefix)
        timings["upload_ms"] = int((time.time() - t0) * 1000)
        result.video_url = url

        t0 = time.time()
        text, usage = call_qwen_video(
            url, user_prompt=user_prompt, system_prompt=system_prompt, fps=fps,
            video_duration=video_duration,
        )
        timings["llm_ms"] = int((time.time() - t0) * 1000)
        timings.update(usage)
        result.raw_response = text

        parsed = _extract_json(text)
        if not parsed:
            result.error = "LLM 输出无法解析为 JSON"
            return result
        result.parsed_json = parsed
        seg_stats: dict = {}
        result.segments = _segments_from_json(
            parsed, video_duration=video_duration, stats=seg_stats,
        )
        result.segment_stats = seg_stats
        # 解析出 highlights 但全部越界 → 明确报错, 不再静默走默认切片
        if not result.segments and seg_stats.get("parsed_total", 0) > 0:
            result.error = (
                f"LLM 解析出 {seg_stats['parsed_total']} 段但全部无效 "
                f"(越界 {seg_stats.get('dropped_oob', 0)} / 零长 {seg_stats.get('dropped_zero', 0)}; "
                f"max_ts={seg_stats.get('max_timestamp')}s vs 时长 {seg_stats.get('video_duration')}s)"
            )
    except Exception as e:
        logger.exception("[LLM] inference failed")
        result.error = f"{type(e).__name__}: {e}"
    return result
