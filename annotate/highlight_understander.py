"""
本地高光理解器: 用小参数量 Qwen2-VL (MLX) 对候选片段做语义判断。

定位:
  - CLIP/KB 负责快速召回候选片段
  - 本模块只对少量候选做慢速理解与重排
  - 也可用本地滑窗方式做轻量高光切分, 供上传标注绕开云端 LLM
  - 懒加载 MLX 模型, 未启用时不占内存
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import cfg
from annotate.auto_kb import PRESET_CLUSTER_LABELS, normalize_cluster_label
from ingest.video_processor import VideoMeta, VideoProcessor

logger = logging.getLogger(__name__)

MANJU_ANALYST_PROMPT = """# 漫剧内容分析师

你是一位专业的漫剧内容分析师，擅长深度解构视频内容，精准识别可用于短视频引流的高光时刻
（吸引观众停留的开头素材）和精准时间点悬念钩子（促使观众看完并点击下一集的关键中断点）。

判断标准:
1. 区分台词驱动型高光与画面驱动型高光，严格匹配短视频留存和转化价值。
2. 识别强情绪、强冲突、强反转、身份揭露、危机紧迫、爽点爆发、系统/金手指触发等核心高光。
3. 识别戛然而止的高转化悬念钩子，例如关键动作前一秒、神秘电话、震惊回头、比赛结果公布前。
4. 高光必须基于实际画面/台词，不得虚构不存在的剧情和对白。
5. 单个片段建议 2-8 秒，描述应短平快、爽点前置，适合抖音/快手平台吸睛。

典型分类:
- 台词驱动型: 冲突打脸型、家庭伦理型、身份反差型、特殊设定型、脑洞猎奇型、数字夸张型、强烈情绪型、系统提示型
- 画面驱动型: 冲突打脸画面、擦边画面型、萌娃怪兽型、危机紧迫型
- 悬念钩子: 冲突顶点型、悬念型"""


@dataclass
class SegmentUnderstanding:
    """本地 VLM 对一个候选高光片段的结构化理解。"""
    is_highlight: bool | None = None
    score: float | None = None          # 0.0-1.0
    tags: list[str] = field(default_factory=list)
    caption: str = ""                   # 对片段画面/剧情的简短描述, 用于多模态向量入库
    reason: str = ""
    cut_advice: str = ""
    raw_response: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_highlight": self.is_highlight,
            "score": round(self.score, 4) if self.score is not None else None,
            "tags": self.tags,
            "caption": self.caption,
            "reason": self.reason,
            "cut_advice": self.cut_advice,
            "error": self.error,
        }


def _extract_json(text: str) -> dict | None:
    """从 VLM 输出中抽 JSON, 兼容 fenced code 和额外说明。"""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else None
    if not candidate:
        s = text.find("{")
        e = text.rfind("}")
        if s >= 0 and e > s:
            candidate = text[s:e + 1]
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"true", "yes", "y", "1", "是", "适合", "高光"}:
            return True
        if t in {"false", "no", "n", "0", "否", "不适合", "不是"}:
            return False
    return None


def _normalize_score(v) -> float | None:
    try:
        s = float(v)
    except Exception:
        return None
    if s > 1.0:
        s = s / 100.0
    return float(min(1.0, max(0.0, s)))


def _normalize_tags(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()][:8]
    if isinstance(v, str):
        parts = re.split(r"[,，/、\s]+", v)
        return [p.strip() for p in parts if p.strip()][:8]
    return []


_NEGATIVE_HIGHLIGHT_PATTERNS = (
    "不适合作为高光",
    "不适合高光",
    "不适合",
    "不是高光",
    "无明显",
    "没有明显",
    "画面信息不足",
    "无法判断",
    "与AI漫剧高光切分无关",
    "与ai漫剧高光切分无关",
    "普通对话",
    "普通画面",
    "过渡",
    "铺垫",
)

def _has_negative_highlight_text(*parts: str) -> bool:
    text = " ".join(str(p or "") for p in parts)
    return any(p in text for p in _NEGATIVE_HIGHLIGHT_PATTERNS)


class LocalHighlightUnderstander:
    """基于 MLX Qwen2-VL 的候选片段理解器。

    当前 Qwen2-VL MLX 后端以多张代表帧近似视频理解。后续接入字幕/ASR 后,
    可把片段对白加入 prompt, 效果会比纯画面判断更稳。
    """

    def __init__(
        self,
        processor: VideoProcessor | None = None,
        model_name: str | None = None,
        max_new_tokens: int | None = None,
    ):
        self.processor = processor or VideoProcessor()
        self.model_name = model_name or cfg.reranker.model_name
        self.max_new_tokens = max_new_tokens or min(int(cfg.reranker.max_new_tokens), 256)
        self._model = None
        self._processor = None
        self._apply_chat_template = None
        self._generate = None
        self._disabled_reason: str | None = None

    def _resolve_model_ref(self) -> str:
        """优先使用项目 models/ 下的本地缓存, 否则交给 mlx-vlm 解析 model id。"""
        project_root = Path(__file__).resolve().parent.parent
        candidates = [
            project_root / "models" / "modelscope" / self.model_name,
            project_root / "models" / self.model_name,
            Path("/Users/alex/coding/vision-rag/models/modelscope") / self.model_name,
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return self.model_name

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._disabled_reason:
            return False
        try:
            from mlx_vlm import load as mlx_load, generate as mlx_generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as e:
            self._disabled_reason = f"mlx-vlm not installed: {e}"
            logger.warning("LocalHighlightUnderstander disabled: %s", self._disabled_reason)
            return False

        ref = self._resolve_model_ref()
        try:
            logger.info("Loading local highlight VLM: %s", ref)
            model, processor = mlx_load(ref)
            self._model = model
            self._processor = processor
            self._apply_chat_template = apply_chat_template
            self._generate = mlx_generate
            logger.info("Local highlight VLM ready")
            return True
        except Exception as e:
            self._disabled_reason = f"load failed: {type(e).__name__}: {e}"
            logger.warning("LocalHighlightUnderstander disabled: %s", self._disabled_reason)
            return False

    @staticmethod
    def segment_frame_indices(
        meta: VideoMeta,
        start_time: float,
        end_time: float,
        max_frames: int,
    ) -> list[int]:
        start = max(0.0, float(start_time))
        end = min(float(end_time), float(meta.duration))
        if end <= start:
            end = min(meta.duration, start + 0.5)
        n = max(1, int(max_frames))
        if n == 1:
            times = [(start + end) / 2.0]
        else:
            times = np.linspace(start, max(end - 1e-3, start), n).tolist()
        return [min(int(t * meta.fps), meta.num_frames - 1) for t in times]

    @staticmethod
    def frames_to_images(frames: list[np.ndarray], max_side: int = 448) -> list[Image.Image]:
        out: list[Image.Image] = []
        resample = getattr(Image, "Resampling", Image).LANCZOS
        for f in frames:
            if f is None:
                continue
            img = Image.fromarray(f).convert("RGB")
            if max_side > 0 and max(img.size) > max_side:
                img.thumbnail((max_side, max_side), resample)
            out.append(img)
        return out

    def _load_segment_frames(
        self,
        meta: VideoMeta,
        start_time: float,
        end_time: float,
        max_frames: int,
        max_side: int = 448,
    ) -> list[Image.Image]:
        """从片段中均匀抽代表帧, 返回 PIL RGB 图片列表。"""
        from ingest.video_processor import _HAS_DECORD  # type: ignore

        idxs = self.segment_frame_indices(meta, start_time, end_time, max_frames)

        if _HAS_DECORD:
            from decord import VideoReader, cpu
            vr = VideoReader(meta.path, ctx=cpu(0))
            frames = vr.get_batch(idxs).asnumpy()
        else:
            frame_cache = self.processor._decode_frames_sequential(meta, idxs)
            frame_list = [frame_cache[fi] for fi in idxs if fi in frame_cache]
            frames = np.stack(frame_list) if frame_list else np.zeros((0, meta.height, meta.width, 3), dtype=np.uint8)

        return self.frames_to_images([frames[i] for i in range(len(frames))], max_side=max_side)

    @staticmethod
    def _qdrant_context_text(qdrant_context: dict[str, Any] | None) -> str:
        if not qdrant_context:
            return "无"
        rows: list[str] = []

        def add(label: str, value, max_len: int = 700):
            if value is None:
                return
            if isinstance(value, (dict, list)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                text = str(value)
            text = text.strip()
            if not text:
                return
            rows.append(f"- {label}: {text[:max_len]}")

        def add_clip_context(prefix: str, ctx: dict[str, Any]) -> None:
            add(f"{prefix} caption_text / caption 向量文本", ctx.get("caption_text"))
            add(f"{prefix} qwen_caption", ctx.get("qwen_caption"))
            add(f"{prefix} transcript_text / transcript 向量文本", ctx.get("transcript_text"))
            add(f"{prefix} transcript_source", ctx.get("transcript_source"), 120)
            add(f"{prefix} 已有 understanding", ctx.get("understanding"), 900)
            add(f"{prefix} qwen_tags", ctx.get("qwen_tags"), 240)
            add(f"{prefix} qwen_reason", ctx.get("qwen_reason"))
            add(f"{prefix} qwen_cut_advice", ctx.get("qwen_cut_advice"), 500)
            add(f"{prefix} qwen_score", ctx.get("qwen_score"), 80)
            add(f"{prefix} qwen_is_highlight", ctx.get("qwen_is_highlight"), 80)

        current_clip = qdrant_context.get("current_clip")
        reference_samples = qdrant_context.get("reference_samples") or qdrant_context.get("matched_samples")
        if isinstance(current_clip, dict):
            rows.append("【当前候选片段上下文: 来自用户视频, 可用于描述当前画面/剧情】")
            add("当前候选时间", f"{current_clip.get('candidate_start_time')} - {current_clip.get('candidate_end_time')}", 80)
            add("当前候选 clip_indices", current_clip.get("clip_indices"), 120)
            add_clip_context("当前片段", current_clip)

        if isinstance(reference_samples, list) and reference_samples:
            rows.append("【相似高光样例: 只作为召回参考, 不是当前候选片段画面】")
            compact_refs = []
            for sample in reference_samples[:5]:
                if not isinstance(sample, dict):
                    continue
                compact_refs.append({
                    "label": sample.get("kb_label") or sample.get("label"),
                    "sample_id": sample.get("kb_sample_id") or sample.get("sample_id"),
                    "score": sample.get("kb_score"),
                    "score_breakdown": sample.get("score_breakdown"),
                    "qwen_is_highlight": sample.get("qwen_is_highlight"),
                    "qwen_score": sample.get("qwen_score"),
                })
            add("参考样例检索摘要", compact_refs, 900)
            add("向量召回 score_breakdown", qdrant_context.get("score_breakdown"), 300)
            add("候选来源", qdrant_context.get("candidate_reason"), 160)

        if not rows:
            rows.append("【当前片段 Qdrant 上下文】")
            add_clip_context("当前片段", qdrant_context)
            add("向量召回 score_breakdown", qdrant_context.get("score_breakdown"), 300)
            add("候选来源", qdrant_context.get("candidate_reason"), 160)
        return "\n".join(rows) or "无"

    @staticmethod
    def _evidence_text(evidence_samples: list[dict] | None, limit: int = 3) -> str:
        if not evidence_samples:
            return "无"
        rows = []
        for e in evidence_samples[:limit]:
            label = e.get("label") or ""
            score = e.get("score")
            score_txt = f"{float(score):.3f}" if isinstance(score, (int, float)) else ""
            bits = []
            if label:
                bits.append(f"label={label}")
            if score_txt:
                bits.append(f"score={score_txt}")
            breakdown = e.get("score_breakdown")
            if isinstance(breakdown, dict):
                compact = {
                    k: round(float(v), 3)
                    for k, v in breakdown.items()
                    if isinstance(v, (int, float))
                }
                if compact:
                    bits.append(f"modal_scores={compact}")
            if bits:
                rows.append(" / ".join(bits))
        return "\n".join(f"- 参考样例(不是当前片段): {x}" for x in rows) or "无"

    @staticmethod
    def _build_prompt(
        *,
        label: str,
        start_time: float,
        end_time: float,
        evidence_text: str,
        dialogue_text: str = "",
        qdrant_context_text: str = "",
    ) -> str:
        dialogue_text = dialogue_text.strip() or "无"
        qdrant_context_text = qdrant_context_text.strip() or "无"
        analyst_prompt = MANJU_ANALYST_PROMPT.strip() or "你是一位专业的漫剧内容分析师，擅长识别短视频引流高光和悬念钩子。"
        return f"""你是一名专业的视频内容分析师。下面这套标准必须沿用原 qwen3.5-plus 的“漫剧内容分析师”判断口径。

【原漫剧内容分析师标准】
{analyst_prompt}

【当前任务】
现在不是分析全片，也不要输出上面标准里的 highlights/hook 数组。
请只根据当前候选片段，综合判断它是否符合“高光时刻”或“悬念钩子”标准。
当前候选片段就是本次输入给你的图片帧 + 下面的“当前候选片段上下文” + “片段字幕/台词”。
“相似高光样例/相似样例证据”只说明为什么向量检索召回了这个候选, 不是当前候选片段的事实描述。

候选时间段: {start_time:.2f}s - {end_time:.2f}s
候选标签: {label or "未知"}

Qdrant 多模态上下文:
{qdrant_context_text}

相似样例证据:
{evidence_text}

片段字幕/台词:
{dialogue_text}

重点关注:
- 身份揭露、剧情反转、冲突升级、强情绪、危机悬念、爽点、钩子感
- 台词驱动型高光与画面驱动型高光都要识别; 如果提供了 Qdrant transcript/Whisper 台词, 优先结合台词判断反转和冲突
- 当前视觉帧和当前片段台词优先级最高; 当前片段 Qdrant caption/qwen_caption、transcript_text、已有 understanding 只能辅助你理解当前视频
- 相似高光样例只能辅助判断“这个候选为什么被召回”和“可能属于什么高光类型”, 不能当作当前画面来描述
- 如果参考样例中的内容没有出现在当前输入帧或当前台词里, 输出 caption/reason/cut_advice 里禁止写这些样例内容
- 若多模态互相矛盾, 以当前视觉帧 + 当前台词 + 漫剧分析师标准为准
- 如果没有字幕/台词, 不要虚构具体台词
- 如果只是过渡、铺垫、普通对话或画面信息不足, 应降低分数
- 可以把符合标准的悬念钩子也视为高光候选, tags 里标注“悬念”或“钩子”

只输出一个 JSON 对象, 不要额外解释:
{{
  "is_highlight": true 或 false,
  "score": 0-100,
  "tags": ["反转", "冲突"],
  "caption": "一句话描述这段画面和剧情正在发生什么",
  "reason": "一句话说明为什么是或不是高光",
  "cut_advice": "建议从哪里开始/结束, 不确定则写保留原片段"
}}

注意:
- caption、reason、cut_advice 必须描述当前用户视频候选片段, 不得复述相似样例的画面或剧情
- 如果 reason 里判断为普通画面、过渡、铺垫、信息不足、不适合作为高光, is_highlight 必须是 false, score 必须低于 50
- 如果没有明确剧情冲突/反转/强情绪/悬念, 不要为了迎合候选标签而输出 true"""

    def analyze_segment(
        self,
        meta: VideoMeta,
        start_time: float,
        end_time: float,
        *,
        label: str = "",
        evidence_samples: list[dict] | None = None,
        max_frames: int = 4,
        max_tokens: int | None = None,
        dialogue_text: str = "",
        qdrant_context: dict[str, Any] | None = None,
        frames: list[Image.Image] | None = None,
    ) -> SegmentUnderstanding:
        if not self._ensure_loaded():
            return SegmentUnderstanding(error=self._disabled_reason or "local VLM unavailable")

        try:
            images = frames if frames is not None else self._load_segment_frames(meta, start_time, end_time, max_frames=max_frames)
            if not images:
                return SegmentUnderstanding(error="no frames decoded")
            gen_tokens = max(32, min(int(max_tokens or self.max_new_tokens), self.max_new_tokens))

            prompt = self._build_prompt(
                label=label,
                start_time=start_time,
                end_time=end_time,
                evidence_text=self._evidence_text(evidence_samples),
                dialogue_text=dialogue_text[:900],
                qdrant_context_text=self._qdrant_context_text(qdrant_context),
            )

            # 多图近似视频片段; 若 MLX 后端不接受多图, 自动退化到中间帧。
            try:
                formatted = self._apply_chat_template(
                    self._processor, self._model.config, prompt, num_images=len(images),
                )
                resp = self._generate(
                    self._model, self._processor, formatted,
                    image=images, max_tokens=gen_tokens, verbose=False,
                )
            except Exception as multi_error:
                logger.debug("multi-image highlight analysis failed, fallback to mid frame: %s", multi_error)
                mid = images[len(images) // 2]
                formatted = self._apply_chat_template(
                    self._processor, self._model.config, prompt, num_images=1,
                )
                resp = self._generate(
                    self._model, self._processor, formatted,
                    image=[mid], max_tokens=gen_tokens, verbose=False,
                )

            text = resp.text if hasattr(resp, "text") else str(resp)
            parsed = _extract_json(text)
            if not parsed:
                return SegmentUnderstanding(raw_response=text, error="VLM output is not JSON")

            score = _normalize_score(parsed.get("score"))
            is_highlight = _as_bool(parsed.get("is_highlight"))
            caption = str(parsed.get("caption") or "")[:500]
            reason = str(parsed.get("reason") or "")[:500]
            cut_advice = str(parsed.get("cut_advice") or "")[:500]
            if _has_negative_highlight_text(caption, reason, cut_advice):
                is_highlight = False
                score = min(score if score is not None else 0.0, 0.35)
            elif is_highlight is True and score is not None and score < 0.5:
                is_highlight = False
            return SegmentUnderstanding(
                is_highlight=is_highlight,
                score=score,
                tags=_normalize_tags(parsed.get("tags")),
                caption=caption,
                reason=reason,
                cut_advice=cut_advice,
                raw_response=text,
            )
        except Exception as e:
            logger.warning("local highlight analysis failed: %s", e)
            return SegmentUnderstanding(error=f"{type(e).__name__}: {e}")

    @staticmethod
    def _clean_cluster_label(text: str) -> str:
        return normalize_cluster_label(text)

    @staticmethod
    def _cluster_description_snippet(text: str) -> str:
        lines = []
        for line in str(text or "").splitlines():
            t = line.strip()
            if not t:
                continue
            compact = re.sub(r"\s+", "", t)
            if compact in {"反转", "冲突", "反转冲突"}:
                continue
            if "符合" in t and "高光" in t:
                continue
            if t.startswith("这段画面展示了反转") or t.startswith("这段画面展示了剧情的反转"):
                continue
            lines.append(t)
            if len(lines) >= 3:
                break
        return "\n".join(lines)[:500]

    def name_highlight_cluster(
        self,
        descriptions: list[str],
        thumbnails: list[str] | None = None,
        *,
        max_images: int = 4,
        max_tokens: int = 48,
    ) -> str:
        """用本地 Qwen2-VL 给一组同类高光样本生成 4-8 个字的抽象标签。"""
        if not self._ensure_loaded():
            raise RuntimeError(self._disabled_reason or "local VLM unavailable")

        descs = [str(d or "").strip() for d in descriptions if str(d or "").strip()]
        if not descs:
            raise ValueError("cluster descriptions are empty")

        images: list[Image.Image] = []
        for thumb in (thumbnails or [])[:max_images]:
            try:
                p = Path(str(thumb))
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                if max(img.size) > 448:
                    resample = getattr(Image, "Resampling", Image).LANCZOS
                    img.thumbnail((448, 448), resample)
                images.append(img)
            except Exception as e:
                logger.debug("skip cluster thumbnail %s: %s", thumb, e)
        if not images:
            images = [Image.new("RGB", (32, 32), "white")]

        listed = "\n".join(
            f"{i + 1}. {self._cluster_description_snippet(d) or d[:300]}"
            for i, d in enumerate(descs[:6])
        )
        coarse_refs = "、".join(PRESET_CLUSTER_LABELS)
        prompt = f"""你是一位专业的漫剧内容分析师。现在这些图片和文字来自同一个自动聚类出来的漫剧高光簇。
请综合视觉缩略图、剧情描述、台词和判断理由，给这一簇取一个“分类标签”。
当前任务只是“给簇取名”，不是重新判断是否高光。

标签契约:
1. 必须输出“细分簇标签”，下面这些只是不允许直接输出的粗粒度参考方向: {coarse_refs}
2. 自创 4-8 个汉字；必须像“类型名/主题名”，不能像一句话或画面描述。
3. 标签必须概括共性，不得复述具体人物、武器、动作、场景或完整剧情。
4. 严禁输出“反转冲突”“高光时刻”“剧情高光”“悬念钩子”等无区分度标签。
5. 严禁输出以“这段/这个/画面/镜头/场景/视频/片段/内容/展示/描述/一个/一位/一名”开头的内容。
6. 严禁输出“这段画面展示”“画面展示”“符合高光”“高光标准”“正在发生”“发生什么”这类描述片段。
7. 如果你想写“这段画面展示了……”，说明你正在写描述而不是标签；请改成抽象分类名。
8. 自创标签必须表达高光类型，包含冲突、反击、反转、危机、悬念、身份反差、特殊设定、强情绪、猎奇、觉醒等语义。
9. 不要输出普通物体、动作、场景、主题、行业或素材标签；若簇不像明确高光类型，也要提炼成细分高光标签，不要退回粗粒度参考方向。

簇内样本描述:
{listed}

只输出 JSON，不要解释:
{{"label":"4-8个汉字标签"}}"""

        gen_tokens = max(16, min(int(max_tokens), self.max_new_tokens))
        try:
            formatted = self._apply_chat_template(
                self._processor, self._model.config, prompt, num_images=len(images),
            )
            resp = self._generate(
                self._model, self._processor, formatted,
                image=images, max_tokens=gen_tokens, verbose=False,
            )
        except Exception as multi_error:
            logger.debug("multi-image cluster naming failed, fallback to first frame: %s", multi_error)
            formatted = self._apply_chat_template(
                self._processor, self._model.config, prompt, num_images=1,
            )
            resp = self._generate(
                self._model, self._processor, formatted,
                image=[images[0]], max_tokens=gen_tokens, verbose=False,
            )

        text = resp.text if hasattr(resp, "text") else str(resp)
        label = self._clean_cluster_label(text)
        if not label:
            raise RuntimeError(f"invalid cluster label from VLM: {text[:120]}")
        return label

    def segment_video(
        self,
        meta: VideoMeta,
        *,
        window_s: float = 8.0,
        stride_s: float = 4.0,
        max_windows: int = 20,
        max_frames: int = 4,
        min_score: float = 0.55,
        merge_gap: float = 1.5,
        should_cancel=None,
        dialogue_lookup=None,
        on_progress=None,
    ) -> list[dict[str, Any]]:
        """用本地 Qwen2-VL 滑窗判断整条视频的高光候选段。

        Qwen2-VL-2B MLX 不是原生长视频切分模型, 所以这里采用:
        固定窗口抽帧 -> 单窗口高光判断 -> 相邻高光窗口合并。
        返回值可直接转成 VideoProcessor 的 custom_segments。
        """
        if not self._ensure_loaded():
            return []

        duration = max(0.0, float(meta.duration))
        if duration <= 0.1:
            return []

        window_s = max(2.0, float(window_s))
        stride_s = max(1.0, float(stride_s))
        max_windows = int(max_windows)
        max_frames = max(1, min(int(max_frames), 2))

        windows: list[tuple[float, float]] = []
        if duration <= window_s:
            windows.append((0.0, duration))
        else:
            t = 0.0
            while t < duration:
                e = min(duration, t + window_s)
                if e - t >= 1.0:
                    windows.append((t, e))
                if e >= duration:
                    break
                t += stride_s

        if max_windows > 0 and len(windows) > max_windows:
            idxs = np.linspace(0, len(windows) - 1, max_windows, dtype=int).tolist()
            windows = [windows[i] for i in sorted(set(idxs))]

        candidates: list[dict[str, Any]] = []
        total_windows = len(windows)
        for i, (start, end) in enumerate(windows, start=1):
            if should_cancel and should_cancel():
                return []
            if on_progress:
                try:
                    on_progress(i, total_windows, start, end)
                except Exception:
                    pass
            u = self.analyze_segment(
                meta,
                start,
                end,
                label="AI漫剧高光切分候选",
                evidence_samples=None,
                max_frames=max_frames,
                max_tokens=96,
                dialogue_text=dialogue_lookup(start, end) if dialogue_lookup else "",
            )
            if u.error:
                continue
            score = float(u.score) if u.score is not None else (0.6 if u.is_highlight else 0.0)
            if (u.is_highlight is True and score >= min_score) or score >= min_score:
                candidates.append({
                    "start": start,
                    "end": end,
                    "score": score,
                    "understanding": u.as_dict(),
                })

        if not candidates:
            return []

        candidates.sort(key=lambda x: (float(x["start"]), -float(x["score"])))
        merged: list[dict[str, Any]] = []
        for c in candidates:
            if not merged or float(c["start"]) > float(merged[-1]["end"]) + merge_gap:
                merged.append(dict(c))
                continue
            prev = merged[-1]
            prev["end"] = max(float(prev["end"]), float(c["end"]))
            if float(c["score"]) > float(prev["score"]):
                prev["score"] = float(c["score"])
                prev["understanding"] = c.get("understanding")

        return merged
