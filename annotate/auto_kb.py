"""
自动高光归纳: Qdrant 高光候选 → 跨视频聚类 → Qwen2-VL 命名 → 写 highlight_example_v1

核心链路 (B 方案: 文本+视觉融合):
  1. 收集 N 个 highlight (视觉向量 + LLM 描述)
  2. 文本向量 = encode_texts(描述); 视觉向量 = 已有 embedding
  3. fused = α*text + (1-α)*visual, L2 normalize
  4. HDBSCAN 聚类 (无需指定 K)
  5. 每簇取靠近 centroid 的描述/缩略图, 调本地 Qwen2-VL 一次命名
  6. 写入高光样例表: kb_id="auto_llm" / label=簇名 / sample_id=<video>:<clip_idx>

幂等:
  - sample_id = source_video_id + ":" + clip_index
  - 已存在则跳过 (insert 前用 query 查一次)
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


AUTO_KB_ID = "auto_llm"
TEXT_WEIGHT = 0.5         # 融合权重: text * 0.5 + visual * 0.5 (实测 0.7+ 会塌成 1 大簇)
HDBSCAN_MIN_CLUSTER = 3   # 簇至少含多少 highlight (兼容旧 API, 实际很少用)
NOISE_LABEL = "其他"

PRESET_CLUSTER_LABELS = (
    "冲突打脸型", "家庭伦理型", "身份反差型", "特殊设定型", "脑洞猎奇型",
    "数字夸张型", "强烈情绪型", "系统提示型", "冲突打脸画面", "擦边画面型",
    "萌娃怪兽型", "危机紧迫型", "冲突顶点型", "悬念型",
)
_COARSE_LABEL_STEMS = tuple(
    sorted(
        {
            label.replace("画面型", "").replace("画面", "").removesuffix("型")
            for label in PRESET_CLUSTER_LABELS
        },
        key=len,
        reverse=True,
    )
)
_GENERIC_CLUSTER_LABELS = {
    "反转冲突", "冲突反转", "高光时刻", "剧情高光", "悬念钩子",
    "反转高光", "冲突高光", "普通画面", "这段画面", "画面展示",
    "普通场景", "普通片段", "普通镜头",
}
_BAD_LABEL_PREFIXES = (
    "这段", "这个", "画面", "镜头", "场景", "视频", "片段", "内容",
    "展示", "呈现", "描述", "符合", "正在", "一个", "一位", "一名",
    "普通", "常规", "一般",
)
_BAD_LABEL_FRAGMENTS = (
    "这段画面", "画面展示", "这段展示", "这个画面", "这段视频",
    "符合高光", "高光标准", "正在发生", "发生什么", "剧情正在",
    "普通", "相机", "镜头", "场景", "主题", "片段", "标签",
    "素材", "示例", "样本", "汉字", "分类名",
)
_HIGHLIGHT_LABEL_HINTS = (
    "冲突", "打脸", "反击", "反转", "危机", "紧迫", "悬念", "身份",
    "反差", "设定", "系统", "情绪", "崩溃", "愤怒", "绝望", "狂喜",
    "觉醒", "背叛", "羞辱", "嘲讽", "抛弃", "开除", "伦理", "萌娃",
    "怪兽", "猎奇", "夸张", "顶点", "危险", "生死", "追杀", "绑架",
    "绝境", "逆袭", "打斗", "袭击", "惊吓", "震惊", "悬疑", "爆发",
    "揭露", "对峙", "逼近", "失控", "突发", "异兽", "兵刃", "奇物",
)
_LABEL_KEYWORD_FALLBACKS = (
    ("打脸", "羞辱反击"),
    ("羞辱", "嘲讽羞辱"),
    ("嘲讽", "嘲讽羞辱"),
    ("背叛", "背叛反击"),
    ("小三", "伦理背叛"),
    ("婆媳", "伦理冲突"),
    ("身份", "身份揭露"),
    ("总裁", "身份揭露"),
    ("乞丐", "身份反转"),
    ("系统", "系统逼近"),
    ("任务", "任务危机"),
    ("奖励", "奖励反转"),
    ("千亿", "巨额反转"),
    ("三千年", "数字震惊"),
    ("亿", "巨额反转"),
    ("龙", "异兽危机"),
    ("火", "能量爆发"),
    ("枪", "生死对峙"),
    ("血", "血迹危机"),
    ("怪兽", "异兽危机"),
    ("追杀", "追杀逃亡"),
    ("危机", "危机逼近"),
    ("哭", "情绪崩溃"),
    ("怒", "愤怒爆发"),
    ("崩溃", "情绪崩溃"),
    ("震惊", "震惊反转"),
    ("神秘", "神秘揭露"),
    ("黑影", "黑影悬念"),
    ("悬念", "未知悬念"),
    ("萌娃", "萌娃危机"),
    ("婴儿", "萌娃危机"),
    ("荒诞", "荒诞反转"),
    ("动物", "异兽惊吓"),
    ("狐狸", "异兽惊吓"),
    ("狼", "异兽危机"),
)
_LABEL_DEDUPE_SUFFIXES = (
    "爆发", "升级", "逼近", "揭露", "反击", "对峙",
    "失控", "震惊", "逃亡", "突发", "觉醒", "反转",
    "分化", "延展", "再起", "递进", "并发", "变奏",
    "支线", "转折", "余波", "暗涌",
)


@dataclass
class HighlightCandidate:
    """待归纳的单个 highlight."""
    video_id: str
    clip_index: int
    visual_vec: np.ndarray      # (D,) 视觉向量, ingest 已写入
    description: str            # LLM 给的描述
    start_time: float
    end_time: float
    thumbnail: str = ""
    caption_text: str = ""
    transcript_text: str = ""
    transcript_source: str = ""

    @property
    def sample_id(self) -> str:
        return f"{self.video_id}:{self.clip_index}"


@dataclass
class ClusterResult:
    label: str                  # LLM 命名的"打脸反击" / "实力觉醒"
    members: list[HighlightCandidate]


def _han_label_len(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))


def _clean_label_text(text: str) -> str:
    text = str(text or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        text = match.group(1)

    def _extract_label_from_json(raw: str) -> str | None:
        for candidate in (raw, raw.replace('\\"', '"')):
            try:
                parsed = json.loads(candidate)
            except Exception:
                s = candidate.find("{")
                e = candidate.rfind("}")
                if s < 0 or e <= s:
                    continue
                try:
                    parsed = json.loads(candidate[s:e + 1])
                except Exception:
                    continue
            if isinstance(parsed, dict):
                value = parsed.get("label") or parsed.get("name")
                if value:
                    return str(value)
            if isinstance(parsed, str) and parsed != raw:
                nested = _extract_label_from_json(parsed)
                if nested:
                    return nested
        return None

    parsed_label = _extract_label_from_json(text)
    if parsed_label is not None:
        text = parsed_label
    lines = text.splitlines()
    if not lines:
        return ""
    text = lines[0].strip(' "\'`，。：:!?.')
    text = re.sub(r"^(标签|名称|簇名|类别)[:：]\s*", "", text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return text[:12]


def is_valid_cluster_label(label: str) -> bool:
    label = _clean_label_text(label)
    if not label:
        return False
    if label in PRESET_CLUSTER_LABELS:
        return False
    if any(label == stem or re.match(rf"^{re.escape(stem)}(?:型)?[0-9]+$", label) for stem in _COARSE_LABEL_STEMS):
        return False
    if label in _GENERIC_CLUSTER_LABELS or label == NOISE_LABEL:
        return False
    if any(fragment in label for fragment in _BAD_LABEL_FRAGMENTS):
        return False
    if any(label.startswith(prefix) for prefix in _BAD_LABEL_PREFIXES):
        return False
    if re.search(r"[0-9]", label):
        return False
    if re.match(r"^[0-9一二三四五六七八九十百千万亿]", label):
        return False
    if _han_label_len(label) < 2 or _han_label_len(label) > 8:
        return False
    if "高光" in label or "反转冲突" in label:
        return False
    if not any(hint in label for hint in _HIGHLIGHT_LABEL_HINTS):
        return False
    return True


def normalize_cluster_label(label: str) -> str:
    cleaned = _clean_label_text(label)
    return cleaned if is_valid_cluster_label(cleaned) else ""


def fallback_cluster_label(members: list[HighlightCandidate]) -> str:
    texts: list[str] = []
    for member in members:
        texts.extend([
            member.caption_text or "",
            member.description or "",
            member.transcript_text or "",
        ])
    joined = "\n".join(t for t in texts if t)
    for keyword, label in _LABEL_KEYWORD_FALLBACKS:
        if keyword in joined:
            return label
    return "未知悬念"


def _unique_cluster_label(
    cluster_id: int,
    raw_label: str,
    members: list[HighlightCandidate],
    used: set[str],
) -> str:
    """Return a visible cluster label that preserves the actual clustering granularity."""
    fallback = fallback_cluster_label(members)
    cleaned = normalize_cluster_label(raw_label)
    if not cleaned:
        cleaned = fallback
    base = cleaned[:8]
    label = base
    for suffix in _LABEL_DEDUPE_SUFFIXES:
        if label not in used:
            break
        root = base[: max(2, 8 - _han_label_len(suffix))]
        label = f"{root}{suffix}"
    if label in used:
        roots = [fallback[:4], base[:4], "悬念"]
        for root in roots:
            for suffix in _LABEL_DEDUPE_SUFFIXES:
                candidate = f"{root}{suffix}"[:8]
                if candidate not in used:
                    label = candidate
                    break
            if label not in used:
                break
    used.add(label)
    return label


# ---------- ① 融合向量 ----------
def fuse_vectors(
    visual_vecs: np.ndarray,    # (N, D)
    text_vecs: np.ndarray,      # (N, D)
    text_weight: float = TEXT_WEIGHT,
) -> np.ndarray:
    """文本+视觉加权 → L2 normalize."""
    if visual_vecs.shape != text_vecs.shape:
        raise ValueError(f"shape mismatch: visual {visual_vecs.shape} vs text {text_vecs.shape}")
    fused = text_weight * text_vecs + (1.0 - text_weight) * visual_vecs
    norms = np.linalg.norm(fused, axis=1, keepdims=True).clip(min=1e-8)
    return fused / norms


# ---------- ② 聚类 ----------
def cluster_hdbscan(
    fused: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER,
) -> np.ndarray:
    """对融合向量做聚类, 返回每条样本的簇标签 (-1 = 噪声).

    策略 (经验性: HDBSCAN 在描述类向量上常塌成单一大簇):
      - 默认优先 KMeans, K = max(3, min(12, round(sqrt(N))))
      - 样本 < 6 时直接归 1 类
      - HDBSCAN 走兜底分支, 当 KMeans 全失败时启用
    """
    n = len(fused)
    if n < 6:
        return np.zeros(n, dtype=int)

    # ---- 主策略: KMeans (经验法则 K=sqrt(N)) ----
    try:
        from sklearn.cluster import KMeans
        k = max(3, min(12, round(n ** 0.5)))
        logger.info(f"cluster: KMeans K={k} (n={n})")
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(fused)
        # 把单成员的"孤立簇"标记为噪声
        from collections import Counter
        sizes = Counter(labels.tolist())
        for cid, sz in sizes.items():
            if sz < 2:
                labels = np.where(labels == cid, -1, labels)
        return labels
    except Exception as e:
        logger.warning(f"KMeans failed: {e}; fallback HDBSCAN")

    # ---- 兜底: HDBSCAN ----
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        return clusterer.fit_predict(fused)
    except Exception as e:
        logger.error(f"HDBSCAN fallback failed: {e}; 全归 0 类")
        return np.zeros(n, dtype=int)

# ---------- ③ 簇命名 ----------
NAMING_REFERENCE = """## 参考标签体系 (来自漫剧内容分析师 prompt, 只作为高光方向参考; 最终标签要比这些大类更细)

### 台词驱动型高光
- 冲突打脸型: 主角遭羞辱/贬低/诬陷/背叛/抛弃/开除等不公对待, 即将反击
- 家庭伦理型: 嫌贫爱富/婚内出轨/小三上门/娘家吸血/重男轻女/彩礼纠纷/婆媳矛盾/忘恩负义
- 身份反差型: 表面身份与真实身份/能力存在巨大落差 (保安是亿万总裁/乞丐是武林高手/小孩有上古法力)
- 特殊设定型: 主角拥有独特身份/特殊能力/非常规武器设定首次曝光
- 脑洞猎奇型: 明显违背常理/突破认知的荒诞设定 (父亲是女性/动物开口说话/非人类配偶)
- 数字夸张型: 巨额财富/超长时长/超大数量等极端数字 (千亿资产/被追杀十年/秒杀十万大军)
- 强烈情绪型: 极致愤怒/崩溃大哭/绝望嘶吼/狂喜大笑等爆发式情绪
- 系统提示型: 系统/金手指/老爷爷等外挂载体发布任务/播报奖励/触发生死危机

### 画面驱动型高光
- 冲突打脸画面: 肢体冲突/视觉化羞辱 (扇耳光/围堵嘲讽/众人轻视特写)
- 擦边画面型: 性暗示的男女互动或身体特写
- 萌娃怪兽型: 萌娃/婴儿/可爱怪兽/炫酷机器人特写
- 危机紧迫型: 角色处于生死攸关或极端危险 (怪兽袭击/野兽追赶/绑架追杀/刀架脖子)

### 悬念钩子
- 冲突顶点型: 关键动作发生前一秒中断 (拳头即将相撞/扣动扳机前/巴掌即将落下)
- 悬念型: 制造未知感 (脚步声+震惊回头/神秘电话/比赛结果即将公布/黑影一闪而过)"""

NAMING_PROMPT = """你是一个视频高光分类专家。下面是从同一类高光场景中抽取的若干描述,
请用 4-8 个汉字总结这一类高光的**共性主题**, 严格遵循以下规则:

{reference}

---

## 命名要求 (依次执行):
1. 必须给当前簇取一个**细分标签**, 不要直接输出上面的预设大类名
2. 自创 4-8 个汉字, 必须是**抽象类型**而非剧情描述
   - ✓ 好示例: "实力觉醒""嘲讽羞辱""背叛抛弃""绝境反击"
   - ✗ 反面: "冲突打脸型"(太粗)/"特殊设定型"(太粗)/"孤胆破阵"(具象)/"长老逐出"(剧情)
3. 抓共性, 不要复述具体剧情、人物、武器、场景
4. 必须用汉字, 4-8 个字
5. 输出必须像一个分类标签, 不能像句子的开头或画面描述
6. 严禁输出以"这段/这个/画面/场景/视频/片段/一个/一位/一名/展示/描述"开头的内容
7. 严禁输出"这段画面展示/画面展示/符合高光/高光标准/正在发生/发生什么"这类描述片段
8. 自创标签必须表达高光类型, 例如冲突、反击、反转、危机、悬念、身份反差、特殊设定、强情绪、猎奇、觉醒等
9. 不要输出普通物体、动作、场景、主题、行业或素材标签; 若簇不像明确高光类型, 也要提炼成细分高光标签, 不要退回预设大类
10. 直接输出标签名, 不要任何额外文字

---

## 这一簇的描述列表
{descs}"""

def name_clusters_with_llm(
    clusters: dict[int, list[HighlightCandidate]],
    samples_per_cluster: int = 5,
) -> dict[int, str]:
    """对每个簇生成名字. 调用 qwen3.5-plus, 一个簇一次调用 (token 极少).

    失败兜底: 用第一个样本描述前 6 字 (比簇号更有信息量).
    """
    from ingest.llm_segments import _env
    import requests

    api_key = _env("DASHSCOPE_API_KEY")
    base = _env("DASHSCOPE_BASE_URI").rstrip("/")
    model = _env("LLM_SEGMENT_MODEL")

    out: dict[int, str] = {}
    used_labels: set[str] = set()
    for cluster_id, members in clusters.items():
        if cluster_id == -1:
            out[-1] = NOISE_LABEL
            continue
        descs = [m.description for m in members[:samples_per_cluster] if m.description]
        if not descs:
            out[cluster_id] = _unique_cluster_label(cluster_id, "", members, used_labels)
            continue
        listed = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(descs))
        prompt = NAMING_PROMPT.format(reference=NAMING_REFERENCE, descs=listed)
        try:
            r = requests.post(
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-DataInspection": '{"input": "disable", "output": "disable"}',
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "你是一个简洁精准的视频高光分类专家, 擅长从抽象层提炼细分共性主题, 严禁直接使用粗粒度预设大类。"},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 32,
                    "enable_thinking": False,
                },
                timeout=60,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            label = _unique_cluster_label(cluster_id, text, members, used_labels)
            out[cluster_id] = label
            logger.info(f"  cluster {cluster_id} ({len(members)} 个) → '{label}'")
        except Exception as e:
            logger.warning(f"naming cluster {cluster_id} failed: {e}")
            out[cluster_id] = _unique_cluster_label(cluster_id, "", members, used_labels)
    return out


def name_clusters_with_vlm(
    clusters: dict[int, list[HighlightCandidate]],
    cluster_namer: Any,
    samples_per_cluster: int = 5,
) -> dict[int, str]:
    """对每个簇生成短标签, 优先使用本地 Qwen2-VL 的多模态簇命名。"""

    out: dict[int, str] = {}
    used_labels: set[str] = set()
    for cluster_id, members in clusters.items():
        if cluster_id == -1:
            out[-1] = NOISE_LABEL
            continue
        selected = members[:samples_per_cluster]
        descs = [m.description for m in selected if m.description]
        thumbs = [m.thumbnail for m in selected if m.thumbnail]
        if not descs:
            out[cluster_id] = _unique_cluster_label(cluster_id, "", members, used_labels)
            continue
        try:
            name = cluster_namer.name_highlight_cluster(descs, thumbs)
            label = _unique_cluster_label(cluster_id, name, members, used_labels)
            out[cluster_id] = label
            logger.info(f"  cluster {cluster_id} ({len(members)} 个) → '{label}' (Qwen2-VL)")
        except Exception as e:
            logger.warning(f"qwen2-vl naming cluster {cluster_id} failed: {e}")
            out[cluster_id] = _unique_cluster_label(cluster_id, "", members, used_labels)
    return out


# ---------- ④ 写 KB ----------
def upsert_to_kb(
    kb_retriever,
    candidates: list[HighlightCandidate],
    cluster_labels: np.ndarray,
    name_map: dict[int, str],
    visual_vecs: np.ndarray,       # (N, D), visual named vector
    caption_vecs: np.ndarray,      # (N, D), caption named vector
    transcript_vecs: np.ndarray,   # (N, D), transcript named vector
    kb_id: str = AUTO_KB_ID,
    force: bool = False,
) -> dict:
    """把每个 highlight 写入 KB named vectors.

    幂等: sample_id 重复时跳过.
    force=True 时跳过 sample_id 去重检查 (调用方应已先 delete_kb 清空).
    """
    from .knowledge_base import HighlightSample

    if force:
        existing_ids: set = set()
    else:
        try:
            existing_ids = {r["sample_id"] for r in kb_retriever.list_samples(kb_id)}
        except Exception:
            existing_ids = set()

    samples = []
    skipped = 0
    label_count: dict[str, int] = {}
    for cand, cl_id, visual_vec, caption_vec, transcript_vec in zip(
        candidates,
        cluster_labels.tolist(),
        visual_vecs,
        caption_vecs,
        transcript_vecs,
    ):
        sid = cand.sample_id
        if sid in existing_ids:
            skipped += 1
            continue
        label = name_map.get(int(cl_id), NOISE_LABEL)
        label_count[label] = label_count.get(label, 0) + 1
        samples.append(HighlightSample(
            embedding=visual_vec.astype(np.float32),
            caption_embedding=caption_vec.astype(np.float32),
            transcript_embedding=transcript_vec.astype(np.float32),
            kb_id=kb_id,
            label=label,
            sample_id=sid,
            source_video_id=cand.video_id,
            start_time=float(cand.start_time),
            end_time=float(cand.end_time),
            thumbnail=cand.thumbnail or "",
            note=cand.description,
            caption_text=cand.caption_text or cand.description,
            transcript_text=cand.transcript_text or "",
            transcript_source=cand.transcript_source or "",
        ))
    inserted = []
    if samples:
        inserted = kb_retriever.insert_samples(samples)
    return {
        "inserted": len(inserted),
        "skipped": skipped,
        "label_count": label_count,
        "kb_id": kb_id,
    }


# ---------- 端到端编排 ----------
@dataclass
class AutoKbResult:
    inserted: int
    skipped: int
    label_count: dict[str, int]
    cluster_count: int
    kb_id: str
    error: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__


def auto_categorize_and_upsert(
    encoder,                       # CLIPEncoder / ChineseCLIPEncoder
    kb_retriever,                  # KBRetriever
    candidates: list[HighlightCandidate],
    text_weight: float = TEXT_WEIGHT,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER,
    kb_id: str = AUTO_KB_ID,
    name_with_llm: bool = True,
    cluster_namer: Any | None = None,
    force: bool = False,
) -> AutoKbResult:
    """端到端: 候选 → 融合向量 → 聚类 → 命名 → 写 KB."""
    if not candidates:
        return AutoKbResult(0, 0, {}, 0, kb_id, error="no candidates")

    try:
        # 1. 文本编码: caption lane 用高光描述; transcript lane 用 ASR/字幕文本.
        descs = [c.caption_text or c.description or "无描述" for c in candidates]
        caption_vecs = encoder.encode_texts(descs).astype(np.float32)
        transcript_texts = [
            c.transcript_text.strip() or c.description or c.caption_text or "无台词字幕"
            for c in candidates
        ]
        transcript_vecs = encoder.encode_texts(transcript_texts).astype(np.float32)
        # 2. 视觉向量整理
        visual = np.stack([c.visual_vec for c in candidates]).astype(np.float32)
        # 已 L2 norm 兜底 (encoder 默认 normalize)
        v_norms = np.linalg.norm(visual, axis=1, keepdims=True).clip(min=1e-8)
        visual = visual / v_norms

        # 3. 融合: 只用于聚类命名; 写入 KB 时仍保留三路 named vectors.
        fused = fuse_vectors(visual, caption_vecs, text_weight=text_weight)
        # 4. 聚类
        cluster_ids = cluster_hdbscan(fused, min_cluster_size=min_cluster_size)
        # 5. 分组
        groups: dict[int, list[HighlightCandidate]] = {}
        for cand, cl in zip(candidates, cluster_ids.tolist()):
            groups.setdefault(int(cl), []).append(cand)
        # 6. 命名
        if name_with_llm and cluster_namer is not None:
            name_map = name_clusters_with_vlm(groups, cluster_namer)
        elif name_with_llm:
            name_map = name_clusters_with_llm(groups)
        else:
            name_map = {i: f"簇{i}" if i != -1 else NOISE_LABEL for i in groups}
        # 7. 写库
        ins = upsert_to_kb(
            kb_retriever,
            candidates,
            cluster_ids,
            name_map,
            visual,
            caption_vecs,
            transcript_vecs,
            kb_id=kb_id,
            force=force,
        )
        return AutoKbResult(
            inserted=ins["inserted"],
            skipped=ins["skipped"],
            label_count=ins["label_count"],
            cluster_count=len([k for k in groups if k != -1]),
            kb_id=kb_id,
        )
    except Exception as e:
        logger.exception("auto_categorize_and_upsert failed")
        return AutoKbResult(0, 0, {}, 0, kb_id, error=f"{type(e).__name__}: {e}")
