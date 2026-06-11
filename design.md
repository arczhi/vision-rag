# Vision RAG - 设计文档与待办事项

## 方案概述

**方案 D: 全开源可商用生产级视频语义检索系统**

```
核心链路:
  Ingest:  视频 → FFmpeg/decord 抽帧切片 → EVA-CLIP-8B 生成 embedding → 存入 Milvus
  Query:   文本/图片 → EVA-CLIP 编码 → Milvus top-K 粗排 → InternVL rerank 精排 → 返回片段
  生成:    检索到的帧/片段 + query → 多模态 LLM (InternVL/Qwen-VL) → 生成回答
```

---

## 技术选型

| 层级 | 组件 | 版本/型号 | License | 说明 |
|---|---|---|---|---|
| 视频解码 | FFmpeg + decord | - | LGPL / Apache 2.0 | decord 比 opencv 快 3-5x |
| Embedding | EVA-CLIP-8B (BAAI) | open_clip_torch | MIT | 1024d, 支持中英文 |
| 向量数据库 | Milvus | v2.4.x | Apache 2.0 | 支持 ANN 检索、混合过滤 |
| Reranker | InternVL2-8B | transformers | Apache 2.0 | 多模态精排 |
| 生成 LLM | InternVL2 / Qwen2-VL | transformers | Apache 2.0 | 可选，RAG 回答生成 |
| API 服务 | FastAPI + Uvicorn | - | MIT / BSD | REST API |
| 基础设施 | Docker Compose | - | - | Milvus + etcd + MinIO |

---

## 项目结构

```
vision-rag/
├── config.py                  # [已完成] 全局配置
├── requirements.txt           # [已完成] Python 依赖
├── docker-compose.yml         # [已完成] Milvus 基础设施
├── design.md                  # [已完成] 本文档
│
├── ingest/                    # --- 入库模块 ---
│   ├── __init__.py            # [已完成]
│   ├── video_processor.py     # [待开发] 视频解码、抽帧、切片
│   ├── embedding.py           # [待开发] EVA-CLIP embedding 编码器
│   └── pipeline.py            # [待开发] 入库编排: 视频→切片→embedding→Milvus
│
├── search/                    # --- 检索模块 ---
│   ├── __init__.py            # [待创建]
│   ├── retriever.py           # [待开发] Milvus 向量检索 (粗排)
│   ├── reranker.py            # [待开发] InternVL 多模态重排序 (精排)
│   └── pipeline.py            # [待开发] 检索编排: query→粗排→精排→结果
│
├── api/                       # --- API 服务 ---
│   ├── __init__.py            # [待创建]
│   └── server.py              # [待开发] FastAPI REST 接口
│
├── scripts/                   # --- 运行脚本 ---
│   ├── ingest_videos.py       # [待开发] 批量入库脚本
│   └── start_server.sh        # [待开发] 启动服务脚本
│
├── data/                      # --- 数据目录 ---
│   ├── videos/                # 原始视频存放
│   ├── frames/                # 抽帧结果缓存
│   └── thumbnails/            # 缩略图缓存
│
└── models/                    # 模型缓存目录
```

---

## 待办事项

### P0 - 核心模块 (Must Have)

- [x] 全局配置 `config.py`
- [x] Python 依赖 `requirements.txt`
- [x] Docker Compose 基础设施 `docker-compose.yml`
- [x] **视频处理模块** `ingest/video_processor.py`
  - [x] decord 视频解码 (含 opencv 兜底)
  - [x] 均匀抽帧 (按 max_frames_per_clip 等距采样)
  - [x] 滑动窗口切片 (clip_duration + clip_stride)
  - [ ] 关键帧提取 (场景切换检测) — 留作 P3
  - [x] 缩略图生成
  - [x] 视频元信息提取 (时长、分辨率、帧率)
- [x] **EVA-CLIP Embedding** `ingest/embedding.py`
  - [x] 模型加载 (open_clip_torch, 支持 fp16/bf16)
  - [x] 图像编码 (单帧 → 1024d vector)
  - [x] 文本编码 (query → 1024d vector)
  - [x] 批量编码 (batch inference)
  - [x] 多帧聚合策略 (mean / max pooling)
- [x] **入库 Pipeline** `ingest/pipeline.py`
  - [x] 单视频入库流程编排
  - [x] 批量入库 + 进度条
  - [x] 去重检测 (基于 video_id 哈希)
  - [ ] 错误重试机制 — 当前仅 try/except 收集，未做 backoff
- [x] **Milvus 检索** `search/retriever.py`
  - [x] Collection 创建 (schema 定义)
  - [x] 索引创建 (IVF_FLAT / HNSW / IVF_SQ8)
  - [x] 向量插入
  - [x] ANN 检索 (文本→视频, 图片→视频)
  - [x] 混合过滤 (按 video_id 列表过滤)
- [x] **Reranker** `search/reranker.py`
  - [x] CLIP 二次打分 (默认轻量实现)
  - [x] InternVL 多模态评分 (use_internvl=True 启用)
  - [x] 片段聚合 (相邻片段合并)
- [x] **检索 Pipeline** `search/pipeline.py`
  - [x] 文本检索流程 (text → embedding → Milvus → rerank → result)
  - [x] 图片检索流程 (image → embedding → Milvus → rerank → result)
  - [ ] 视频片段检索 — 暂未实现 (可基于 image 流程扩展)

### P1 - API 与服务 (Should Have)

- [x] **FastAPI 服务** `api/server.py`
  - [x] `POST /ingest` 上传视频并入库
  - [x] `POST /search/text` 文本检索视频片段
  - [x] `POST /search/image` 图片检索视频片段
  - [x] `GET /videos` 已入库视频列表
  - [ ] `GET /videos/{id}/clips` 视频片段列表 — TODO
  - [x] `DELETE /videos/{id}` 删除视频及其向量
  - [x] 健康检查 + Milvus 连接状态

### P2 - 运行脚本 (Nice to Have)

- [x] `scripts/ingest_videos.py` 批量入库脚本
  - [x] 支持目录扫描、文件匹配
  - [x] 断点续传 (基于 video_id 去重 = skip_existing)
- [x] `scripts/start_server.sh` 一键启动
  - [x] 检查 Docker/Milvus 状态
  - [x] 启动 FastAPI 服务
- [x] **Web UI** `web/index.html`
  - [x] 文本搜 / 图像搜 / 上传入库 / 视频列表 / 删除
  - [x] 健康状态指示器，缩略图渲染
  - [x] 挂载方式: FastAPI `/web/` 静态目录 + `/thumbnails/` 静态目录, `/` → `/web/`

### P3 - 增强功能 (Future)

- [ ] 音频转文字 (Whisper) 作为辅助文本索引
- [ ] 视频场景分割 (PySceneDetect) 替代固定窗口切片
- [ ] 多模态 RAG 生成回答 (InternVL/Qwen-VL 作为 generator)
- [ ] 前端可视化界面 (Gradio / Streamlit)
- [ ] 分布式入库 (Celery / Ray)
- [ ] 增量更新索引

---

## Milvus Schema 设计

```python
schema = {
    "fields": [
        {"name": "id",           "type": "INT64",        "is_primary": True, "auto_id": True},
        {"name": "embedding",    "type": "FLOAT_VECTOR", "dim": 1024},
        {"name": "video_id",     "type": "VARCHAR",      "max_length": 256},   # 视频唯一标识
        {"name": "video_path",   "type": "VARCHAR",      "max_length": 1024},  # 视频文件路径
        {"name": "clip_index",   "type": "INT32"},                              # 片段序号
        {"name": "start_time",   "type": "FLOAT"},                              # 片段起始时间 (秒)
        {"name": "end_time",     "type": "FLOAT"},                              # 片段结束时间 (秒)
        {"name": "frame_index",  "type": "INT32"},                              # 帧在片段内的序号
        {"name": "timestamp",    "type": "FLOAT"},                              # 帧的绝对时间戳 (秒)
        {"name": "thumbnail",    "type": "VARCHAR",      "max_length": 1024},  # 缩略图路径
    ]
}
```

---

## 数据流

```
                        ┌──────────────────────────────────────────┐
                        │              Ingest Pipeline             │
                        │                                          │
  视频文件 ──►  decord 解码 ──► 滑动窗口切片 ──► EVA-CLIP encode ──► Milvus
                  │                │                                   │
                  ▼                ▼                                   │
              元信息提取       缩略图生成                              │
              (时长/分辨率)   (320x180 jpg)                           │
                                                                      │
                        ┌─────────────────────────────────────────────┘
                        │
                        │         ┌──────────────────────────────────────┐
                        │         │            Search Pipeline           │
                        │         │                                      │
  文本/图片 query ──► EVA-CLIP encode ──► Milvus ANN (top-50) ──► InternVL rerank (top-10)
                                                                        │
                                                                        ▼
                                                                   返回结果
                                                                 (视频片段 + 时间 + 缩略图)
```

---

## 运行方式

```bash
# 1. 启动 Milvus
docker compose up -d

# 2. 安装依赖
pip install -r requirements.txt

# 3. 入库视频
python scripts/ingest_videos.py --video-dir ./data/videos

# 4. 启动 API
python -m api.server

# 5. 检索
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "一个人在跑步", "top_k": 10}'
```

---

## 当前进度

P0 / P1 / P2 主链路已打通 (2026-05-25):
  - ingest: video_processor / embedding / pipeline 完成
  - search: retriever / reranker (CLIP + InternVL 可选) / pipeline 完成
  - api:    /ingest, /search/text, /search/image, /videos, DELETE, /health
  - scripts: ingest_videos.py, start_server.sh

未做: 关键帧场景检测、错误重试 backoff、`/videos/{id}/clips`、视频片段检索、P3 增强项

验证状态: 全部 12 个 Python 文件通过 `python3 -m py_compile`；运行时验证需要先
`docker compose up -d` 起 Milvus，并 `pip install -r requirements.txt` 后实测。
