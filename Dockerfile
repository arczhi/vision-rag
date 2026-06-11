# syntax=docker/dockerfile:1.6
# Vision RAG API 容器
# linux/amd64 平台: decord/PyAV 有官方 wheel, 比 Apple Silicon 原生兼容性好
FROM --platform=linux/amd64 python:3.11-slim

# 系统依赖: ffmpeg 用于 PyAV/decord, libgl/libglib 用于 opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 第一层: torch CPU 版 (~200 MB, 比 GPU 版小一个数量级)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchvision==0.18.1

# 第二层: 其他依赖 (走清华镜像加速; torch 已装不会被重装)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r /app/requirements.txt

# 第三层: 项目代码 (后改也只 invalidate 这一层)
COPY . /app

# 缺省环境: HF 镜像 + Qdrant 服务名 + 模型缓存挂卷路径
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/models \
    HUGGINGFACE_HUB_CACHE=/app/models \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    QDRANT_GRPC_PORT=6334 \
    API_HOST=0.0.0.0 \
    API_PORT=28765

EXPOSE 28765

# 健康检查 (lifespan 完成后 /health 返回 200)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=120s \
    CMD curl -fs http://localhost:28765/health || exit 1

CMD ["python", "-m", "api.server"]
