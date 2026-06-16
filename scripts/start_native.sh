#!/usr/bin/env bash
# 用 Mac native venv + MPS GPU 加速跑 vision-rag API
# Qdrant 在 docker 容器里, API/模型全部 native, 速度提升 10-20x
set -eo pipefail   # 不开 -u, 因为 HF_ENDPOINT 默认未设

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# launchctl / GUI services usually start with a minimal PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
# Some desktop shells inherit proxy settings from GUI agents. qdrant-client does
# not always bypass proxies for localhost unless NO_PROXY is explicit.
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "ERROR: neither 'docker compose' nor 'docker-compose' is available" >&2
  exit 1
fi

# ---- 默认走中文 + MPS 高质量模式 ----
export EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-cn_clip}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-OFA-Sys/chinese-clip-vit-large-patch14-336px}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-768}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-mps}"
export EMBEDDING_PRECISION="${EMBEDDING_PRECISION:-fp32}"   # MPS 上 fp32 最稳

# ---- HF 镜像 + 模型缓存 ----
# 模型已下载到 ./models, 启用 OFFLINE 模式跳过 HEAD 验证调用 (hf-mirror.com 在国内偶尔超时)
export HF_HOME="${HF_HOME:-$ROOT/models}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$ROOT/models}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# 仅当模型未下载时才需要这两行 (取消 OFFLINE 后启用):
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- Qdrant 主视频多模态索引, 走 host:6333 ----
export QDRANT_HOST="${QDRANT_HOST:-localhost}"
export QDRANT_PORT="${QDRANT_PORT:-6333}"
export QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"

# ---- API ----
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-28765}"

echo "============================================================"
echo "  Vision RAG · Native venv 模式 (MPS 加速 + 中文 CLIP)"
echo "============================================================"
echo "  EMBEDDING_BACKEND  = $EMBEDDING_BACKEND"
echo "  EMBEDDING_MODEL    = $EMBEDDING_MODEL"
echo "  EMBEDDING_DEVICE   = $EMBEDDING_DEVICE (MPS = Apple Silicon GPU)"
echo "  HF_ENDPOINT        = $HF_ENDPOINT"
echo "  QDRANT             = $QDRANT_HOST:$QDRANT_PORT"
echo "  QDRANT DASHBOARD   = http://localhost:$QDRANT_PORT/dashboard"
echo "  API                = http://$API_HOST:$API_PORT/"
echo "============================================================"

# ---- 1. 确保容器 Qdrant 起来 (API 容器可以不起) ----
echo "==> [1/3] 检查 Qdrant 容器"
if ! "${COMPOSE[@]}" ps --status running --services 2>/dev/null | grep -q '^qdrant$'; then
  echo "    Qdrant 未运行, 启动..."
  "${COMPOSE[@]}" up -d qdrant
fi
until curl -fs "http://localhost:${QDRANT_PORT}/healthz" >/dev/null 2>&1; do
  echo "    等待 Qdrant healthy..."; sleep 3
done
echo "    Qdrant ready."

# ---- 2. 停掉已有 API (避免端口冲突) ----
echo "==> [2/3] 停掉已有 API (避免端口 $API_PORT 冲突)"
"${COMPOSE[@]}" stop api 2>/dev/null || true
if command -v lsof >/dev/null 2>&1; then
  api_pids="$(lsof -tiTCP:"$API_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$api_pids" ]; then
    echo "    停止占用端口 $API_PORT 的 native API: $api_pids"
    kill $api_pids 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      sleep 1
      api_pids="$(lsof -tiTCP:"$API_PORT" -sTCP:LISTEN 2>/dev/null || true)"
      [ -z "$api_pids" ] && break
    done
    if [ -n "$api_pids" ]; then
      echo "    native API 未退出, 强制停止: $api_pids"
      kill -9 $api_pids 2>/dev/null || true
    fi
  fi
fi
echo "    OK"

# ---- 3. native venv 起 uvicorn ----
echo "==> [3/3] 用 venv 启动 API (MPS GPU 加速)"
echo "    首次加载 CN-CLIP-Large ~75s, 之后会很快"
echo
exec "$ROOT/.venv/bin/python" -m api.server
