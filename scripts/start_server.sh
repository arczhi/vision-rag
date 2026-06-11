#!/usr/bin/env bash
# 一键启动 Vision RAG (容器化版)
#
# 链路: docker compose up -d → 等 Qdrant + API 健康 → 打印访问地址
#
# 首次启动会:
#   1. 构建 vision-rag-api 镜像 (~2 GB, 含 torch CPU + decord + cv2)
#   2. API 容器内首次启动会自动从 hf-mirror.com 下载 ViT-B/32 (~577 MB)
#      模型缓存挂在 ./models, 后续启动直接命中
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-28765}"
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "ERROR: neither 'docker compose' nor 'docker-compose' is available" >&2
  exit 1
fi

echo "==> [1/3] docker compose up -d (build if needed)"
"${COMPOSE[@]}" up -d --build

echo
echo "==> [2/3] 等待 Qdrant 健康检查..."
for i in $(seq 1 90); do
  if curl -fs http://localhost:6333/healthz >/dev/null 2>&1; then
    echo "    Qdrant ready."
    break
  fi
  sleep 2
done

echo
echo "==> [3/3] 等待 API 服务就绪 (首次启动需下载模型, 可能 1-3 分钟)..."
for i in $(seq 1 180); do
  if curl -fs "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
    echo "    API ready."
    break
  fi
  sleep 2
done

echo
echo "================================================================"
echo "  Web UI : http://localhost:${API_PORT}/"
echo "  API    : http://localhost:${API_PORT}/docs"
echo "  Qdrant : http://localhost:6333/dashboard"
echo "  日志   : ${COMPOSE[*]} logs -f api"
echo "================================================================"
