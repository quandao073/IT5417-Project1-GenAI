#!/usr/bin/env bash
# One command for the local demo (README, "Chạy demo"): Qdrant + Neo4j in Docker, the API with
# uvicorn, the web UI with `pnpm dev`. Ctrl+C stops the API and the web; the containers stay up
# (`docker compose --profile kg stop` stops them). Git Bash on Windows, or any bash.
set -euo pipefail
cd "$(dirname "$0")/.."

API=http://127.0.0.1:8000
docker compose --profile kg up -d qdrant neo4j

if curl -sf "$API/api/v1/health" > /dev/null; then
  echo "API already running at $API"
else
  echo "starting the API (the first start loads the encoder, SapBERT and the score matrix: ~1 min)"
  PYTHONIOENCODING=utf-8 uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
  API_PID=$!
  trap 'kill $API_PID 2>/dev/null || true' EXIT
  for _ in $(seq 1 180); do
    curl -sf "$API/api/v1/health" > /dev/null && break
    kill -0 $API_PID 2>/dev/null || { echo "the API exited; see its log above"; exit 1; }
    sleep 1
  done
  # 503 means not ready: the reason is in the body (an index, Qdrant or the encoder disagree)
  curl -sf "$API/api/v1/health" > /dev/null || { curl -s "$API/api/v1/health"; echo; exit 1; }
fi
curl -s "$API/api/v1/health"; echo

cd web
[ -d node_modules ] || pnpm install --frozen-lockfile
echo "web UI: http://localhost:3000"
pnpm dev
