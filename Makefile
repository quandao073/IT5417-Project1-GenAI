# Makefile cho POC CXR semantic retrieval.
# Trên Windows dùng Git Bash hoặc WSL. Cần `make` (choco install make).

SHELL := /bin/bash
COMPOSE := docker compose
COMPOSE_GPU := docker compose -f compose.yaml -f compose.gpu.yaml

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ps seed smoke test lint typecheck \
        audit ingest preprocess embed index graph evaluate benchmark disk clean

help: ## Liệt kê target
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

# --- môi trường ---
setup: ## Tạo .env, dựng thư mục data/artifacts (bị gitignore nên không có sau khi clone), cài dependency dev
	@test -f .env || cp .env.example .env
	@mkdir -p data/{raw,staging,processed/images,manifests,indexes} artifacts/{metrics,reports,screenshots}
	pip install -e ".[api,dev]"

# --- vòng đời stack ---
up: ## Khởi động web + api + qdrant + neo4j
	$(COMPOSE) up -d --build
down: ## Dừng stack, giữ nguyên volume
	$(COMPOSE) down
logs: ## Theo dõi log
	$(COMPOSE) logs -f
ps: ## Trạng thái service
	$(COMPOSE) ps

# --- dữ liệu demo ---
seed: ## Nạp fixture tổng hợp để demo chạy được khi không có dataset
	$(COMPOSE) --profile worker run --rm worker seed --fixtures data/fixtures/synthetic

# --- pipeline ingestion (cần GPU) ---
audit: ## Phase 1: đo tỉ lệ join - CỔNG CHẶN, thoát khác 0 nếu < 70%
	PYTHONPATH=src python -m cxr_retrieval.cli.main audit
ingest: ## Join metadata, dựng master table + sample manifest
	$(COMPOSE) --profile worker run --rm worker ingest
preprocess: ## Chuẩn hóa ảnh, cap cạnh dài 512 px (không upscale)
	$(COMPOSE) --profile worker run --rm worker preprocess
embed: ## Sinh image embedding trên GPU
	$(COMPOSE_GPU) --profile worker run --rm worker embed
index: ## Index vector vào Qdrant + dựng SQLite FTS
	$(COMPOSE) --profile worker run --rm worker index
graph: ## RadGraph -> study_facts.parquet -> Neo4j
	$(COMPOSE) --profile worker run --rm worker graph

# --- kiểm thử & đánh giá ---
test: ## Unit + integration test
	pytest -q
smoke: ## Smoke test trên stack đang chạy
	./scripts/smoke_test.sh
evaluate: ## Chạy benchmark ba trục
	$(COMPOSE) --profile worker run --rm worker evaluate
benchmark: evaluate ## Alias của evaluate
lint: ## ruff
	ruff check src tests
typecheck: ## mypy
	mypy

# --- vận hành ---
disk: ## Kiểm tra ngân sách dung lượng
	python scripts/check_disk_budget.py
clean: ## Xóa stack VÀ toàn bộ volume (mất index/graph)
	$(COMPOSE) down -v
