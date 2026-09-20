# Makefile for the CXR semantic retrieval POC.
# On Windows use Git Bash or WSL. Requires `make` (choco install make).

SHELL := /bin/bash
COMPOSE := docker compose
COMPOSE_GPU := docker compose -f compose.yaml -f compose.gpu.yaml

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ps seed smoke test lint typecheck \
        audit ingest preprocess embed index graph evaluate benchmark clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

# --- environment ---
setup: ## Create .env, make the gitignored data/artifacts dirs, install dev deps
	@test -f .env || cp .env.example .env
	@mkdir -p data/{raw,staging,processed/images,manifests,indexes} artifacts/{metrics,reports,screenshots}
	pip install -e ".[api,dev]"

# --- stack lifecycle ---
up: ## Start web + api + qdrant + neo4j
	$(COMPOSE) up -d --build
down: ## Stop the stack, keep volumes
	$(COMPOSE) down
logs: ## Follow logs
	$(COMPOSE) logs -f
ps: ## Service status
	$(COMPOSE) ps

# --- demo data ---
seed: ## Load synthetic fixtures so the demo runs without the dataset
	$(COMPOSE) --profile worker run --rm worker seed --fixtures data/fixtures/synthetic

# --- ingestion pipeline (GPU) ---
audit: ## Phase 1 GATE: measure join rate, exit nonzero below 70%
	PYTHONPATH=src python -m cxr_retrieval.cli.main audit
ingest: ## Join metadata, build the master table and sample manifest
	$(COMPOSE) --profile worker run --rm worker ingest
preprocess: ## Normalize images, cap long side at 512 px (never upscale)
	$(COMPOSE) --profile worker run --rm worker preprocess
embed: ## Generate image embeddings on the GPU
	$(COMPOSE_GPU) --profile worker run --rm worker embed
index: ## Index vectors into Qdrant and build the SQLite FTS
	$(COMPOSE) --profile worker run --rm worker index
graph: ## RadGraph -> study_facts.parquet -> Neo4j
	$(COMPOSE) --profile worker run --rm worker graph

# --- testing and evaluation ---
test: ## Unit and integration tests
	pytest -q
smoke: ## Smoke test against the running stack
	./scripts/smoke_test.sh
evaluate: ## Run the three-axis benchmark
	$(COMPOSE) --profile worker run --rm worker evaluate
benchmark: evaluate ## Alias for evaluate
lint: ## ruff
	ruff check src tests
typecheck: ## mypy
	mypy

# --- operations ---
clean: ## Remove the stack AND all volumes (destroys index and graph)
	$(COMPOSE) down -v
