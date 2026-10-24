# Makefile for the CXR semantic retrieval POC.
# On Windows use Git Bash or WSL. Requires `make` (choco install make) and `uv`.

SHELL := /bin/bash
COMPOSE := docker compose
COMPOSE_GPU := docker compose -f compose.yaml -f compose.gpu.yaml
PY := uv run

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ps seed smoke test lint typecheck \
        audit-data build-corpus build-probe-pool embed index \
        build-report-index validate-indexes \
        evaluate benchmark eval-parser eval-retrieval clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'

# --- environment ---
setup: ## Create .env, make the gitignored data/artifacts dirs, install deps
	@test -s .env || cp .env.example .env
	@mkdir -p data/{raw,canonical,indexes,cache/thumbnails,fixtures/synthetic} \
	          artifacts/{audit,metrics,reports,screenshots}
	uv sync --all-extras

# --- stack lifecycle ---
up: ## Start web + api + qdrant
	$(COMPOSE) up -d --build
down: ## Stop the stack, keep volumes
	$(COMPOSE) down
logs: ## Follow logs
	$(COMPOSE) logs -f
ps: ## Service status
	$(COMPOSE) ps

# --- demo data ---
seed: ## Load synthetic fixtures so the demo runs without the dataset
	$(PY) cxr seed --fixtures data/fixtures/synthetic

# --- corpus and index pipeline ---
audit-data: ## Measure the join rate between CheXpert Plus and CheXpert-small
	$(PY) cxr audit
build-corpus: ## Join, sample and write the canonical Parquet tables
	$(PY) cxr build-corpus
embed: ## Embed the corpus into shard files (CPU, native, ~35 min, resumable)
	$(PY) cxr embed
# GPU alternative via the Docker worker (needs WSL2 + NVIDIA Container Toolkit):
#   $(COMPOSE_GPU) --profile worker run --rm worker embed
index: ## Upsert the shards into Qdrant and register the build (needs `make up` or `docker compose up -d qdrant`)
	$(PY) cxr index
build-probe-pool: ## Pool of studies outside the corpus for the image probe
	$(PY) cxr build-probe-pool
build-report-index: ## Build the SQLite FTS5 report index
	$(PY) cxr build-report-index
validate-indexes: ## Check canonical tables, Qdrant and the report index agree
	$(PY) cxr validate-indexes

# --- testing and evaluation ---
test: ## Unit, data-contract and integration tests
	$(PY) pytest -q
smoke: ## Smoke test against the running stack
	./scripts/smoke_test.sh
evaluate: ## Run the three-axis benchmark
	$(PY) cxr evaluate
benchmark: evaluate ## Alias for evaluate
eval-parser: ## Score rule / llm / auto parsers on the held-out benchmark (needs Ollama)
	$(PY) cxr eval-parser
eval-retrieval: ## Score the one search path on the held-out benchmark (needs Qdrant, report index; OpenAI key in .env for auto)
	$(PY) cxr eval-retrieval
lint: ## ruff
	$(PY) ruff check src tests
typecheck: ## mypy
	$(PY) mypy

# --- operations ---
clean: ## Remove the stack AND all volumes (destroys the vector index)
	$(COMPOSE) down -v
