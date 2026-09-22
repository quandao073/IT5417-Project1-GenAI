# Makefile for the CXR semantic retrieval POC.
# On Windows use Git Bash or WSL. Requires `make` (choco install make) and `uv`.

SHELL := /bin/bash
COMPOSE := docker compose
COMPOSE_GPU := docker compose -f compose.yaml -f compose.gpu.yaml
PY := uv run

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ps seed smoke test lint typecheck \
        audit-data build-corpus embed build-report-index validate-indexes \
        evaluate benchmark clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n",$$1,$$2}'

# --- environment ---
setup: ## Create .env, make the gitignored data/artifacts dirs, install deps
	@test -s .env || cp .env.example .env
	@mkdir -p data/{raw,canonical,indexes,cache/thumbnails,fixtures/synthetic} \
	          artifacts/{audit,metrics,reports,screenshots}
	uv sync --extra api --extra dev

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
embed: ## Embed the corpus and index the vectors into Qdrant (GPU)
	$(COMPOSE_GPU) --profile worker run --rm worker embed
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
lint: ## ruff
	$(PY) ruff check src tests
typecheck: ## mypy
	$(PY) mypy

# --- operations ---
clean: ## Remove the stack AND all volumes (destroys the vector index)
	$(COMPOSE) down -v
