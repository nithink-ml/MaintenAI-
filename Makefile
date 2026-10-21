# ============================================================
# MaintenAIMakefile
# ============================================================

.PHONY: help install dev lint format typecheck test test-unit test-integration \
        test-ml test-rag test-api test-e2e coverage \
        infra-up infra-down db-migrate db-downgrade \
        data-download data-prepare data-version \
        eda eda-cmapss eda-ims eda-ai4i \
        train-anomaly train-rul train-failure train-all \
        simulate dev-api dev-frontend \
        docker-build up down logs clean

PYTHON := python
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# ---- Help ----
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

# ---- Installation ----
install: ## Install production dependencies
	pip install -e .

dev: ## Install with dev dependencies
	pip install -e ".[dev,notebooks]"
	pre-commit install || true

# ---- Code Quality ----
lint: ## Run linter
	$(RUFF) check .

format: ## Format code
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck: ## Run type checker
	$(MYPY) factorypulse/ services/ ml/ rag/ agents/

quality: lint typecheck ## Run all code quality checks

# ---- Testing ----
test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests
	$(PYTEST) tests/unit/ -v -m unit

test-integration: ## Run integration tests (requires infrastructure)
	$(PYTEST) tests/integration/ -v -m integration

test-ml: ## Run ML model tests
	$(PYTEST) tests/ml/ -v -m ml

test-rag: ## Run RAG pipeline tests
	$(PYTEST) tests/rag/ -v -m rag

test-api: ## Run API endpoint tests
	$(PYTEST) tests/api/ -v -m api

test-e2e: ## Run end-to-end tests
	$(PYTEST) tests/e2e/ -v -m e2e

coverage: ## Run tests with coverage report
	$(PYTEST) tests/ --cov --cov-report=html --cov-report=term-missing

# ---- Infrastructure ----
infra-up: ## Start infrastructure (TimescaleDB, Redis, Qdrant, MLflow)
	docker compose up -d timescaledb redis qdrant mlflow

infra-down: ## Stop infrastructure
	docker compose down

# ---- Database ----
db-migrate: ## Run database migrations
	alembic upgrade head

db-downgrade: ## Rollback last migration
	alembic downgrade -1

db-reset: ## Reset database (DESTRUCTIVE)
	alembic downgrade base
	alembic upgrade head

# ---- Data ----
data-download: ## Download raw datasets
	$(PYTHON) -m ml.common.data_download

data-prepare: ## Run data preparation pipeline
	$(PYTHON) -m ml.common.data_prepare

data-version: ## Version current data with DVC
	dvc add data/raw/ data/processed/
	dvc push

# ---- Exploratory Data Analysis ----
eda: ## Run EDA for all datasets
	$(PYTHON) -m ml.common.eda.runner --output-dir docs/eda -v

eda-cmapss: ## Run EDA for C-MAPSS only
	$(PYTHON) -m ml.common.eda.runner --dataset cmapss --output-dir docs/eda -v

eda-ims: ## Run EDA for IMS Bearing only
	$(PYTHON) -m ml.common.eda.runner --dataset ims --output-dir docs/eda -v

eda-ai4i: ## Run EDA for AI4I only
	$(PYTHON) -m ml.common.eda.runner --dataset ai4i --output-dir docs/eda -v

# ---- ML Training ----
train-anomaly: ## Train anomaly detection models
	$(PYTHON) -m ml.anomaly.train

train-rul: ## Train RUL prediction models
	$(PYTHON) -m ml.rul.train

train-failure: ## Train failure classification models
	$(PYTHON) -m ml.failure_classification.train

train-all: train-anomaly train-rul train-failure ## Train all models

# ---- Evaluation ----
evaluate: ## Run unified model evaluation
	$(PYTHON) -m ml.evaluation.run

# ---- Services (Local Dev) ----
simulate: ## Start sensor simulator
	$(PYTHON) -m services.simulator --machines 20 --rate 200

dev-ingestion: ## Start ingestion service
	uvicorn services.ingestion.app:app --host 0.0.0.0 --port 8003 --reload

dev-inference: ## Start inference service
	uvicorn services.inference.app:app --host 0.0.0.0 --port 8001 --reload

dev-api: ## Start API gateway
	uvicorn services.api_gateway.app:app --host 0.0.0.0 --port 8000 --reload

dev-copilot: ## Start copilot agent service
	uvicorn services.copilot.app:app --host 0.0.0.0 --port 8002 --reload

dev-frontend: ## Start frontend dev server
	cd apps/frontend && npm run dev

# ---- Docker ----
docker-build: ## Build all Docker images
	docker compose build

up: ## Start full stack via Docker Compose
	docker compose up -d

down: ## Stop full stack
	docker compose down

logs: ## Tail Docker Compose logs
	docker compose logs -f --tail=100

# ---- RAG ----
rag-ingest: ## Ingest maintenance documents into vector DB
	$(PYTHON) -m rag.ingestion.run

rag-evaluate: ## Evaluate RAG retrieval quality
	$(PYTHON) -m rag.evaluation.run

# ---- Cleanup ----
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage
