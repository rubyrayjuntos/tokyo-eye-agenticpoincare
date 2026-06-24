.PHONY: help up down kill build rebuild logs ps migrate psql dev test test-integration test-all lint format typecheck clean

# Configure make to use bash instead of sh for better error handling
SHELL := /bin/bash -o pipefail

# ---------------------------------------------------------------------------
# Docker Compose shortcuts
# ---------------------------------------------------------------------------

up: ## Start all services (detached)
	docker compose up -d
	@echo "✓ All services started. Check status with: make ps"

down: ## Stop all services gracefully
	docker compose down

kill: ## Force-stop and remove all containers and volumes
	docker compose down -v --remove-orphans
	docker rm -f tokyoeye_db tokyoeye_agent tokyoeye_science 2>/dev/null || true
	@echo "✓ All containers and volumes removed"

build: ## Rebuild all images from scratch (no cache)
	docker compose build --no-cache

rebuild: ## Rebuild and restart all services
	docker compose up -d --build
	@echo "✓ Services rebuilt and restarted"

build-agent: ## Rebuild agent image only
	docker compose build --no-cache agent

build-science: ## Rebuild science image only
	docker compose build --no-cache science

build-db: ## Rebuild database (normally not needed)
	docker compose build --no-cache db

# ---------------------------------------------------------------------------
# Individual services
# ---------------------------------------------------------------------------

up-db: ## Start only the database
	docker compose up -d db
	@docker compose logs -f db | grep -q "ready to accept connections" && echo "✓ Database ready"

up-agent: ## Start DB + agent (for API-only development)
	docker compose up -d db agent
	@echo "✓ Database and agent started. API available at http://localhost:8000"

up-science: ## Start DB + science container (for compute jobs)
	docker compose up -d db science
	@echo "✓ Database and science container ready"

# ---------------------------------------------------------------------------
# Logs & status
# ---------------------------------------------------------------------------

logs: ## Tail logs from all services
	docker compose logs -f

logs-agent: ## Tail agent logs only
	docker compose logs -f agent

logs-science: ## Tail science container logs only
	docker compose logs -f science

logs-db: ## Tail database logs only
	docker compose logs -f db

ps: ## Show running containers and status
	docker compose ps

status: ## Show detailed service status
	@echo "=== Container Status ===" && \
	docker compose ps && \
	echo "" && \
	echo "=== Database Health ===" && \
	docker compose exec -T db pg_isready -U tokyoeye -d tokyoeye_dev || echo "Database not ready" && \
	echo "" && \
	echo "=== Agent Health ===" && \
	curl -s http://localhost:8000/health | jq . || echo "Agent not responding"

# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

migrate: ## Run database migrations
	docker compose exec agent python -c "import asyncio; from data.db import run_migrations; asyncio.run(run_migrations())"
	@echo "✓ Migrations complete"

psql: ## Open interactive psql shell
	docker compose exec db psql -U tokyoeye -d tokyoeye_dev

db-reset: ## Reset database (destructive - use with caution)
	@read -p "Are you sure? This will delete all data. Type 'yes' to confirm: " confirm && \
	[ "$$confirm" = "yes" ] && \
	docker compose down -v && \
	docker compose up -d db && \
	echo "✓ Database reset complete" || echo "Aborted"

# ---------------------------------------------------------------------------
# Development (local, no Docker)
# ---------------------------------------------------------------------------

dev: ## Run agent locally with uvicorn (hot-reload, requires local setup)
	uv run uvicorn agent.coordinator.app:app --reload --port 8000

dev-science: ## Run science scripts locally (requires CUDA/GPU setup)
	uv run python -m science.dtie.v5.orchestrator.pipeline

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run unit tests (no integration tests)
	uv run pytest tests/ -v --tb=short -m "not integration"

test-integration: ## Run integration tests (requires running DB)
	uv run pytest tests/ -v --tb=short -m "integration"

test-all: ## Run all tests (unit + integration)
	uv run pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=agent --cov=science --cov=data --cov-report=html --tb=short

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Check code style and imports
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format code and fix issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run mypy type checking
	uv run mypy science/ data/ agent/ --ignore-missing-imports

quality: lint typecheck ## Run all quality checks

# ---------------------------------------------------------------------------
# Cleanup and maintenance
# ---------------------------------------------------------------------------

clean: ## Remove local build artifacts, caches, and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -delete
	@echo "✓ Cleaned up artifacts"

docker-clean: ## Remove dangling Docker images and build cache
	docker image prune -f
	docker builder prune -f
	@echo "✓ Docker cleanup complete"

prune: clean docker-clean ## Deep cleanup of all build artifacts

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

pip-compile: ## Regenerate pinned requirements files from pyproject.toml
	uv pip compile pyproject.toml --extra agent -o requirements-agent.txt
	uv pip compile pyproject.toml --extra science -o requirements-science.txt
	@echo "✓ Requirements files regenerated"

# ---------------------------------------------------------------------------
# Info and help
# ---------------------------------------------------------------------------

help: ## Show this help message
	@echo "Tokyo Eye Agenticpoincare — Make Commands" && \
	echo "" && \
	grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

version: ## Show version and dependency info
	@echo "Tokyo Eye Agenticpoincare v0.1.0" && \
	echo "" && \
	echo "Python:" && python --version && \
	echo "" && \
	echo "Docker:" && docker --version && \
	echo "Docker Compose:" && docker compose --version

.DEFAULT_GOAL := help
