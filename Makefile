.PHONY: up down kill build logs ps migrate test lint

# ---------------------------------------------------------------------------
# Docker Compose shortcuts
# ---------------------------------------------------------------------------

up: ## Start all services (detached)
	docker compose up -d

down: ## Stop all services gracefully
	docker compose down

kill: ## Force-stop and remove all containers, volumes
	docker compose down -v --remove-orphans
	docker rm -f tokyoeye_db tokyoeye_agent tokyoeye_science 2>/dev/null || true

build: ## Rebuild all images from scratch
	docker compose build --no-cache

rebuild: ## Rebuild and restart
	docker compose up -d --build

# ---------------------------------------------------------------------------
# Individual services
# ---------------------------------------------------------------------------

up-db: ## Start only the database
	docker compose up -d db

up-agent: ## Start DB + agent
	docker compose up -d db agent

up-science: ## Start DB + science container
	docker compose up -d db science

# ---------------------------------------------------------------------------
# Logs & status
# ---------------------------------------------------------------------------

logs: ## Tail logs from all services
	docker compose logs -f

logs-agent: ## Tail agent logs
	docker compose logs -f agent

logs-db: ## Tail database logs
	docker compose logs -f db

ps: ## Show running containers
	docker compose ps

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

migrate: ## Run database migrations
	docker compose exec agent python -c "import asyncio; from data.db import run_migrations; asyncio.run(run_migrations())"

psql: ## Open psql shell
	docker compose exec db psql -U tokyoeye -d tokyoeye_dev

# ---------------------------------------------------------------------------
# Development (local, no Docker)
# ---------------------------------------------------------------------------

dev: ## Run agent locally with uvicorn (hot-reload)
	uv run uvicorn agent.coordinator.app:app --reload --port 8000

test: ## Run unit tests
	uv run pytest tests/ -v --tb=short -m "not integration"

test-integration: ## Run integration tests (requires DB)
	uv run pytest tests/ -v --tb=short -m "integration"

test-all: ## Run all tests
	uv run pytest tests/ -v --tb=short

lint: ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format code
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run mypy
	uv run mypy science/ data/normalizer/ agent/ --ignore-missing-imports

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
