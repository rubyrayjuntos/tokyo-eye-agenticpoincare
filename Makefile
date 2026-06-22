.PHONY: up down kill build logs ps migrate test lint start stop dev

# ---------------------------------------------------------------------------
# Primary commands — use these for daily development
# ---------------------------------------------------------------------------

up: ## Start dev environment: DB + Agent in Docker, Frontend locally (Vite)
	@echo "🔬 Starting Tokyo Eye (dev mode)..."
	@echo "   Dashboard: http://localhost:3000  (Vite dev server — local)"
	@echo "   Agent API: http://localhost:8000  (Docker)"
	@echo ""
	@lsof -ti :3000 | xargs -r kill 2>/dev/null || true
	docker compose up -d db agent science
	@echo "⏳ Waiting for agent to be healthy..."
	@sleep 5
	@echo "✓ Backend ready. Starting frontend..."
	cd visualizer/frontend && npm run dev

down: ## Stop everything (Docker + local frontend) — preserves DB data
	@lsof -ti :3000 | xargs -r kill 2>/dev/null || true
	docker compose down

stop: down ## Alias for down

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build: ## Build Docker images (DB + Agent + Science only — frontend runs locally)
	docker compose build db agent science

build-all: ## Build ALL Docker images including frontend (for CI/production)
	docker compose --profile full build --no-cache

# ---------------------------------------------------------------------------
# Alternative start modes
# ---------------------------------------------------------------------------

start: up ## Alias for up

dev: ## Run agent locally (no Docker for agent, just DB in Docker)
	@lsof -ti :8000 | xargs -r kill 2>/dev/null || true
	docker compose up -d db
	@sleep 3
	PYTHONPATH=. uv run uvicorn agent.coordinator.app:app --reload --port 8000

dev-frontend: ## Run frontend only (assumes agent is already running)
	cd visualizer/frontend && npm run dev

dev-all: ## Run everything locally: DB in Docker, agent + frontend native
	@lsof -ti :3000 | xargs -r kill 2>/dev/null || true
	@lsof -ti :8000 | xargs -r kill 2>/dev/null || true
	docker compose up -d db
	@sleep 3
	PYTHONPATH=. uv run uvicorn agent.coordinator.app:app --reload --port 8000 &
	cd visualizer/frontend && npm run dev

up-docker: ## Start full stack in Docker (production-like, all containers)
	docker compose --profile full up -d
	@echo "✓ All containers starting. Run 'make logs' to watch."

# ---------------------------------------------------------------------------
# Logs & status
# ---------------------------------------------------------------------------

logs: ## Tail logs from all running containers
	docker compose logs -f

logs-agent: ## Tail agent logs only
	docker compose logs -f agent

logs-science: ## Tail science container logs
	docker compose logs -f science

ps: ## Show running containers
	docker compose ps

health: ## Check health of all services
	@echo "Database:"
	@docker compose exec db pg_isready -U tokyoeye -d tokyoeye_dev 2>/dev/null && echo "  ✓ healthy" || echo "  ✗ down"
	@echo "Agent API:"
	@curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "  ✓ healthy" || echo "  ✗ down"
	@echo "Frontend:"
	@curl -sf http://localhost:3000 > /dev/null 2>&1 && echo "  ✓ healthy" || echo "  ✗ down"
	@echo "Science container:"
	@docker compose ps science --format '{{.State}}' 2>/dev/null | grep -q running && echo "  ✓ running (idle)" || echo "  ✗ down"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

migrate: ## Run database migrations
	docker compose exec agent python -c "import asyncio; from data.db import open_pool, run_migrations; asyncio.run(open_pool()); asyncio.run(run_migrations())"

psql: ## Open psql shell
	docker compose exec db psql -U tokyoeye -d tokyoeye_dev

# ---------------------------------------------------------------------------
# Science container (manual dispatch)
# ---------------------------------------------------------------------------

run-pipeline: ## Run full pipeline on a structure. Usage: make run-pipeline STRUCTURE=4obe
	docker compose run --rm science python -m science.dtie.v5.orchestrator.pipeline --structure $(STRUCTURE)

run-gnn: ## Run GNN inference only. Usage: make run-gnn STRUCTURE=4obe
	docker compose run --rm science python -m science.dtie.v5.gnn.runner --structure $(STRUCTURE)

science-check: ## Verify science container can load the model
	docker compose run --rm science python -m science.dtie.v5.gnn.runner --check

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run unit tests
	PYTHONPATH=. uv run pytest tests/ -v --tb=short -m "not integration"

test-integration: ## Run integration tests (requires DB)
	PYTHONPATH=. uv run pytest tests/ -v --tb=short -m "integration"

test-all: ## Run all tests
	PYTHONPATH=. uv run pytest tests/ -v --tb=short

lint: ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format code
	uv run ruff format .
	uv run ruff check --fix .

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

kill: ## Force-stop and remove all containers (⚠️  KEEPS volumes/data — use nuke for full wipe)
	@lsof -ti :3000 | xargs -r kill 2>/dev/null || true
	@lsof -ti :8000 | xargs -r kill 2>/dev/null || true
	docker compose down --remove-orphans
	docker rm -f tokyoeye_db tokyoeye_agent tokyoeye_science tokyoeye_frontend 2>/dev/null || true

nuke: ## ⚠️  DESTROY everything including database data. Irreversible.
	@echo "⚠️  This will PERMANENTLY DELETE your database (hypotheses, agent memory, pipeline results)."
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ] || (echo "Aborted." && exit 1)
	@lsof -ti :3000 | xargs -r kill 2>/dev/null || true
	@lsof -ti :8000 | xargs -r kill 2>/dev/null || true
	docker compose down -v --remove-orphans
	docker rm -f tokyoeye_db tokyoeye_agent tokyoeye_science tokyoeye_frontend 2>/dev/null || true
	@echo "💀 All data destroyed."

backup: ## Dump the database to a timestamped SQL file
	@mkdir -p backups
	docker compose exec db pg_dump -U tokyoeye tokyoeye_dev > backups/tokyoeye_$$(date +%Y%m%d_%H%M%S).sql
	@echo "✓ Backup saved to backups/"

clean-vite: ## Clear Vite cache (fixes permission errors after Docker runs)
	sudo rm -rf visualizer/frontend/node_modules/.vite

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
