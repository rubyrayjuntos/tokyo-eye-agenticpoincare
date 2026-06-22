# Tokyo Eye — Agentic Poincaré Platform

A clean, modern hybrid research platform for advanced hyperbolic Graph Neural Networks, DTIE pipelines, agentic scientific workflows, and interactive structural biology visualization.

**Status:** Fresh consolidation repository (June 2026) — Recently optimized for performance and local development

This repo is a deliberate, clean-slate synthesis of the best surviving components from previous fragmented work on the Tokyo Eyes project.

## Quick Links

- [AGENTS.md](AGENTS.md) — Authoritative project context
- [CONSOLIDATION_PLAN.md](CONSOLIDATION_PLAN.md) — Strategy and effort estimate
- [MIGRATION_MAP.md](MIGRATION_MAP.md) — Detailed source-to-destination mapping
- [docs/adr/](docs/adr/) — Architecture Decision Records
- [data/governance/](data/governance/) — Data package contract and schemas

## High-Level Structure

- `science/dtie/` — Core DTIE pipelines (v3 mature + v4 advanced)
- `agent/` — Coordinator + tool interfaces (FastAPI)
- `visualizer/` — Poincaré disc frontend (React)
- `data/` — Schemas, governance, and database layer
- `experiments/` — Training and analysis notebooks
- `docs/` — Findings, architecture, specifications
- `infra/` — Deployment configurations (AWS, etc.)
- `tests/` — Unit and integration tests

## Getting Started

### Prerequisites

- **Python 3.11+** (uv recommended for fast installs)
- **Docker & Docker Compose** (for PostgreSQL + containerized services)
- **Node.js 18+** (optional, for the visualizer frontend)
- **NVIDIA GPU + CUDA 12.1** (optional, for GPU-accelerated science container)

### Local Development Setup

#### 1. Clone & Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/rubyrayjuntos/tokyo-eye-agenticpoincare.git
cd tokyo-eye-agenticpoincare

# Install Python dependencies (lightweight — no torch needed on host)
uv sync --extra dev

# Copy environment file and configure as needed
cp .env.example .env
```

#### 2. Start Local Services

```bash
# Start PostgreSQL + pgvector database
make up-db

# Or start all services (DB + agent + science containers)
make up

# Check service status
make ps
```

#### 3. Run Tests

```bash
# Unit tests (no integration, no heavy deps needed)
make test

# Integration tests (requires running DB)
make test-integration

# All tests with coverage
make test-coverage
```

#### 4. Local Development (Lightweight)

```bash
# Run agent coordinator locally with hot-reload (no Docker needed)
make dev

# Access API at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

#### 5. Compute-Heavy Operations (GPU)

```bash
# Run GNN inference / full pipeline in Docker with GPU
docker compose run science python -m science.dtie.v5.orchestrator.pipeline --structure 4OBE

# Or for specific phases
docker compose run science python -m science.dtie.v5.orchestrator.pipeline --phase 3 --structure 4OBE
```

#### 6. Visualizer Frontend (Optional)

```bash
cd visualizer/server && npm install && npm run dev
```

### Docker Architecture

The project uses **separate containers** to isolate heavy ML dependencies and optimize for different workloads:

#### Containers

| Service | Purpose | Dependencies | Memory | GPU |
|---------|---------|--------------|--------|-----|
| `db` | PostgreSQL 16 + pgvector | Lightweight | ~256MB | ✗ |
| `agent` | FastAPI coordinator | FastAPI, Uvicorn, boto3 | ~300MB | ✗ |
| `science` | Compute engine | PyTorch, GNNs, RDKit, CUDA | ~4GB | ✓ |

#### Network

- Internal Docker network `tokyoeye_net` isolates services
- Database shared between agent and science containers
- Agent dispatches compute jobs to science container asynchronously

### Database Setup

#### Local Development

```bash
# PostgreSQL 16 + pgvector via Docker
# Connection: postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/tokyoeye_dev

# Start database
make up-db

# Open psql shell
make psql

# Run migrations
make migrate

# Reset database (destructive)
make db-reset
```

#### Migrations

All 29 migrations apply identically to local PostgreSQL and Aurora in production. The same schema is used everywhere.

### Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key variables:

```bash
# Database
DATABASE_URL=postgresql://tokyoeye:tokyoeye_dev_local@db:5432/tokyoeye_dev

# Agent API
JWT_SECRET=dev-secret-change-in-production
REQUIRE_AUTH=false
PORT=8000

# AWS (if using Bedrock)
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# LLM provider
ANTHROPIC_API_KEY=...
```

## Development Workflow

### Make Targets

Use `make help` to see all available commands:

```bash
# Start/stop
make up              # Start all services
make down            # Stop gracefully
make kill            # Force-stop and remove all

# Rebuild
make build           # Rebuild all images (no cache)
make rebuild         # Rebuild and restart services
make build-agent     # Rebuild agent only
make build-science   # Rebuild science only

# Logs & monitoring
make logs            # Tail all logs
make logs-agent      # Agent logs only
make logs-science    # Science container logs
make ps              # Show container status
make status          # Detailed status + health checks

# Database
make migrate         # Run migrations
make psql            # Open psql shell
make db-reset        # Reset database (destructive)

# Testing & quality
make test            # Unit tests
make test-all        # All tests
make test-coverage   # Coverage report
make quality         # Lint + typecheck
make lint            # Code style check
make format          # Auto-format code
make typecheck       # mypy type checking

# Cleanup
make clean           # Remove artifacts, caches
make docker-clean    # Prune Docker images
make prune           # Deep cleanup
```

### Code Quality

```bash
# Check code style and imports
make lint

# Auto-format code
make format

# Type checking (mypy)
make typecheck

# All checks combined
make quality
```

### Running Tests

```bash
# Unit tests only (no integration)
make test

# Integration tests (requires running DB)
make test-integration

# All tests
make test-all

# With coverage report
make test-coverage
```

## Architecture Notes

### Container Optimization

- **Agent container** (`python:3.11-slim`): Minimal, fast startup for API serving
- **Science container** (`pytorch/pytorch:2.3.1-cuda12.1`): GPU-enabled, heavy dependencies isolated
- **Database container** (`pgvector:pg16`): Shared persistence layer with vector support

### Performance

- **Connection pooling**: `psycopg-pool` with 2-10 connections
- **Uvicorn workers**: 4 concurrent workers (configurable via `WEB_CONCURRENCY`)
- **Layer caching**: Docker layers ordered by change frequency for fast rebuilds
- **Dependency pinning**: All transitive dependencies pinned for reproducibility

### Data Governance

- **Structure-centric**: `structure.cif` is the canonical immutable input
- **Sidecar outputs**: All Tokyo Eyes outputs are derived (never modify structure)
- **Package contract**: See [data/governance/README.md](data/governance/README.md)

## Important Notes

### Research Platform

This is a **research platform**, not a production product. All scientific claims must be grounded in validation documents under `docs/findings/`.

### Parallel Lineages

The platform maintains both v3 (mature) and v4 (advanced) scientific pipelines in parallel:

- **v3**: Stable, production-ready inference
- **v4**: Experimental advanced techniques

See [docs/adr/ADR-002](docs/adr/) for decision rationale.

### Recent Optimizations (June 2026)

- ✅ Optimized Docker Compose for network isolation and connection pooling
- ✅ Improved Dockerfiles for layer caching and size efficiency
- ✅ Enhanced Makefile with service-specific targets and health checks
- ✅ Pinned transitive dependencies for reproducibility
- ✅ Added comprehensive Make targets for development workflow

---

**Work in this repository replaces previous scattered locations.**

For questions or issues, see [AGENTS.md](AGENTS.md) for project context.
