# Tokyo Eye — Agentic Poincaré Platform

A clean, modern hybrid research platform for advanced hyperbolic Graph Neural Networks, DTIE pipelines, agentic scientific workflows, and interactive structural biology visualization.

**Status:** Fresh consolidation repository (May 2026)

This repo is being built as a deliberate, clean-slate synthesis of the best surviving components from previous fragmented work on the Tokyo Eyes project.

## Quick Links

- [AGENTS.md](AGENTS.md) — Authoritative project context
- [CONSOLIDATION_PLAN.md](CONSOLIDATION_PLAN.md) — Strategy and effort estimate
- [MIGRATION_MAP.md](MIGRATION_MAP.md) — Detailed source-to-destination mapping

## High-Level Structure

- `science/dtie/` — Core DTIE pipelines (v3 mature + v4 advanced)
- `agent/` — Coordinator + tool interfaces
- `visualizer/` — Poincaré disc frontend
- `data/` — Schemas and governance
- `experiments/` — Training and analysis
- `docs/` — Findings, architecture, specs

## Getting Started

### Prerequisites
- Python 3.11+ (uv recommended)
- Docker (for local PostgreSQL with pgvector)
- Node.js 18+ (for the visualizer frontend)

### Local Development Setup

```bash
# 1. Install Python dependencies (lightweight — no torch needed on host)
uv sync --extra dev

# 2. Start local database and run migrations
docker compose up -d db
python scripts/dev_setup.py

# 3. Run tests (all mocked, no heavy deps needed)
PYTHONPATH=. pytest tests/

# 4. Start the agent coordinator
docker compose up agent
# Or locally: PYTHONPATH=. uvicorn agent.coordinator.app:app --reload --port 8000

# 5. Run GNN inference / full pipeline (heavy deps in container)
docker compose run science python -m science.dtie.v5.gnn.runner --structure 4OBE

# 6. (Optional) Start the visualizer frontend
cd visualizer/frontend && npm install && npm run dev
```

### Docker Architecture

The project uses separate containers to isolate heavy ML dependencies:

- `db` — PostgreSQL 16 + pgvector (always running)
- `agent` — Lightweight FastAPI coordinator (no torch, fast startup)
- `science` — Heavy compute (torch, geoopt, e3nn, gudhi, rdkit, GPU support)

The agent container handles API requests and dispatches compute jobs to the science container. Both share the same database.

### Local Database

Local dev uses PostgreSQL 16 with pgvector via Docker:
- **Connection:** `postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/tokyoeye_dev`
- **Start:** `docker compose up -d db`
- **Reset:** `python scripts/dev_setup.py reset`
- **Migrations:** `python scripts/dev_setup.py migrate`

The same schema runs locally and on Aurora in production. All 29 migrations apply identically to both environments.

### Environment Variables

Copy `.env.example` to `.env` and adjust as needed:
```bash
cp .env.example .env
```

## Important

This is a **research platform**, not a production product. All scientific claims must be grounded in the validation documents under `docs/findings/`.

---

**Work in this repository replaces previous scattered locations.**