# AGENTS.md — Tokyo Eye Agentic Poincaré

**Repository:** tokyo-eye-agenticpoincare  
**Status:** Production-ready hybrid platform (reviewed and hardened — May 2026)  
**Purpose:** Hyperbolic GNN platform for protein allostery detection, source-leak identification, and agentic structural biology workflows.

---

## Project Identity

A **hybrid research + production platform** for:

- Graph Neural Networks in hyperbolic space (Poincaré disc/ball) for protein source-leak and allosteric site detection
- Full DTIE pipeline (Phases 1–6d) for structural biology and in silico drug discovery
- Agentic orchestration (LLM coordinator + 20 tools) driving multi-phase scientific workflows
- Interactive Poincaré disc visualization tightly integrated with the agent
- Governed data layer with provenance tracking, audit trails, and idempotent writes

### Core Technical Thesis

A GNN using per-residue dehydron density (ρ) + Cα graph connectivity can surface biologically meaningful allosteric and vulnerability signals in oncogenic proteins (KRAS, etc.) when operating in hyperbolic geometry, without functional labels.

---

## Architecture Overview

```
agent/coordinator/          FastAPI app (routers: chat, tools, poincare)
agent/coordinator/auth.py   JWT auth, bounded session store, rate limiting
agent/coordinator/viewport.py  WebSocket viewport protocol
agent/llm/                  LLM framework (Agent loop, Bedrock/Anthropic/Mock providers)
agent/tools/dtie/           Core DTIE tools (GNN inference, source leaks, uncertainty)
agent/tools/data_tools.py   Data access tools (export, search, annotate, provenance)
agent/tools/plotting/       Matplotlib figure generation from pipeline data
agent/tools/diagnostics.py  Startup self-diagnostic for all tools
agent/models/viewport.py    Pydantic models for viewport directives

data/db.py                  Connection pool (psycopg + psycopg-pool)
data/normalizer/core.py     THE governed write path (single entry for all data writes)
data/views/refresh.py       Materialized view management
data/aurora/migrations/     SQL migrations (001–030)
data/provenance/            Provenance lineage tracing

science/dtie/v5/            Production GNN + orchestrator (decoupled radial-angular)
science/dtie/v4/            Hyperbolic-aware Phase 3 (used by v5 orchestrator)
science/dtie/v3/            Phase implementations (1–6d, reused by v5 via adapters)
science/dtie/common/        Shared: keys, adapters, interfaces, payloads, graph builder

shared/context.py           Request context propagation (ContextVar)
shared/logging.py           Structured JSON logging with correlation IDs
shared/middleware.py        FastAPI middleware for request tracing

infra/terraform/            Aurora PostgreSQL + S3 (AWS)
```

---

## Key Design Decisions

1. **Single write path**: ALL data writes go through `data/normalizer/core.py`. No exceptions.
2. **Provenance-first**: Every computation produces a `provenance_run` record before writing results.
3. **Idempotent upserts**: Re-running the same pipeline is safe (ON CONFLICT on natural keys).
4. **V5 is production**: V3/V4 code exists for backward compat only. New analysis uses V5 exclusively.
5. **Adapters bridge science→data**: Science code never touches the Normalizer directly.
6. **Connection pooling**: All DB access goes through `AsyncConnectionPool` (min=2, max=10).
7. **Auth required by default**: `REQUIRE_AUTH=true` unless explicitly opted out for local dev.
8. **Structured logging**: JSON in prod, human-readable in dev. Correlation IDs propagate via ContextVar.
9. **Phase functions run in threads**: CPU-bound numpy/scipy work uses `asyncio.to_thread`.
10. **Token budget on LLM calls**: Agent loop stops at `max_tokens` (default 100k) to prevent runaway costs.

---

## Agent Tools (20 total)

### DTIE Core (6)
- `run_gnn_inference` — V5 GNN on a structure
- `run_full_pipeline` — Complete DTIE pipeline (Phases 1–6d + source-leak + allosteric)
- `get_source_leaks` — High uncertainty + deep residues
- `get_high_uncertainty_residues` — Top-N by uncertainty type
- `get_residue_state` — Current governed state of residues
- `compare_wt_mutant` — WT vs mutant displacement in hyperbolic space

### Data Access (8)
- `export_structure_data` — CSV/JSON export for external tools
- `get_allosteric_sites` — Site clusters with member residues
- `get_provenance_lineage` — Run ancestry and history
- `search_residues` — Flexible filtering (chain, name, uncertainty, depth)
- `annotate_structure` — Store findings/hypotheses/notes
- `list_structures` — All analyzed structures
- `get_run_summary` — Run timing, assets, errors
- `compare_runs` — Per-residue diff between two runs

### Visualization (4)
- `highlight_residues` — Color residues in the Poincaré viewer
- `set_metric` — Change coloring metric
- `focus_residues` — Animate camera to residues
- `clear_highlights` — Remove all highlights

### Plotting (1)
- `generate_plot` — Matplotlib figures (poincare_disc, uncertainty_profile, cone_depth_histogram, wt_vs_mutant, persistence_barcode, source_leak_map)

### Planned
- `fetch_structure_from_rcsb` — Ingest from RCSB PDB + auto-run pipeline (see `docs/feature-briefs/02_rcsb_integration.md`)
- Graph topology tools — Edge diff, H-bond networks, betweenness, bridges (see `docs/feature-briefs/01_graph_compare.md`)
- Hypothesis engine — Propose, test, track scientific hypotheses with guardrails (see `docs/feature-briefs/03_hypothesis_engine.md`)

---

## Critical Rules for AI Assistants

1. **All writes go through the Normalizer** — never INSERT directly into fact tables.
2. **Validate user input** — uncertainty_type against allowlist, structure_id format, residue_id count limits.
3. **No raw exceptions to users** — log details, return generic messages in prod.
4. **Run tests after changes** — `make test` must pass before considering work done.
5. **V5 is the only production model** — V3/V4 are for provenance/backward compat only.
6. **Canonical keys** — Always use `science/dtie/common/keys.py` for ID generation.
7. **Thread-safe model loading** — GNN runner uses asyncio.Lock to prevent duplicate loads.
8. **Never hardcode secrets** — JWT_SECRET must come from env, fail fast in prod if missing.

---

## Development Commands

```bash
make up          # Start all services (Docker)
make down        # Stop gracefully
make kill        # Force-stop + remove volumes
make build       # Rebuild images
make dev         # Run agent locally (hot-reload)
make test        # Unit tests
make test-all    # Unit + integration tests
make lint        # Ruff lint + format check
make migrate     # Run DB migrations
make psql        # Open psql shell
```

---

## Environment Variables

See `.env.example` for the full list. Key ones:
- `DATABASE_URL` — PostgreSQL connection string
- `JWT_SECRET` — Required in prod, defaults in dev
- `ENVIRONMENT` — dev/staging/prod (controls CORS, error messages, logging format)
- `REQUIRE_AUTH` — true by default, set false for local dev
- `LLM_TIMEOUT_SECONDS` — LLM call timeout (default 120)
- `CHAT_RATE_LIMIT_PER_MINUTE` — Rate limit for /api/chat (default 20)

---

**This file is the single source of truth for project context. Update it when major decisions are made.**
