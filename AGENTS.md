# AGENTS.md — Tokyo Eye Agentic Poincaré

**Repository:** tokyo-eye-agenticpoincare  
**Status:** Production-ready hybrid platform (reviewed and hardened — May 2026)  
**Purpose:** Hyperbolic GNN platform for protein allostery detection, source-leak identification, and agentic structural biology workflows.

> **New developers:** Read [`docs/DEVELOPER_ONBOARDING.md`](docs/DEVELOPER_ONBOARDING.md) before starting feature work — it walks through write path, compute path, onboard contract, geometry, and audit compliance.  
> **Project Hub (roadmap + doc map):** [`docs/PROJECT_HUB.md`](docs/PROJECT_HUB.md) — stitch point for Fix-1 biology status, open work, and key specs. Live print: `make project-hub`.  
> **AI agents:** Follow [`docs/AI_DEVELOPER_CONTRACT.md`](docs/AI_DEVELOPER_CONTRACT.md) (STOP conditions + gate output). Enforcement map: [`docs/ENFORCEMENT_MATRIX.md`](docs/ENFORCEMENT_MATRIX.md).

---

## Project Identity

A **hybrid research + production platform** for:

- Graph Neural Networks in hyperbolic space (Poincaré disc/ball) for protein source-leak and allosteric site detection
- Discovery Story pathway (Signal → Persistent Leak → Cryptic Pocket → Fragment → Verdict) driving onboard compute and UX
- Atomic compute jobs (former pipeline stages) scheduled as a DAG at ingest — **DTIE is deprecated** as a term and pathway
- Agentic orchestration (LLM coordinator + tools) driving discovery workflows over pre-computed data
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
agent/tools/dtie/           Discovery signal tools (source leaks, uncertainty, residue state)
agent/tools/data_tools.py   Data access tools (export, search, annotate, provenance)
agent/tools/plotting/       Matplotlib figure generation from pipeline data
agent/tools/diagnostics.py  Startup self-diagnostic for all tools
agent/models/viewport.py    Pydantic models for viewport directives

data/db.py                  Connection pool (psycopg + psycopg-pool)
data/normalizer/core.py     THE governed write path (single entry for all data writes)
data/audit/                 Pipeline runtime audit (query + retention; not governed facts)
data/views/refresh.py       Materialized view management
data/aurora/migrations/     SQL migrations (001–050)
data/provenance/            Provenance lineage tracing

science/compute/            Job registry, scheduler, 15 peeled atomic runners (Acts 01–05); monolith = gnn_inference only
science/contracts/          Master onboard contract (artifacts, geometry, readiness, API types)
science/tokyo_eye/          Tokyo Eye GNN product (v7+); production module TokyoEye.py — not versioned under dtie
science/dtie/               Structural biology (historical name; do not version the GNN here). Future rename → structural-biology
science/dtie/common/        Shared: keys, adapters, interfaces, payloads, graph builder

shared/audit/               Structured pipeline audit event bus (logs + persistence)
shared/context.py           Request context propagation (ContextVar)
shared/logging.py           Structured JSON logging with correlation IDs
shared/middleware.py        FastAPI middleware for request tracing

infra/terraform/            Aurora PostgreSQL + S3 (AWS)
```

---

## Key Design Decisions

1. **Single write path**: ALL data writes go through `data/normalizer/core.py`. No exceptions.
2. **Single compute path**: ALL structure-scoped science computation runs at **ingest onboarding** via the ingest orchestrator — the agent consumes pre-computed data and must not schedule GNN, pipeline, scan, or MD jobs. Pathways follow the **Discovery Story** acts; job catalog in `docs/specs/discovery-story-pathway/`. See `docs/specs/ingest-compute-contract/requirements.md`.
3. **Provenance-first**: Every computation produces a `provenance_run` record before writing results.
4. **Idempotent upserts**: Re-running the same pipeline is safe (ON CONFLICT on natural keys).
5. **V5 is production**: V3/V4 code exists for backward compat only. New analysis uses V5 exclusively.
6. **Adapters bridge science→data**: Science code never touches the Normalizer directly.
7. **Connection pooling**: All DB access goes through `AsyncConnectionPool` (min=2, max=10).
8. **Auth required by default**: `REQUIRE_AUTH=true` unless explicitly opted out for local dev.
9. **Structured logging**: JSON in prod, human-readable in dev. Correlation IDs propagate via ContextVar.
10. **Phase functions run in threads**: CPU-bound numpy/scipy work uses `asyncio.to_thread`.
11. **Token budget on LLM calls**: Agent loop stops at `max_tokens` (default 100k) to prevent runaway costs.
12. **Hyperbolic geometry by default**: All geometric computations run in hyperbolic (Poincaré) space unless the onboard contract marks an artifact or job as `euclidean` or `mixed`. The contract (`science/contracts/onboard_contract.yaml`) is the SSOT for which jobs and artifacts are hyperbolic. **Curvature is learned at GNN inference** (`gnn_inference` → `embedding_space.curvature`) and must passthrough to downstream hyperbolic jobs — never hardcode a numeric curvature. Enforcement defaults to `warning`; set `GEOMETRIC_ENFORCEMENT_LEVEL=error` and `GEOMETRIC_ENFORCEMENT_ERROR_JOBS` in CI/staging for critical jobs. See `science/contracts/README.md`.
13. **Tokyo Eye v8 — active trunk SSOT (2026-07-23):** Production GNN is [`science/tokyo_eye/v8/`](science/tokyo_eye/v8/) (`TokyoEye-v8`, contract `tokyo_eye_v8`). Equiformer/SE(3) frontend + hyperbolic spine + MoE. Checkpoints: `checkpoints/v8/`. **Spine restore:** `HEALTHY_V8_SPINE_CKPT` (`tokyo_eye_v8_mode_c_moe_rebalance_s9/v8_best.pt`). **Affinity seal:** `HEALTHY_V8_AFFINITY_CKPT` (CASF Core \(R\approx0.407\)). Spec: [`docs/specs/tokyo-eye-v8/README.md`](docs/specs/tokyo-eye-v8/README.md). Biology TODOs: [`biology-roadmap.md`](docs/specs/tokyo-eye-v8/biology-roadmap.md). **v7 is dead archaeology** (different architecture; never production) — do not open `HEALTHY_V7_CKPT` / `docs/specs/tokyo-eye-v7/` for new work. Trunk stamp: [`data/gates/tokyo_eye_v8_trunk_ssot.json`](data/gates/tokyo_eye_v8_trunk_ssot.json).
14. **July-19 Euc-MP lock historical:** [`docs/audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md`](docs/audit/EUCLIDEAN_CONSTRUCTION_HYPERBOLIC_INFERENCE.md) remains historical for **frozen v6.x** only. Complements [`docs/audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md`](docs/audit/DISC_PROJECTION_NOT_TRUNK_PROXY.md).
15. **v6.x / v7 frozen archaeology:** Sealed Fix-1 under `checkpoints/v66/` and Hyp-MP v7 under `checkpoints/v7/` are compare-only. Chem-MVP **PARKED**: [`docs/specs/chem-mvp-reengage/README.md`](docs/specs/chem-mvp-reengage/README.md). Run names containing `s4` are historical labels, not a backbone version.
16. **Pipeline runtime audit**: Compute orchestration emits structured events to logs and `audit_pipeline_events` (migrations 049–050) for historical query — geometric validation, curvature passthrough, preconditions, pathway lifecycle, enforcement. Complements `normalization_audit` (governed writes). Full reference: `docs/audit/PIPELINE_AUDIT.md`.

---

## Agent Tools (20 total)

### Discovery signals (read-only; 4)
- `get_source_leaks` — Physics-rim triage (high cone_depth, prefer τ=1); evidential ranking opt-in only
- `get_high_uncertainty_residues` — Prefer `cone_depth`; epistemic/aleatoric marked experimental (G5b)
- `get_residue_state` — Current governed state of residues
- `compare_wt_mutant` — WT vs mutant displacement in hyperbolic space

> **Ingest–compute contract:** Compute is only triggered by `POST /api/ingest`. Agent tools are read-only over governed artifacts. See `docs/specs/ingest-compute-contract/requirements.md`.

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

> **Investigation coloring (CLOSED 2026-07-16):** Default GNN HTML viewers use ρ/τ physics underwrap, not evidential `ale×(1−epi)`. See [`docs/audit/VIEWER_INVESTIGATION_CORRECTNESS.md`](docs/audit/VIEWER_INVESTIGATION_CORRECTNESS.md).

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
5. **TokyoEye-v8 is the production GNN** — `science.tokyo_eye.v8` via contract `tokyo_eye_v8`. **Never open v7 as an active trunk** (`TokyoEye.py` / `HEALTHY_V7_CKPT` are archaeology). Legacy `GOSPConeMapper-v*` / `gospc_v*` remain compare-only. Pathway/orchestrator code under `science/dtie/v5` is not the GNN trunk.
6. **Canonical keys** — Always use `science/dtie/common/keys.py` for ID generation.
7. **Thread-safe model loading** — GNN runner uses asyncio.Lock to prevent duplicate loads.
8. **Never hardcode secrets** — JWT_SECRET must come from env, fail fast in prod if missing.
9. **Follow the compliance guide** — [`docs/DEVELOPER_ONBOARDING.md`](docs/DEVELOPER_ONBOARDING.md) for contract, geometry, audit, and PR checklists on every new effort. See [`docs/ENFORCEMENT_MATRIX.md`](docs/ENFORCEMENT_MATRIX.md) for which rules CI actually enforces.

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
make contract-sync  # Regenerate job_schema.json + frontend onboard types

# Pipeline audit (requires DB + migrations 049–050)
make audit-structure STRUCTURE_ID=4obe SINCE=7d
make audit-summary SINCE=7d
make audit-retention RETENTION_DAYS=90
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
- `GEOMETRIC_ENFORCEMENT_LEVEL` / `GEOMETRIC_ENFORCEMENT_ERROR_JOBS` — geometric contract enforcement (default warning)
- `AUDIT_PIPELINE_PERSIST` / `AUDIT_PIPELINE_MIN_SEVERITY` / `AUDIT_RETENTION_DAYS` — pipeline runtime audit (see `docs/audit/PIPELINE_AUDIT.md`)

---

**This file is the single source of truth for project context. Update it when major decisions are made.**
