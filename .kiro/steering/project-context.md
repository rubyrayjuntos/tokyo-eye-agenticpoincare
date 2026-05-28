---
inclusion: always
---

# Tokyo Eye — Project Context

This is a structural biology platform with a hyperbolic GNN (GOSPConeMapper-v5) that detects allosteric sites and source leaks in proteins. The codebase has been through a comprehensive code review and hardening pass.

## Architecture at a Glance

- **Agent** (FastAPI): `agent/coordinator/app.py` — routers in `agent/coordinator/routers/`
- **LLM Framework**: `agent/llm/base.py` (Agent loop), `agent/llm/providers.py` (Bedrock/Anthropic/Mock)
- **Tools**: `agent/tools/dtie/tools.py`, `agent/tools/data_tools.py`, `agent/tools/plotting/tools.py`
- **Data Layer**: `data/db.py` (pooled connections), `data/normalizer/core.py` (governed write path)
- **Science**: `science/dtie/v5/` (production GNN + orchestrator), phases in `science/dtie/v3/phases/`
- **Shared**: `shared/context.py` (request tracing), `shared/logging.py` (structured JSON logs)

## Key Patterns

- All DB writes go through `data/normalizer/core.py` — never write to fact tables directly
- Canonical IDs generated via `science/dtie/common/keys.py`
- Phase functions are CPU-bound and run via `asyncio.to_thread`
- GNN model loading is protected by `asyncio.Lock`
- Auth required by default (`REQUIRE_AUTH=true`)
- Structured logging with correlation IDs via `shared/context.py` ContextVar

## When Making Changes

1. Run `make test` after any code change
2. Validate SQL column names against allowlists (never interpolate user input into SQL)
3. New tools go in `agent/tools/` and get registered in `agent/llm/agents.py`
4. New migrations go in `data/aurora/migrations/` with sequential numbering (next: 031)
5. Update `AGENTS.md` if you add tools or change architecture

## File References

- #[[file:AGENTS.md]] — Full project context, tool inventory, design decisions
- #[[file:data/ARCHITECTURE.md]] — Data layer design
- #[[file:data/RESIDUE_ID_KEY_STRATEGY.md]] — Canonical key rationale
