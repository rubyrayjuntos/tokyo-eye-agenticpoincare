# AGENTS.md — Tokyo Eye Agentic Poincaré (Hybrid Platform)

**Repository:** tokyo-eye-agenticpoincare  
**Status:** Fresh hybrid monorepo (consolidation in progress — May 2026)  
**Purpose:** A clean, maintainable platform combining the best surviving components of the Tokyo Eyes / DTIE research system.

---

## Project Identity

This is a **hybrid research platform** for:

- Advanced Graph Neural Networks operating in hyperbolic space (Poincaré disc) for protein allostery and "source leak" detection.
- Full DTIE (Dynamic Topology Inference Engine) pipelines for structural biology and in silico drug discovery analysis.
- Agentic orchestration (Coordinator + tools) that can drive complex multi-phase scientific workflows.
- Interactive visualization (Poincaré disc viewer) tightly integrated with the scientific outputs.

We are deliberately building this as a **clean slate hybrid** after fragmentation and loss of prior consolidated work.

### Core Technical Thesis

A well-trained GNN using only per-residue dehydron density (ρ) + Cα graph connectivity can surface biologically meaningful allosteric and vulnerability signals (especially in oncogenic proteins like KRAS) when operating in hyperbolic geometry, without being given functional labels.

---

## Current State (May 2026)

This repository is in **active consolidation** from three source locations:

- `adk-samples/python/agents/data-science` — Agent wrapper, infrastructure, governance, documentation.
- `Tokyo-eye-demensional-investigator` — Most complete v3 full DTIE pipeline (orchestrator + phases 1-6d).
- `tokyo-eyes-visualizer` — Current best v4 GNN training code + React Poincaré visualizer.

**Important:** v3 and v4 science tracks are intentionally kept separate during the transition.

---

## Repository Structure (Canonical)

```
science/
  dtie/
    common/          # Shared contracts, provenance, ingestion, utils
    v3/              # Mature full pipeline (best orchestrator + complete phases)
    v4/              # Modern hyperbolic source-leak focused work
  shared/            # Cross-cutting modules (spectral, source_leak_scoring, etc.)

agent/               # Agentic layer
  coordinator/       # FastAPI + chat interface
  tools/             # Clean @tool wrappers for DTIE v3/v4
  models/            # Viewport directives, session state, etc.

visualizer/          # Frontend
  frontend/          # React + TypeScript Poincaré disc viewer
  server/            # Thin serving layer

data/
  aurora/            # Schemas and migrations
  governance/        # FAIR data schemas

infra/
  terraform/

experiments/
  training/          # GNN training scripts (v4 focus)
  analysis/

docs/
  findings/          # Scientific results (version-tagged)
  architecture/
  specs/
```

---

## Critical Rules for Any AI Working Here

1. **Never mix v3 and v4 internals** without explicit approval and clear boundaries.
2. Science core (`science/`) must remain framework-free (no FastAPI, no React dependencies).
3. Always preserve provenance and reproducibility when porting code.
4. When in doubt about scientific claims, ground them in specific checkpoint validation documents.
5. The existing fragmented locations are deprecated. All new work happens here.

---

## Key Documents

- `CONSOLIDATION_PLAN.md` — Overall strategy and LOE
- `MIGRATION_MAP.md` — Detailed source → destination mapping
- `docs/architecture/` — Layer and interface decisions
- `docs/findings/` — The scientific record

---

**This file is the single source of truth for project context.**

Update it whenever major architectural decisions are made.