---

## 10. Detailed Component Specifications

### 10.1 Science / DTIE Layer (`science/dtie/`)

#### v3 Subsystem (`science/dtie/v3/`)
**Origin**: Primarily `Tokyo-eye-demensional-investigator/DTIE_GNN_ORCHESTRATION/` + supporting services.

**Responsibilities**:
- Implement the full DTIE phase pipeline (1–6d) as defined in the original `dtie_pipeline.py`.
- Provide both deterministic (physics kernel) and probabilistic (GNN) execution paths.
- Support ensemble and comparative (WT vs mutant, GDP vs GTP) runs.
- Emit all outputs through the governed data model via the Normalizer.

**Key Modules (to be created/adapted)**:
- `orchestrator.py` (or equivalent) — coordinates the phase sequence.
- Individual phase modules (adapted from source).
- `Gnnv3.py` + inference runner.
- Ingestion and graph construction utilities (moved to `common/` where possible).

**Constraints**:
- Must not contain any web framework code.
- Must not write directly to the database.

#### v4 Subsystem (`science/dtie/v4/`)
**Origin**: Primarily `tokyo-eyes-visualizer/src/components/TokyoEyesv4/`.

**Responsibilities**:
- Host the advanced Gnnv4 architecture and associated training/inference code.
- Support the specific loss functions and hardening developed for v4.
- Produce native hyperbolic outputs (`hyp_projections`, `x_hyp`, etc.).
- Support differential analysis and validation workflows.

**Key Modules**:
- `Gnnv4.py` (core model).
- Training scripts (`train_v4.py`, retrain, fine-tune variants) — moved to `experiments/training/v4/`.
- v4-specific phase improvements (e.g., `phase1_witness_embedding_v4.py`).

**Constraints**:
- Same framework-free rule as v3.
- Training runs must be clearly distinguished from inference runs (per ADR-003).

#### Common (`science/dtie/common/`)
Shared utilities that can be safely used by both lineages:
- Contracts / payload definitions.
- Provenance helpers.
- Ingestion utilities (BinaryCIF parsing, etc.).
- Graph construction logic.

### 10.2 Agent Layer (`agent/`)

**Coordinator (`agent/coordinator/`)**
- FastAPI application.
- Exposes `/chat` endpoint for the main agent.
- Exposes specialized endpoints (health, poincaré data, schema setup, etc.).
- Manages session state.
- Issues `ViewportDirective` messages to the visualizer.

**Tools (`agent/tools/`)**
- Clean, version-aware wrappers around DTIE v3 and v4 capabilities.
- Must go through the Normalizer for any writes.
- Should return governed asset references where possible.

**Models (`agent/models/`)**
- `ViewportState`, `ViewportDirective`, `HighlightGroup`, `ChatRequest/Response`, etc. (originally defined in the adk `main.py`).

### 10.3 Visualizer Layer (`visualizer/`)

**Frontend (`visualizer/frontend/`)**
- React + TypeScript + Three.js application.
- Core components: Visualizer2D, Visualizer3D, OverlayUI.
- Data loading from live server or static files.
- Real-time application of viewport directives from the agent.

**Thin Server (`visualizer/server/`)**
- Serves data required by the frontend (embeddings, metrics, structure metadata).
- Should read from governed views/materialized views where possible.
- Handles live data serving for the Poincaré viewer.

### 10.4 Data Layer (`data/`)

Defined in detail in `data/ARCHITECTURE.md` (v1.0) and the Dimensional Model Draft. It is the foundation for the entire platform.

---

## 11. Data Flow Examples

### Example 1: New Structure Analysis (Agent-Driven)

1. User or automated trigger sends message to Coordinator.
2. Coordinator calls ingestion tool → structure normalized into dimensions.
3. Coordinator triggers Tier 1 fast path (via tools) → outputs written via Normalizer.
4. GNN inference (v3 or v4) runs → embeddings + metrics written with full provenance.
5. Events published.
6. Visualizer receives updates and renders.
7. Agent can now query results or issue viewport directives.

All steps carry or reference a `run_id` for provenance.

### Example 2: Historical Backfill

1. Backfill job (Phase 1 tooling) creates a synthetic `provenance_run` marked as historical.
2. Legacy outputs are imported or re-derived and registered in `governed_asset`.
3. Data is written to the appropriate fact tables with the synthetic run linkage.

---

## 12. Non-Functional Requirements (Detailed)

- **Provenance & Reproducibility**: Every asset must be traceable to its producing run, model version, code version, and upstream assets.
- **Extensibility**: Adding a new GNN version or analysis phase should primarily require changes in `science/` and the data model, not a rewrite of the agent or visualizer.
- **Governance**: No ad-hoc storage of computed scientific results.
- **Separation of Concerns**: Strict enforcement of the four pillars and the framework-free science rule.
- **Auditability**: It must be possible to answer "What was the full lineage and parameters for this residue embedding?" efficiently.

---

## 13. Risks & Mitigations

- **Risk**: Over-coupling between layers during migration.  
  **Mitigation**: Define strict interfaces early (Normalizer contract, tool contracts, viewport directive contract).

- **Risk**: Loss of validated scientific behavior during porting.  
  **Mitigation**: Maintain side-by-side validation against original outputs during integration (Phase 3). Use the existing validation documents from the source repos.

- **Risk**: Data model becoming too rigid for future science.  
  **Mitigation**: Follow the extensibility strategy defined in the Dimensional Model Draft and ARCHITECTURE.md.

---

## 14. Open Questions (to be resolved in Phase 1+)

- Exact payload contracts between science code and the Normalizer.
- Degree of refactoring vs. adapter wrapping for the legacy v3 and v4 codebases.
- Concrete refresh and caching strategy for materialized views under realistic agent + visualizer load.
- Detailed backfill tooling and historical data import process.

---

**End of Specification (Current Draft)**

This document, in conjunction with `AGENTS.md`, the locked ADRs, `data/ARCHITECTURE.md` (v1.0), and the Dimensional Model Draft, provides the authoritative technical and functional specification for the consolidated Tokyo Eye platform.

It will be updated as Phase 1 work clarifies interfaces and requirements.