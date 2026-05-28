# DTIE Pipeline Integration Plan

## Module Structure

```
data_science/
├── sub_agents/
│   ├── pipeline/              # Existing Tier 1 (structural biology — unchanged)
│   │   ├── compute.py         # Biotite-based dehydron/void/SASA/energy
│   │   ├── models.py          # Tier 1 Pydantic models
│   │   ├── store.py           # Aurora read/write for GOSP star-schema
│   │   ├── tools.py           # @tool wrappers for Tier 1
│   │   └── loader.py          # S3 fetch + Biotite parse
│   │
│   └── dtie/                  # NEW — self-contained DTIE pipeline
│       ├── __init__.py        # Re-exports run_dtie_pipeline tool
│       ├── contracts.py       # ← from DTIE_GNN_ORCHESTRATION/contracts.py
│       ├── Gnnv3.py           # ← GOSPConeMapper model (torch, torch_geometric, geoopt, e3nn)
│       ├── gnn_runner.py      # ← 28-dim features, Cα graph, forward pass, gnn_output.npz
│       ├── ingestion.py       # ← BioPython PDB parse → per-residue features
│       ├── source_leak_scoring.py  # ← weighted scoring functions
│       ├── spectral.py        # ← Laplacian spectrum, Fiedler, fragment stabilization
│       ├── compare_graphs.py  # ← WT vs mutant graph comparison
│       │
│       ├── phases/            # Phase modules (unchanged from DTIE_GNN_ORCHESTRATION)
│       │   ├── __init__.py
│       │   ├── phase1_witness_embedding.py
│       │   ├── phase2_vulnerability_scan.py
│       │   ├── phase3_witness_persistence.py
│       │   ├── phase35_topological_lift.py
│       │   ├── phase4_resistance_mapping.py
│       │   ├── phase5_pharmacophore.py
│       │   ├── phase6a_virtual_screening.py
│       │   ├── phase6b_binding_affinity.py
│       │   ├── phase6c_admet_filter.py
│       │   └── phase6d_state_selectivity.py
│       │
│       ├── orchestrator.py    # Thin wrapper: runs the full pipeline, returns results dict
│       ├── store.py           # Aurora writeback for DTIE results (post-hoc, not inline)
│       └── tools.py           # @tool functions for agent integration
```

## Design Principles

1. **Self-contained compute**: The `dtie/` module runs the full pipeline
   (ingestion → GNN → phases 1-6) without touching Aurora or S3 during
   execution. All intermediate state flows through in-memory dicts and
   local .npz files in a temp directory.

2. **Aurora stores results after the fact**: Once the pipeline completes,
   `dtie/store.py` writes the results to Aurora fact tables. This keeps
   the compute chain pure and the database writes atomic.

3. **No modification to DTIE science code**: The phase modules, GNN model,
   ingestion, and scoring functions are copied verbatim from
   `DTIE_GNN_ORCHESTRATION/`. Import paths are adjusted (relative imports
   within the `dtie` package) but logic is untouched.

4. **Agent orchestrates, doesn't compute**: The Coordinator calls
   `run_dtie_pipeline` as a tool. The tool handles PDB file resolution
   (from S3 or local), creates a temp working directory, runs the
   orchestrator, writes results to Aurora, and returns a summary.

## Tool Interface

```python
@tool
def run_dtie_pipeline(
    gdp_pdb_id: str,
    gtp_pdb_id: str,
    pipeline_mode: str = "source_leak_v4",
    n_landmarks: int = 200,
    effector_sites: str = "",
) -> str:
    """Run the full DTIE drug target identification pipeline.

    Executes: PDB ingestion → GNN inference → Phase 1-6 analysis.
    Results are written to Aurora and returned as a JSON summary.

    Args:
        gdp_pdb_id: PDB ID for the GDP (inactive) state structure.
        gtp_pdb_id: PDB ID for the GTP (active) state structure.
        pipeline_mode: "source_leak_v4" (default) or "legacy_witness".
        n_landmarks: Number of witness landmarks for Phase 1 (default 200).
        effector_sites: Comma-separated effector site indices, or empty.

    Returns:
        JSON summary with phase results, doorways, pathways, and
        pharmacophores.
    """
```

Additional tools for the agent to query DTIE results:

```python
@tool
def get_dtie_doorways(structure_id: str) -> str:
    """Get state-selective doorways from a completed DTIE run."""

@tool
def get_dtie_pathways(structure_id: str) -> str:
    """Get ranked allosteric pathways from a completed DTIE run."""

@tool
def get_poincare_graph(structure_id: str) -> str:
    """Get the Poincaré disc projections and cone geometry for a structure.
    Returns per-residue: projections, cone_depth, cone_width,
    expert_weights, uncertainty decomposition."""

@tool
def compare_wt_mutant(
    wt_pdb_id: str,
    mutant_pdb_id: str,
    mutation_residue: int,
) -> str:
    """Run WT vs mutant graph comparison using compare_graphs.py."""
```

## Aurora Schema Additions

The existing `002_schema_extensions.sql` already defines:
- `fact_gnn_inference` — one row per GNN run
- `fact_gnn_node_output` — per-residue GNN outputs (projections, cone, uncertainty)
- `fact_gnn_node_embedding` — pgvector projection embeddings

New tables needed for DTIE phase results:
- `fact_dtie_run` — one row per pipeline execution (provenance, timing, mode)
- `fact_dtie_doorway` — state-selective doorways from Phase 2
- `fact_dtie_pathway` — allosteric pathways from Phase 4
- `fact_dtie_pharmacophore` — pharmacophore models from Phase 5
- `fact_dtie_candidate` — ranked drug candidates from Phase 6d

## Dependencies

New Python dependencies required (not currently in pyproject.toml):
- `torch` + `torch_geometric` — GNN inference
- `geoopt` — Poincaré ball operations
- `e3nn` — equivariant neural network layers
- `gudhi` — witness complex persistence (Phase 3)
- `biopython` — PDB parsing for DTIE ingestion
- `networkx` — resistance graph (Phase 4)
- `meeko` + `vina` — AutoDock Vina docking (Phase 6b, optional)
- `fpocket` — pocket detection (Phase 5, optional external binary)

These are heavy dependencies. Recommended approach:
- Core deps (torch, torch_geometric, geoopt, e3nn, gudhi, biopython,
  networkx) go in pyproject.toml as required.
- Docking deps (meeko, vina, fpocket) stay optional — Phase 6b-6d
  gracefully skip if unavailable.

## Checkpoint

The `robust_experts.pt` file (GNN trained weights) ships with the module
at `data_science/sub_agents/dtie/robust_experts.pt`. The orchestrator
defaults to this path. Override via `DTIE_CHECKPOINT_PATH` env var.

## Execution Flow

```
Agent receives: "Analyze KRAS G12D for drug targets"
    │
    ▼
Coordinator → run_dtie_pipeline(gdp_pdb_id="6GOF", gtp_pdb_id="6GOD")
    │
    ▼
dtie/tools.py:
    1. Resolve PDB files (download from RCSB or fetch from S3)
    2. Create temp working directory
    3. Call orchestrator.run(gdp_path, gtp_path, output_dir, ...)
    │
    ▼
dtie/orchestrator.py (pure compute, no DB):
    1. ingestion.ingest_pdb(gdp) → gdp_ingestion dict
    2. ingestion.ingest_pdb(gtp) → gtp_ingestion dict
    3. gnn_runner.run_gnn(gdp, gtp, checkpoint) → gnn_output dict
    4. Phase 1: witness embedding (landmarks in Poincaré space)
    5. Phase 2: vulnerability scan (GDP vs GTP doorways)
    6. Phase 3: witness persistence (H1 terminal leaks)
    7. Phase 3.5: topological lift (persistence → 3D coords)
    8. Phase 4: resistance mapping (allosteric pathways)
    9. Phase 5: pharmacophore generation
   10. Phase 6a-6d: screening → docking → ADMET → selectivity
    │
    ▼
dtie/tools.py (continued):
    4. Write results to Aurora via dtie/store.py
       - fact_gnn_inference + fact_gnn_node_output
       - fact_dtie_run + fact_dtie_doorway + fact_dtie_pathway
       - fact_dtie_pharmacophore + fact_dtie_candidate
    5. Return JSON summary to Coordinator
    │
    ▼
Coordinator synthesizes results for user
```

## Migration Steps

1. Copy DTIE_GNN_ORCHESTRATION files into data_science/sub_agents/dtie/
2. Adjust import paths (relative imports within dtie package)
3. Write orchestrator.py (thin wrapper around dtie_pipeline.run_dtie_full_pipeline)
4. Write store.py (Aurora writeback for DTIE results)
5. Write tools.py (@tool wrappers for agent integration)
6. Add Aurora schema migration for DTIE fact tables
7. Add dependencies to pyproject.toml
8. Wire tools into Coordinator agent (agent.py)
9. Test with a known protein pair (KRAS G12D: 6GOF/6GOD)
