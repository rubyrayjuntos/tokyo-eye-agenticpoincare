# Feature Brief: RCSB PDB Integration

## Goal

Allow the agent to ingest structures directly from RCSB PDB by ID, normalize them for compatibility with existing structures, and automatically kick off the full compute pipeline. The user says "analyze 6LU7" and the system handles everything.

## Why This Matters

Currently, structures must be pre-loaded. Researchers want to say "compare my structure against 6LU7" without manual setup. The ingestion must also handle normalization so that multi-structure comparisons are valid (consistent chain labeling, residue numbering alignment).

## Design Requirements

1. **Fetch**: Download structure from RCSB (mmCIF preferred, PDB fallback)
2. **Parse**: Extract chains, residues, atoms using biotite
3. **Normalize**: Ensure compatibility with existing structures in the DB
   - Consistent chain labeling
   - Author vs. PDB residue numbering reconciliation
   - Handle insertion codes, alternate conformations
4. **Persist**: Write dimensional data (dim_structure, dim_chain, dim_residue) via Normalizer
5. **Compute**: Auto-trigger the DTIE pipeline on the new structure
6. **Multi-structure normalization**: When comparing structures, ensure residue alignment is valid

## Data Flow

```
User: "analyze 4obe"
  → agent calls fetch_structure_from_rcsb(pdb_id="4obe")
    → download mmCIF from RCSB API
    → parse with biotite
    → generate canonical IDs (make_structure_id, make_chain_id, make_residue_id)
    → check if already ingested (idempotent)
    → write dim_structure, dim_chain, dim_residue via Normalizer
    → trigger DTIEOrchestrator.run(PipelineConfig(structure_id="4obe"))
    → return summary (residue count, chains, pipeline status)
```

## Tool Interface

### `fetch_structure_from_rcsb(pdb_id, auto_compute=True, compute_config?)`

Parameters:
- `pdb_id`: 4-character PDB ID
- `auto_compute`: Whether to run the pipeline after ingestion (default True)
- `compute_config`: Optional PipelineConfig overrides (source_leak_only, etc.)

Returns:
- structure_id (canonical)
- chains found
- residue count
- pipeline run_id (if auto_compute)
- warnings (missing atoms, alternate conformations, etc.)

### `align_structures(structure_id_a, structure_id_b)`

For multi-structure comparison normalization:
- Sequence alignment between two structures
- Residue correspondence mapping
- Warns about insertions/deletions that affect comparison validity

## Implementation Location

- `science/dtie/common/ingestion.py` — already exists (used by backfill script), extend it
- `agent/tools/rcsb_tools.py` — new tool file
- RCSB API client: use `biotite.database.rcsb` (already in requirements-science.txt)

## Key Considerations

1. **Idempotent**: Re-fetching the same PDB ID should not duplicate data
2. **Validation**: Reject invalid PDB IDs early (4 chars, alphanumeric)
3. **Rate limiting**: RCSB has rate limits — add backoff
4. **Large structures**: Some PDB entries have 100k+ atoms — handle gracefully
5. **Obsolete entries**: Check if PDB ID is obsolete, warn user
6. **AlphaFold support**: Consider also supporting AF2 structure fetch (UniProt ID → AF DB)

## Dependencies

Already available:
- `biotite` (structure parsing, RCSB fetch)
- `biopython` (sequence alignment for multi-structure normalization)

May need:
- `requests` or `httpx` for direct RCSB REST API calls (for metadata, obsolete checks)

## Testing

- Unit test: mock RCSB response, verify parsing and ID generation
- Integration test: fetch a small real structure (1CRN — 46 residues), verify DB state
- Idempotency test: fetch same structure twice, verify no duplicates
- Error test: invalid PDB ID, obsolete entry, network failure
