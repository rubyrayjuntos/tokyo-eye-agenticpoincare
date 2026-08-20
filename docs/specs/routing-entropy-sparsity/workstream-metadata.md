# Workstream metadata (status JSON schema v3)

**SSOT:** [`data/gates/fix1_sparsity_biology_phase_status.json`](../../../data/gates/fix1_sparsity_biology_phase_status.json)  
**Render:** `make project-hub`

## Why

Flat ledger rows record Pass/Fail. **Workstreams** record *progression*. **`meta`** on parents and children records provenance so later readers know *when*, *which model*, *what went in*, and *what came out*.

## Parent (`workstreams.<id>.meta`)

| Field | Meaning |
|-------|---------|
| `opened_at` | ISO date the workstream started |
| `updated_at` | ISO date of last child status change |
| `model` | Checkpoint constant, path, run_id, epoch, gate stamp |
| `lineage` | Sealed baseline + relationship (compare-only / specialized trunk) |
| `roster` | Structures in scope |
| `gnn_input_mode` | e.g. `topology_three_vector` |
| `docs` | Spec / design pointers |

## Child (`workstreams.<id>.children[].meta`)

| Field | Meaning |
|-------|---------|
| `dates` | `prereg_at`, `graded_at`, `closed_at` (null if N/A) |
| `model` | Usually inherits parent; override if different ckpt |
| `inputs` | PDBs, arms, graft definition, controls, metric |
| `outputs` | Artifact paths, headline numbers, verdict |
| `docs` | Pre-reg, closeout, make target |
| `depends_on` | Sibling child ids (also top-level on child) |

## Rules

1. Fill `meta` when opening a child (prereg) — outputs may be null.  
2. Update `outputs` + `dates.graded_at` on grade; `dates.closed_at` on closeout.  
3. Do not invent Pass claims in meta that contradict `ledger` / child `status`.  
4. Flat `ledger` remains the grade SSOT; meta is narrative provenance.
