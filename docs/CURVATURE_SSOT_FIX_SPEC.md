# Curvature SSOT Fix — Implementation Specification

## Tokyo Eye / DTIE · Eidetix Bio

| | |
|---|---|
| **Status** | SPEC — ready for implementation (no code until approved) |
| **Spec date** | 2026-07-02 |
| **Canonical value** | `0.7026273608207703` |
| **Space name** (`embedding_space.name`) | `gospconemapper_v6_hyp128` |
| **Space id** (`embedding_space.space_id`) | `space_gospconemapper_v6_hyp128` |
| **Checkpoint** | `lever_a_clean_slate_v1/v6_best_disc.pt` |
| **Phase 1a** | Code consumer fix + DB hash column (no Secrets Manager) |
| **Phase 1b** | Secrets Manager hardening (deferred after 1a stable) |
| **Phase 2** | Re-ingest (separate sprint; gated on 1a + staging/prod check) |

---

## 1. Problem Statement

DTIE's Poincaré ball embedding uses curvature `c` in every hyperbolic distance, normalization, and vector query. A **split-brain** occurs when components read `c` from different sources.

**What pre-work (2026-07-02) established:**

| Source | Value | Status |
|--------|-------|--------|
| `ckpt['curvature']` / `V6GNNRunner` | `0.7026273608207703` | **Canonical** |
| Local dev `embedding_space.curvature` | `0.7026273608207703` | **Matches checkpoint** |
| `vector_queries.py` default | `0.6054343` | **Stale — bug** |
| `curvature_registry.CANONICAL_TS002` | `0.6054342985153198` | **Stale — bug** |
| `hyperbolic_distance_populator.V6_C` | `0.6054342985153198` | **Stale — bug** |
| `discover_hyperbolic_motifs.DEFAULT_CURVATURE` | `0.6054342985153198` | **Stale — bug** |

The **DB is already correct** on local dev. The bug is **four hardcoded constants + two CLI flags**, not Aurora vs checkpoint on this machine.

At c=0.605 vs c=0.703, Poincaré distances differ by ~7–14% depending on disc position — silent errors on every path through `vector_queries` or the distance populator.

**This spec is not:** a training change, model change, or full schema redesign. It is surgical code hygiene.

**Legacy (~40% of 514 grep hits):** v3/v4 training scripts and archived orchestrators — **do not touch in Phase 1**. Document as non-runtime-critical for GNNv6.

---

## 2. Canonical Value — Pin at Full Precision

```
Source:     lever_a_clean_slate_v1/v6_best_disc.pt
Key:        ckpt['curvature']  (also top-level checkpoint field)
Value:      0.7026273608207703
Derivation: softplus(log_c) + 1e-4
            log_c = 0.01867313
            softplus(0.01867313) = 0.70252733
            + 1e-4 = 0.70262734 (matches runner property)
```

### Confirmed pin (do not substitute)

| Value | Verdict |
|-------|---------|
| **`0.7026273608207703`** | **Use this everywhere** |
| `0.7033` | Reject — conversation rounding artifact |
| `0.605434…` | Reject — stale TS-002 constant |
| `0.695…` | **Unexplained** — see §8 (pre-Phase-2 gate) |

Store at **full float64 precision**. Rounding creates hash-mismatch bugs once `curvature_hash` is enforced.

**Do not read curvature from the checkpoint at runtime** after pin. Checkpoint is a training artifact; pinned `c` is infrastructure configuration in `embedding_space` (later Secrets Manager in 1b).

---

## 3. Consumer Inventory (514 lines — `/tmp/curvature_consumers.txt`)

### 3.1 Phase 1a — MUST fix (runtime)

| File | Current | Fix |
|------|---------|-----|
| `data/db_helpers/vector_queries.py:100` | `curvature_c=0.6054343` | `get_curvature('gospconemapper_v6_hyp128')` |
| `science/dtie/common/curvature_registry.py` | `CANONICAL_TS002`, `VECTOR_QUERIES_DEFAULT_C` | Deprecate; align probe with `get_curvature()` |
| `science/dtie/v5/workers/hyperbolic_distance_populator.py` | `V6_C = 0.605434…` | `get_curvature('gospconemapper_v6_hyp128')` |
| `scripts/discover_hyperbolic_motifs.py` | `DEFAULT_CURVATURE = 0.605434…` | `get_curvature('gospconemapper_v6_hyp128')` |

### 3.2 Phase 1a — CLI removal

| File | Flag | Action |
|------|------|--------|
| `science/dtie/v3/orchestrator/pipeline.py` | `--curvature-c` | Remove; no override path |
| `scripts/discover_hyperbolic_motifs.py` | `--curvature` | Remove |

### 3.3 Do not modify (correct passthrough)

- `science/contracts/geometric_runtime.py` — reads `embedding_space.curvature`
- `science/dtie/v6/gnn/runner.py` — `softplus(log_c)+eps` at inference
- `data/normalizer/core.py` — upserts `embedding_space.curvature` on write
- `science/compute/runner_dispatch.py` — audit + passthrough

### 3.4 Test fixtures — update, do not delete

- `tests/test_poincare_conventions.py` — `0.605` → `0.7026273608207703` or fixture via `get_curvature()`
- `tests/test_discover_hyperbolic_motifs.py` — same

### 3.5 Legacy — DO NOT TOUCH in Phase 1

v3/v4 orchestrators, training scripts, notebooks (~40% of grep). Not on GNNv6 inference path.

---

## 4. Phase 1a — Implementation

### 4.1 New module: `science/dtie/common/curvature_loader.py`

```python
def get_curvature(space_name: str) -> float:
    """Single SSOT read for pinned hyperbolic curvature.

    Phase 1a: Aurora embedding_space only (no Secrets Manager).
    Phase 1b: adds Secrets Manager cross-check (deferred).
    """
```

**Behavior (Phase 1a):**

1. Query `embedding_space` by `name = space_name` (not `space_id`).
2. **Fail loud** (`CurvatureSSOTError`) if row missing, or `curvature IS NULL`.
3. If `curvature_hash` column present and non-null: verify `SHA256(repr(c))` matches stored hash (Python `repr` of float — same rule used in migration population script).
4. Return `float` at full precision.
5. **No fallback constants.** No silent warnings.
6. **No Secrets Manager** in 1a.
7. Importable without torch/GPU (DB scripts and tests).

**Default space for v6 production:** `gospconemapper_v6_hyp128`

### 4.2 Migration `051_embedding_space_curvature_hash.sql`

**Table:** `embedding_space` (not `dim_space_metadata` — that table does not exist in this repo).

```sql
-- Migration 051: curvature_hash for SSOT verification
ALTER TABLE embedding_space
    ADD COLUMN IF NOT EXISTS curvature_hash VARCHAR(64);

-- Populate for production hyp space (run after column add)
UPDATE embedding_space
SET curvature_hash = encode(
    digest(CAST(curvature AS TEXT), 'sha256'), 'hex'
)
WHERE name = 'gospconemapper_v6_hyp128'
  AND curvature IS NOT NULL;
```

**Note:** Migration SQL uses `CAST(curvature AS TEXT)` for population; `curvature_loader` verification must use the **same stringification** as migration (document and test in `test_curvature_loader.py`). If Python `repr(c)` ≠ SQL `CAST`, align both to one canonical string format in Commit 1.

Non-breaking: existing reads of `embedding_space.curvature` unchanged.

**Verify locally after migration:**

```sql
SELECT name, curvature, curvature_hash, created_at
FROM embedding_space
WHERE name = 'gospconemapper_v6_hyp128';
-- Expected curvature: 0.7026273608207703
```

### 4.3 Commit discipline (five commits — independently revertable)

```
Commit 1 — Migration only
  data/aurora/migrations/051_embedding_space_curvature_hash.sql
  Run make migrate; verify SELECT above
  NO application code changes

Commit 2 — curvature_loader.py only
  science/dtie/common/curvature_loader.py
  science/dtie/common/exceptions.py (or local CurvatureSSOTError)
  tests/test_curvature_loader.py (mock DB + hash round-trip)
  NO consumer changes yet

Commit 3 — Consumer replacements + CLI removals
  vector_queries.py
  curvature_registry.py (deprecate CANONICAL_TS002 for runtime; keep bucket labels if needed)
  hyperbolic_distance_populator.py
  discover_hyperbolic_motifs.py (+ remove --curvature)
  science/dtie/v3/orchestrator/pipeline.py (remove --curvature-c)

Commit 4 — Test fixture updates
  test_poincare_conventions.py
  test_discover_hyperbolic_motifs.py

Commit 5 — Gate tests in property suite
  P_CURV_01, P_CURV_02, P_CURV_03 (see §5)
```

**Do not bundle migration + consumer replacement in one commit.**

---

## 5. Gate Tests (Phase 1a complete)

### P_CURV_01 — `probe_curvature_sources()` (existing tool)

```python
from science.dtie.common.curvature_registry import probe_curvature_sources
import asyncio

report = asyncio.run(probe_curvature_sources("11qe"))  # any ingested structure

# Today (pre-1a):
#   model_vs_db_match = True
#   model_vs_vector_queries_match = False  ← the bug

# After Phase 1a (once vector_queries uses get_curvature):
assert report.model_vs_db_match is True
assert report.model_vs_vector_queries_match is True
assert report.model_checkpoint_c is not None
assert abs(report.model_checkpoint_c - 0.7026273608207703) < 1e-12
```

Extend `probe_curvature_sources` if needed so `vector_queries_default_c` reflects `get_curvature()` rather than module-level constant.

### P_CURV_02 — No stale literals in runtime paths

Lint or test: no `0.605434`, `0.6054343`, or bare `0.605` in `data/db_helpers/`, `science/dtie/v5/workers/`, `scripts/discover_hyperbolic_motifs.py`, excluding legacy v3/v4 paths (see `scripts/lint_curvature_literals.py` if present).

### P_CURV_03 — No CLI curvature override

```bash
python scripts/discover_hyperbolic_motifs.py --help
# must not contain --curvature
```

### P_CURV_04 — Inference non-regression (manual, after P_CURV_01–03)

Run lever_a inference on one structure (e.g. `11QE`) **after** Phase 1a. Metrics must match pre-1a baseline exactly:

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| σ₂/σ₁ | 0.665 | ±0.001 |
| disc_thick | 0.219 | ±0.001 |
| r(d,s) | 0.730 | ±0.005 |
| r(e,s) | 0.780 | ±0.005 |

**If anything shifts:** stop. Do not proceed to 1b or Phase 2. Loader introduced a regression.

Also run: `make test` (full unit suite).

---

## 6. Phase 1b — Secrets Manager (Deferred)

**Not required to fix the current bug.** Implement after Phase 1a is stable in dev/staging.

- Write `0.7026273608207703` to AWS Secrets Manager: `dtie/curvature/gospconemapper_v6_hyp128`
- Extend `get_curvature()` with SM read + Aurora `curvature_hash` cross-check
- Fail loud on SM ↔ DB mismatch
- One-time manual SM write (not code-automated on first deploy)

---

## 7. Phase 2 — Re-ingest (Separate Sprint)

Gated on:

1. Phase 1a complete (P_CURV_01–04 pass)
2. **Staging/prod curvature check** (§8) — scope finalized only after this

Preliminary scope (local dev already aligned on `c`):

```
Corpus:      manifests/v6_corpus_120.json (minus holdouts)
Checkpoint:  lever_a_clean_slate_v1/v6_best_disc.pt
Space:       gospconemapper_v6_hyp128
Curvature:   get_curvature() — never hardcoded
Verify:      embedding row count ≈ corpus residue count
               curvature_hash matches post-1a hash
               σ₂/σ₁, disc_thick, r(d,s) stable vs 6-structure baseline
```

Phase 2 does **not** change model or canonical `c`. It refreshes embeddings for expanded corpus under consistent geometry.

---

## 8. Pre-Phase-2 Gate — Staging/Prod Curvature Check (DO NOT SKIP)

### The ~0.695 figure is unexplained

Earlier sprint notes referenced ~0.695 for stored curvature. Local dev holds `0.7026273608207703` (matches checkpoint). **Origin of 0.695 is unknown** — possibly staging/prod, pre-normalizer writes, or a different checkpoint era.

**This is not a Phase 1 concern.** It **is** a hard gate before Phase 2 re-ingest scope is finalized.

### Required SQL (schema-accurate for this repo)

There is no `embeddings` table and no `space_name` column — use `embedding_space.name` and `fact_gnn_node_embedding`:

```sql
SELECT
    es.name,
    es.space_id,
    es.curvature,
    es.curvature_hash,
    COUNT(e.embedding_id) AS embedding_rows,
    MIN(e.computed_at) AS first_embedding,
    MAX(e.computed_at) AS last_embedding
FROM embedding_space es
LEFT JOIN fact_gnn_node_embedding e ON e.space_id = es.space_id
WHERE es.space_type = 'hyperbolic'
GROUP BY es.name, es.space_id, es.curvature, es.curvature_hash
ORDER BY last_embedding DESC NULLS LAST;
```

Run on **staging** and **production** before Phase 2 design.

### Decision table

| Staging/prod `curvature` | Phase 2 implication |
|--------------------------|---------------------|
| `0.7026273608207703` | Re-ingest for corpus expansion only (same geometry) |
| `0.695…` or other | **Full re-ingest required** — stored embeddings on wrong coordinate system |
| `0.605…` | **Full re-ingest + audit** — TS-002 era data |
| NULL / missing rows | Fix embedding_space registration before re-ingest |

Record results in §10 audit trail before opening Phase 2 sprint.

---

## 9. What Not To Do

- Bundle migration + consumer fix in one commit
- Touch legacy v3/v4 scripts in Phase 1
- Use rounded pin (`0.7033`, `0.703`, `0.70263`)
- Read `c` from checkpoint at runtime after pin
- Skip staging/prod check before Phase 2
- Proceed to Phase 2 if P_CURV_04 fails

---

## 10. Audit Trail

| Event | Date | Artifact / result |
|-------|------|-------------------|
| Split-brain identified | prior sprint | conversation + `probe_curvature_sources` |
| Pre-work: canonical from checkpoint | 2026-07-02 | `c = 0.7026273608207703` |
| Pre-work: consumer grep | 2026-07-02 | 514 lines → `/tmp/curvature_consumers.txt` |
| Pre-work: local Aurora | 2026-07-02 | `embedding_space.curvature` matches checkpoint |
| Spec approved | — | this document |
| Commit 1–5 (Phase 1a) | — | — |
| P_CURV_01–04 pass | — | — |
| Staging prod curvature check | — | **required before Phase 2** |
| Phase 1b (Secrets Manager) | — | — |
| Phase 2 re-ingest | — | — |

---

## 11. Related Science (out of scope for this spec)

**Exploratory (not pre-registered):** G12D oncogenic substitution may produce angular dehydron reorganization in Poincaré embedding (WT partial → G12D pass → G12D/I55E stronger). Document separately in `TIER2_CRESCENT_DECLARATION.md` § Exploratory; confirmation protocol: NRAS Q61R + BRAF V600E crescent projections. Do not promote to confirmed finding without pre-reg.

**Pharmacophore surface SBIR panel:** `checkpoints/v6/diagnostics/pharmacophore_surface/` — use honest caption (mixed transitional band); do not extend before review.
