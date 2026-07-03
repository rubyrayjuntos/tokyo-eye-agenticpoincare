# Stage A Corpus Selection — Design & Redundancy Script Spec

**Status:** Pre-implementation (blocks Stage A training)  
**Governance coupling:** `docs/TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md`  
**Date:** 2026-07-03

---

## 1. Problem statement

Stage A corpus size (~20–28 structures) is a **derived endpoint**, not a prior. The diversity axis is **CATH topology** (fold architecture), not functional gene family and not the manifest's legacy `fold_class` field (CATH *Class* tier — 4 buckets only).

The existing `manifests/v6_corpus_120.json` already encodes a leakage pattern (multiple KRAS/NRAS/HRAS variants, `max_sequence_identity_pct: 30` unenforced). `per_family_loss.{family}` in MLflow governance is the wrong imbalance metric — same class of units mismatch as the pre-fix routing gate.

---

## 2. Non-negotiable sequencing

These steps are **not parallel**. Step 2 depends on step 1's `fold_id` vocabulary; step 3 must not run before step 2.

```
1. Redundancy script on candidate pool
   → fetch CATH Architecture/Topology per structure
   → emit fold_id vocabulary + pairwise audit
   → produce locked Stage A manifest draft

2. per_family_loss → per_fold_loss rename (mlflow_governance + gate + docs)
   → keys on script's fold_id (CATH topology code, not manifest fold_class)

3. Stage A training
   → never before step 2; manifest locked only after P_CORPUS_01 passes
```

---

## 3. fold_id granularity (load-bearing)

| Tier | CATH example | Usable as fold_id? |
|------|----------------|-------------------|
| Class | `alpha`, `alpha_beta` | **No** — manifest `fold_class` is here; masks imbalance |
| Architecture | `3.40.50` | **Yes (minimum)** |
| Topology | `3.40.50.720` | **Yes (preferred)** |
| Superfamily | finer | Optional; usually too fine for ~25 corpus |

**Rule:** The script **overwrites** any manifest `fold_class` / `fold_id` with fetched CATH codes. `FOLD_UNVERIFIED` means API/SIFTS returned nothing — manual resolution required before lock. Do **not** compare against or preserve manifest Class-tier labels.

**Canonical key:** `fold_id` = CATH topology code when available; else CATH architecture code with `_arch` suffix and `FOLD_UNVERIFIED` flag until topology resolved.

---

## 3.5 Phase 0 — CATH coverage probe (run before building matrix)

**Module:** `experiments/training/v6/cath_coverage_probe.py`  
**Makefile:** `make cath-coverage-probe MANIFEST=manifests/v6_corpus_120.json`  
**Cache:** `manifests/cache/cath_pdbe/{pdb_id}.json` (frozen PDBe responses; re-run uses cache unless `--refresh`)  
**Report:** `manifests/cache/cath_coverage_report.json` (committed snapshot after probe)

Chain matching: match manifest `chain` against PDBe `chain_id` **or** `struct_asym_id` (author vs asym mismatch is common).

### Probe result — `v6_corpus_120` enabled (2026-07-03)

| Tier | Count | % |
|------|-------|---|
| CATH Topology `fold_id` | **40 / 46** | **87.0%** |
| Architecture-only | 0 | 0% |
| Unverified | 6 | 13.0% |
| Mixed-tier vocabulary | No | — |

**Conclusion:** 87% topology, zero architecture-only, no mixed-tier — fallback branch is dead code on this pool. Matrix input is **40 verified structures** after §3.6 dispositions applied (not raw 46).

### CATH fetch — cache, fallback, selection behavior

| Situation | Policy |
|-----------|--------|
| PDBe returns Topology (4-part code) | `fold_id` = code; `fold_id_tier: topology` |
| PDBe returns Architecture only (3-part) | **Corpus-wide rule:** if *any* structure is architecture-only, entire locked manifest uses architecture tier for all keys (no mixed topology/architecture in `per_fold_loss`). On current pool: N/A (0 architecture-only). |
| PDBe 404 / no mapping | `FOLD_UNVERIFIED` — **excluded from Stage A train** by default; eligible for `manual_fold_id` in manifest with `fold_id_source: manual` + reviewer note |
| SCOP fallback | **Deferred** — only if manual assignment insufficient; not in MVP |
| Re-fetch in CI | **Never** — P_CORPUS_01 uses frozen report (§6) |

**Rate limits:** probe caches per-PDB JSON; 150ms delay on live fetch; CI reads cache/report only.

---

## 3.6 Locked dispositions — six unverified entries (2026-07-03)

Resolved **before** redundancy matrix run. No new policy — each row cites an existing rule. Matrix input = 40 verified structures (46 enabled − 6 below).

| PDB:chain | Gene | Disposition | `fold_id` | Governing rule |
|-----------|------|-------------|-----------|----------------|
| 1D0T:A | D0T | **exclude** — not in matrix input | — | Sub-fold / peptide: below CATH fold threshold; degenerate residue graph; category error for fold-diversity corpus |
| 1VII:A | VII | **exclude** | — | Miniprotein (~36 res villin headpiece); no CATH topology; routing unlike full-length targets |
| 1E0L:A | E0L | **exclude** | — | Miniprotein (trp-cage / WW-scale); same as above |
| 1BEN:A | BEN | **exclude** | — | Generic filler; PDBe CATH 404; not oncogene spine |
| 1CHO:A | CHO | **exclude** (Stage A) | — | Manifest bug: chain A unmapped; CATH on F/G/I — see manifest corrections below |
| 6OIM:A | KRAS | **eval-only** (never Stage A train) | `3.40.50.300` manual | `fold_id_source: manual` (G-domain, sibling KRAS topology). Disposition from **per-fold cap** / redundancy — another `3.40.50.300` train slot, not CATH gap |

**Matrix input pool:** 40 structures with verified PDBe CATH topology (`cath_coverage_report.json`, excluding above six).

### Manifest corrections (deferred fixes, logged)

| PDB | Issue | Action |
|-----|--------|--------|
| 1CHO | `chain: A` has no CATH; domains on F/G/I | **Defer fix, exclude Stage A.** Re-entry requires chain correction + re-probe |

### 3.7 Locked review dispositions — TM-align draft (2026-07-03)

Applied to `corpus_redundancy_report_tmalign.json` via `lock_stage_a_corpus.py`. Metric: `tm_align_tmtools`.

| PDB:chain | Action | Rule |
|-----------|--------|------|
| 1SUP:A | train → holdout | `CROSS_FOLD_HIGH_TM` vs 4OBE (CATH sibling `3.40.50.200`); SOD filler yields to GTPase centrality |
| 2ABD:A | train → holdout | `CROSS_FOLD_HIGH_TM` hub (9 pairs); acyl-carrier filler mimics multiple fold trains |
| 3CON:A | holdout → train | GTPase **cap-2 biological-centrality override** — structurally farthest RAS paralog from 4OBE (TM=0.894) |

**GTPase cap-2 principle (locked):** fold `3.40.50.300` carries 2 train slots (`4OBE:A` + `3CON:A`) by documented biological-centrality override, not algorithm-default cap-1. Identity dedup still governs KRAS-variant collapse; the second slot is paralog diversity (NRAS), not a third KRAS mutant.

**Locked output:** 25 train / 16 holdout, 0 cross-fold high-TM violations among train pairs.

---

## 4. Redundancy script

### 4.1 Location & invocation

| Item | Value |
|------|--------|
| Module | `experiments/training/v6/corpus_redundancy.py` |
| CLI | `python -m experiments.training.v6.corpus_redundancy` |
| Makefile | `make corpus-redundancy-check MANIFEST=manifests/v6_corpus_120.json` |
| Default input | **40 verified structures** — `v6_corpus_120` enabled entries minus §3.6 exclusions; CATH from `manifests/cache/cath_pdbe/` |
| Outputs | `corpus_redundancy_report.json`, `corpus_fold_vocabulary.json`, `manifests/v6_corpus_stage_a_draft.json` |

**Filename discipline:**

| File | Role |
|------|------|
| `v6_corpus_stage_a_draft.json` | Script output — **not** CI-gated; human review required |
| `v6_corpus_stage_a.json` | **Locked** train manifest — committed; P_CORPUS_01 target |
| `corpus_redundancy_report.json` | Frozen pairwise decisions — artifact of record for CI |

Draft → lock is a manual gate. Never point P_CORPUS_01 at the draft.

### 4.2 Structural similarity tool (locked)

**TM-align** (or repo-standard US-align wrapper if already present) for all locking decisions.

- Pool size ≤ ~300 pairs: exact TM-align is tractable.
- **Do not** use Foldseek TM for the lock — different calibration; margins disagree with the &lt;0.5 fold boundary literature.
- Report records `structural_metric: "tm_align"` and version/commit of the binary.

### 4.3 Split-metric routing (do not use one threshold for both jobs)

| Decision | Metric | Threshold | Rationale |
|----------|--------|-----------|-----------|
| **Cross-fold distinctness** | TM-score (TM-align) | &lt; **0.5** required across different `fold_id` | Fold/not-fold boundary ([Zhang & Skolnick 2005](https://doi.org/10.1093/bioinformatics/btag058); van Kempen et al. 2024) |
| **Within-fold dedup** | Sequence identity (%), same `fold_id` | &lt; **30%** to keep both; ≥ threshold → drop one | Sharper for near-duplicates; TM ~0.55 is ambiguous |
| **Per-fold train cap** | same `fold_id` | **≤ `max_train_per_fold_id`** (default **2**) in Stage A train | Generalizes KRAS rule: kinase cluster, G-domain cluster, any over-represented topology |
| **Train selection within fold** | TM-align distance | farthest-first among identity-eligible; maximize structural spread | Sequence spread ≠ structural spread; geometry is the training signal |

**Note:** KRAS/NRAS/HRAS at `3.40.50.300` are one fold_id — not a special gene rule. The per-fold cap handles them; 6OIM is eval-only because the fold is already saturated, not because CATH failed.

**Flag codes:**

| Flag | Meaning |
|------|---------|
| `CROSS_FOLD_HIGH_TM` | Different `fold_id`, TM ≥ 0.5 — review fold assignment or drop one |
| `WITHIN_FOLD_HIGH_IDENTITY` | Same `fold_id`, identity ≥ threshold — near-duplicate |
| `FOLD_TRAIN_CAP_EXCEEDED` | &gt;`max_train_per_fold_id` structures from same `fold_id` in proposed train set |
| `FOLD_UNVERIFIED` | No CATH topology/architecture resolved |
| `EXCLUDED_PRE_MATRIX` | Removed by §3.6 before matrix (peptide, filler, manifest bug) |
| `MISSING_STRUCTURE` | PDB/chain not loadable |

### 4.4 CATH / fold label fetch

1. **Primary:** PDBe SIFTS or CATH API → topology code per chain (document which API in report metadata).
2. **Multidomain:** use domain covering the ingested chain segment; if multiple domains, pick highest residue-coverage domain and log `domain_selection` in per-structure record.
3. **Output per structure:**

```json
{
  "pdb_id": "4OBE",
  "chain": "A",
  "gene": "KRAS",
  "fold_id": "3.40.50.300",
  "fold_id_tier": "topology",
  "cath_class_legacy": "alpha_beta",
  "cath_class_warning": "manifest fold_class is Class tier — ignored for fold_id",
  "flags": []
}
```

### 4.5 Pairwise matrix

For each unordered pair (i, j):

```json
{
  "a": "4OBE:A",
  "b": "11QE:A",
  "fold_id_a": "3.40.50.300",
  "fold_id_b": "3.40.50.300",
  "same_fold_id": true,
  "sequence_identity_pct": 94.2,
  "tm_score": 0.91,
  "decision": "WITHIN_FOLD_HIGH_IDENTITY",
  "action": "drop_one_for_stage_a",
  "metric_used": "sequence_identity"
}
```

Cross-fold example:

```json
{
  "a": "4OBE:A",
  "b": "1IVO:A",
  "same_fold_id": false,
  "tm_score": 0.31,
  "decision": "CROSS_FOLD_DISTINCT",
  "metric_used": "tm_score"
}
```

### 4.6 Stage A draft selection (greedy)

**Config (report metadata):**

```json
{
  "max_train_per_fold_id": 2,
  "sequence_identity_threshold_pct": 30,
  "cross_fold_tm_threshold": 0.5,
  "within_fold_spread_metric": "tm_align"
}
```

**Cap policy vs train count:** `max_train_per_fold_id: 2` is a **balance** policy (no single fold dominates training signal). It generalizes the KRAS leakage rule but is not identical to it — leakage is handled by identity dedup; the cap limits per-fold representation. The resulting Stage A **train count is emergent** (depends on fold distribution in the 40-structure pool) and is validated only after the matrix report — not asserted at ~25 upfront. Example: 6 well-populated folds × 2 + 4 singletons × 1 → 16 train, not 25; if the report yields a thin train set, raise the cap or accept smaller Stage A as a **data-dependent** decision at draft review, not in this spec.

**Within-fold train-slot selection:** After identity dedup, pick up to `max_train_per_fold_id` structures per fold by **farthest-first spread on structural distance** (TM-align score between structures; maximize minimum pairwise TM distance among selected set — prefer conformationally/topologically distinct pairs). Do **not** use sequence identity for spread ranking — sequence-divergent pairs can be structurally silent; the training signal is geometric (dehydron topology, wrapping). Sequence identity is only for dedup/removal, not for spread ranking.

After pairwise audit on **40-structure input**:

1. Group by `fold_id` (topology).
2. **Within each fold:** drop near-duplicates (identity ≥ threshold); keep structures that pairwise identity &lt; threshold.
3. **Per-fold train cap:** select up to `max_train_per_fold_id` structures per `fold_id` for Stage A **train**, using farthest-first on **TM-align distance** among identity-eligible structures (see above).
4. All other structures in pool → `role: eval_holdout` (includes saturated folds, 6OIM, redundant KRAS/kinase variants).
5. Target ~8–10 distinct `fold_id` with 1–2 train each → ~20–28 train total; exact count is output.
6. Emit `manifests/v6_corpus_stage_a_draft.json` — every entry has `fold_id`, `role`, and `disposition_rule` citation.

**Important:** 40 verified ≠ 40 non-redundant. Within-fold collapse (e.g. multiple `3.40.50.300` GTPases, kinase cluster) is where spine count drops from ~40 toward ~25.

### 4.7 Report root schema

```json
{
  "spec_version": "stage-a-corpus-redundancy:2026-07-03",
  "input_manifest": "manifests/v6_corpus_120.json",
  "structural_metric": "tm_align",
  "tm_align_version": "...",
  "sequence_identity_threshold_pct": 30,
  "cross_fold_tm_threshold": 0.5,
  "max_train_per_fold_id": 2,
  "input_structure_count": 40,
  "excluded_pre_matrix": ["1D0T:A", "1VII:A", "1E0L:A", "1BEN:A", "1CHO:A"],
  "eval_only_manual": ["6OIM:A"],
  "structures": [ "... per-structure records ..." ],
  "pairs": [ "... pairwise records ..." ],
  "fold_vocabulary": ["3.40.50.300", "1.20.120.340", "..."],
  "stage_a_draft": {
    "train_count": 24,
    "eval_holdout_count": 3,
    "fold_count": 9,
    "violations": []
  },
  "pass": true
}
```

`pass: false` if any blocking flag remains in proposed Stage A train set.

---

## 5. Governance rename (step 2 — after script)

### 5.1 MLflow metrics

| Old | New |
|-----|-----|
| `per_family_loss.{family}` | `per_fold_loss.{fold_id}` |
| `corpus_families()` | `corpus_fold_ids(manifest)` — reads `fold_id` from locked manifest |

`fold_id` strings in metrics: sanitize CATH dots to underscores for MLflow keys (e.g. `per_fold_loss.3_40_50_300`) with reverse map in report metadata.

### 5.2 Stage gate (`stage_a_gate_passed`)

Replace:

```
per_family_loss max/min ratio < 3.0
```

With:

```
per_fold_loss max/min ratio < 3.0   # across fold_ids present in locked Stage A manifest
```

Update `docs/TRAINING_GOVERNANCE_AND_MLFLOW_SCHEMA.md` §3.2, §5 gate definitions, and P_ROUTING / P_MLFLOW tests.

---

## 6. P_CORPUS_01 — CI property gate

**Purpose:** Enforce redundancy constraints on the **locked** Stage A manifest without live CATH/TM-align in CI.

**Test:** `tests/test_corpus_redundancy_gate.py::test_p_corpus_01_locked_stage_a_manifest_passes`

**Behavior (frozen report — no recompute in CI):**

1. Load committed `manifests/v6_corpus_stage_a.json` (locked — never the `_draft` file).
2. Load committed `manifests/corpus_redundancy_report.json`; SHA256 pins in `science/training/corpus_governance.py` must match (fail-closed with explicit regeneration message if desynced).
3. Assert locked manifest `proteins` list matches report `stage_a_locked.train` entries (same pdb_id:chain set).
4. Assert report `pass is True`, `status == "locked"`, and `structural_metric` is TM-align-backed (`tm_align`, `tm_align_binary`, or `tm_align_tmtools` — **not** `biotite_ca_proxy`).
5. Assert every train entry has `fold_id` + `fold_id_tier` (topology or corpus-uniform architecture).
6. Assert no train pair violates flags in report (identity / cross-fold TM / per-fold cap).
7. Assert `max_sequence_identity_pct` from manifest honored.

**Recompute:** deliberate, on-demand only — `make corpus-redundancy-check` locally; commit updated report + locked manifest together.

**Skip:** If locked manifest or frozen report absent, `pytest.skip`.

**CI:** Add to `gates.yml` after first lock — no network, no TM-align binary required on runner.

---

## 7. Literature anchors (what we claim vs don't)

| Claim | Source |
|-------|--------|
| Fold-level split is hardest OOD axis for structure embeddings | SCOPe benchmarks: family / superfamily / fold tiers ([btag058](https://doi.org/10.1093/bioinformatics/btag058)); Topotein Fold split |
| TM &lt; 0.5 ≈ different fold | Zhang & Skolnick; used in van Kempen et al. FP definition |
| MoE routing minimum corpus size | **Not in literature** — empirical; `effective_experts` + `per_fold_loss` ratio gate |

---

## 8. Open governance items (cross-ref)

| Item | Status after this spec |
|------|------------------------|
| Define Stage A corpus (family balance) | **Superseded** → fold-topology selection via script |
| `max_sequence_identity_pct` enforcement | **P_CORPUS_01** + redundancy script |
| `per_family_loss` imbalance gate | **Superseded** → `per_fold_loss` (CATH fold_id) |
| Stage A training | **Blocked** until steps 1–2 + P_CORPUS_01 green |

---

## 9. Implementation tasks (ordered)

0. `cath_coverage_probe.py` — **done**; `cath_coverage_report.json` committed
1. Six unverified dispositions — **locked** §3.6
2. `corpus_redundancy.py` — TM-align + identity matrix on 40-structure input; per-fold cap selection; draft manifest
3. `make corpus-redundancy-check` — human review → lock `v6_corpus_stage_a.json` + `corpus_redundancy_report.json`
4. `per_fold_loss` rename + gate update + governance doc
5. `test_p_corpus_01` (frozen report) + `gates.yml` entry
6. Stage A training launch
