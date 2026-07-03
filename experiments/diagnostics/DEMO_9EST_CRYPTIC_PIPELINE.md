# Pipeline Demo — 9EST Cryptic Pocket Finder (Exploratory)

**Status:** Exploratory demonstration — **not** part of the frozen inference-only pre-registration.  
**Companion:** `experiments/diagnostics/demo_9est_cryptic_pipeline.py`  
**Inference prereg (closed):** `experiments/diagnostics/PREREG_9EST_1FLE_cryptic_interface.md` → **REFUTE**

---

## Purpose

Validate whether the **ingest cryptic-pocket pipeline** (GNN seeds + fpocket merge + optional SMD) can surface the 1FLE elafin interface on static 9EST — without amending the locked S1 inference test.

This demo also answers: **does the binding-site scan need v6 architecture updates?**

---

## Pipeline under test

```
lever_a forward pass (9EST)
    → channel profile (v5_legacy vs v6_lever_a)
    → DBSCAN seed clusters
    → fpocket geometry pockets
    → merge + classify + rank
    → overlap with I_gold (26 residues, from lock JSON)
```

SMD validation (`md_validate_top_n`) is part of production ingest but **not** required for this offline demo (run separately on science container when GPU path is available).

---

## Endpoints (exploratory — not frozen)

| Metric | Definition |
|---|---|
| Qualifying residues | Count passing profile epistemic + shell gates |
| Sites found | Ranked `UnifiedCandidate` count after merge |
| Top-site gold recall | \|site₁ ∩ I_gold\| / \|I_gold\| |
| Top-3 union recall | \|⋃rank≤3 sites ∩ I_gold\| / \|I_gold\| |

---

## v6 channel profile change (implemented)

| | v5_legacy | v6_lever_a |
|---|---|---|
| Epistemic gate | absolute ≥ 9.5 | p75 on structure (~1.586 on 9EST) |
| Shell gate | cone_depth ≥ 6.0 | disc_r ≥ p50 (~0.40 on 9EST) |
| Shell field | `cone_depth` | `‖hyp_projections_2d‖` |

**Finding:** v5 thresholds on a lever_a checkpoint yield **0 qualifying residues** on 9EST (epistemic ~1.57–1.60, cone_depth ~0.33–1.35). The scan phase must use `profile_for_model_version()` — wired in `scan_phase.py` and `gnn_channel_profile.py`.

---

## Run

```bash
make demo-9est-cryptic-pipeline
# or with artifact:
python -m experiments.diagnostics.demo_9est_cryptic_pipeline --write-json
```

## Initial run (2026-07-02, lever_a, fpocket unavailable)

| Profile | Qualifying | Sites | Top-site gold recall | Top3 union recall |
|---|---|---|---|---|
| v5_legacy | 0 | 0 | 0.000 | 0.000 |
| v6_lever_a | 60 | 6 | 0.000 | 0.000 |

v6 profile unblocks seed generation; interface overlap still zero without geometry merge (fpocket missing on host) and before SMD. Re-run with fpocket installed for hybrid sites.

| | Inference prereg | This demo |
|---|---|---|
| Question | Does static S1 recover interface? | Does ingest scan tag interface region? |
| MD | Excluded | SMD optional at ingest (separate) |
| Verdict | REFUTE (closed) | Exploratory — open |
| Amend frozen S1? | No | No |
