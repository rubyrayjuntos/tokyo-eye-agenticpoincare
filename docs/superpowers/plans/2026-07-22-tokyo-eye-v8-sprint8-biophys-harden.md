# TokyoEye-v8 Sprint 8 — Biophys Harden + Mode C

> **For agentic workers:** Execute milestones in order. Design SSOT:
> [`docs/superpowers/specs/2026-07-22-tokyo-eye-v8-sprint8-biophys-harden-design.md`](../specs/2026-07-22-tokyo-eye-v8-sprint8-biophys-harden-design.md)

**Goal:** Replace isotropic proximity H-bond/wrap rules with Kabsch–Sander electrostatic gating + directional double-cone wrapping; cache graphs; Mode C KRAS basin train.

**Architecture:** Energy + H reconstruction in `biophysics.py`; R1/R2 gated in `r0_r5_graph.py`; hashed cache in `loader.py`; manifest `manifests/v8_kras_nucleotide_basin_v1.json`.

## Frozen contracts

| Item | Value |
|------|-------|
| H placement | 1.01 Å along planar bisector ∠C(prev)–N–Cα |
| DSSP admit | `E ≤ -0.5` kcal/mol |
| Double cone | `abs(dot(v̂, û)) ≥ cos(45°) ≈ 0.7071` |
| Wrap τ | `DEHYDRON_WRAP_MAX=19`; Epoch-0 auto-median if 4OBE `dehydron_frac ≥ 0.60` |
| Cache version | `v8_biophys_s8` |

## Global constraints

- Isolation under `science/tokyo_eye/v8/`
- No Normalizer / ingest writes
- Do not change hyp attention / MoE / SE(3)-lite contracts

---

### Milestone 1 — Native electrostatic & angle-gated engine

- [x] `place_backbone_amide_h`, `kabsch_sander_energy` in `science/tokyo_eye/v8/biophysics.py`
- [x] `compute_bond_wrapping_count` / double-cone via PyTorch vector ops
- [x] Wire `_detect_backbone_hbonds` through energy gate + cone wrap
- [x] Unit tests: energy admit/reject; on-axis vs equatorial carbon

### Milestone 2 — Automated local graph cache

- [x] Serialize under `pdb_cache/v8_graph_cache/{pdb}_{chain}_{hash}.pt`
- [x] Hash includes `v8_biophys_s8`, DSSP cutoff, cone angle, wrap radius, τ
- [x] CLI `--no-graph-cache` / Makefile `NO_GRAPH_CACHE=1` forces rebuild

### Milestone 3 — Mode C multi-structure production

- [x] `manifests/v8_kras_nucleotide_basin_v1.json` (4LPK, 5US4, 6GOD, 6GOF)
- [x] Epoch-0 4OBE frac gate + wrap histogram median-then-descend retune
- [x] Train round-robin ≥1 full pass; report `dehydron_frac` + `val_dehydron_auprc`; no NaN

## Verification

```bash
PYTHONPATH=. python -m pytest tests/v8/test_biophys_sprint8.py tests/v8/test_r0_r5_graph.py -q
PDB=4OBE EPOCHS=1 NO_MLFLOW=1 RUN_NAME=tokyo_eye_v8_s8_4obe_gate make train-v8-experiment
MANIFEST=manifests/v8_kras_nucleotide_basin_v1.json EPOCHS=4 NO_MLFLOW=1 \
  RUN_NAME=tokyo_eye_v8_mode_c_kras make train-v8-experiment
```

## Acceptance

| Gate | Pass |
|------|------|
| 4OBE dehydron_frac | ≪ 0.90 (target 0.15–0.55) or median-retuned |
| Energy / cone unit tests | green |
| Cache | second load hits cache |
| Mode C | 4 structures, no NaN |
