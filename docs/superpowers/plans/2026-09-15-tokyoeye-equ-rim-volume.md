# Tokyo Eye EQU — Rim Volume Implementation Plan

**Gate ID:** `tokyo_eye_equ_rim_volume`  
**Status:** APPROVED_LOCKED  
**Design:** `docs/superpowers/specs/2026-09-15-tokyoeye-equ-rim-volume-design.md`  
**Date:** 2026-09-15

## Steps

1. **Land stamps** — `data/gates/tokyo_eye_equ_rim_volume.json` APPROVED_LOCKED + pins JSON (reuse correct_start PDB panels / MPtrj sha).
2. **Rewrite `RadialAngularProjector`** — invariant scalar lift only; backup prior Option-B file.
3. **Add `hyperbolic_volume_loss`** — barrier + spread hinge; wire into `run_epoch(..., volume_coeff=...)`.
4. **Helpers + runner** — `equ_rim_volume.py` + `run_tokyo_eye_equ_rim_volume.py` (fork correct_start): pure-hyp strict, L_vol on, sealed hygiene using mean ‖z‖ and std(r_H), lift+spine Δ_equiv, MLflow run name `tokyo_eye_equ_rim_volume`.
5. **Preflight** — static pure_hyp + smoke forward in `tokyoeye_science`.
6. **Train** — 20 epochs science CUDA; PDB `/tmp/dtie_pdb_cache`; stamp writable.
7. **Seal** — QUALIFIED or FAILED from binary gates; log artifacts to MLflow.

## Locked knobs

| Knob | Value |
| --- | --- |
| epochs | 20 |
| τ | quadratic 0.60 → 0.95 |
| L_vol σ_target / λ_spread / coeff | 0.20 / 0.5 / 1.0 |
| lr_hyp / wd | 3e-4 / 1e-4 |
| freeze | entire frontend |
| forbidden init | champion / correct_start / affinity / C1 |

## Forbidden

- Loading Fail θ from correct_start
- Tangent Linear post-lift
- Directional `â` Poincaré lift
- Changing sealed gate thresholds mid-run
