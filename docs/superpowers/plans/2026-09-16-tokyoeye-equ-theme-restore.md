# Plan — tokyo_eye_equ_theme_restore

## Goal
Theme dehydron AUPRC biology Pass on the geoopt_restore QUALIFIED Equiformer-pool + geoopt spine.

## Files
- `docs/superpowers/specs/2026-09-16-tokyoeye-equ-theme-restore-design.md`
- `experiments/training/v8/equ_theme_restore.py`
- `experiments/training/v8/run_tokyo_eye_equ_theme_restore.py`
- `data/gates/tokyo_eye_equ_theme_restore.json` + `_pins.json`

## Tasks
1. Pins: Stage-A panels; `spine_init` → geoopt_restore best; wrap_max=1; forbid_se3_lite.
2. Helpers: reuse theme AUPRC evaluators from `equ_theme_biology`; stamp adds restore_frontend_gate.
3. Runner: `build_equiformer_pool_system` + `assert_geoopt_restore_preflight` + warm spine_init; probe AUPRC; seal.
4. Execute in `tokyoeye_science`; MLflow `tokyo_eye_equ_theme_restore`.
5. Honest QUALIFIED/FAILED.

## Probe themes
| Theme | PDB | Chain |
|-------|-----|-------|
| ig_like | 1HNG | A |
| lysozyme_like | 1ALC | A |
| ubiquitin_grasp | 1A5R | A |
| tim_barrel | 1NAL | 1 |
| globin | 2HHB | B |
| ploop_ntpase | 1GKY | A |
