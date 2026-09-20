# Plan — tokyo_eye_equ_theme_biology

## Files

- `docs/superpowers/specs/2026-09-15-tokyoeye-equ-theme-biology-design.md`
- `experiments/training/v8/equ_theme_biology.py` — biology gate helpers
- `experiments/training/v8/run_tokyo_eye_equ_theme_biology.py` — sealed runner
- `data/gates/tokyo_eye_equ_theme_biology.json` + `_pins.json`

## Tasks

1. Pins: copy lift_radius panels; add `spine_init` → lift_radius best; `wrap_max=1`.
2. Helpers: theme AUPRC rows; `evaluate_theme_biology`; stamp builder (geometry HOLD + biology NEW).
3. Runner: fork lift_radius; load spine_init; enrich probe with AUPRC; log per-theme metrics; seal.
4. Execute in `tokyoeye_science`; MLflow run name `tokyo_eye_equ_theme_biology`.
5. Report QUALIFIED/FAILED honestly.

## PDB themes (probe)

| Theme | PDB | Chain |
|-------|-----|-------|
| ig_like | 1HNG | A |
| lysozyme_like | 1ALC | A |
| ubiquitin_grasp | 1A5R | A |
| tim_barrel | 1NAL | 1 |
| globin | 2HHB | B |
| ploop_ntpase | 1GKY | A |
