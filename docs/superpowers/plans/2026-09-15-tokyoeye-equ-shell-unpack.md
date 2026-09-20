# Tokyo Eye EQU — Shell Unpack Implementation Plan

**Gate ID:** `tokyo_eye_equ_shell_unpack`  
**Status:** APPROVED_LOCKED  
**Design:** `docs/superpowers/specs/2026-09-15-tokyoeye-equ-shell-unpack-design.md`

## Steps

1. Land stamp + pins (reuse rim PDB panels / MPtrj sha).
2. `TokyoEyesHyperbolicV8.tau_clamp_mode`: `per_stage` (default) | `final_only`.
3. Extend volume loss → `L_rad` with mean-radius target; wire coeffs into `run_epoch`.
4. Helpers + runner forked from rim_volume; set `spine.tau_clamp_mode = "final_only"`; forbid rim/correct_start θ.
5. Preflight pure_hyp + projector SO(3) + smoke forward.
6. Train 20 epochs in `tokyoeye_science`; seal QUALIFIED/FAILED.

## Locked knobs

| Knob | Value |
| --- | --- |
| tau_clamp_mode | final_only |
| μ* / σ* | 0.35 / 0.20 |
| λ_mean / λ_spread / λ_bar | 3.0 / 1.0 / 0.5 |
| volume_coeff | 1.0 |
| epochs / τ | 20 / 0.60→0.95 quadratic |
| lr_hyp / wd | 3e-4 / 1e-4 |
