# Plan — tokyo_eye_equ_lift_radius

1. Stamp/pins APPROVED_LOCKED.
2. Rewrite `RadialAngularProjector` → MLP_α + MLP_u as §2.1; backup prior.
3. Keep `final_only` + `L_rad` knobs from shell_unpack.
4. Runner `run_tokyo_eye_equ_lift_radius.py` (fork shell_unpack); forbid prior Fail θ.
5. Preflight SO(3) on projector + pure_hyp; train; seal.
