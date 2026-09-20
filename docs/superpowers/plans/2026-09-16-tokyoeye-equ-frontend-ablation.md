# Plan — tokyo_eye_equ_frontend_ablation

1. Stage 0 script: `scripts/frontend_dehydron_separability.py` → JSON under `data/local_objects/frontend_ablation/`.
2. Pins + stamp for card; MLflow optional for Stage 0 (file artifact OK).
3. Stage 1 runner only after Stage 0 reported.
4. Parallel (small): joint-gate ckpt selection helper for future trains — not required to finish Stage 0.
