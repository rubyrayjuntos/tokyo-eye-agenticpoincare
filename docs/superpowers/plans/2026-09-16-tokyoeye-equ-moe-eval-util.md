# Plan — tokyo_eye_equ_moe_eval_util

1. Evidence pack: stripe crosstab + eval_vs_train loads + MLflow audit (done / finishing).
2. Implement soft Switch LB in `science/tokyo_eye/v8/moe.py`; wire coeff in engine + runners.
3. Seal helper: eval-mode load / entropy / n_alive on probe; fail card if below floors.
4. Retro-annotate geoopt_restore stamp: QUALIFIED geometry HOLD but MoE eval collapsed (regression note).
5. Short sealed retrain card after LB lands (separate execution, same gate id or child slug).
6. Framing pass on specs mentioning E0–E3 physical tiers.
