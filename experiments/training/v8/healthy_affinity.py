"""Sealed TokyoEye-v8 affinity SSOT (CASF Core Pearson ≥ 0.40).

Compare-only / affinity trunk — does **not** replace ``HEALTHY_V7_CKPT`` for
Mol* / onboard production geometry.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

HEALTHY_V8_AFFINITY_RUN_ID = "tokyo_eye_v8_affinity_s1011_finetune_all"
HEALTHY_V8_AFFINITY_RUN_DIR = (
    REPO / "checkpoints" / "v8" / "runs" / HEALTHY_V8_AFFINITY_RUN_ID
)
HEALTHY_V8_AFFINITY_CKPT = HEALTHY_V8_AFFINITY_RUN_DIR / "v8_affinity_best.pt"
HEALTHY_V8_AFFINITY_POINTER = (
    REPO / "checkpoints" / "v8" / "HEALTHY_V8_AFFINITY_CKPT.pt"
)
HEALTHY_V8_AFFINITY_CLOSEOUT = (
    REPO / "data" / "gates" / "tokyo_eye_v8_affinity_s1011_finetune_all_closeout.json"
)
