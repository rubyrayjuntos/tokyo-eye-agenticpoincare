"""TokyoEye-v8 trunk SSOTs (spine + affinity).

v7 is dead archaeology — never resume ``HEALTHY_V7_CKPT`` for new work.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Spine / MoE bank (Mode C rematch)
HEALTHY_V8_SPINE_RUN_ID = "tokyo_eye_v8_mode_c_moe_rebalance_s9"
HEALTHY_V8_SPINE_CKPT = (
    REPO / "checkpoints" / "v8" / "runs" / HEALTHY_V8_SPINE_RUN_ID / "v8_best.pt"
)
HEALTHY_V8_SPINE_POINTER = REPO / "checkpoints" / "v8" / "HEALTHY_V8_SPINE_CKPT.pt"

# Affinity seal (CASF Core ≥ 0.40) — compare-only for ligand ranking claims
from experiments.training.v8.healthy_affinity import (  # noqa: E402
    HEALTHY_V8_AFFINITY_CKPT,
    HEALTHY_V8_AFFINITY_CLOSEOUT,
    HEALTHY_V8_AFFINITY_POINTER,
    HEALTHY_V8_AFFINITY_RUN_DIR,
    HEALTHY_V8_AFFINITY_RUN_ID,
)

HEALTHY_V8_CKPT = HEALTHY_V8_SPINE_CKPT  # default Θ for biology / forward smoke
HEALTHY_V8_TRUNK_GATE = REPO / "data" / "gates" / "tokyo_eye_v8_trunk_ssot.json"
HEALTHY_V8_BIOLOGY_ROADMAP = (
    REPO / "data" / "gates" / "tokyo_eye_v8_biology_roadmap.json"
)

__all__ = [
    "HEALTHY_V8_AFFINITY_CKPT",
    "HEALTHY_V8_AFFINITY_CLOSEOUT",
    "HEALTHY_V8_AFFINITY_POINTER",
    "HEALTHY_V8_AFFINITY_RUN_DIR",
    "HEALTHY_V8_AFFINITY_RUN_ID",
    "HEALTHY_V8_BIOLOGY_ROADMAP",
    "HEALTHY_V8_CKPT",
    "HEALTHY_V8_SPINE_CKPT",
    "HEALTHY_V8_SPINE_POINTER",
    "HEALTHY_V8_SPINE_RUN_ID",
    "HEALTHY_V8_TRUNK_GATE",
]
