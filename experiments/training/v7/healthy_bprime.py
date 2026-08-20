"""Healthy Tokyo Eye v7 B′ trunk SSOT (disc-health Pass; uncertainty deferred).

Prefer ``HEALTHY_V7_CKPT`` (sealed fat checkpoint) over bare ``phase_12.pt`` /
epoch snapshots for restore and uncertainty-head continues.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

HEALTHY_V7_RUN_ID = "tokyo_eye_v7_bprime_core_floor_continue_v1"
HEALTHY_V7_RUN_DIR = REPO / "checkpoints" / "v7" / "runs" / HEALTHY_V7_RUN_ID
HEALTHY_V7_EPOCH = 41
HEALTHY_V7_EPOCH_CKPT = HEALTHY_V7_RUN_DIR / "epochs" / f"epoch_{HEALTHY_V7_EPOCH:03d}.pt"
HEALTHY_V7_CKPT = HEALTHY_V7_RUN_DIR / "v7_healthy_sealed.pt"
HEALTHY_V7_CKPT_APP = (
    Path("/app/checkpoints/v7/runs") / HEALTHY_V7_RUN_ID / "v7_healthy_sealed.pt"
)
HEALTHY_V7_SEAL_GATE = REPO / "data" / "gates" / "tokyo_eye_v7_bprime_healthy_sealed.json"
HEALTHY_V7_HEALTH_CLOSEOUT = (
    REPO / "data" / "gates" / "tokyo_eye_v7_bprime_core_radial_floor_closeout.json"
)
