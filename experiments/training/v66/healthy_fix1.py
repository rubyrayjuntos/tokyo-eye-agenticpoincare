"""Healthy Fix-1 + S4 trunk SSOT for v6.6 (post chem-MVP park).

Prefer ``HEALTHY_FIX1_CKPT`` (sealed fat checkpoint) over bare ``phase_12.pt``,
which omits ``architecture`` / ``training_config`` and mis-reconstructs the
forward shell under ``load_model_from_checkpoint``.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

HEALTHY_FIX1_RUN_ID = "fix1_s4_stack_initseed_controlled_3d_seed2_v1"
HEALTHY_FIX1_RUN_DIR = REPO / "checkpoints" / "v66" / "runs" / HEALTHY_FIX1_RUN_ID
HEALTHY_FIX1_CKPT = HEALTHY_FIX1_RUN_DIR / "v66_healthy_sealed.pt"

# Historical weight twin — do not use as default load path.
HEALTHY_FIX1_WEIGHTS_ONLY = HEALTHY_FIX1_RUN_DIR / "phase_12.pt"
# Early saver — origin-collapsed; never promote.
HEALTHY_FIX1_BEST_DISC_COLLAPSED = HEALTHY_FIX1_RUN_DIR / "v66_best_disc.pt"

# Post-restore expansion MLflow lineage (not legacy tokyo-eyes-v66).
FIX1_EXPAND_MLFLOW_EXPERIMENT = "tokyo-eyes-v66-fix1-expand"
FIX1_EXPAND_LINEAGE_STAMP = (
    REPO / "data" / "gates" / "fix1_expand_mlflow_lineage_root.json"
)
HEALTHY_FIX1_REMATCH_RUN_ID = "fix1_s4_stack_initseed_controlled_3d_seed2_restore_v1"
HEALTHY_FIX1_CKPT_APP = (
    Path("/app/checkpoints/v66/runs")
    / HEALTHY_FIX1_RUN_ID
    / "v66_healthy_sealed.pt"
)

# Sparsity phase champion (banked confirm continue; save-band governor).
# Specialized routing trunk for next biology probes (e.g. KRAS G12D hub migration).
# Does NOT replace HEALTHY_FIX1_CKPT for restore / sealed-baseline compares.
FIX1_SPARSITY_CHAMPION_RUN_ID = "fix1_s4_sparsity_confirm_continue_v2"
FIX1_SPARSITY_CHAMPION_RUN_DIR = (
    REPO / "checkpoints" / "v66" / "runs" / FIX1_SPARSITY_CHAMPION_RUN_ID
)
FIX1_SPARSITY_CHAMPION_CKPT = (
    FIX1_SPARSITY_CHAMPION_RUN_DIR / "v66_sparsity_champion.pt"
)
FIX1_SPARSITY_CHAMPION_EPOCH = 48
FIX1_SPARSITY_CHAMPION_GATE = (
    FIX1_SPARSITY_CHAMPION_RUN_DIR / "sparsity_gate.json"
)
FIX1_SPARSITY_CHAMPION_CKPT_APP = (
    Path("/app/checkpoints/v66/runs")
    / FIX1_SPARSITY_CHAMPION_RUN_ID
    / "v66_sparsity_champion.pt"
)
