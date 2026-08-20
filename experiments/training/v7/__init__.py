"""Tokyo Eye v7 training entrypoints (isolated from experiments.training.v66)."""

from __future__ import annotations

from pathlib import Path

V7_CHECKPOINT_ROOT = Path("checkpoints/v7/runs")
V7_DIAGNOSTICS_ROOT = Path("checkpoints/v7/diagnostics")
PRODUCTION_MODULE = "science.tokyo_eye.TokyoEye"
LINEAGE_ID = "v7"
V7_MLFLOW_EXPERIMENT = "tokyo-eyes-v7"
V7_HYP_SPACE_NAME = "tokyoeye_v7_hyp128"
V7_MLFLOW_LINEAGE_STAMP = Path("data/gates/tokyo_eye_v7_mlflow_lineage_root.json")
