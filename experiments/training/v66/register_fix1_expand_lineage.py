"""Register Fix-1 expand MLflow lineage root (sealed SSOT).

Creates experiment ``tokyo-eyes-v66-fix1-expand`` and a lineage-root run
tagged with ``checkpoint_path`` so resume children resolve ``parent_run_id``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiments.training.v66.healthy_fix1 import (
    FIX1_EXPAND_LINEAGE_STAMP,
    FIX1_EXPAND_MLFLOW_EXPERIMENT,
    HEALTHY_FIX1_CKPT,
    HEALTHY_FIX1_CKPT_APP,
    HEALTHY_FIX1_REMATCH_RUN_ID,
    HEALTHY_FIX1_RUN_DIR,
    HEALTHY_FIX1_RUN_ID,
)
from science.training.mlflow_governance import corpus_manifest_hash


def _import_mlflow():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "false")
    import mlflow

    return mlflow


def register_lineage_root(
    *,
    tracking_uri: str,
    experiment: str = FIX1_EXPAND_MLFLOW_EXPERIMENT,
    stamp_path: Path = FIX1_EXPAND_LINEAGE_STAMP,
) -> dict:
    if not HEALTHY_FIX1_CKPT.is_file():
        raise FileNotFoundError(f"missing sealed ckpt: {HEALTHY_FIX1_CKPT}")

    mlf = _import_mlflow()
    mlf.set_tracking_uri(tracking_uri)
    mlf.set_experiment(experiment)

    manifest = Path("manifests/v6_corpus_stage_a_small_v1.json")
    if not manifest.is_file():
        manifest = Path("/app/manifests/v6_corpus_stage_a_small_v1.json")

    metrics_path = HEALTHY_FIX1_RUN_DIR / "metrics.json"
    rematch_metrics = (
        Path("checkpoints/v66/runs")
        / HEALTHY_FIX1_REMATCH_RUN_ID
        / "metrics.json"
    )
    ge30 = None
    for candidate in (metrics_path, rematch_metrics):
        if candidate.is_file():
            rows = json.loads(candidate.read_text())
            if rows:
                last = rows[-1]
                h = last.get("health") or {}
                ge30 = {
                    "source": str(candidate),
                    "global_epoch": last.get("global_epoch"),
                    "disc_r_mean": h.get("disc_r_mean"),
                    "disc_sigma2_sigma1_mean": h.get("disc_sigma2_sigma1_mean"),
                    "probe_r_proj_depth": h.get("probe_r_proj_depth"),
                    "probe_r_depth_tau": h.get("probe_r_depth_tau"),
                }
                break

    ckpt_app = str(HEALTHY_FIX1_CKPT_APP)
    ckpt_rel = f"checkpoints/v66/runs/{HEALTHY_FIX1_RUN_ID}/v66_healthy_sealed.pt"

    with mlf.start_run(run_name=f"fix1_s4_lineage_root_{HEALTHY_FIX1_RUN_ID}") as run:
        run_id = run.info.run_id
        mlf.set_tag("lineage_root", "true")
        mlf.set_tag("parent_run_id", "null")
        mlf.set_tag("warm_start", "none")
        mlf.set_tag("gnn_lineage", "v6.6")
        mlf.set_tag("fix1_ssot_run_id", HEALTHY_FIX1_RUN_ID)
        mlf.set_tag("fix1_rematch_run_id", HEALTHY_FIX1_REMATCH_RUN_ID)
        # Primary lookup path used inside science containers.
        mlf.set_tag("checkpoint_path", ckpt_app)
        mlf.set_tag("checkpoint_path_rel", ckpt_rel)
        mlf.set_tag("promoted_trunk", "false")
        mlf.set_tag("phase", "P0_lineage_root")

        mlf.log_params(
            {
                "lineage_root": "true",
                "parent_run_id": "null",
                "warm_start": "none",
                "gnn_lineage": "v6.6",
                "model_version": "GOSPConeMapper-v6.6",
                "corpus_manifest_hash": corpus_manifest_hash(manifest),
                "corpus_size": "12",
                "sealed_checkpoint": ckpt_app,
                "spec_version": "fix1-s4-expand-v1",
            }
        )
        if ge30:
            for k, v in ge30.items():
                if isinstance(v, (int, float)) and v is not None:
                    mlf.log_metric(f"root_{k}", float(v))

        # Log lightweight artifacts (not the full 3.6MB ckpt by default).
        note = HEALTHY_FIX1_RUN_DIR / "README.md"
        if note.is_file():
            mlf.log_artifact(str(note), artifact_path="ssot")
        if metrics_path.is_file():
            mlf.log_artifact(str(metrics_path), artifact_path="ssot")
        monitor = (
            Path("checkpoints/v66/runs")
            / HEALTHY_FIX1_REMATCH_RUN_ID
            / "MONITOR_EP001.md"
        )
        if monitor.is_file():
            mlf.log_artifact(str(monitor), artifact_path="ssot")

        summary = {
            "experiment": experiment,
            "run_id": run_id,
            "lineage_root": True,
            "parent_run_id": None,
            "checkpoint_path": ckpt_app,
            "checkpoint_path_rel": ckpt_rel,
            "sealed_host_path": str(HEALTHY_FIX1_CKPT.resolve()),
            "ssot_run_id": HEALTHY_FIX1_RUN_ID,
            "rematch_run_id": HEALTHY_FIX1_REMATCH_RUN_ID,
            "ge30_summary": ge30,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "tracking_uri": tracking_uri,
        }
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(summary, indent=2) + "\n")
        mlf.log_artifact(str(stamp_path), artifact_path="ssot")
        return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    p.add_argument("--experiment", default=FIX1_EXPAND_MLFLOW_EXPERIMENT)
    p.add_argument("--stamp", type=Path, default=FIX1_EXPAND_LINEAGE_STAMP)
    args = p.parse_args()
    summary = register_lineage_root(
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        stamp_path=args.stamp,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
