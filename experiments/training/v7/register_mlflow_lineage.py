"""Register Tokyo Eye v7 MLflow experiment + lineage-root run.

Experiment: ``tokyo-eyes-v7`` (isolated from tokyo-eyes-v66 / fix1-expand).
Registered model name: ``TokyoEye-v7`` (from lineage registry).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiments.training.v7 import (
    LINEAGE_ID,
    PRODUCTION_MODULE,
    V7_CHECKPOINT_ROOT,
    V7_DIAGNOSTICS_ROOT,
    V7_HYP_SPACE_NAME,
    V7_MLFLOW_EXPERIMENT,
    V7_MLFLOW_LINEAGE_STAMP,
)

SPEC = "docs/specs/tokyo-eye-v7/README.md"


def _import_mlflow():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "false")
    import mlflow

    return mlflow


def register_v7_lineage_root(
    *,
    tracking_uri: str,
    experiment: str = V7_MLFLOW_EXPERIMENT,
    stamp_path: Path = V7_MLFLOW_LINEAGE_STAMP,
) -> dict:
    mlf = _import_mlflow()
    mlf.set_tracking_uri(tracking_uri)
    mlf.set_experiment(experiment)

    smoke = V7_DIAGNOSTICS_ROOT / "tokyo_eye_v7_forward_smoke.json"
    smoke_payload = None
    if smoke.is_file():
        smoke_payload = json.loads(smoke.read_text())

    with mlf.start_run(run_name="tokyo_eye_v7_lineage_root") as run:
        run_id = run.info.run_id
        mlf.set_tag("lineage_root", "true")
        mlf.set_tag("parent_run_id", "null")
        mlf.set_tag("warm_start", "none")
        mlf.set_tag("gnn_lineage", LINEAGE_ID)
        mlf.set_tag("production_module", PRODUCTION_MODULE)
        mlf.set_tag("hyp_mp_primary", "true")
        mlf.set_tag("se3_label", "SE3_aux_not_S4")
        mlf.set_tag("dtie_policy", "untouched_structural_biology")
        mlf.set_tag("july19_mp_euclidean_lock", "superseded_for_v7")
        mlf.set_tag("v6x_policy", "frozen_compare_only")
        mlf.set_tag("chem_mvp", "parked")
        mlf.set_tag("checkpoint_root", str(V7_CHECKPOINT_ROOT))
        mlf.set_tag("space_name", V7_HYP_SPACE_NAME)
        mlf.set_tag("phase", "P0_lineage_root")
        mlf.set_tag("spec", SPEC)

        from science.training.model_registry_mlflow import ensure_registered_model

        ensure_registered_model("TokyoEye-v7", tracking_uri=tracking_uri)

        mlf.log_params(
            {
                "lineage_root": "true",
                "parent_run_id": "null",
                "warm_start": "none",
                "gnn_lineage": LINEAGE_ID,
                "model_version": "TokyoEye-v7",
                "production_module": PRODUCTION_MODULE,
                "hyp_mp_primary": "true",
                "se3_aux_default": "false",
                "space_name": V7_HYP_SPACE_NAME,
                "curvature_mode": "learned_log_c",
                "spec_version": "tokyo-eye-v7-open-2026-07-21",
                "checkpoint_root": str(V7_CHECKPOINT_ROOT),
            }
        )
        if smoke_payload and smoke_payload.get("pass"):
            mlf.log_metric(
                "forward_smoke_curvature",
                float(smoke_payload.get("curvature_learned", float("nan"))),
            )
            mlf.log_metric(
                "forward_smoke_cone_depth_mean",
                float(smoke_payload.get("cone_depth_mean", float("nan"))),
            )
            mlf.log_artifact(str(smoke), artifact_path="diagnostics")

        readme = Path(SPEC)
        if readme.is_file():
            mlf.log_artifact(str(readme), artifact_path="ssot")
        design = Path("docs/specs/tokyo-eye-v7/design.md")
        if design.is_file():
            mlf.log_artifact(str(design), artifact_path="ssot")

    stamp = {
        "schema_version": 1,
        "gate": "tokyo_eye_v7_mlflow_lineage_root",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "gnn_lineage": LINEAGE_ID,
        "production_module": PRODUCTION_MODULE,
        "registered_model_name": "TokyoEye-v7",
        "hyp_mp_primary": True,
        "space_name": V7_HYP_SPACE_NAME,
        "checkpoint_root": str(V7_CHECKPOINT_ROOT),
        "spec": SPEC,
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    return stamp


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    p.add_argument("--experiment", default=V7_MLFLOW_EXPERIMENT)
    p.add_argument("--stamp", type=Path, default=V7_MLFLOW_LINEAGE_STAMP)
    args = p.parse_args(argv)
    stamp = register_v7_lineage_root(
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        stamp_path=args.stamp,
    )
    print(json.dumps(stamp, indent=2))
    print(f"wrote {args.stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
