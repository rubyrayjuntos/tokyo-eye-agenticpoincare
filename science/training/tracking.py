"""MLflow tracking wrapper for v6 GNN training runs."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from science.training.config import PhaseConfig, TrainingConfig
from science.training.mlflow_governance import build_governance_params
from science.training.mlflow_run import find_run_id_by_checkpoint_path

logger = logging.getLogger(__name__)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _resolve_tracking_uri(uri: str) -> str:
    if not uri.startswith("file:"):
        return uri
    root = Path(uri.replace("file:", "", 1)).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root.as_uri()


def _ensure_experiment(mlf: Any, tracking_uri: str, experiment_name: str) -> str:
    """Create experiment when missing. Returns active name.

    HTTP/Postgres tracking servers use the server default artifact root
    (``MLFLOW_DEFAULT_ARTIFACT_ROOT``). File-store URIs still get an explicit
    sibling ``mlflow-artifacts/<experiment>`` location so artifacts are not
    nested under the tracking store root.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    is_file_store = tracking_uri.startswith("file:")

    if not is_file_store:
        if exp is None:
            client.create_experiment(experiment_name)
        return experiment_name

    store_root = Path(tracking_uri.replace("file:", "", 1))
    if not store_root.is_absolute():
        store_root = (Path.cwd() / store_root).resolve()
    # Artifacts must NOT live directly under the tracking store root — MLflow treats
    # every top-level subdirectory as an experiment and 500s the UI without meta.yaml.
    artifact_root = store_root.parent / "mlflow-artifacts"
    artifact_location = (artifact_root / experiment_name.replace(" ", "_")).as_uri()
    if exp is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
        return experiment_name
    if exp.artifact_location.startswith("file:///app/") and not Path("/app").exists():
        local_name = f"{experiment_name}-local"
        local_exp = client.get_experiment_by_name(local_name)
        if local_exp is None:
            client.create_experiment(local_name, artifact_location=artifact_location)
        return local_name
    return experiment_name


def _import_mlflow() -> Any:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "false")
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for training tracking. "
            "Install with: uv pip install -e '.[science,training]'"
        ) from exc
    return mlflow


class TrainingTracker:
    """Thin MLflow facade with repo-specific tags and artifact helpers."""

    def __init__(self, config: TrainingConfig, run_name: str | None = None) -> None:
        self.config = config
        self.run_name = run_name or f"v6-{config.model_version}"
        self._mlflow: Any | None = None
        self._active = False
        self._active_experiment: str | None = None
        self.run_id: str | None = None

    def _mlf(self) -> Any:
        if self._mlflow is None:
            self._mlflow = _import_mlflow()
            uri = _resolve_tracking_uri(self.config.mlflow_tracking_uri)
            self._mlflow.set_tracking_uri(uri)
            active_experiment = _ensure_experiment(
                self._mlflow, uri, self.config.mlflow_experiment
            )
            self._mlflow.set_experiment(active_experiment)
            self._active_experiment = active_experiment
        return self._mlflow

    @contextmanager
    def start_run(
        self,
        *,
        proteins: list[dict[str, Any]] | None = None,
        phases: list[PhaseConfig] | None = None,
    ) -> Iterator["TrainingTracker"]:
        mlf = self._mlf()
        with mlf.start_run(run_name=self.run_name) as run:
            self._active = True
            self.run_id = run.info.run_id
            parent_id: str | None = None
            if self.config.master_cold_lineage or self.config.slim_moe_structural_ssot:
                mlf.set_tag("parent_run_id", "null")
                mlf.set_tag("lineage_root", "true")
            elif self.config.resume is not None:
                parent_id = find_run_id_by_checkpoint_path(
                    str(self.config.resume),
                    tracking_uri=self.config.mlflow_tracking_uri,
                    experiment_name=self._active_experiment or self.config.mlflow_experiment,
                )
            config_params = self.config.to_mlflow_params()
            mlf.log_params(config_params)
            gov_params = build_governance_params(
                self.config,
                proteins=proteins,
                parent_run_id=parent_id,
                phases=phases,
            )
            # Config params win; governance may repeat keys (e.g. topology_only_gate).
            mlf.log_params({k: v for k, v in gov_params.items() if k not in config_params})
            sha = _git_sha()
            if sha:
                mlf.set_tag("git_sha", sha)
            mlf.set_tag("model_version", self.config.model_version)
            preset = self.config.phase_preset_name()
            if preset:
                mlf.set_tag("phase_preset", preset)
            if self.config.resume is not None:
                mlf.set_tag("resume_from", str(self.config.resume))
            from science.training.edge_telemetry import (
                MLFLOW_TAG_NOT_GATE,
                MLFLOW_TAG_TELEMETRY,
            )

            mlf.set_tag(MLFLOW_TAG_TELEMETRY, "true")
            mlf.set_tag(MLFLOW_TAG_NOT_GATE, "true")
            mlf.set_tag("gnn_lineage", str(self.config.gnn_lineage))
            if str(self.config.gnn_lineage) == "v7":
                from experiments.training.v7 import (
                    PRODUCTION_MODULE,
                    V7_HYP_SPACE_NAME,
                )

                mlf.set_tag("production_module", PRODUCTION_MODULE)
                mlf.set_tag(
                    "hyp_mp_primary",
                    str(bool(getattr(self.config, "hyp_mp_primary", True))).lower(),
                )
                mlf.set_tag("se3_label", "SE3_aux_not_S4")
                mlf.set_tag("space_name", V7_HYP_SPACE_NAME)
            if parent_id and not (
                self.config.master_cold_lineage or self.config.slim_moe_structural_ssot
            ):
                mlf.set_tag("parent_run_id", parent_id)
            try:
                yield self
            finally:
                self._active = False

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        self._mlf().log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            return
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        self._mlf().log_metrics(clean, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if not self._active:
            return
        try:
            if path.is_file():
                self._mlf().log_artifact(str(path), artifact_path=artifact_path)
            elif path.is_dir():
                self._mlf().log_artifacts(str(path), artifact_path=artifact_path)
        except OSError as exc:
            logger.warning("MLflow artifact upload failed for %s: %s", path, exc)

    def set_checkpoint_tag(self, checkpoint_path: Path, sha256: str | None) -> None:
        if not self._active:
            return
        mlf = self._mlf()
        mlf.set_tag("checkpoint_path", str(checkpoint_path))
        if sha256:
            mlf.set_tag("checkpoint_sha256", sha256)

    def log_best_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        score: float,
        global_epoch: int,
        sha256: str | None,
    ) -> None:
        """Tag + upload eligible v6_best.pt (idempotent overwrite on each new best)."""
        if not self._active:
            return
        mlf = self._mlf()
        self.set_checkpoint_tag(checkpoint_path, sha256)
        mlf.set_tag("best_score", f"{score:.6f}")
        mlf.set_tag("best_epoch", str(global_epoch))
        if checkpoint_path.is_file():
            self.log_artifact(checkpoint_path, artifact_path="checkpoints")

    def log_eval_json(self, eval_report: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(eval_report, indent=2, default=str))
        self.log_artifact(path, artifact_path="eval")

    def log_focus_summary(self, summary: dict[str, Any], path: Path) -> None:
        """Persist training focus assessment (what needs work vs saturated)."""
        if not self._active:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str))
        self.log_artifact(path, artifact_path="focus")
        mlf = self._mlf()
        mlf.set_tag("focus_primary", summary.get("primary_focus_str", "none"))
        mlf.set_tag("focus_recommendation", (summary.get("recommendation") or "")[:250])
        counts = summary.get("counts", {})
        mlf.set_tag("focus_critical_count", str(counts.get("critical", 0)))
        mlf.set_tag("focus_matters_needs_work_count", str(counts.get("matters_needs_work", 0)))
