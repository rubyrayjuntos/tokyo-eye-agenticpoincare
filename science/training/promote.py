"""Promote a trained checkpoint into onboard_contract.yaml as candidate/production."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml

from science.contracts.model_registry import compute_checkpoint_sha256
from science.training.config import PromotionConfig
from science.training.mlflow_run import DEFAULT_TRACKING_URI, resolve_run_checkpoint_path, run_summary

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONTRACT_PATH = _REPO_ROOT / "science" / "contracts" / "onboard_contract.yaml"
_ALLOW_LEGACY_PROMOTE_ENV = "TOKYOEYE_ALLOW_LEGACY_CONTRACT_PROMOTE"


def _contract_path(override: Path | None = None) -> Path:
    return override or _DEFAULT_CONTRACT_PATH


def _load_contract(contract_path: Path | None = None) -> dict:
    path = _contract_path(contract_path)
    with path.open() as handle:
        return yaml.safe_load(handle)


def _write_contract(contract: dict, contract_path: Path | None = None) -> None:
    path = _contract_path(contract_path)
    with path.open("w") as handle:
        yaml.dump(contract, handle, default_flow_style=False, sort_keys=False, width=120)


def _dest_subdir_for_model(model_id: str) -> str:
    mid = model_id.lower()
    if "tokyo_eye" in mid or mid.endswith("_v7") or mid == "v7":
        return "v7"
    if "v65" in mid or "v6.5" in mid or mid.endswith("_v65"):
        return "v65"
    if "v5" in mid and "v6" not in mid:
        return "v5"
    return "v6"


def promote_checkpoint(
    config: PromotionConfig,
    *,
    contract_path: Path | None = None,
) -> dict[str, str]:
    """Copy checkpoint into repo checkpoints tree and register in contract."""
    if config.status == "production" and not _legacy_contract_promote_allowed():
        raise RuntimeError(
            "Legacy promote_checkpoint cannot mutate active production. "
            "Use science.tokyo_eye.governance promote/set-alias for TokyoEye, "
            f"or set {_ALLOW_LEGACY_PROMOTE_ENV}=true for archaeology-only work."
        )
    src = config.checkpoint_path.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    subdir = _dest_subdir_for_model(config.model_id)
    dest_dir = _REPO_ROOT / "checkpoints" / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"{config.checkpoint_id}.pt"
    dest = dest_dir / dest_name
    if src != dest:
        shutil.copy2(src, dest)

    logical_path = f"checkpoints/{subdir}/{dest.name}"
    sha256 = compute_checkpoint_sha256(logical_path) or compute_checkpoint_sha256(str(dest))
    if sha256 is None:
        raise RuntimeError(f"Could not compute SHA-256 for {dest}")

    contract = _load_contract(contract_path)
    gnn = contract.setdefault("gnn_models", {})
    checkpoints = gnn.setdefault("checkpoints", {})

    checkpoints[config.checkpoint_id] = {
        "model_id": config.model_id,
        "path": logical_path,
        "status": config.status,
    }

    if config.status == "production":
        gnn["production_model_id"] = config.model_id
        models = gnn.setdefault("models", {})
        if config.model_id in models:
            models[config.model_id]["production_checkpoint_id"] = config.checkpoint_id
            models[config.model_id]["status"] = "production"
        for ckpt_id, ckpt in list(checkpoints.items()):
            if ckpt_id == config.checkpoint_id:
                ckpt["status"] = "production"
            elif ckpt.get("status") == "production":
                ckpt["status"] = "legacy"

    _write_contract(contract, contract_path)

    result = {
        "checkpoint_id": config.checkpoint_id,
        "logical_path": logical_path,
        "resolved_path": str(dest),
        "sha256": sha256,
        "status": config.status,
    }
    if config.run_id:
        result["mlflow_run_id"] = config.run_id
    return result


def _legacy_contract_promote_allowed() -> bool:
    raw = os.environ.get(_ALLOW_LEGACY_PROMOTE_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Promote v6 checkpoint into onboard contract")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint-path", help="Path to .pt checkpoint file")
    src.add_argument("--mlflow-run-id", help="MLflow run id (uses checkpoint_path tag or artifact)")
    parser.add_argument("--checkpoint-id", default="tokyo_eye_v7_candidate")
    parser.add_argument("--model-id", default="tokyo_eye_v7")
    parser.add_argument("--status", choices=["candidate", "production"], default="candidate")
    parser.add_argument("--run-id", default=None, help="Optional MLflow run id for audit (with --checkpoint-path)")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI when using --mlflow-run-id",
    )
    args = parser.parse_args()

    run_id = args.run_id
    if args.mlflow_run_id:
        run_id = args.mlflow_run_id
        summary = run_summary(args.mlflow_run_id, tracking_uri=args.mlflow_tracking_uri)
        logger.info("MLflow run: %s", summary)
        checkpoint_path = resolve_run_checkpoint_path(
            args.mlflow_run_id,
            tracking_uri=args.mlflow_tracking_uri,
        )
    else:
        checkpoint_path = Path(args.checkpoint_path)

    config = PromotionConfig(
        checkpoint_path=checkpoint_path,
        checkpoint_id=args.checkpoint_id,
        model_id=args.model_id,
        status=args.status,
        run_id=run_id,
    )
    result = promote_checkpoint(config)
    logger.info("Promoted checkpoint: %s", result)


if __name__ == "__main__":
    main()
