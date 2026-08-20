"""GNN lifecycle control-plane helpers (status, corpus, promote bridge)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from science.contracts.model_registry import (
    compute_checkpoint_sha256,
    get_checkpoint_catalog,
    get_gnn_model_catalog,
    get_production_checkpoint_path,
    get_production_model,
)
from science.training.gnn_lineage import LINEAGE_REGISTRY, get_lineage
from science.training.model_registry_mlflow import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    DEFAULT_TRACKING_URI,
    get_version_by_alias,
    list_model_versions,
    mlflow_server_health,
    register_checkpoint_version,
    set_alias,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JOBS_PATH = _REPO_ROOT / "data" / "lifecycle" / "jobs.json"
_GATE_STAMP = _REPO_ROOT / "data" / "gates" / "p_feature_01_passed.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jobs() -> dict[str, Any]:
    if not _JOBS_PATH.is_file():
        return {"active": None, "history": []}
    try:
        return json.loads(_JOBS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {"active": None, "history": []}


def _save_jobs(payload: dict[str, Any]) -> None:
    _JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _JOBS_PATH.write_text(json.dumps(payload, indent=2, default=str))


def gate_stamp_status() -> dict[str, Any]:
    if not _GATE_STAMP.is_file():
        return {"p_feature_01_passed": False, "path": str(_GATE_STAMP)}
    try:
        data = json.loads(_GATE_STAMP.read_text())
        return {
            "p_feature_01_passed": True,
            "path": str(_GATE_STAMP),
            "stamp": data,
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {"p_feature_01_passed": False, "path": str(_GATE_STAMP), "error": str(exc)}


def lifecycle_status(*, tracking_uri: str | None = None) -> dict[str, Any]:
    """360° MLOps snapshot for the cockpit."""
    uri = tracking_uri or DEFAULT_TRACKING_URI
    prod_model = None
    prod_path = None
    prod_sha = None
    try:
        prod_model = get_production_model()
        prod_path = get_production_checkpoint_path()
        prod_sha = compute_checkpoint_sha256(prod_path)
    except Exception as exc:
        logger.warning("production model resolve failed: %s", exc)

    lineages: list[dict[str, Any]] = []
    mlflow_health = mlflow_server_health(uri)
    for lid, spec in LINEAGE_REGISTRY.items():
        champion = None
        challenger = None
        if mlflow_health.get("available"):
            try:
                champion = get_version_by_alias(
                    lineage_id=lid, alias=ALIAS_CHAMPION, tracking_uri=uri
                )
                challenger = get_version_by_alias(
                    lineage_id=lid, alias=ALIAS_CHALLENGER, tracking_uri=uri
                )
            except Exception as exc:
                logger.debug("alias lookup failed for %s: %s", lid, exc)
        lineages.append(
            {
                "lineage_id": lid,
                "model_version": spec.model_version,
                "mlflow_experiment": spec.mlflow_experiment,
                "checkpoint_root": str(spec.checkpoint_root),
                "frozen_baseline": spec.frozen_baseline,
                "champion": champion,
                "challenger": challenger,
            }
        )

    jobs = _load_jobs()
    return {
        "generated_at": _now_iso(),
        "mlflow": mlflow_health,
        "production": {
            "model_id": prod_model.model_id if prod_model else None,
            "model_version": prod_model.model_version if prod_model else None,
            "runner_module": prod_model.runner_module if prod_model else None,
            "runner_class": prod_model.runner_class if prod_model else None,
            "checkpoint_path": prod_path,
            "checkpoint_sha256": prod_sha,
            "checkpoint_id": prod_model.production_checkpoint_id if prod_model else None,
        },
        "lineages": lineages,
        "gates": gate_stamp_status(),
        "active_job": jobs.get("active"),
        "contract_models": {
            mid: {
                "model_version": m.model_version,
                "status": m.status,
                "production_checkpoint_id": m.production_checkpoint_id,
                "runner_module": m.runner_module,
                "runner_class": m.runner_class,
            }
            for mid, m in get_gnn_model_catalog().items()
        },
        "contract_checkpoints": {
            cid: {"path": c.path, "status": c.status, "model_id": c.model_id}
            for cid, c in get_checkpoint_catalog().items()
        },
    }


def list_lineages() -> list[dict[str, Any]]:
    return [
        {
            "lineage_id": lid,
            "model_version": spec.model_version,
            "package": spec.package,
            "mlflow_experiment": spec.mlflow_experiment,
            "checkpoint_prefix": spec.checkpoint_prefix,
            "checkpoint_root": str(spec.checkpoint_root),
            "frozen_baseline": spec.frozen_baseline,
        }
        for lid, spec in LINEAGE_REGISTRY.items()
    ]


def register_and_alias(
    *,
    lineage_id: str,
    checkpoint_path: str | Path,
    alias: Literal["champion", "challenger"],
    run_id: str | None = None,
    sync_contract: bool = True,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Register checkpoint, set alias, optionally sync onboard_contract for champion."""
    path = Path(checkpoint_path)
    reg = register_checkpoint_version(
        lineage_id=lineage_id,
        checkpoint_path=path,
        run_id=run_id,
        tracking_uri=tracking_uri,
    )
    aliased = set_alias(
        lineage_id=lineage_id,
        version=reg["version"],
        alias=alias,
        tracking_uri=tracking_uri,
    )
    result: dict[str, Any] = {"register": reg, "alias": aliased}
    if sync_contract and alias == ALIAS_CHAMPION:
        from science.training.config import PromotionConfig
        from science.training.promote import promote_checkpoint

        spec = get_lineage(lineage_id)
        if lineage_id == "v7":
            model_id = "tokyo_eye_v7"
        elif lineage_id == "v6.5":
            model_id = "gospc_v65"
        else:
            model_id = "gospc_v6"
        ckpt_id = f"{spec.checkpoint_prefix}_champion"
        status = "production"
        promo = promote_checkpoint(
            PromotionConfig(
                checkpoint_path=path,
                checkpoint_id=ckpt_id,
                model_id=model_id,
                status=status,
                run_id=run_id,
            )
        )
        # Ensure model entry exists for new lineages
        _ensure_contract_model(lineage_id=lineage_id, model_id=model_id)
        # Re-promote so production_model_id points correctly after model ensure
        promo = promote_checkpoint(
            PromotionConfig(
                checkpoint_path=path,
                checkpoint_id=ckpt_id,
                model_id=model_id,
                status="production",
                run_id=run_id,
            )
        )
        result["contract"] = promo
        try:
            get_gnn_model_catalog.cache_clear()
            get_checkpoint_catalog.cache_clear()
        except Exception:
            pass
    elif sync_contract and alias == ALIAS_CHALLENGER:
        spec = get_lineage(lineage_id)
        if lineage_id == "v7":
            model_id = "tokyo_eye_v7"
        elif lineage_id == "v6.5":
            model_id = "gospc_v65"
        else:
            model_id = "gospc_v6"
        ckpt_id = f"{spec.checkpoint_prefix}_challenger"
        promo = promote_checkpoint(
            PromotionConfig(
                checkpoint_path=path,
                checkpoint_id=ckpt_id,
                model_id=model_id,
                status="candidate",
                run_id=run_id,
            )
        )
        result["contract"] = promo
        try:
            get_gnn_model_catalog.cache_clear()
            get_checkpoint_catalog.cache_clear()
        except Exception:
            pass
    return result


def _ensure_contract_model(*, lineage_id: str, model_id: str) -> None:
    from science.training.promote import _load_contract, _write_contract

    contract = _load_contract()
    gnn = contract.setdefault("gnn_models", {})
    models = gnn.setdefault("models", {})
    if model_id in models:
        return
    spec = get_lineage(lineage_id)
    if lineage_id == "v7":
        models[model_id] = {
            "model_version": spec.model_version,
            "api_alias": "v7",
            "status": "candidate",
            "runner_module": "science.tokyo_eye.runner",
            "runner_class": "TokyoEyeRunner",
            "production_checkpoint_id": f"{spec.checkpoint_prefix}_champion",
        }
    elif lineage_id == "v6.5":
        models[model_id] = {
            "model_version": spec.model_version,
            "api_alias": "v65",
            "status": "candidate",
            "runner_module": "science.dtie.v65.gnn.runner",
            "runner_class": "V65GNNRunner",
            "production_checkpoint_id": f"{spec.checkpoint_prefix}_champion",
        }
    else:
        models[model_id] = {
            "model_version": spec.model_version,
            "api_alias": "v6",
            "status": "legacy",
            "runner_module": "science.dtie.v6.gnn.runner",
            "runner_class": "V6GNNRunner",
            "production_checkpoint_id": f"{spec.checkpoint_prefix}_champion",
        }
    _write_contract(contract)


def add_structure_to_corpus(
    *,
    structure_id: str,
    manifest_path: str | Path,
    pdb_id: str | None = None,
    chain: str = "A",
    role: str = "train",
    enabled: bool = True,
) -> dict[str, Any]:
    """Pin an ingested structure into a training corpus manifest."""
    path = Path(manifest_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {path}")
    data = json.loads(path.read_text())
    proteins = data.setdefault("proteins", [])
    pdb = (pdb_id or structure_id).upper()
    for entry in proteins:
        if str(entry.get("pdb_id", "")).upper() == pdb:
            entry["enabled"] = enabled
            entry["role"] = role
            entry["chain"] = entry.get("chain") or chain
            entry["lifecycle_added_at"] = _now_iso()
            entry["structure_id"] = structure_id
            path.write_text(json.dumps(data, indent=2) + "\n")
            return {"action": "updated", "pdb_id": pdb, "manifest": str(path), "entry": entry}

    entry = {
        "pdb_id": pdb,
        "chain": chain,
        "gene": pdb,
        "fold_id": "unknown",
        "fold_id_tier": "lifecycle",
        "fold_id_source": "lifecycle_panel",
        "role": role,
        "disposition_rule": "lifecycle_manual",
        "enabled": enabled,
        "stage0": False,
        "structure_id": structure_id,
        "lifecycle_added_at": _now_iso(),
    }
    proteins.append(entry)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return {"action": "added", "pdb_id": pdb, "manifest": str(path), "entry": entry}


def enqueue_train_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a train request for the lifecycle panel (async worker may pick up later)."""
    jobs = _load_jobs()
    job = {
        "job_id": f"train_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "status": "queued",
        "created_at": _now_iso(),
        "payload": payload,
    }
    jobs["active"] = job
    history = jobs.setdefault("history", [])
    history.insert(0, job)
    jobs["history"] = history[:50]
    _save_jobs(jobs)
    return job


def resolve_checkpoint_for_preview(
    *,
    checkpoint_path: str | None = None,
    alias: str | None = None,
    lineage_id: str = "v6",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    if checkpoint_path:
        p = Path(checkpoint_path)
        return {
            "checkpoint_path": str(p),
            "exists": p.is_file(),
            "sha256": compute_checkpoint_sha256(str(p)) if p.is_file() else None,
            "source": "path",
        }
    if alias:
        mv = get_version_by_alias(
            lineage_id=lineage_id, alias=alias, tracking_uri=tracking_uri
        )
        if mv is None:
            return {"checkpoint_path": None, "exists": False, "source": "alias", "alias": alias}
        cp = mv.get("checkpoint_path")
        p = Path(cp) if cp else None
        return {
            "checkpoint_path": cp,
            "exists": bool(p and p.is_file()),
            "sha256": compute_checkpoint_sha256(str(p)) if p and p.is_file() else None,
            "source": "alias",
            "alias": alias,
            "version": mv.get("version"),
        }
    prod = get_production_checkpoint_path()
    return {
        "checkpoint_path": prod,
        "exists": Path(prod).is_file() if prod else False,
        "sha256": compute_checkpoint_sha256(prod) if prod else None,
        "source": "production",
    }
