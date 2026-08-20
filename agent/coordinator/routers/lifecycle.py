"""GNN model lifecycle control plane — status, train, promote, corpus, preview."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lifecycle", tags=["lifecycle"])

_REPO_ROOT = Path(__file__).resolve().parents[3]


class ExperimentRequest(BaseModel):
    lineage_id: Literal["v6", "v6.5", "v6.6", "v7"] = "v7"
    experiment_name: str | None = None


class TrainRequest(BaseModel):
    lineage_id: Literal["v6", "v6.5", "v6.6", "v7"] = "v7"
    run_id: str = "tokyo_eye_v7_cold_v1"
    preset: Literal[
        "master_cold",
        "slim_moe_structural_ssot",
        "v66_feeler",
        "tokyo_eye_v7",
        "custom",
    ] = "tokyo_eye_v7"
    corpus: str = "manifests/v6_corpus_stage_a_small_v1.json"
    device: str = "cuda"
    max_proteins: int | None = 12
    epochs: int | None = None
    no_warm_start: bool = True


class AssessRequest(BaseModel):
    checkpoint_path: str
    corpus: str = "manifests/v6_corpus_stage_a_small_v1.json"
    max_proteins: int = 5
    device: str = "cpu"


class RegisterRequest(BaseModel):
    lineage_id: Literal["v6", "v6.5", "v6.6", "v7"] = "v7"
    checkpoint_path: str
    run_id: str | None = None
    alias: Literal["champion", "challenger"] = "challenger"
    sync_contract: bool = True


class PromoteRequest(BaseModel):
    lineage_id: Literal["v6", "v6.5", "v6.6", "v7"] = "v7"
    checkpoint_path: str | None = None
    version: str | None = None
    alias: Literal["champion", "challenger"] = "champion"
    run_id: str | None = None
    sync_contract: bool = True


class CorpusAddRequest(BaseModel):
    structure_id: str
    manifest_path: str = "manifests/v6_corpus_stage_a_small_v1.json"
    pdb_id: str | None = None
    chain: str = "A"
    enabled: bool = True


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@router.get("/status")
async def get_lifecycle_status() -> dict[str, Any]:
    from science.training.lifecycle import lifecycle_status

    try:
        return lifecycle_status(tracking_uri=_tracking_uri())
    except Exception as exc:
        logger.exception("lifecycle status failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/lineages")
async def get_lineages() -> dict[str, Any]:
    from science.training.lifecycle import list_lineages

    return {"lineages": list_lineages()}


@router.get("/runs")
async def get_lifecycle_runs(
    experiment: str | None = Query(None),
    lineage_id: str | None = Query(None),
    max_results: int = Query(25, ge=1, le=100),
) -> dict[str, Any]:
    from science.training.gnn_lineage import get_lineage
    from science.training.model_registry_mlflow import ensure_tracking

    try:
        mlf = ensure_tracking(_tracking_uri())
        exp_name = experiment
        if not exp_name and lineage_id:
            exp_name = get_lineage(lineage_id).mlflow_experiment
        if not exp_name:
            exp_name = "tokyo-eyes-v7"
        runs = mlf.search_runs(
            experiment_names=[exp_name],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        records: list[dict[str, Any]] = []
        if runs is not None and not runs.empty:
            for _, row in runs.iterrows():
                records.append(
                    {
                        "run_id": row.get("run_id"),
                        "run_name": row.get("tags.mlflow.runName")
                        or row.get("run_name"),
                        "status": row.get("status"),
                        "start_time": str(row.get("start_time")),
                        "checkpoint_path": row.get("tags.checkpoint_path"),
                        "focus_primary": row.get("tags.focus_primary"),
                        "focus_recommendation": row.get("tags.focus_recommendation"),
                    }
                )
        return {"experiment": exp_name, "runs": records}
    except Exception as exc:
        logger.exception("lifecycle runs failed")
        return {"experiment": experiment, "runs": [], "error": str(exc)}


@router.post("/experiments")
async def create_experiment(body: ExperimentRequest) -> dict[str, Any]:
    from science.training.gnn_lineage import get_lineage
    from science.training.model_registry_mlflow import ensure_tracking

    spec = get_lineage(body.lineage_id)
    name = body.experiment_name or spec.mlflow_experiment
    try:
        mlf = ensure_tracking(_tracking_uri())
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        existing = client.get_experiment_by_name(name)
        if existing is None:
            exp_id = client.create_experiment(name)
        else:
            exp_id = existing.experiment_id
        return {
            "experiment_name": name,
            "experiment_id": exp_id,
            "lineage_id": body.lineage_id,
        }
    except Exception as exc:
        logger.exception("create experiment failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/train")
async def enqueue_train(body: TrainRequest) -> dict[str, Any]:
    from science.training.lifecycle import enqueue_train_job

    job = enqueue_train_job(body.model_dump())
    # Best-effort: spawn science-container training via docker compose when available
    try:
        import asyncio
        import shlex

        run_id = body.run_id
        if body.lineage_id == "v7" or body.preset == "tokyo_eye_v7":
            lineage_flag = "train-v7"
        elif body.lineage_id == "v6.6" or body.preset == "v66_feeler":
            # Default enqueue is P1; P2 continue is host make train-v66-feeler-p2.
            lineage_flag = "train-v66-feeler"
        elif body.lineage_id == "v6.5":
            if body.preset == "slim_moe_structural_ssot":
                lineage_flag = "train-v65-slim-cold-start"
            else:
                lineage_flag = "train-v65-master-cold"
        else:
            if body.preset == "slim_moe_structural_ssot":
                lineage_flag = "train-v6-slim-moe-structural-ssot"
            else:
                lineage_flag = "train-v6-stage-a-small-master-cold"
        cmd = (
            f"make {lineage_flag} RUN_ID={shlex.quote(run_id)} "
            f"DEVICE={shlex.quote(body.device)}"
        )
        job["suggested_command"] = cmd
        job["note"] = (
            "Job queued in lifecycle store. Run suggested_command on the host "
            "or wait for a worker to pick it up."
        )
    except Exception as exc:
        job["spawn_error"] = str(exc)
    return job


@router.post("/assess")
async def assess_checkpoint(body: AssessRequest) -> dict[str, Any]:
    """Run assess_checkpoint harness (CPU-friendly subset)."""
    import asyncio

    ckpt = Path(body.checkpoint_path)
    if not ckpt.is_absolute():
        ckpt = _REPO_ROOT / ckpt
    if not ckpt.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt}")

    corpus = Path(body.corpus)
    if not corpus.is_absolute():
        corpus = _REPO_ROOT / corpus

    cmd = [
        "python",
        "-m",
        "experiments.training.v6.assess_checkpoint",
        "--checkpoint",
        str(ckpt),
        "--corpus",
        str(corpus),
        "--max-proteins",
        str(body.max_proteins),
        "--device",
        body.device,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": stdout.decode(errors="replace")[-8000:],
            "stderr": stderr.decode(errors="replace")[-4000:],
            "checkpoint_path": str(ckpt),
        }
    except Exception as exc:
        logger.exception("assess failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/register")
async def register_model(body: RegisterRequest) -> dict[str, Any]:
    from science.training.lifecycle import register_and_alias

    path = Path(body.checkpoint_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        return register_and_alias(
            lineage_id=body.lineage_id,
            checkpoint_path=path,
            alias=body.alias,
            run_id=body.run_id,
            sync_contract=body.sync_contract,
            tracking_uri=_tracking_uri(),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("register failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/promote")
async def promote_model(body: PromoteRequest) -> dict[str, Any]:
    from science.training.lifecycle import register_and_alias
    from science.training.model_registry_mlflow import (
        get_version_by_alias,
        set_alias,
    )

    try:
        if body.version and not body.checkpoint_path:
            aliased = set_alias(
                lineage_id=body.lineage_id,
                version=body.version,
                alias=body.alias,
                tracking_uri=_tracking_uri(),
            )
            mv = get_version_by_alias(
                lineage_id=body.lineage_id,
                alias=body.alias,
                tracking_uri=_tracking_uri(),
            )
            result: dict[str, Any] = {"alias": aliased, "model_version": mv}
            if body.sync_contract and mv and mv.get("checkpoint_path"):
                result = register_and_alias(
                    lineage_id=body.lineage_id,
                    checkpoint_path=mv["checkpoint_path"],
                    alias=body.alias,
                    run_id=mv.get("run_id") or body.run_id,
                    sync_contract=True,
                    tracking_uri=_tracking_uri(),
                )
            return result

        if not body.checkpoint_path:
            raise HTTPException(
                status_code=400,
                detail="Provide checkpoint_path or version",
            )
        path = Path(body.checkpoint_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return register_and_alias(
            lineage_id=body.lineage_id,
            checkpoint_path=path,
            alias=body.alias,
            run_id=body.run_id,
            sync_contract=body.sync_contract,
            tracking_uri=_tracking_uri(),
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("promote failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/corpus/add")
async def corpus_add(body: CorpusAddRequest) -> dict[str, Any]:
    from science.training.lifecycle import add_structure_to_corpus

    try:
        return add_structure_to_corpus(
            structure_id=body.structure_id,
            manifest_path=body.manifest_path,
            pdb_id=body.pdb_id,
            chain=body.chain,
            enabled=body.enabled,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("corpus add failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/checkpoint-preview")
async def checkpoint_preview(
    structure_id: str = Query(...),
    checkpoint_path: str | None = Query(None),
    alias: str | None = Query(None),
    lineage_id: str = Query("v6"),
) -> dict[str, Any]:
    """Resolve checkpoint for viewport hydrate; structure must already be in DB."""
    from science.training.lifecycle import resolve_checkpoint_for_preview

    resolved = resolve_checkpoint_for_preview(
        checkpoint_path=checkpoint_path,
        alias=alias,
        lineage_id=lineage_id,
        tracking_uri=_tracking_uri(),
    )
    return {
        "structure_id": structure_id,
        "preview_mode": alias or ("path" if checkpoint_path else "production"),
        "checkpoint": resolved,
        "hydrate_hint": (
            "Select this structure in the cockpit TripleViewport. "
            "Embeddings use production champion unless a scoped re-infer is run."
        ),
    }
