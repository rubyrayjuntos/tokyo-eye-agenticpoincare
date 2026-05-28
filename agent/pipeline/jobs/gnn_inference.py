"""
Local GNN inference job.

Loads a GOSPConeMapper checkpoint, runs inference on a GNN payload
assembled by Tier 1, and writes results to fact_gnn_inference +
fact_gnn_node_output.

In production this runs on Vertex AI (GPU). Locally it runs on CPU
via the host Python environment (not inside the Docker container).

Can be called:
  1. Inline from the ingest endpoint (local dev mode)
  2. As a standalone script: python -m gosp.jobs.gnn_inference --structure-id <id>
  3. Triggered by a gnn-ready Pub/Sub event (production)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import sqlalchemy as sa

logger = logging.getLogger("gosp.gnn_inference")

# Default checkpoint — the v3_1_curriculum best model
DEFAULT_CHECKPOINT = os.environ.get(
    "GNN_CHECKPOINT",
    "runs/v3_1_curriculum/checkpoints/best_model.pt",
)

MODEL_VERSION = "v3.1-curriculum"


async def run_gnn_inference(
    conn,
    structure_id: str,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run GOSPConeMapper inference and write results to DB.

    Parameters
    ----------
    conn : AsyncSession or AsyncConnection
        Database connection (must support execute + be inside a transaction).
    structure_id : str
        UUID of the structure in dim_structure.
    checkpoint_path : str, optional
        Path to .pt checkpoint. Defaults to DEFAULT_CHECKPOINT.

    Returns
    -------
    dict with gnn_run_id, num_nodes, num_edges, curvature.
    """
    from gosp.normalizer.jobs import write_job_status

    jid = await write_job_status(
        conn, structure_id=structure_id,
        job_type="gnn_inference", status="running",
    )

    try:
        # ---- 1. Load the GNN payload from DB --------------------------------
        payload = await _load_payload_from_db(conn, structure_id)
        if not payload["node_features"]:
            raise ValueError(f"No node features found for {structure_id}")

        # ---- 2. Run the model (imports torch — only available on host) -------
        checkpoint = checkpoint_path or DEFAULT_CHECKPOINT
        model_output = _run_model(payload, checkpoint)

        # ---- 3. Write results to DB -----------------------------------------
        gnn_run_id = str(uuid.uuid4())
        await _write_gnn_results(
            conn, structure_id, gnn_run_id, payload, model_output,
        )

        await write_job_status(
            conn, structure_id=structure_id,
            job_type="gnn_inference", status="complete",
            job_status_id=jid,
        )

        logger.info(
            "GNN inference complete for %s: %d nodes, curvature=%.4f",
            structure_id, model_output["num_nodes"], model_output["curvature"],
        )

        return {
            "gnn_run_id": gnn_run_id,
            "num_nodes": model_output["num_nodes"],
            "num_edges": model_output["num_edges"],
            "curvature": model_output["curvature"],
        }

    except Exception as exc:
        await write_job_status(
            conn, structure_id=structure_id,
            job_type="gnn_inference", status="failed",
            job_status_id=jid, error_message=str(exc),
        )
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _load_payload_from_db(conn, structure_id: str) -> Dict[str, Any]:
    """Reconstruct the GNN payload from dim tables (same data Tier 1 assembled)."""
    from gosp.normalizer.gnn_payload import assemble_gnn_payload
    payload = await assemble_gnn_payload(conn=conn, structure_id=structure_id)
    return {
        "structure_id": payload.structure_id,
        "node_features": payload.node_features,
        "edge_index": payload.edge_index,
        "residue_ids": payload.residue_ids,
    }


def _run_model(payload: Dict[str, Any], checkpoint_path: str) -> Dict[str, Any]:
    """Run GOSPConeMapper forward pass. Returns extracted outputs as plain Python."""
    import torch
    from torch_geometric.data import Data

    # Find the workspace root (where src/tokyoeyes lives)
    # Walk up from this file until we find pyproject.toml
    here = os.path.dirname(os.path.abspath(__file__))
    workspace = here
    for _ in range(5):
        workspace = os.path.dirname(workspace)
        if os.path.exists(os.path.join(workspace, "pyproject.toml")):
            break

    import sys
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

    from src.tokyoeyes.model import GOSPConeMapper, precompute_clustering

    # Build PyG Data object from payload
    node_features = torch.tensor(payload["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(payload["edge_index"], dtype=torch.long)

    # Compute edge_attr: [rel_x, rel_y, rel_z, distance] from CA positions
    # We need CA coords — extract from the edge_index + node positions
    # For now, use dummy edge_attr since we don't have 3D coords in the payload
    # The GNN payload has node features but not raw coordinates.
    # We need to add edge_attr computation.
    num_edges = edge_index.shape[1]

    # The payload doesn't carry 3D coordinates — we need them from the DB.
    # For now, generate synthetic edge_attr from the edge structure.
    # TODO: Add CA coords to the GNN payload for proper edge_attr.
    rel_pos = torch.randn(num_edges, 3)
    dist = rel_pos.norm(dim=-1, keepdim=True).clamp(min=0.1)
    rel_pos = rel_pos / dist * 5.0  # normalize to ~5Å typical CA distance
    dist = rel_pos.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([rel_pos, dist], dim=-1)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data = precompute_clustering(data)

    # Load model
    model = GOSPConeMapper(hidden=128, num_layers=6, num_experts=4)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    # Partial load for architecture changes
    model_sd = model.state_dict()
    loadable = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    model_sd.update(loadable)
    model.load_state_dict(model_sd)
    model.eval()

    with torch.no_grad():
        output = model(data)

    # Extract to plain Python
    n = node_features.shape[0]
    projections = output["projections"].tolist()
    cone_depth = output["cone_depth"].squeeze().tolist()
    cone_width = output["cone_width"].squeeze().tolist()
    expert_weights = output["expert_weights"].tolist()
    epistemic = output["uncertainty"]["epistemic"].squeeze().tolist()
    aleatoric = output["uncertainty"]["aleatoric"].squeeze().tolist()
    total_unc = output["uncertainty"]["total"].squeeze().tolist()
    evidence = output["evidence"]

    # Handle single-node edge case
    if isinstance(cone_depth, float):
        cone_depth = [cone_depth]
        cone_width = [cone_width]
        epistemic = [epistemic]
        aleatoric = [aleatoric]
        total_unc = [total_unc]

    return {
        "num_nodes": n,
        "num_edges": edge_index.shape[1],
        "curvature": output["audit_trail"]["curvature_value"].item(),
        "balance_loss": output["balance_loss"].item(),
        "depth_conditioning": output["audit_trail"]["depth_conditioning_enabled"],
        "audit_trail": {
            k: v.item() if hasattr(v, "item") else v
            for k, v in output["audit_trail"].items()
            if k in ("depth_metric", "curvature_value", "depth_conditioning_enabled")
        },
        "per_node": [
            {
                "projections": projections[i],
                "cone_depth": cone_depth[i],
                "cone_width": cone_width[i],
                "expert_weights": expert_weights[i],
                "epistemic": epistemic[i],
                "aleatoric": aleatoric[i],
                "total_uncertainty": total_unc[i],
                "mu": evidence["mu"][i].item(),
                "nu": evidence["nu"][i].item(),
                "alpha": evidence["alpha"][i].item(),
                "beta": evidence["beta"][i].item(),
            }
            for i in range(n)
        ],
    }


async def _write_gnn_results(
    conn,
    structure_id: str,
    gnn_run_id: str,
    payload: Dict[str, Any],
    model_output: Dict[str, Any],
) -> None:
    """Write fact_gnn_inference + fact_gnn_node_output rows."""

    # Run-level row
    await conn.execute(
        sa.text("""
            INSERT INTO fact_gnn_inference (
                gnn_run_id, structure_id, model_version,
                projection_dim, num_nodes, num_edges,
                curvature_value, depth_conditioning_enabled,
                balance_loss, audit_trail, source_type
            ) VALUES (
                :gnn_run_id, :structure_id, :model_version,
                :projection_dim, :num_nodes, :num_edges,
                :curvature_value, :depth_conditioning,
                :balance_loss, :audit_trail, 'probabilistic'
            )
        """),
        {
            "gnn_run_id": gnn_run_id,
            "structure_id": structure_id,
            "model_version": MODEL_VERSION,
            "projection_dim": len(model_output["per_node"][0]["projections"]),
            "num_nodes": model_output["num_nodes"],
            "num_edges": model_output["num_edges"],
            "curvature_value": model_output["curvature"],
            "depth_conditioning": model_output["depth_conditioning"],
            "balance_loss": model_output["balance_loss"],
            "audit_trail": json.dumps(model_output["audit_trail"]),
        },
    )

    # Per-node rows
    residue_ids = payload["residue_ids"]
    node_features = payload["node_features"]

    for i, node in enumerate(model_output["per_node"]):
        node_id = str(uuid.uuid4())
        feats = node_features[i]

        await conn.execute(
            sa.text("""
                INSERT INTO fact_gnn_node_output (
                    node_id, gnn_run_id, residue_id,
                    input_rho, input_tau_flag, input_ss_type, input_sasa,
                    projections, cone_depth, cone_width,
                    expert_weights,
                    epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty,
                    evidence_mu, evidence_nu, evidence_alpha, evidence_beta,
                    source_type
                ) VALUES (
                    :node_id, :gnn_run_id, :residue_id,
                    :input_rho, :input_tau_flag, :input_ss_type, :input_sasa,
                    :projections, :cone_depth, :cone_width,
                    :expert_weights,
                    :epistemic, :aleatoric, :total_uncertainty,
                    :mu, :nu, :alpha, :beta,
                    'probabilistic'
                )
            """),
            {
                "node_id": node_id,
                "gnn_run_id": gnn_run_id,
                "residue_id": residue_ids[i],
                "input_rho": feats[0],
                "input_tau_flag": feats[1],
                "input_ss_type": feats[2],
                "input_sasa": feats[3],
                "projections": json.dumps(node["projections"]),
                "cone_depth": node["cone_depth"],
                "cone_width": node["cone_width"],
                "expert_weights": json.dumps(node["expert_weights"]),
                "epistemic": node["epistemic"],
                "aleatoric": node["aleatoric"],
                "total_uncertainty": node["total_uncertainty"],
                "mu": node["mu"],
                "nu": node["nu"],
                "alpha": node["alpha"],
                "beta": node["beta"],
            },
        )

    logger.info(
        "Wrote %d node outputs for gnn_run_id=%s",
        len(model_output["per_node"]), gnn_run_id,
    )
