"""Write GNN inference results to fact_gnn_inference and fact_gnn_node_output.

Write path:
    Vertex AI GNN response  →  fact_gnn_inference  (one row per run)
                            →  fact_gnn_node_output (one row per residue)

fact_gnn_node_output stores both the INPUT features sent to the GNN (for
provenance) and all OUTPUT values (projections, cone geometry, expert routing,
evidential uncertainty).
"""
import json
import uuid
from typing import Optional

import sqlalchemy as sa


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT_INFERENCE = sa.text("""
    INSERT INTO fact_gnn_inference
        (gnn_run_id, structure_id, model_version, projection_dim, num_nodes, num_edges,
         sasa_null_warnings, audit_trail, source_type)
    VALUES
        (:gnn_run_id, :structure_id, :model_version, :projection_dim, :num_nodes, :num_edges,
         :sasa_null_warnings, :audit_trail, :source_type)
    ON CONFLICT (gnn_run_id) DO NOTHING
""")

_INSERT_NODE = sa.text("""
    INSERT INTO fact_gnn_node_output
        (node_id, gnn_run_id, residue_id,
         input_rho, input_tau_flag, input_ss_type, input_sasa,
         projections, cone_depth, cone_width,
         expert_weights,
         epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty,
         evidence_mu, evidence_nu, evidence_alpha, evidence_beta,
         source_type)
    VALUES
        (:node_id, :gnn_run_id, :residue_id,
         :input_rho, :input_tau_flag, :input_ss_type, :input_sasa,
         :projections, :cone_depth, :cone_width,
         :expert_weights,
         :epistemic_uncertainty, :aleatoric_uncertainty, :total_uncertainty,
         :evidence_mu, :evidence_nu, :evidence_alpha, :evidence_beta,
         :source_type)
    ON CONFLICT (node_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_gnn_inference(
    conn,
    structure_id: str,
    model_version: str,
    gnn_run_id: str,
    node_outputs: list,
    sasa_null_warnings: Optional[list] = None,
    audit_trail: Optional[dict] = None,
    num_edges: int = 0,
    source_type: str = "probabilistic",
) -> str:
    """Write GNN inference results to fact_gnn_inference and fact_gnn_node_output.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID identifying the parent structure in dim_structure.
    model_version:
        GNN model version string, e.g. ``"v3"``.
    gnn_run_id:
        Stable identifier for this inference run (used as PK in
        fact_gnn_inference and FK in fact_gnn_node_output).
    node_outputs:
        List of dicts, one per residue, each containing:
        - ``residue_id``        — FK to dim_residue
        - ``input_rho``         — wrapping count (input feature)
        - ``input_tau_flag``    — 1.0 if rho < threshold (input feature)
        - ``input_ss_type``     — encoded secondary structure (input feature)
        - ``input_sasa``        — solvent-accessible surface area (input feature)
        - ``projections``       — list of floats (Poincaré disc output)
        - ``cone_depth``        — float
        - ``cone_width``        — float
        - ``expert_weights``    — list of floats
        - ``epistemic_uncertainty``  — float or None
        - ``aleatoric_uncertainty``  — float or None
        - ``total_uncertainty``      — float or None
        - ``evidence_mu``            — float or None (NIG parameter)
        - ``evidence_nu``            — float or None (NIG parameter)
        - ``evidence_alpha``         — float or None (NIG parameter)
        - ``evidence_beta``          — float or None (NIG parameter)
    sasa_null_warnings:
        List of residue_id strings where SASA was NULL (sentinel 0.0 used).
    audit_trail:
        Free-form dict stored as JSONB for downstream traceability.
    num_edges:
        Number of edges in the GNN graph (stored for diagnostics).
    source_type:
        Provenance tag; defaults to ``"probabilistic"``.

    Returns
    -------
    str
        The ``gnn_run_id`` passed in (for chaining).
    """
    projection_dim = len(node_outputs[0]["projections"]) if node_outputs else 0

    await conn.execute(_INSERT_INFERENCE, {
        "gnn_run_id":          gnn_run_id,
        "structure_id":        structure_id,
        "model_version":       model_version,
        "projection_dim":      projection_dim,
        "num_nodes":           len(node_outputs),
        "num_edges":           num_edges,
        "sasa_null_warnings":  json.dumps(sasa_null_warnings or []),
        "audit_trail":         json.dumps(audit_trail or {}),
        "source_type":         source_type,
    })

    for node in node_outputs:
        await conn.execute(_INSERT_NODE, {
            "node_id":                  str(uuid.uuid4()),
            "gnn_run_id":               gnn_run_id,
            "residue_id":               node["residue_id"],
            "input_rho":                node["input_rho"],
            "input_tau_flag":           node["input_tau_flag"],
            "input_ss_type":            node["input_ss_type"],
            "input_sasa":               node["input_sasa"],
            "projections":              json.dumps(node["projections"]),
            "cone_depth":               node["cone_depth"],
            "cone_width":               node["cone_width"],
            "expert_weights":           json.dumps(node["expert_weights"]),
            "epistemic_uncertainty":    node.get("epistemic_uncertainty"),
            "aleatoric_uncertainty":    node.get("aleatoric_uncertainty"),
            "total_uncertainty":        node.get("total_uncertainty"),
            "evidence_mu":              node.get("evidence_mu"),
            "evidence_nu":              node.get("evidence_nu"),
            "evidence_alpha":           node.get("evidence_alpha"),
            "evidence_beta":            node.get("evidence_beta"),
            "source_type":              source_type,
        })

    return gnn_run_id
