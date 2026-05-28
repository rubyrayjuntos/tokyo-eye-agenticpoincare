from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger_p1_v4 = logging.getLogger("DTIE_Phase1_v4")


def _residue_id_to_key(res_id) -> str:
    return str(res_id)


def _extract_ca_coords(residue_set) -> np.ndarray:
    coords = np.zeros((len(residue_set.residues), 3), dtype=np.float64)
    for i, residue in enumerate(residue_set.residues):
        try:
            coords[i] = residue["CA"].get_coord()
        except KeyError:
            pass
    return coords


def _condition_payload(label: str, residue_set, preds) -> Dict:
    ids = [_residue_id_to_key(r.id) for r in residue_set.residues]
    ca = _extract_ca_coords(residue_set)
    cone_depth = np.asarray(preds.cone_depth, dtype=np.float64).reshape(-1)
    if cone_depth.size != len(ids):
        raise ValueError(f"{label}: cone_depth size mismatch for residue ids")

    cone_width = np.asarray(
        getattr(preds, "cone_width", np.exp(-cone_depth)), dtype=np.float64
    ).reshape(-1)
    if cone_width.size != len(ids):
        cone_width = np.exp(-cone_depth)

    expert_weights = np.asarray(
        getattr(preds, "expert_weights", np.zeros((len(ids), 1), dtype=np.float64)),
        dtype=np.float64,
    )
    if expert_weights.ndim != 2 or expert_weights.shape[0] != len(ids):
        expert_weights = np.zeros((len(ids), 1), dtype=np.float64)

    audit_trail = getattr(preds, "audit_trail", {})
    if not isinstance(audit_trail, dict):
        audit_trail = {}

    return {
        "label": label,
        "residue_ids": ids,
        "ca_coords": ca,
        "projections": np.asarray(preds.projections, dtype=np.float64),
        "cone_depth": cone_depth,
        "cone_width": cone_width,
        "expert_weights": expert_weights,
        "aleatoric": np.asarray(preds.aleatoric, dtype=np.float64).reshape(-1),
        "epistemic": np.asarray(preds.epistemic, dtype=np.float64).reshape(-1),
        "total_uncertainty": np.asarray(preds.total, dtype=np.float64).reshape(-1),
        "audit_trail": audit_trail,
    }


def execute_phase_1_v4_residue_hyperbolic(
    gdp_set,
    gtp_set,
    evidential_gnn,
) -> Dict:
    """Residue-level hyperbolic inference for v4 source-leak mode."""
    gdp_preds = evidential_gnn.predict_nig(gdp_set)
    gtp_preds = evidential_gnn.predict_nig(gtp_set)

    gdp = _condition_payload("GDP", gdp_set, gdp_preds)
    gtp = _condition_payload("GTP", gtp_set, gtp_preds)

    logger_p1_v4.info(
        "Phase 1 v4: residue hyperbolic inference complete | "
        f"gdp={len(gdp['residue_ids'])} residues | gtp={len(gtp['residue_ids'])} residues"
    )

    return {
        "conditions": {
            "gdp": gdp,
            "gtp": gtp,
        }
    }
