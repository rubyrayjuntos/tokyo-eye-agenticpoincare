"""Build training PyG graphs from governed DB rows (read-only, no feature recompute)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from science.dtie.common import residue_features as rf
from science.dtie.common.ingest_master_features import DEFAULT_CONDITION

logger = logging.getLogger(__name__)

EDGE_CUTOFF = 8.0


async def fetch_training_rows(
    db: Any,
    structure_id: str,
    chain_label: str,
) -> list[dict[str, Any]]:
    return await db.fetch_all(
        """
        SELECT
            r.residue_id,
            r.residue_index,
            c.chain_label,
            r.sasa,
            r.sse_code,
            f.rho,
            f.tau_flag,
            f.ss_type,
            a.x AS ca_x,
            a.y AS ca_y,
            a.z AS ca_z,
            a.b_factor AS ca_b_factor
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN fact_ingestion_features f
          ON f.residue_id = r.residue_id
         AND f.structure_id = c.structure_id
         AND f.condition = :condition
         AND f.is_current = TRUE
        LEFT JOIN dim_atom a
          ON a.residue_id = r.residue_id
         AND a.atom_name = 'CA'
        WHERE c.structure_id = :structure_id
          AND c.chain_label = :chain_label
          AND r.sasa IS NOT NULL
          AND r.sse_code IS NOT NULL
          AND a.x IS NOT NULL
        ORDER BY r.residue_index
        """,
        {
            "structure_id": structure_id,
            "chain_label": chain_label,
            "condition": DEFAULT_CONDITION,
        },
    )


def build_protein_graph_dict(
    pdb_id: str,
    chain: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Assemble the load_protein_graph return dict from DB rows."""
    from science.dtie.v5.gnn.model import precompute_clustering

    if len(rows) < 10:
        logger.warning("%s chain %s: only %d DB residues", pdb_id, chain, len(rows))
        return None

    rho_list, tau_list, ss_list, sasa_list = [], [], [], []
    ca_list, bf_list, bf_present, res_ids = [], [], [], []

    for row in rows:
        rho_list.append(float(row["rho"]))
        tau_list.append(float(row["tau_flag"]))
        ss_list.append(float(row["ss_type"]))
        sasa_list.append(float(row["sasa"]))
        ca_list.append([row["ca_x"], row["ca_y"], row["ca_z"]])
        bf = row.get("ca_b_factor")
        if bf is None:
            bf_list.append(float("nan"))
            bf_present.append(False)
        else:
            bf_f = float(bf)
            bf_list.append(bf_f if np.isfinite(bf_f) else float("nan"))
            bf_present.append(np.isfinite(bf_f))
        res_ids.append(f"{chain}:{row['residue_index']}:")

    rho_arr = np.array(rho_list, dtype=np.float64)
    ca_coords = np.array(ca_list, dtype=np.float64)
    n = len(rho_arr)

    tau_flag = np.array(tau_list, dtype=np.float64)
    ss_type = np.array(ss_list, dtype=np.float64)
    sasa = np.array(sasa_list, dtype=np.float64)
    x = rf.stack_gnn_node_features(rho_arr, tau_flag, ss_type, sasa)

    dists = cdist(ca_coords, ca_coords)
    src, dst = np.where((dists < EDGE_CUTOFF) & (dists > 0.1))
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    rel_pos = ca_coords[dst] - ca_coords[src]
    edge_dist = dists[src, dst]
    edge_attr = np.column_stack([rel_pos, edge_dist]).astype(np.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    data.sasa = torch.tensor(sasa, dtype=torch.float32)
    data = precompute_clustering(data)

    domain_labels = torch.full((n,), -1, dtype=torch.long)
    if pdb_id in ("4OBE", "4DSO", "6OIM", "5VQ2", "3CON", "4G0N"):
        for i, rid in enumerate(res_ids):
            resnum = int(rid.split(":")[1])
            if 10 <= resnum <= 17:
                domain_labels[i] = 0
            elif 25 <= resnum <= 40:
                domain_labels[i] = 1
            elif 57 <= resnum <= 75:
                domain_labels[i] = 2
            elif 87 <= resnum <= 104:
                domain_labels[i] = 3
            elif 116 <= resnum <= 126:
                domain_labels[i] = 4
            elif 145 <= resnum <= 170:
                domain_labels[i] = 5

    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "structure_id": rows[0].get("structure_id") if rows else None,
        "data": data,
        "target_rho": torch.tensor(rho_arr, dtype=torch.float32).unsqueeze(1),
        "target_dehydron": torch.tensor(tau_flag, dtype=torch.float32).unsqueeze(1),
        "target_sasa": torch.tensor(sasa, dtype=torch.float32).unsqueeze(1),
        "b_factor_ca": torch.tensor(bf_list, dtype=torch.float32).unsqueeze(1),
        "b_factor_present": torch.tensor(bf_present, dtype=torch.bool),
        "ca_coords": torch.tensor(ca_coords, dtype=torch.float32),
        "domain_labels": domain_labels if (domain_labels >= 0).any() else None,
        "residue_ids": res_ids,
        "n_residues": n,
        "source": "db",
    }


async def load_protein_graph_from_db(
    db: Any,
    structure_id: str,
    pdb_id: str,
    chain: str,
) -> dict[str, Any] | None:
    rows = await fetch_training_rows(db, structure_id, chain)
    for row in rows:
        row["structure_id"] = structure_id
    return build_protein_graph_dict(pdb_id, chain, rows)
