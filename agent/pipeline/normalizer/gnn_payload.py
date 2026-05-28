"""
Assemble GNN input payload from DB, serialize to GCS, publish gnn-ready event.

The GNN (Tokyo Eyes v3 / GOSPConeMapper) takes:
  node features: [rho, tau_flag, ss_type, sasa] per residue
  edges: C-alpha distance cutoff (CALPHA_CUTOFF_ANGSTROM)

This module does NOT call the GNN — it only prepares and publishes the input.

Schema notes (from 001_base_schema.sql + 002_schema_extensions.sql):
  dim_residue:  residue_id, chain_id, residue_index, residue_name, sse_code, sasa (002)
  dim_atom:     atom_id, residue_id, atom_name, x, y, z, ...
  dim_chain:    chain_id, structure_id, chain_label, sequence
  fact_dehydron: structure_id (002), donor_chain, donor_residue_index,
                 acceptor_chain, acceptor_residue_index, wrapping_count
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
import sqlalchemy as sa

TAU_THRESHOLD = 13.0
CALPHA_CUTOFF_ANGSTROM = 10.0

_SS_CODE_MAP = {"H": 0.0, "E": 0.5, "C": 1.0}


# ---------------------------------------------------------------------------
# Pure helper functions (no DB dependency — easy to unit-test)
# ---------------------------------------------------------------------------

def compute_tau_flag(rho: float) -> float:
    """Return 1.0 if rho < TAU_THRESHOLD, else 0.0."""
    return 1.0 if rho < TAU_THRESHOLD else 0.0


def encode_ss_type(sse_code: Optional[str]) -> float:
    """Encode secondary structure code: H→0.0, E→0.5, C or unknown→1.0."""
    return _SS_CODE_MAP.get(sse_code or "C", 1.0)


# ---------------------------------------------------------------------------
# GNNPayload dataclass
# ---------------------------------------------------------------------------

@dataclass
class GNNPayload:
    structure_id: str
    node_features: list          # [[rho, tau_flag, ss_type, sasa], ...] — shape [N, 4]
    edge_index: list             # [[src_indices], [dst_indices]]         — shape [2, E]
    residue_ids: list            # DB residue_id UUIDs in node order
    sasa_null_residue_ids: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

_RESIDUES_SQL = sa.text("""
    SELECT r.residue_id, r.residue_index, r.sse_code, r.sasa,
           c.chain_label
    FROM dim_residue r
    JOIN dim_chain c ON r.chain_id = c.chain_id
    WHERE c.structure_id = :structure_id
    ORDER BY c.chain_label, r.residue_index
""")

_DEHYDRONS_SQL = sa.text("""
    SELECT fd.donor_chain, fd.donor_residue_index,
           fd.acceptor_chain, fd.acceptor_residue_index,
           fd.wrapping_count
    FROM fact_dehydron fd
    WHERE fd.structure_id = :structure_id
""")

_CA_COORDS_SQL = sa.text("""
    SELECT a.residue_id, a.x, a.y, a.z
    FROM dim_atom a
    JOIN dim_residue r ON a.residue_id = r.residue_id
    JOIN dim_chain c ON r.chain_id = c.chain_id
    WHERE c.structure_id = :structure_id
      AND a.atom_name = 'CA'
    ORDER BY c.chain_label, r.residue_index
""")


# ---------------------------------------------------------------------------
# Core assembler
# ---------------------------------------------------------------------------

async def assemble_gnn_payload(conn, structure_id: str) -> GNNPayload:
    """
    Read dim_residue + fact_dehydron + dim_atom for structure_id and
    build the GNN input payload.

    Steps
    -----
    1. Fetch all residues (ordered) — provides sse_code, sasa.
    2. Fetch all dehydron rows — aggregate max wrapping_count per
       (chain_label, residue_index) key for both donor and acceptor.
    3. Build node feature vectors [rho, tau_flag, ss_type, sasa] per residue.
       Track residues whose sasa was NULL (sentinel 0.0).
    4. Fetch C-alpha coordinates from dim_atom.
    5. Build edge_index from the C-alpha distance cutoff.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine whose result supports ``fetchall()``).
    structure_id:
        UUID identifying the structure in dim_structure.

    Returns
    -------
    GNNPayload
    """
    params = {"structure_id": structure_id}

    # ---- 1. Residues --------------------------------------------------------
    residue_result = await conn.execute(_RESIDUES_SQL, params)
    residues = residue_result.fetchall()

    # ---- 2. Dehydrons → rho map keyed by (chain_label, residue_index) -------
    deh_result = await conn.execute(_DEHYDRONS_SQL, params)
    dehydrons = deh_result.fetchall()

    # Use max wrapping_count seen at each residue position
    rho_map: dict[tuple, float] = {}
    for deh in dehydrons:
        wc = float(deh.wrapping_count or 0)
        for key in [
            (deh.donor_chain, deh.donor_residue_index),
            (deh.acceptor_chain, deh.acceptor_residue_index),
        ]:
            if key[0] is not None and key[1] is not None:
                rho_map[key] = max(rho_map.get(key, 0.0), wc)

    # ---- 3. Build node features ---------------------------------------------
    node_features: list[list[float]] = []
    residue_ids: list[str] = []
    sasa_null_residue_ids: list[str] = []

    for row in residues:
        rid = row.residue_id
        key = (row.chain_label, row.residue_index)
        rho = rho_map.get(key, 0.0)
        tau_flag = compute_tau_flag(rho)
        ss_type = encode_ss_type(row.sse_code)

        sasa = row.sasa
        if sasa is None:
            sasa = 0.0
            sasa_null_residue_ids.append(rid)

        node_features.append([rho, tau_flag, ss_type, float(sasa)])
        residue_ids.append(rid)

    # ---- 4. C-alpha coordinates for edge building ---------------------------
    ca_result = await conn.execute(_CA_COORDS_SQL, params)
    ca_data = ca_result.fetchall()
    ca_by_rid: dict[str, tuple[float, float, float]] = {
        row.residue_id: (float(row.x), float(row.y), float(row.z))
        for row in ca_data
    }

    # ---- 5. Build edge_index (C-alpha distance cutoff via KD-tree) ---------
    n = len(residue_ids)
    src_edges: list[int] = []
    dst_edges: list[int] = []

    # Collect CA coords in node order
    ca_ordered = []
    ca_valid = []
    for i in range(n):
        ci = ca_by_rid.get(residue_ids[i])
        ca_ordered.append(ci)
        ca_valid.append(ci is not None)

    # Build KD-tree from valid CA positions
    valid_indices = [i for i in range(n) if ca_valid[i]]
    if valid_indices:
        coords = np.array([ca_ordered[i] for i in valid_indices])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(CALPHA_CUTOFF_ANGSTROM)
        for a, b in pairs:
            i, j = valid_indices[a], valid_indices[b]
            src_edges.extend([i, j])
            dst_edges.extend([j, i])

    return GNNPayload(
        structure_id=structure_id,
        node_features=node_features,
        edge_index=[src_edges, dst_edges],
        residue_ids=residue_ids,
        sasa_null_residue_ids=sasa_null_residue_ids,
    )


# ---------------------------------------------------------------------------
# GCS serialization
# ---------------------------------------------------------------------------

async def serialize_and_upload(payload: GNNPayload) -> str:
    """Serialize GNNPayload to JSON and upload to GCS. Returns GCS URI.

    In local dev (GCS_BUCKET_NAME unset), writes to a local temp file
    and returns a file:// URI instead.
    """
    data = json.dumps({
        "structure_id": payload.structure_id,
        "node_features": payload.node_features,
        "edge_index": payload.edge_index,
        "residue_ids": payload.residue_ids,
        "sasa_null_residue_ids": payload.sasa_null_residue_ids,
    }).encode("utf-8")

    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    if not bucket_name:
        # Local dev: write to temp directory
        import tempfile
        local_dir = os.path.join(tempfile.gettempdir(), "gosp-gnn-payloads", payload.structure_id)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, "payload.json")
        with open(local_path, "wb") as f:
            f.write(data)
        return f"file://{local_path}"

    from google.cloud import storage  # type: ignore[import]

    blob_name = f"gnn-payloads/{payload.structure_id}/payload.json"
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="application/json")

    return f"gs://{bucket_name}/{blob_name}"
