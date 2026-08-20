"""Chem-typed edges from governed ``fact_covalent_bond`` (Chem-MVP, train-side).

Extends the role multi-rel vocabulary with new IDs:

- ``disulf`` → relation 5
- ``covale`` → relation 6

Does **not** change GraphBuilder / Normalizer / ``fact_graph_edge``.
Loads already-written covalent bonds and appends directed one-hot rows after
``attach_role_edge_graph``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from science.tokyo_eye.role_edge_graph import (
    EDGE_ATTR_ROLE_DIM,
    NUM_ROLE_RELATIONS,
    ROLE_COUPLING,
    ROLE_DEHYDRON,
    ROLE_PACKING,
    ROLE_RIBBON,
    ROLE_SPOKE,
)
from science.dtie.common.dehydron_barcode_features import EDGE_BARCODE_DIM
from science.tokyo_eye.thermo_edge_features import GEO_DIM

ROLE_AUX_DIM = 1 + EDGE_BARCODE_DIM

# New chem IDs — do not remap onto packing/spoke (ablation vocabulary lock).
ROLE_DISULF = 5
ROLE_COVALE = 6
NUM_ROLE_RELATIONS_WITH_CHEM = 7
ROLE_ONEHOT_DIM_CHEM = NUM_ROLE_RELATIONS_WITH_CHEM
SPOKE_RHO_COL_CHEM = GEO_DIM + ROLE_ONEHOT_DIM_CHEM
COUPLING_STRENGTH_COL_CHEM = SPOKE_RHO_COL_CHEM
DEHYDRON_BARCODE_COL_CHEM = SPOKE_RHO_COL_CHEM + 1
EDGE_ATTR_CHEM_DIM = GEO_DIM + ROLE_ONEHOT_DIM_CHEM + ROLE_AUX_DIM

CHEM_BOND_TYPES = ("disulf", "covale")
_BOND_TYPE_TO_ROLE = {
    "disulf": ROLE_DISULF,
    "covale": ROLE_COVALE,
}


def pad_role_edge_attr_for_chem(
    edge_attr: torch.Tensor | np.ndarray,
    *,
    ha_edges: bool = False,
) -> torch.Tensor | np.ndarray:
    """Insert two zero one-hot columns after the base 5 role slots (before aux).

    ``EDGE_ATTR_ROLE_HA_DIM`` equals ``EDGE_ATTR_CHEM_DIM`` (both 17) — when
    ``ha_edges=True``, width 17 is treated as role+HA, not chem-MVP.
    """
    is_torch = isinstance(edge_attr, torch.Tensor)
    ea = edge_attr if is_torch else np.asarray(edge_attr)
    from science.tokyo_eye.ha_edge_graph import (
        EDGE_ATTR_CHEM_HA_DIM,
        EDGE_ATTR_ROLE_HA_DIM,
        HA_AUX_DIM,
    )

    if ea.shape[-1] == EDGE_ATTR_CHEM_HA_DIM:
        return ea
    # Role+HA width collides with chem-MVP width — flag disambiguates.
    if bool(ha_edges) and ea.shape[-1] == EDGE_ATTR_ROLE_HA_DIM:
        head = ea[..., : GEO_DIM + NUM_ROLE_RELATIONS]
        mid_aux = ea[..., GEO_DIM + NUM_ROLE_RELATIONS : EDGE_ATTR_ROLE_DIM]
        ha = ea[..., EDGE_ATTR_ROLE_DIM : EDGE_ATTR_ROLE_DIM + HA_AUX_DIM]
        if is_torch:
            zeros = torch.zeros(
                *ea.shape[:-1],
                2,
                dtype=ea.dtype,
                device=ea.device,
            )
            return torch.cat([head, zeros, mid_aux, ha], dim=-1)
        zeros = np.zeros((*ea.shape[:-1], 2), dtype=ea.dtype)
        return np.concatenate([head, zeros, mid_aux, ha], axis=-1)
    if ea.shape[-1] == EDGE_ATTR_CHEM_DIM:
        return ea
    if ea.shape[-1] == EDGE_ATTR_ROLE_DIM:
        # [geo | role5 | aux...] → [geo | role5 | chem2zeros | aux...]
        head = ea[..., : GEO_DIM + NUM_ROLE_RELATIONS]
        aux = ea[..., GEO_DIM + NUM_ROLE_RELATIONS :]
        if is_torch:
            zeros = torch.zeros(
                *ea.shape[:-1],
                2,
                dtype=ea.dtype,
                device=ea.device,
            )
            return torch.cat([head, zeros, aux], dim=-1)
        zeros = np.zeros((*ea.shape[:-1], 2), dtype=ea.dtype)
        return np.concatenate([head, zeros, aux], axis=-1)
    raise ValueError(
        f"expected role edge_attr width {EDGE_ATTR_ROLE_DIM}, "
        f"{EDGE_ATTR_ROLE_HA_DIM}, {EDGE_ATTR_CHEM_DIM}, or "
        f"{EDGE_ATTR_CHEM_HA_DIM}, got {ea.shape[-1]} (ha_edges={ha_edges})"
    )


def _parse_canonical_residue_id(
    residue_id: str,
) -> tuple[str, str, int, str | None] | None:
    parts = str(residue_id).strip().split(":")
    if len(parts) < 3:
        return None
    structure_id, chain_label, index_str = parts[0], parts[1], parts[2]
    try:
        residue_index = int(index_str)
    except ValueError:
        return None
    insertion = parts[3] if len(parts) >= 4 and parts[3] else None
    return structure_id, chain_label, residue_index, insertion


def _training_residue_key(chain_label: str, residue_index: int, insertion: str | None) -> str:
    # Matches load_graph_from_db training residue_ids (insertion currently dropped).
    del insertion
    return f"{chain_label}:{residue_index}:"


def map_bond_endpoints_to_nodes(
    bonds: Sequence[Mapping[str, Any]],
    residue_ids: Sequence[str],
    *,
    structure_id: str,
    chain_label: str,
) -> tuple[list[tuple[int, int, str]], int]:
    """Map governed bonds onto graph node indices.

    Skips bonds whose endpoints are missing, cross-chain relative to
    ``chain_label``, unknown bond types, or unresolved insertion codes.
    Returns ``(mapped_pairs, n_skipped)``.
    """
    index: dict[str, int] = {str(rid): i for i, rid in enumerate(residue_ids)}
    mapped: list[tuple[int, int, str]] = []
    skipped = 0
    structure_id = str(structure_id).strip().lower()
    chain_label = str(chain_label).strip()

    for bond in bonds:
        bond_type = str(bond.get("bond_type", "")).strip().lower()
        if bond_type not in _BOND_TYPE_TO_ROLE:
            skipped += 1
            continue
        p1 = _parse_canonical_residue_id(str(bond.get("residue_id_1", "")))
        p2 = _parse_canonical_residue_id(str(bond.get("residue_id_2", "")))
        if p1 is None or p2 is None:
            skipped += 1
            continue
        s1, c1, i1, ins1 = p1
        s2, c2, i2, ins2 = p2
        if s1.lower() != structure_id or s2.lower() != structure_id:
            skipped += 1
            continue
        if c1 != chain_label or c2 != chain_label:
            skipped += 1
            continue
        if ins1 or ins2:
            # Training graphs currently drop insertion codes; refuse ambiguous map.
            skipped += 1
            continue
        n1 = index.get(_training_residue_key(c1, i1, ins1))
        n2 = index.get(_training_residue_key(c2, i2, ins2))
        if n1 is None or n2 is None or n1 == n2:
            skipped += 1
            continue
        a, b = (n1, n2) if n1 < n2 else (n2, n1)
        mapped.append((a, b, bond_type))
    # Deduplicate identical undirected (pair, type)
    uniq: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for item in mapped:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq, skipped


def attach_chem_edge_graph(
    data: Data,
    ca_coords: torch.Tensor | np.ndarray,
    bonds: Sequence[Mapping[str, Any]],
    *,
    residue_ids: Sequence[str],
    structure_id: str,
    chain_label: str,
    force: bool = False,
) -> Data:
    """Pad role ``edge_attr`` for chem one-hots and append disulf/covale rows."""
    if not isinstance(data, Data):
        return data
    if getattr(data, "chem_edge_graph", False) and not force:
        return data
    if not getattr(data, "role_edge_graph", False):
        raise ValueError("attach_chem_edge_graph requires role_edge_graph first")

    if isinstance(ca_coords, torch.Tensor):
        coords_np = ca_coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(ca_coords, dtype=np.float64)

    device = data.edge_index.device
    dtype = data.edge_attr.dtype if data.edge_attr is not None else torch.float32

    ha_mode = bool(getattr(data, "ha_edge_graph", False))
    padded = pad_role_edge_attr_for_chem(data.edge_attr, ha_edges=ha_mode)
    if not isinstance(padded, torch.Tensor):
        padded = torch.tensor(padded, dtype=dtype, device=device)
    else:
        padded = padded.to(device=device, dtype=dtype)

    mapped, skipped = map_bond_endpoints_to_nodes(
        bonds,
        residue_ids,
        structure_id=structure_id,
        chain_label=chain_label,
    )

    from science.tokyo_eye.ha_edge_graph import EDGE_ATTR_CHEM_HA_DIM

    ha_mode = ha_mode or (padded.size(-1) == EDGE_ATTR_CHEM_HA_DIM)
    attr_dim = EDGE_ATTR_CHEM_HA_DIM if ha_mode else EDGE_ATTR_CHEM_DIM

    new_src: list[int] = []
    new_dst: list[int] = []
    new_attr_rows: list[torch.Tensor] = []
    counts = {"disulf": 0, "covale": 0}

    for i, j, bond_type in mapped:
        role = _BOND_TYPE_TO_ROLE[bond_type]
        diff = coords_np[j] - coords_np[i]
        dist = float(np.linalg.norm(diff))
        if dist < 1e-6:
            skipped += 1
            continue
        for src, dst, vec in (
            (i, j, diff),
            (j, i, -diff),
        ):
            row = torch.zeros(attr_dim, dtype=dtype, device=device)
            row[0] = float(vec[0])
            row[1] = float(vec[1])
            row[2] = float(vec[2])
            row[3] = dist
            row[GEO_DIM + role] = 1.0
            new_src.append(src)
            new_dst.append(dst)
            new_attr_rows.append(row)
        counts[bond_type] = counts.get(bond_type, 0) + 1

    if new_attr_rows:
        chem_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
        chem_attr = torch.stack(new_attr_rows, dim=0)
        data.edge_index = torch.cat([data.edge_index, chem_index], dim=1)
        data.edge_attr = torch.cat([padded, chem_attr], dim=0)
    else:
        data.edge_attr = padded

    data.chem_edge_graph = True  # type: ignore[attr-defined]
    data.chem_edge_counts = counts  # type: ignore[attr-defined]
    data.chem_edge_skipped = int(skipped)  # type: ignore[attr-defined]
    return data


async def fetch_covalent_bonds(
    db: Any,
    structure_id: str,
    *,
    bond_types: Sequence[str] = CHEM_BOND_TYPES,
) -> list[dict[str, Any]]:
    """Read governed Chem-MVP bonds for one structure (train-side, read-only)."""
    types = tuple(str(t).lower() for t in bond_types)
    rows = await db.fetch_all(
        """
        SELECT structure_id, residue_id_1, residue_id_2,
               atom_name_1, atom_name_2, bond_type, run_id
        FROM fact_covalent_bond
        WHERE structure_id = :structure_id
          AND bond_type = ANY(:bond_types)
        ORDER BY bond_type, residue_id_1, residue_id_2
        """,
        {
            "structure_id": str(structure_id).strip().lower(),
            "bond_types": list(types),
        },
    )
    return list(rows or [])


__all__ = [
    "CHEM_BOND_TYPES",
    "ROLE_DISULF",
    "ROLE_COVALE",
    "NUM_ROLE_RELATIONS_WITH_CHEM",
    "EDGE_ATTR_CHEM_DIM",
    "SPOKE_RHO_COL_CHEM",
    "COUPLING_STRENGTH_COL_CHEM",
    "DEHYDRON_BARCODE_COL_CHEM",
    "ROLE_PACKING",
    "ROLE_DEHYDRON",
    "ROLE_SPOKE",
    "ROLE_RIBBON",
    "ROLE_COUPLING",
    "pad_role_edge_attr_for_chem",
    "map_bond_endpoints_to_nodes",
    "attach_chem_edge_graph",
    "fetch_covalent_bonds",
]
