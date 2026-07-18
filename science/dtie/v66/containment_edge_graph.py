"""SSE parent-node containment edges (Path B, train-side).

Extends the chem multi-rel vocabulary with new IDs:

- ``contain_down`` → relation 7 (parent → child)
- ``contain_up`` → relation 8 (child → parent)

Appends mean-pooled parent nodes after chem attach. Does **not** change
GraphBuilder / Normalizer / ``fact_graph_edge``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.dehydron_barcode_features import EDGE_BARCODE_DIM
from science.dtie.v66.chem_edge_graph import (
    EDGE_ATTR_CHEM_DIM,
    NUM_ROLE_RELATIONS_WITH_CHEM,
)
from science.dtie.v66.sse_hierarchy import map_sse_ranges_to_node_indices, parse_pdb_helix_sheet
from science.dtie.v66.thermo_edge_features import GEO_DIM

ROLE_AUX_DIM = 1 + EDGE_BARCODE_DIM

ROLE_CONTAIN_DOWN = 7
ROLE_CONTAIN_UP = 8
NUM_ROLE_RELATIONS_WITH_CONTAINMENT = 9
ROLE_ONEHOT_DIM_CONTAIN = NUM_ROLE_RELATIONS_WITH_CONTAINMENT
SPOKE_RHO_COL_CONTAIN = GEO_DIM + ROLE_ONEHOT_DIM_CONTAIN
COUPLING_STRENGTH_COL_CONTAIN = SPOKE_RHO_COL_CONTAIN
DEHYDRON_BARCODE_COL_CONTAIN = SPOKE_RHO_COL_CONTAIN + 1
EDGE_ATTR_CONTAIN_DIM = GEO_DIM + ROLE_ONEHOT_DIM_CONTAIN + ROLE_AUX_DIM


def pad_chem_edge_attr_for_containment(
    edge_attr: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Insert two zero one-hot columns after chem's 7 role slots (before aux)."""
    is_torch = isinstance(edge_attr, torch.Tensor)
    ea = edge_attr if is_torch else np.asarray(edge_attr)
    if ea.shape[-1] == EDGE_ATTR_CONTAIN_DIM:
        return ea
    if ea.shape[-1] != EDGE_ATTR_CHEM_DIM:
        raise ValueError(
            f"expected chem edge_attr width {EDGE_ATTR_CHEM_DIM} or "
            f"{EDGE_ATTR_CONTAIN_DIM}, got {ea.shape[-1]}"
        )
    # [geo | role7 | aux...] → [geo | role7 | contain2zeros | aux...]
    head = ea[..., : GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM]
    aux = ea[..., GEO_DIM + NUM_ROLE_RELATIONS_WITH_CHEM :]
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


def _load_pdb_text(pdb_text_or_path: str | Path) -> str:
    path = Path(pdb_text_or_path)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return str(pdb_text_or_path)


def attach_containment_edge_graph(
    data: Data,
    ca_coords: torch.Tensor | np.ndarray,
    pdb_text_or_path: str | Path,
    *,
    residue_ids: Sequence[str],
    force: bool = False,
) -> Data:
    """Pad chem ``edge_attr`` for containment one-hots and append parent rows."""
    if not isinstance(data, Data):
        return data
    if getattr(data, "containment_edge_graph", False) and not force:
        return data
    if not getattr(data, "chem_edge_graph", False) and not force:
        raise ValueError("attach_containment_edge_graph requires chem_edge_graph first")

    if isinstance(ca_coords, torch.Tensor):
        coords_np = ca_coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(ca_coords, dtype=np.float64)

    n_residues = int(data.x.size(0))
    device = data.edge_index.device
    dtype = data.edge_attr.dtype if data.edge_attr is not None else torch.float32

    padded = pad_chem_edge_attr_for_containment(data.edge_attr)
    if not isinstance(padded, torch.Tensor):
        padded = torch.tensor(padded, dtype=dtype, device=device)
    else:
        padded = padded.to(device=device, dtype=dtype)

    pdb_text = _load_pdb_text(pdb_text_or_path)
    mapped = map_sse_ranges_to_node_indices(
        parse_pdb_helix_sheet(pdb_text),
        residue_ids,
    )

    new_src: list[int] = []
    new_dst: list[int] = []
    new_attr_rows: list[torch.Tensor] = []
    parent_x_rows: list[torch.Tensor] = []
    parent_ca_rows: list[np.ndarray] = []
    counts = {"contain_down": 0, "contain_up": 0}

    for _sse_range, child_indices in mapped:
        if not child_indices:
            continue
        parent_idx = n_residues + len(parent_x_rows)
        child_x = data.x[child_indices]
        if isinstance(child_x, torch.Tensor):
            parent_x = child_x.mean(dim=0)
        else:
            parent_x = torch.tensor(np.mean(child_x, axis=0), dtype=dtype, device=device)
        parent_x_rows.append(parent_x)
        parent_ca = coords_np[child_indices].mean(axis=0)
        parent_ca_rows.append(parent_ca)

        for child_idx in child_indices:
            child_ca = coords_np[child_idx]
            diff_down = child_ca - parent_ca
            dist_down = float(np.linalg.norm(diff_down))
            row_down = torch.zeros(EDGE_ATTR_CONTAIN_DIM, dtype=dtype, device=device)
            row_down[0] = float(diff_down[0])
            row_down[1] = float(diff_down[1])
            row_down[2] = float(diff_down[2])
            row_down[3] = dist_down
            row_down[GEO_DIM + ROLE_CONTAIN_DOWN] = 1.0
            new_src.append(parent_idx)
            new_dst.append(child_idx)
            new_attr_rows.append(row_down)
            counts["contain_down"] += 1

            diff_up = parent_ca - child_ca
            dist_up = float(np.linalg.norm(diff_up))
            row_up = torch.zeros(EDGE_ATTR_CONTAIN_DIM, dtype=dtype, device=device)
            row_up[0] = float(diff_up[0])
            row_up[1] = float(diff_up[1])
            row_up[2] = float(diff_up[2])
            row_up[3] = dist_up
            row_up[GEO_DIM + ROLE_CONTAIN_UP] = 1.0
            new_src.append(child_idx)
            new_dst.append(parent_idx)
            new_attr_rows.append(row_up)
            counts["contain_up"] += 1

    if parent_x_rows:
        parent_x_tensor = torch.stack(parent_x_rows, dim=0)
        data.x = torch.cat([data.x, parent_x_tensor], dim=0)
        if new_attr_rows:
            contain_index = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
            contain_attr = torch.stack(new_attr_rows, dim=0)
            data.edge_index = torch.cat([data.edge_index, contain_index], dim=1)
            data.edge_attr = torch.cat([padded, contain_attr], dim=0)
        else:
            data.edge_attr = padded
    else:
        data.edge_attr = padded

    data.containment_edge_graph = True  # type: ignore[attr-defined]
    data.n_residue_nodes = n_residues  # type: ignore[attr-defined]
    data.n_parent_nodes = len(parent_x_rows)  # type: ignore[attr-defined]
    data.containment_edge_counts = counts  # type: ignore[attr-defined]
    return data


__all__ = [
    "ROLE_CONTAIN_DOWN",
    "ROLE_CONTAIN_UP",
    "NUM_ROLE_RELATIONS_WITH_CONTAINMENT",
    "EDGE_ATTR_CONTAIN_DIM",
    "SPOKE_RHO_COL_CONTAIN",
    "COUPLING_STRENGTH_COL_CONTAIN",
    "DEHYDRON_BARCODE_COL_CONTAIN",
    "pad_chem_edge_attr_for_containment",
    "attach_containment_edge_graph",
]
