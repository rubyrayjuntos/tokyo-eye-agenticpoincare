"""Role-typed edge graph for v6.6 feeler (training-only).

Replaces the isotropic Cα contact MP graph with five thermodynamic relations:

- packing: ordered ↔ ordered contacts (high ρ / τ=0)
- dehydron: underwrapped backbone H-bonds (leak / rim)
- spoke: ordered ↔ disordered contacts (disc-filling rays)
- ribbon: sequence-local ±1…±4 (SS angular continuity)
- coupling: cross-subgraph propagation (rim ↔ core, optional)

Does **not** change GraphBuilder / Normalizer / fact_graph_edge.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.dehydron_barcode_features import EDGE_BARCODE_DIM
from science.dtie.common.residue_features import (
    TAU,
    ResidueRecord,
    compute_bond_wrapping_count,
    wrapping_carbon_coords,
)
from science.dtie.v66.thermo_edge_features import (
    GEO_DIM,
    HBOND_SEQ_CUTOFF,
    HBOND_SPATIAL_CUTOFF,
    parse_auth_seq_ids,
    resolve_residue_records_for_prot,
)

ROLE_PACKING = 0
ROLE_DEHYDRON = 1
ROLE_SPOKE = 2
ROLE_RIBBON = 3
ROLE_COUPLING = 4
NUM_ROLE_RELATIONS = 5
ROLE_ONEHOT_DIM = NUM_ROLE_RELATIONS
# Continuous ρ affinity on spoke edges (0 on packing / dehydron / ribbon / coupling).
SPOKE_RHO_COL = GEO_DIM + ROLE_ONEHOT_DIM
# Cross-subgraph coupling strength on ROLE_COUPLING edges (reuses aux column slot).
COUPLING_STRENGTH_COL = SPOKE_RHO_COL
# Local dehydron barcode on dehydron role edges only (0 elsewhere).
DEHYDRON_BARCODE_COL = SPOKE_RHO_COL + 1
ROLE_AUX_DIM = 1 + EDGE_BARCODE_DIM
EDGE_ATTR_ROLE_DIM = GEO_DIM + ROLE_ONEHOT_DIM + ROLE_AUX_DIM

CONTACT_CUTOFF = 8.0
COUPLING_CUTOFF = 12.0
RIBBON_MAX_SEP = 4


def coupling_strength(
    rho_i: float,
    rho_j: float,
    *,
    tau: float,
    dist: float,
    coupling_cutoff: float = COUPLING_CUTOFF,
) -> float:
    """Scalar weight for cross-subgraph coupling edges (rim ↔ core propagation)."""
    tau_f = max(float(tau), 1e-6)
    rho_hi = max(float(rho_i), float(rho_j), tau_f)
    rho_sim = 1.0 - abs(float(rho_i) - float(rho_j)) / rho_hi
    rim_side = (tau_f - min(float(rho_i), float(rho_j))) / tau_f
    dist_scale = max(0.0, 1.0 - float(dist) / float(coupling_cutoff))
    return float(np.clip((0.15 + 0.85 * rho_sim) * (0.35 + 0.65 * rim_side) * dist_scale, 0.1, 1.5))


def _ordered_mask(rho: np.ndarray, tau_flag: np.ndarray | None, tau: float) -> np.ndarray:
    """Ordered/core residues from continuous ρ (ρ ≥ τ). Binary τ is not used."""
    del tau_flag  # kept for call-site compatibility; binary shell cut removed
    return rho >= float(tau)


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def spoke_rho_weight(rho_i: float, rho_j: float, tau: float, *, scale: float = 1.0) -> float:
    """Continuous spoke strength: thermodynamic contrast × rim-side pull."""
    tau_f = max(float(tau), 1e-6)
    rho_lo = float(min(rho_i, rho_j))
    rho_hi = float(max(rho_i, rho_j))
    contrast = (rho_hi - rho_lo) / tau_f
    rim_pull = (tau_f - rho_lo) / tau_f
    return float(np.clip(contrast * rim_pull * float(scale), 0.0, 2.0))


def _append_directed(
    src: list[int],
    dst: list[int],
    geo: list[list[float]],
    roles: list[int],
    coords: np.ndarray,
    i: int,
    j: int,
    role: int,
) -> None:
    diff = coords[j] - coords[i]
    d = float(np.linalg.norm(diff))
    if d < 1e-6:
        return
    src.append(i)
    dst.append(j)
    geo.append([float(diff[0]), float(diff[1]), float(diff[2]), d])
    roles.append(role)
    src.append(j)
    dst.append(i)
    geo.append([-float(diff[0]), -float(diff[1]), -float(diff[2]), d])
    roles.append(role)


def compute_role_edge_graph(
    coords: np.ndarray,
    rho: np.ndarray,
    *,
    tau_flag: np.ndarray | None = None,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
    contact_cutoff: float = CONTACT_CUTOFF,
    ribbon_max_sep: int = RIBBON_MAX_SEP,
    enable_coupling_edges: bool = False,
    dehydron_exclusivity: bool = True,
    coupling_cutoff: float = COUPLING_CUTOFF,
    spoke_edge_scale: float = 1.0,
    ribbon_edge_scale: float = 1.0,
    ha_edges: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``edge_index [2,E]`` and ``edge_attr [E, D]``.

    ``D = EDGE_ATTR_ROLE_DIM`` (chem-MVP) or ``EDGE_ATTR_ROLE_HA_DIM`` when
    ``ha_edges=True`` (appends HA_MIN_DIST_NORM + HA_STRENGTH).
    """
    del ribbon_edge_scale  # reserved for message scaling in the conv, not graph build
    coords = np.asarray(coords, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    n = int(coords.shape[0])
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N,3], got {coords.shape}")
    if rho.shape[0] != n:
        raise ValueError("rho length must match coords")

    if tau_flag is None:
        tau_np = (rho < float(tau)).astype(np.float64)
    else:
        tau_np = np.asarray(tau_flag, dtype=np.float64).reshape(-1)
        if tau_np.shape[0] != n:
            raise ValueError("tau_flag length must match coords")

    ordered = _ordered_mask(rho, tau_np, tau)
    seq = parse_auth_seq_ids(residue_ids, n)

    by_seq: dict[int, ResidueRecord] | None = None
    all_atoms = None
    carbon_coords = None
    if residue_records:
        from science.dtie.v66.thermo_edge_features import (
            _flatten_atoms,
            _index_records_by_auth_seq,
        )

        by_seq = _index_records_by_auth_seq(residue_records)
        all_atoms = _flatten_atoms(residue_records)
        carbon_coords = wrapping_carbon_coords(all_atoms)

    # Pairwise distances (upper triangle)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    dehydron_pairs: set[tuple[int, int]] = set()
    packing_pairs: set[tuple[int, int]] = set()
    spoke_pairs: set[tuple[int, int]] = set()
    ribbon_pairs: set[tuple[int, int]] = set()
    coupling_pairs: dict[tuple[int, int], float] = {}
    # Per undirected pair: (rho_bond, d_no, ha_min) for HA aux scoring.
    dehydron_meta: dict[tuple[int, int], tuple[float, float, float]] = {}
    packing_ha_min: dict[tuple[int, int], float] = {}

    def _add_coupling(i: int, j: int, d: float) -> None:
        if not enable_coupling_edges:
            return
        key = _pair_key(i, j)
        w = coupling_strength(
            float(rho[i]),
            float(rho[j]),
            tau=tau,
            dist=d,
            coupling_cutoff=coupling_cutoff,
        )
        coupling_pairs[key] = max(coupling_pairs.get(key, 0.0), w)

    def _records_for(i: int, j: int) -> tuple[ResidueRecord | None, ResidueRecord | None]:
        if by_seq is None:
            return None, None
        return by_seq.get(int(seq[i])), by_seq.get(int(seq[j]))

    for i in range(n):
        for j in range(i + 1, n):
            d = float(dists[i, j])
            seq_sep = abs(int(seq[i]) - int(seq[j]))
            key = (i, j)

            if 1 <= seq_sep <= int(ribbon_max_sep):
                ribbon_pairs.add(key)

            is_hbond_cand = seq_sep <= HBOND_SEQ_CUTOFF and d < HBOND_SPATIAL_CUTOFF
            if is_hbond_cand:
                rho_bond = -1.0
                donor_used: ResidueRecord | None = None
                acceptor_used: ResidueRecord | None = None
                if by_seq is not None and all_atoms is not None:
                    donor = by_seq.get(int(seq[i]))
                    acceptor = by_seq.get(int(seq[j]))
                    if donor is not None and acceptor is not None:
                        rho_bond = compute_bond_wrapping_count(
                            donor,
                            acceptor,
                            all_atoms,
                            carbon_coords=carbon_coords,
                        )
                        if rho_bond >= 0:
                            donor_used, acceptor_used = donor, acceptor
                    if rho_bond < 0:
                        # try reverse donor/acceptor
                        donor_r = by_seq.get(int(seq[j]))
                        acceptor_r = by_seq.get(int(seq[i]))
                        if donor_r is not None and acceptor_r is not None:
                            rho_bond = compute_bond_wrapping_count(
                                donor_r,
                                acceptor_r,
                                all_atoms,
                                carbon_coords=carbon_coords,
                            )
                            if rho_bond >= 0:
                                donor_used, acceptor_used = donor_r, acceptor_r
                if rho_bond < 0:
                    rho_bond = 0.5 * (float(rho[i]) + float(rho[j]))
                if rho_bond < float(tau):
                    dehydron_pairs.add(key)
                    if ha_edges:
                        from science.dtie.v66.ha_edge_graph import (
                            donor_acceptor_no_distance,
                            heavy_atom_min_distance,
                        )

                        d_no = donor_acceptor_no_distance(
                            donor_used, acceptor_used, ca_fallback=d
                        )
                        ha_min = heavy_atom_min_distance(
                            *_records_for(i, j), ca_fallback=d
                        )
                        dehydron_meta[key] = (float(rho_bond), float(d_no), float(ha_min))
                    _add_coupling(i, j, d)
                    if dehydron_exclusivity:
                        continue  # dehydron wins over packing/spoke for this pair

            oi, oj = bool(ordered[i]), bool(ordered[j])
            if oi and oj:
                if ha_edges:
                    from science.dtie.v66.ha_edge_graph import (
                        heavy_atom_min_distance,
                        packing_contact_ha,
                    )

                    ha_min = heavy_atom_min_distance(
                        *_records_for(i, j), ca_fallback=d
                    )
                    if packing_contact_ha(d, ha_min_dist=ha_min, contact_cutoff=contact_cutoff):
                        packing_pairs.add(key)
                        packing_ha_min[key] = float(ha_min if ha_min >= 0 else d)
                elif d < float(contact_cutoff) and d > 0.1:
                    packing_pairs.add(key)
            elif oi != oj and d < float(contact_cutoff) and d > 0.1:
                # Spoke membership unchanged (Cα band only).
                spoke_pairs.add(key)
                _add_coupling(i, j, d)

            # Rim ↔ core propagation beyond strict contact when coupling enabled.
            if (
                enable_coupling_edges
                and d < float(coupling_cutoff)
                and d > 0.1
                and bool(ordered[i]) != bool(ordered[j])
            ):
                _add_coupling(i, j, d)

    src: list[int] = []
    dst: list[int] = []
    geo: list[list[float]] = []
    roles: list[int] = []

    for i, j in sorted(ribbon_pairs):
        _append_directed(src, dst, geo, roles, coords, i, j, ROLE_RIBBON)
    for i, j in sorted(dehydron_pairs):
        _append_directed(src, dst, geo, roles, coords, i, j, ROLE_DEHYDRON)
    for i, j in sorted(packing_pairs):
        _append_directed(src, dst, geo, roles, coords, i, j, ROLE_PACKING)
    for i, j in sorted(spoke_pairs):
        _append_directed(src, dst, geo, roles, coords, i, j, ROLE_SPOKE)
    for i, j in sorted(coupling_pairs):
        _append_directed(src, dst, geo, roles, coords, i, j, ROLE_COUPLING)

    if not src:
        # Degenerate: fall back to ribbon-only nearest sequence neighbors
        for i in range(max(0, n - 1)):
            _append_directed(src, dst, geo, roles, coords, i, i + 1, ROLE_RIBBON)

    edge_index = np.stack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)])
    e = len(roles)
    edge_attr = np.zeros((e, EDGE_ATTR_ROLE_DIM), dtype=np.float32)
    edge_attr[:, :GEO_DIM] = np.asarray(geo, dtype=np.float32)
    for k, role in enumerate(roles):
        edge_attr[k, GEO_DIM + int(role)] = 1.0
        if int(role) == ROLE_SPOKE:
            i, j = int(src[k]), int(dst[k])
            edge_attr[k, SPOKE_RHO_COL] = spoke_rho_weight(
                float(rho[i]), float(rho[j]), tau, scale=spoke_edge_scale
            )
        if int(role) == ROLE_COUPLING:
            i, j = int(src[k]), int(dst[k])
            key = _pair_key(i, j)
            edge_attr[k, COUPLING_STRENGTH_COL] = float(
                coupling_pairs.get(key, 0.0)
            )

    if ha_edges:
        from science.dtie.v66.ha_edge_graph import (
            append_ha_aux_columns,
            dehydron_strength,
            normalize_ha_min_dist,
            packing_strength,
        )

        ha_min_norm = np.zeros(e, dtype=np.float32)
        ha_str = np.zeros(e, dtype=np.float32)
        for k, role in enumerate(roles):
            i, j = int(src[k]), int(dst[k])
            key = _pair_key(i, j)
            ca_d = float(geo[k][3])
            if int(role) == ROLE_PACKING:
                ha_d = float(packing_ha_min.get(key, ca_d))
                ha_min_norm[k] = normalize_ha_min_dist(ha_d, scale=float(contact_cutoff))
                ha_str[k] = packing_strength(ha_d, scale=float(contact_cutoff))
            elif int(role) == ROLE_DEHYDRON:
                rho_b, d_no, ha_d = dehydron_meta.get(
                    key, (0.5 * (float(rho[i]) + float(rho[j])), ca_d, ca_d)
                )
                ha_min_norm[k] = normalize_ha_min_dist(
                    float(d_no), scale=float(HBOND_SPATIAL_CUTOFF)
                )
                ha_str[k] = dehydron_strength(float(rho_b), float(d_no), tau=tau)
        edge_attr = append_ha_aux_columns(
            edge_attr, ha_min_dist_norm=ha_min_norm, ha_strength=ha_str
        )
    return edge_index, edge_attr


def inject_dehydron_edge_barcode(
    edge_attr: np.ndarray,
    edge_index: np.ndarray,
    lookup: Mapping[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Write local barcode features onto dehydron role edges only."""
    if not lookup:
        return edge_attr
    out = np.asarray(edge_attr, dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    for k in range(out.shape[0]):
        if out[k, GEO_DIM + ROLE_DEHYDRON] < 0.5:
            continue
        key = _pair_key(int(src[k]), int(dst[k]))
        vec = lookup.get(key)
        if vec is None:
            continue
        out[k, DEHYDRON_BARCODE_COL : DEHYDRON_BARCODE_COL + EDGE_BARCODE_DIM] = np.asarray(
            vec, dtype=np.float32
        ).reshape(EDGE_BARCODE_DIM)
    return out


def attach_role_edge_graph(
    data: Data,
    ca_coords: torch.Tensor | np.ndarray,
    *,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
    force: bool = False,
    dehydron_edge_lookup: Mapping[tuple[int, int], np.ndarray] | None = None,
    enable_coupling_edges: bool = False,
    dehydron_exclusivity: bool = True,
    coupling_cutoff: float = COUPLING_CUTOFF,
    spoke_edge_scale: float = 1.0,
    ribbon_edge_scale: float = 1.0,
    ha_edges: bool = False,
) -> Data:
    """Replace ``edge_index`` / ``edge_attr`` with role-typed graph."""
    if not isinstance(data, Data):
        return data
    if getattr(data, "role_edge_graph", False) and not force:
        return data

    if isinstance(ca_coords, torch.Tensor):
        coords_np = ca_coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(ca_coords, dtype=np.float64)

    rho = data.rho if getattr(data, "rho", None) is not None else data.x[:, 0]
    if isinstance(rho, torch.Tensor):
        rho_np = rho.detach().cpu().numpy()
    else:
        rho_np = np.asarray(rho, dtype=np.float64)

    tau_flag = None
    if data.x is not None and data.x.size(-1) >= 2:
        tau_flag = data.x[:, 1].detach().cpu().numpy()

    edge_index_np, edge_attr_np = compute_role_edge_graph(
        coords_np,
        rho_np,
        tau_flag=tau_flag,
        residue_ids=residue_ids,
        residue_records=residue_records,
        tau=tau,
        enable_coupling_edges=enable_coupling_edges,
        dehydron_exclusivity=dehydron_exclusivity,
        coupling_cutoff=coupling_cutoff,
        spoke_edge_scale=spoke_edge_scale,
        ribbon_edge_scale=ribbon_edge_scale,
        ha_edges=ha_edges,
    )
    if dehydron_edge_lookup:
        edge_attr_np = inject_dehydron_edge_barcode(
            edge_attr_np,
            edge_index_np,
            dehydron_edge_lookup,
        )
    device = data.x.device if data.x is not None else torch.device("cpu")
    dtype = data.x.dtype if data.x is not None else torch.float32
    data.edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    data.edge_attr = torch.tensor(edge_attr_np, dtype=dtype, device=device)
    data.role_edge_graph = True  # type: ignore[attr-defined]
    data.ha_edge_graph = bool(ha_edges)  # type: ignore[attr-defined]
    # Counts for logging / tests
    oh = edge_attr_np[:, GEO_DIM:]
    data.role_edge_counts = {  # type: ignore[attr-defined]
        "packing": int(oh[:, ROLE_PACKING].sum() // 2),
        "dehydron": int(oh[:, ROLE_DEHYDRON].sum() // 2),
        "spoke": int(oh[:, ROLE_SPOKE].sum() // 2),
        "ribbon": int(oh[:, ROLE_RIBBON].sum() // 2),
        "coupling": int(oh[:, ROLE_COUPLING].sum() // 2),
    }
    data.role_coupling_edges = enable_coupling_edges  # type: ignore[attr-defined]
    data.role_spoke_edge_scale = float(spoke_edge_scale)  # type: ignore[attr-defined]
    data.role_ribbon_edge_scale = float(ribbon_edge_scale)  # type: ignore[attr-defined]
    return data


__all__ = [
    "ROLE_PACKING",
    "ROLE_DEHYDRON",
    "ROLE_SPOKE",
    "ROLE_RIBBON",
    "ROLE_COUPLING",
    "NUM_ROLE_RELATIONS",
    "SPOKE_RHO_COL",
    "COUPLING_STRENGTH_COL",
    "DEHYDRON_BARCODE_COL",
    "EDGE_ATTR_ROLE_DIM",
    "coupling_strength",
    "spoke_rho_weight",
    "compute_role_edge_graph",
    "inject_dehydron_edge_barcode",
    "attach_role_edge_graph",
    "resolve_residue_records_for_prot",
]
