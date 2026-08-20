"""Training-only thermodynamic edge attributes for v6.6 (Cα topology unchanged).

Augments ``edge_attr`` ``[Δx, Δy, Δz, d]`` with affinity channels for
``EquivariantConvThermo``. Does **not** change GraphBuilder / Normalizer.

For candidate backbone H-bonds, ``rho_bond`` uses SSOT
``compute_bond_wrapping_count`` (donor N → acceptor O midpoint, same carbon
filter as node ρ). Generic contacts fall back to mean endpoint node ρ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import (
    TAU,
    ResidueRecord,
    compute_bond_wrapping_count,
    parse_residue_records_from_pdb_chain,
    wrapping_carbon_coords,
)

# Match GraphBuilder persistence heuristic (not angle-validated H-bonds).
HBOND_SEQ_CUTOFF = 5
HBOND_SPATIAL_CUTOFF = 5.5

EDGE_TYPE_GENERIC = 0
EDGE_TYPE_WRAPPED_HBOND = 1
EDGE_TYPE_DEHYDRON = 2

GEO_DIM = 4
# thermo_weight, rho_bond_norm, one-hot type × 3, used_mean_fallback
THERMO_DIM = 6
EDGE_ATTR_THERMO_DIM = GEO_DIM + THERMO_DIM


def parse_auth_seq_ids(residue_ids: Sequence[str] | None, n: int) -> np.ndarray:
    """Auth sequence indices from ``A:32:``-style ids; fallback to 0..n-1."""
    if residue_ids is None or len(residue_ids) != n:
        return np.arange(n, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    for i, rid in enumerate(residue_ids):
        try:
            out[i] = int(str(rid).split(":")[1])
        except (IndexError, ValueError):
            out[i] = i
    return out


def _index_records_by_auth_seq(
    records: Sequence[ResidueRecord],
) -> dict[int, ResidueRecord]:
    return {int(r.residue_index): r for r in records}


def _flatten_atoms(records: Sequence[ResidueRecord]) -> list:
    atoms: list = []
    for r in records:
        atoms.extend(r.atoms)
    return atoms


def compute_thermo_edge_attr(
    edge_index: torch.Tensor | np.ndarray,
    edge_attr_geo: torch.Tensor | np.ndarray,
    rho: torch.Tensor | np.ndarray,
    *,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
) -> np.ndarray:
    """Build ``[E, THERMO_DIM]`` thermo channels for existing Cα edges."""
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].detach().cpu().numpy().astype(np.int64)
        dst = edge_index[1].detach().cpu().numpy().astype(np.int64)
    else:
        src = np.asarray(edge_index[0], dtype=np.int64)
        dst = np.asarray(edge_index[1], dtype=np.int64)

    if isinstance(edge_attr_geo, torch.Tensor):
        geo = edge_attr_geo.detach().cpu().numpy()
    else:
        geo = np.asarray(edge_attr_geo, dtype=np.float64)
    if geo.ndim != 2 or geo.shape[1] < GEO_DIM:
        raise ValueError(f"edge_attr_geo must be [E,{GEO_DIM}+], got {geo.shape}")

    if isinstance(rho, torch.Tensor):
        rho_np = rho.detach().cpu().numpy().astype(np.float64).reshape(-1)
    else:
        rho_np = np.asarray(rho, dtype=np.float64).reshape(-1)

    n = int(rho_np.shape[0])
    seq = parse_auth_seq_ids(residue_ids, n)
    dist = geo[:, 3].astype(np.float64)
    e = src.shape[0]

    by_seq: dict[int, ResidueRecord] | None = None
    carbon_coords = None
    all_atoms: list | None = None
    if residue_records:
        by_seq = _index_records_by_auth_seq(residue_records)
        all_atoms = _flatten_atoms(residue_records)
        carbon_coords = wrapping_carbon_coords(all_atoms)

    thermo = np.zeros((e, THERMO_DIM), dtype=np.float32)
    for k in range(e):
        i, j = int(src[k]), int(dst[k])
        d = float(dist[k])
        seq_sep = abs(int(seq[i]) - int(seq[j]))
        mean_rho = 0.5 * (float(rho_np[i]) + float(rho_np[j]))
        is_potential_hbond = seq_sep <= HBOND_SEQ_CUTOFF and d < HBOND_SPATIAL_CUTOFF
        used_fallback = 1.0

        if is_potential_hbond and by_seq is not None and all_atoms is not None:
            donor = by_seq.get(int(seq[i]))
            acceptor = by_seq.get(int(seq[j]))
            rho_bond = -1.0
            if donor is not None and acceptor is not None:
                rho_bond = compute_bond_wrapping_count(
                    donor,
                    acceptor,
                    all_atoms,
                    carbon_coords=carbon_coords,
                )
            if rho_bond < 0:
                rho_bond = mean_rho
                used_fallback = 1.0
            else:
                used_fallback = 0.0
            is_dehydron = rho_bond < float(tau)
            edge_type = EDGE_TYPE_DEHYDRON if is_dehydron else EDGE_TYPE_WRAPPED_HBOND
            thermo_weight = float(1.0 / (1.0 + np.exp((rho_bond - float(tau)) / 3.0)))
        elif is_potential_hbond:
            rho_bond = mean_rho
            is_dehydron = rho_bond < float(tau)
            edge_type = EDGE_TYPE_DEHYDRON if is_dehydron else EDGE_TYPE_WRAPPED_HBOND
            thermo_weight = float(1.0 / (1.0 + np.exp((rho_bond - float(tau)) / 3.0)))
            used_fallback = 1.0
        else:
            rho_bond = mean_rho
            edge_type = EDGE_TYPE_GENERIC
            thermo_weight = float(np.clip((rho_bond - 8.0) / 10.0, 0.0, 1.0))
            used_fallback = 1.0

        thermo[k, 0] = thermo_weight
        thermo[k, 1] = float(np.clip(rho_bond / float(tau), 0.0, 3.0))
        thermo[k, 2 + edge_type] = 1.0
        thermo[k, 5] = used_fallback

    return thermo


def resolve_residue_records_for_prot(
    prot: dict,
    *,
    pdb_dir: Path | str | None = None,
) -> list[ResidueRecord] | None:
    """Load/cache ``ResidueRecord`` atoms for bond wrapping (training-only)."""
    cached = prot.get("residue_records")
    if cached is not None:
        return list(cached)
    pdb_id = prot.get("pdb_id")
    chain = prot.get("chain") or "A"
    if not pdb_id:
        return None
    from experiments.training.v66 import _data as training_data

    if pdb_dir is not None:
        cache = Path(pdb_dir)
    elif prot.get("pdb_dir") is not None:
        cache = Path(prot["pdb_dir"])
    else:
        cache = Path("/tmp/dtie_pdb_cache")
    try:
        pdb_path = training_data._download_pdb(str(pdb_id).upper(), cache)
    except Exception:
        return None
    records = parse_residue_records_from_pdb_chain(pdb_path, str(chain))
    prot["residue_records"] = records
    return records


def attach_thermo_edge_features(
    data: Data,
    *,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
    force: bool = False,
) -> Data:
    """Append thermo channels to ``data.edge_attr`` (idempotent unless ``force``)."""
    if not isinstance(data, Data):
        return data
    if data.edge_index is None or data.edge_attr is None:
        return data

    attr = data.edge_attr
    if attr.size(-1) >= EDGE_ATTR_THERMO_DIM and not force:
        data.thermo_edge_features = True  # type: ignore[attr-defined]
        return data

    geo = attr[:, :GEO_DIM] if attr.size(-1) >= GEO_DIM else attr
    rho = data.rho if getattr(data, "rho", None) is not None else data.x[:, 0]
    thermo = compute_thermo_edge_attr(
        data.edge_index,
        geo,
        rho,
        residue_ids=residue_ids,
        residue_records=residue_records,
        tau=tau,
    )
    thermo_t = torch.tensor(thermo, dtype=geo.dtype, device=geo.device)
    data.edge_attr = torch.cat([geo.to(dtype=geo.dtype), thermo_t], dim=-1)
    data.thermo_edge_features = True  # type: ignore[attr-defined]
    data.thermo_bond_wrapping = bool(residue_records)  # type: ignore[attr-defined]
    data.thermo_fallback_frac = float(thermo[:, 5].mean())  # type: ignore[attr-defined]
    return data
