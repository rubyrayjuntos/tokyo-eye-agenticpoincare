"""Heavy-atom–informed edge construction for ``ha_edges_v1`` (train-side).

Frozen construction rules (Part 0 / ablation §3) — 2026-07-19:

**Packing existence** (ordered↔ordered only; spoke/ribbon/dehydron membership
unchanged):

- Baseline band: ``0.1 < Cα–Cα ≤ 8.0 Å`` (same as chem-MVP ``CONTACT_CUTOFF``).
- HA rescue: if residue atoms are available, also accept
  ``0.1 < heavy_atom_min_dist ≤ 4.5 Å`` **and** ``Cα–Cα ≤ 12.0 Å``.
- Heavy-atom min-distance = min pairwise distance among non-H atoms of the two
  residues. Missing atoms → fall back to Cα-only (identical to baseline).

**Dehydron membership** — unchanged underwrap H-bond rule
(``rho_bond < τ`` via ``compute_bond_wrapping_count``).

**Dehydron strength** (aux, not a new relation)::

    deficit = clip((τ − rho_bond) / τ, 0, 1)
    prox    = clip((5.5 − d_NO) / (5.5 − 2.5), 0, 1)   # d_NO = ||N−O|| Å
    strength = 0.5 * deficit + 0.5 * prox

**Packing strength** (aux)::

    strength = clip(1 − ha_min / 8.0, 0, 1)   # ha_min falls back to Cα–Cα

**Aux column map** (appended after chem-MVP aux so one-hot / spoke / barcode
indices stay matched)::

    role+ha:  HA_MIN_DIST_NORM @ EDGE_ATTR_ROLE_DIM, HA_STRENGTH @ +1
    chem+ha:  HA_MIN_DIST_NORM @ EDGE_ATTR_CHEM_DIM, HA_STRENGTH @ +1

``GEO_DIM=4`` Cα Δxyz+d is unchanged (SE(3) stays Cα-based).
No new relation IDs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from science.dtie.common.dehydron_barcode_features import EDGE_BARCODE_DIM
from science.dtie.common.residue_features import ResidueRecord, TAU
from science.tokyo_eye.thermo_edge_features import GEO_DIM, HBOND_SPATIAL_CUTOFF

# Mirror role/chem widths without importing those modules (avoid cycles).
_NUM_ROLE = 5
_NUM_CHEM = 7
_ROLE_AUX = 1 + EDGE_BARCODE_DIM
_EDGE_ATTR_ROLE_DIM = GEO_DIM + _NUM_ROLE + _ROLE_AUX
_EDGE_ATTR_CHEM_DIM = GEO_DIM + _NUM_CHEM + _ROLE_AUX

# --- Frozen cutoffs (Part 0) -------------------------------------------------
HA_PACKING_MIN_DIST_A = 4.5
HA_PACKING_CA_MAX_A = 12.0
HA_NO_IDEAL_A = 2.5  # typical backbone H-bond lower scale for prox

HA_AUX_DIM = 2
HA_MIN_DIST_COL_ROLE = _EDGE_ATTR_ROLE_DIM
HA_STRENGTH_COL_ROLE = _EDGE_ATTR_ROLE_DIM + 1
EDGE_ATTR_ROLE_HA_DIM = _EDGE_ATTR_ROLE_DIM + HA_AUX_DIM

HA_MIN_DIST_COL_CHEM = _EDGE_ATTR_CHEM_DIM
HA_STRENGTH_COL_CHEM = _EDGE_ATTR_CHEM_DIM + 1
EDGE_ATTR_CHEM_HA_DIM = _EDGE_ATTR_CHEM_DIM + HA_AUX_DIM

# Non-regression δ for feeler rim (filed for post-train grade; not Part 0 measurable):
# rim enrichment holds on ≥ (baseline_hold_count − 1) Stage A-12 structures.
FEELER_RIM_DELTA_STRUCTURES = 1


def is_heavy_atom(atom_name: str, element: str) -> bool:
    el = (element or "").strip().upper()
    name = (atom_name or "").strip().upper()
    if el == "H" or name.startswith("H"):
        return False
    return True


def residue_heavy_coords(record: ResidueRecord) -> np.ndarray:
    """Stack non-H atom coordinates ``[M,3]`` (may be empty)."""
    coords = [
        np.asarray(a.coord, dtype=np.float64).reshape(3)
        for a in record.atoms
        if is_heavy_atom(a.atom_name, a.element)
    ]
    if not coords:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(coords, axis=0)


def heavy_atom_min_distance(
    rec_i: ResidueRecord | None,
    rec_j: ResidueRecord | None,
    *,
    ca_fallback: float | None = None,
) -> float:
    """Min pairwise heavy-atom distance; optional Cα fallback when atoms missing."""
    if rec_i is None or rec_j is None:
        return float(ca_fallback) if ca_fallback is not None else -1.0
    ci = residue_heavy_coords(rec_i)
    cj = residue_heavy_coords(rec_j)
    if ci.shape[0] == 0 or cj.shape[0] == 0:
        return float(ca_fallback) if ca_fallback is not None else -1.0
    # (Mi, Mj) pairwise
    d2 = np.sum((ci[:, None, :] - cj[None, :, :]) ** 2, axis=-1)
    return float(np.sqrt(d2.min()))


def packing_contact_ha(
    ca_dist: float,
    *,
    ha_min_dist: float | None,
    contact_cutoff: float = 8.0,
    ha_min_cutoff: float = HA_PACKING_MIN_DIST_A,
    ha_ca_max: float = HA_PACKING_CA_MAX_A,
) -> bool:
    """Frozen packing existence: Cα 8 Å band OR HA-min rescue within Cα max."""
    d = float(ca_dist)
    if d <= 0.1:
        return False
    if d <= float(contact_cutoff):
        return True
    if ha_min_dist is None or ha_min_dist < 0:
        return False
    return float(ha_min_dist) <= float(ha_min_cutoff) and d <= float(ha_ca_max)


def packing_strength(ha_or_ca_dist: float, *, scale: float = 8.0) -> float:
    return float(np.clip(1.0 - float(ha_or_ca_dist) / float(scale), 0.0, 1.0))


def dehydron_strength(
    rho_bond: float,
    d_no: float,
    *,
    tau: float = TAU,
    hbond_spatial: float = HBOND_SPATIAL_CUTOFF,
    no_ideal: float = HA_NO_IDEAL_A,
) -> float:
    """Strength from wrapping deficit + donor–acceptor heavy-atom proximity."""
    tau_f = max(float(tau), 1e-6)
    deficit = float(np.clip((tau_f - float(rho_bond)) / tau_f, 0.0, 1.0))
    denom = max(float(hbond_spatial) - float(no_ideal), 1e-6)
    prox = float(np.clip((float(hbond_spatial) - float(d_no)) / denom, 0.0, 1.0))
    return float(np.clip(0.5 * deficit + 0.5 * prox, 0.0, 1.0))


def donor_acceptor_no_distance(
    donor: ResidueRecord | None,
    acceptor: ResidueRecord | None,
    *,
    ca_fallback: float,
) -> float:
    if donor is None or acceptor is None:
        return float(ca_fallback)
    n_atom = donor.get_atom("N")
    o_atom = acceptor.get_atom("O")
    if n_atom is None or o_atom is None:
        return float(ca_fallback)
    return float(np.linalg.norm(n_atom.coord - o_atom.coord))


def normalize_ha_min_dist(dist: float, *, scale: float = 8.0) -> float:
    if dist < 0:
        return 0.0
    return float(np.clip(float(dist) / float(scale), 0.0, 1.0))


def append_ha_aux_columns(
    edge_attr: np.ndarray,
    *,
    ha_min_dist_norm: np.ndarray,
    ha_strength: np.ndarray,
) -> np.ndarray:
    """Append two HA aux columns; no-op if already present at expected width."""
    ea = np.asarray(edge_attr, dtype=np.float32)
    e = ea.shape[0]
    if ea.ndim != 2:
        raise ValueError(f"edge_attr must be [E,D], got {ea.shape}")
    if ea.shape[1] in (EDGE_ATTR_ROLE_HA_DIM, EDGE_ATTR_CHEM_HA_DIM):
        return ea
    if ea.shape[1] not in (_EDGE_ATTR_ROLE_DIM, _EDGE_ATTR_CHEM_DIM):
        raise ValueError(
            f"append_ha_aux_columns expected width {_EDGE_ATTR_ROLE_DIM} or "
            f"{_EDGE_ATTR_CHEM_DIM}, got {ea.shape[1]}"
        )
    if ha_min_dist_norm.shape[0] != e or ha_strength.shape[0] != e:
        raise ValueError("HA aux length must match edge count")
    extra = np.stack(
        [
            np.asarray(ha_min_dist_norm, dtype=np.float32).reshape(-1),
            np.asarray(ha_strength, dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    return np.concatenate([ea, extra], axis=1)


def ha_column_map(*, chem: bool) -> dict[str, int | str]:
    """Frozen aux column map for Part 0 writeup / ablation."""
    if chem:
        return {
            "layout": "geo4 | chem_onehot7 | spoke_rho1 | barcode5 | ha_min_dist_norm | ha_strength",
            "GEO_DIM": 4,
            "HA_MIN_DIST_NORM": HA_MIN_DIST_COL_CHEM,
            "HA_STRENGTH": HA_STRENGTH_COL_CHEM,
            "EDGE_ATTR_DIM": EDGE_ATTR_CHEM_HA_DIM,
            "packing_existence": (
                "ordered↔ordered AND (Cα≤8.0 OR (HA_min≤4.5 AND Cα≤12.0))"
            ),
            "dehydron_strength": "0.5*deficit(rho_bond,τ)+0.5*prox(d_NO)",
            "spoke_ribbon": "unchanged vs chem_mvp",
            "feeler_rim_delta_structures": FEELER_RIM_DELTA_STRUCTURES,
        }
    return {
        "layout": "geo4 | role_onehot5 | spoke_rho1 | barcode5 | ha_min_dist_norm | ha_strength",
        "GEO_DIM": 4,
        "HA_MIN_DIST_NORM": HA_MIN_DIST_COL_ROLE,
        "HA_STRENGTH": HA_STRENGTH_COL_ROLE,
        "EDGE_ATTR_DIM": EDGE_ATTR_ROLE_HA_DIM,
        "packing_existence": (
            "ordered↔ordered AND (Cα≤8.0 OR (HA_min≤4.5 AND Cα≤12.0))"
        ),
        "dehydron_strength": "0.5*deficit(rho_bond,τ)+0.5*prox(d_NO)",
        "spoke_ribbon": "unchanged vs chem_mvp",
        "feeler_rim_delta_structures": FEELER_RIM_DELTA_STRUCTURES,
    }


__all__ = [
    "HA_PACKING_MIN_DIST_A",
    "HA_PACKING_CA_MAX_A",
    "HA_AUX_DIM",
    "HA_MIN_DIST_COL_ROLE",
    "HA_STRENGTH_COL_ROLE",
    "EDGE_ATTR_ROLE_HA_DIM",
    "HA_MIN_DIST_COL_CHEM",
    "HA_STRENGTH_COL_CHEM",
    "EDGE_ATTR_CHEM_HA_DIM",
    "FEELER_RIM_DELTA_STRUCTURES",
    "is_heavy_atom",
    "residue_heavy_coords",
    "heavy_atom_min_distance",
    "packing_contact_ha",
    "packing_strength",
    "dehydron_strength",
    "donor_acceptor_no_distance",
    "normalize_ha_min_dist",
    "append_ha_aux_columns",
    "ha_column_map",
]
