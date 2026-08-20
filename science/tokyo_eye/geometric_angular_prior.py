"""Geometric angular prior for pre-MoE Poincaré disc grounding (v6.6 Fix 1).

Builds a per-residue θ prior from dehydron wrapping-deficit vectors blended with
a peptide-frame fallback, then applies a bounded residual:
``θ = θ_prior + α tanh(δ)`` with ``α = π/4`` and fixed handoff ``κ``.

Training-only / forward auxiliary — does not write through the Normalizer.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from science.dtie.common.residue_features import (
    TAU,
    ResidueRecord,
    compute_bond_wrapping_count,
    wrapping_carbon_coords,
)
from science.tokyo_eye.thermo_edge_features import (
    HBOND_SEQ_CUTOFF,
    HBOND_SPATIAL_CUTOFF,
    _flatten_atoms,
    _index_records_by_auth_seq,
    parse_auth_seq_ids,
)

DEFAULT_KAPPA = 1.0
DEFAULT_ALPHA = math.pi / 4.0
EPS = 1e-8


def protein_pca_plane(ca_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal in-plane axes (e1, e2) from Cα PCA."""
    ca = np.asarray(ca_coords, dtype=np.float64)
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError(f"ca_coords must be [N,3], got {ca.shape}")
    n = ca.shape[0]
    if n < 3:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    centered = ca - ca.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    e1 = vt[0]
    e2 = vt[1]
    e1 = e1 / (np.linalg.norm(e1) + EPS)
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + EPS)
    return e1, e2


def project_to_plane(v: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Project 3D vector into (e1, e2) plane as 2D coords."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([np.dot(v, e1), np.dot(v, e2)], dtype=np.float64)


def _unit_2d(v: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    if nrm < EPS:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / nrm


def peptide_direction_2d(
    record: ResidueRecord | None,
    ca_i: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    """Peptide fallback: Cα→C projected; else N–Cα axis; else [1,0]."""
    if record is not None:
        ca = record.get_atom("CA")
        c_atom = record.get_atom("C")
        n_atom = record.get_atom("N")
        ca_coord = np.asarray(ca.coord if ca is not None else ca_i, dtype=np.float64)
        if c_atom is not None:
            return _unit_2d(project_to_plane(c_atom.coord - ca_coord, e1, e2))
        if n_atom is not None:
            return _unit_2d(project_to_plane(ca_coord - n_atom.coord, e1, e2))
    return np.array([1.0, 0.0], dtype=np.float64)


def softplus_np(x: float) -> float:
    ax = abs(float(x))
    return float(math.log1p(math.exp(-ax)) + max(float(x), 0.0))


def compute_geometric_angular_prior(
    ca_coords: np.ndarray,
    *,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
    kappa: float = DEFAULT_KAPPA,
) -> dict[str, np.ndarray]:
    """Compute per-residue θ_prior and diagnostics.

    Returns dict with:
      - theta_prior: [N] radians in (-π, π]
      - u_prior: [N, 2] unit vectors in PCA plane
      - blend_mass: [N] m = ||Σ w û||
      - sigma: [N] 1 - exp(-m/κ)
      - peptide_u: [N, 2]
    """
    ca = np.asarray(ca_coords, dtype=np.float64)
    n = int(ca.shape[0])
    e1, e2 = protein_pca_plane(ca)
    kappa = max(float(kappa), EPS)

    peptide = np.zeros((n, 2), dtype=np.float64)
    by_seq: dict[int, ResidueRecord] | None = None
    all_atoms = None
    carbon_coords = None
    seq = parse_auth_seq_ids(residue_ids, n)
    if residue_records:
        by_seq = _index_records_by_auth_seq(residue_records)
        all_atoms = _flatten_atoms(residue_records)
        carbon_coords = wrapping_carbon_coords(all_atoms)

    for i in range(n):
        rec = by_seq.get(int(seq[i])) if by_seq is not None else None
        peptide[i] = peptide_direction_2d(rec, ca[i], e1, e2)

    v_dh = np.zeros((n, 2), dtype=np.float64)
    if by_seq is not None and all_atoms is not None:

        def _try_bond(
            donor_idx: int, acceptor_idx: int
        ) -> tuple[float, np.ndarray | None]:
            don = by_seq.get(int(seq[donor_idx]))
            acc = by_seq.get(int(seq[acceptor_idx]))
            if don is None or acc is None:
                return -1.0, None
            rb = compute_bond_wrapping_count(
                don,
                acc,
                all_atoms,
                carbon_coords=carbon_coords,
            )
            n_a = don.get_atom("N")
            o_a = acc.get_atom("O")
            if n_a is None or o_a is None:
                return float(rb), None
            return float(rb), np.asarray(o_a.coord - n_a.coord, dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(ca[i] - ca[j]))
                seq_sep = abs(int(seq[i]) - int(seq[j]))
                if not (seq_sep <= HBOND_SEQ_CUTOFF and d < HBOND_SPATIAL_CUTOFF):
                    continue
                candidates: list[tuple[int, int, float, np.ndarray]] = []
                rho_f, dir_f = _try_bond(i, j)
                rho_r, dir_r = _try_bond(j, i)
                if rho_f >= 0 and dir_f is not None and rho_f < float(tau):
                    candidates.append((i, j, rho_f, dir_f))
                if rho_r >= 0 and dir_r is not None and rho_r < float(tau):
                    candidates.append((j, i, rho_r, dir_r))
                for di, aj, rb, dvec in candidates:
                    u = _unit_2d(project_to_plane(dvec, e1, e2))
                    w = softplus_np(float(tau) - rb)
                    v_dh[di] += w * u
                    v_dh[aj] += w * u

    m = np.linalg.norm(v_dh, axis=1)
    sigma = 1.0 - np.exp(-m / kappa)
    v_blend = v_dh + ((1.0 - sigma)[:, None] * peptide)
    u_prior = np.stack([_unit_2d(v_blend[i]) for i in range(n)], axis=0)
    theta_prior = np.arctan2(u_prior[:, 1], u_prior[:, 0])

    return {
        "theta_prior": theta_prior.astype(np.float64),
        "u_prior": u_prior.astype(np.float64),
        "blend_mass": m.astype(np.float64),
        "sigma": sigma.astype(np.float64),
        "peptide_u": peptide.astype(np.float64),
    }


def attach_geometric_angular_prior(
    data: Data,
    ca_coords: torch.Tensor | np.ndarray,
    *,
    residue_ids: Sequence[str] | None = None,
    residue_records: Sequence[ResidueRecord] | None = None,
    tau: float = TAU,
    kappa: float = DEFAULT_KAPPA,
) -> Data:
    """Attach ``geom_theta_prior`` [N] to ``data`` for the forward pass."""
    if isinstance(ca_coords, torch.Tensor):
        ca_np = ca_coords.detach().cpu().numpy()
    else:
        ca_np = np.asarray(ca_coords, dtype=np.float64)
    prior = compute_geometric_angular_prior(
        ca_np,
        residue_ids=residue_ids,
        residue_records=residue_records,
        tau=tau,
        kappa=kappa,
    )
    device = data.x.device if hasattr(data, "x") and data.x is not None else "cpu"
    data.geom_theta_prior = torch.tensor(  # type: ignore[attr-defined]
        prior["theta_prior"], dtype=torch.float32, device=device
    )
    data.geom_blend_mass = torch.tensor(  # type: ignore[attr-defined]
        prior["blend_mass"], dtype=torch.float32, device=device
    )
    return data


class DiscAngularResidual(nn.Module):
    """Scalar residual δ for θ = θ_prior + α tanh(δ)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def apply_geometric_disc_prior(
    hyp_proj_2d: torch.Tensor,
    theta_prior: torch.Tensor,
    residual_delta: torch.Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-angle disc xy keeping radius; return (xy_new, theta, r)."""
    r = hyp_proj_2d.norm(dim=-1).clamp(min=EPS)
    delta = float(alpha) * torch.tanh(residual_delta)
    theta = theta_prior + delta
    xy = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    return xy, theta, r


def geometric_angular_fidelity_loss(
    disc_xy: torch.Tensor,
    theta_prior: torch.Tensor,
    *,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """r²-weighted cosine fidelity of disc θ to geometric prior."""
    r = disc_xy.norm(dim=-1).clamp(min=EPS)
    theta = torch.atan2(disc_xy[:, 1], disc_xy[:, 0])
    cos_term = torch.cos(theta - theta_prior)
    w = torch.clamp(r * r, max=1.0)
    loss = (w * (1.0 - cos_term)).mean() * float(scale)
    return {
        "geometric_angular_fidelity": loss,
        "geometric_angular_fidelity_mean_cos": cos_term.detach().mean(),
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_KAPPA",
    "DiscAngularResidual",
    "apply_geometric_disc_prior",
    "attach_geometric_angular_prior",
    "compute_geometric_angular_prior",
    "geometric_angular_fidelity_loss",
    "peptide_direction_2d",
    "protein_pca_plane",
    "project_to_plane",
]
