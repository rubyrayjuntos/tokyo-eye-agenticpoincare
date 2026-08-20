"""Path 2: explicit directionality objective (diam≤9 only).

Train-time surrogate for A→B ≠ B→A on graphs already within the ~6-hop MP
budget. Large-diameter structures are masked (loss contributes 0).

Grading on z-norm-on lineages must use forward knockout / passive geometry —
not Jacobian flow-influence (see JACOBIAN_ZNORM_DEFECT.md).
"""

from __future__ import annotations

from typing import Iterable

import torch

# Locked from asymmetry_vs_diameter.json (Stage A-12 role-edge diameters).
DIAM_LE_9_PASS_GROUP: frozenset[str] = frozenset(
    {
        "1UBQ",  # diam 5
        "1TEN",  # diam 6
        "1HHP",  # diam 7
        "1LYZ",  # diam 7
        "4OBE",  # diam 7
        "1TIM",  # diam 8
        "1MBN",  # diam 9
    }
)

ASYMMETRY_FLOOR = 0.05  # locked probe floor (where defined)
EPS = 1e-6


def pdb_in_diam_le9_pass_group(pdb_id: str | None) -> bool:
    """True when directionality loss may apply (hop budget already covers)."""
    if not pdb_id:
        return False
    return str(pdb_id).strip().upper() in DIAM_LE_9_PASS_GROUP


def pairwise_asymmetry_index(influence: torch.Tensor) -> torch.Tensor:
    """Mean |I_ab − I_ba| / (|I_ab| + |I_ba|) over off-diagonal finite pairs.

    Matches ``jacobian_flow_influence.asymmetry_index`` (torch, differentiable).
    Returns a scalar tensor; NaN-safe via EPS floors (never zero-fills pairs).
    """
    if influence.ndim != 2 or influence.shape[0] != influence.shape[1]:
        raise ValueError(f"influence must be square, got {tuple(influence.shape)}")
    n = int(influence.shape[0])
    if n < 2:
        return influence.new_tensor(float("nan"))

    # Upper triangle pairs only (each unordered pair once).
    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=influence.device)
    i_ab = influence[idx_i, idx_j]
    i_ba = influence[idx_j, idx_i]
    denom = i_ab.abs() + i_ba.abs()
    ok = denom >= EPS
    if not bool(ok.any()):
        return influence.new_tensor(float("nan"))
    vals = (i_ab - i_ba).abs() / denom.clamp_min(EPS)
    return vals[ok].mean()


def source_norm_cosine_influence(encoder_h: torch.Tensor) -> torch.Tensor:
    """Cheap directed influence proxy on trunk embeddings (train surrogate).

    ``I(a→b) = ‖h_a‖ · softplus(⟨ĥ_a, ĥ_b⟩)``.

    Source-norm weighting breaks ``I(a→b) = I(b→a)`` whenever norms differ;
    cosine couples direction of states. Not a Jacobian / knockout substitute —
    grading must use those instruments separately.
    """
    if encoder_h.ndim != 2:
        raise ValueError(f"encoder_h must be [N,D], got {tuple(encoder_h.shape)}")
    n = int(encoder_h.shape[0])
    if n < 2:
        return encoder_h.new_zeros(n, n)

    norms = encoder_h.norm(dim=-1).clamp_min(EPS)  # [N]
    hat = encoder_h / norms.unsqueeze(-1)
    cos = hat @ hat.T  # [N,N]
    # softplus(cos) ∈ (log2, +∞) roughly for cos∈[-1,1]
    return norms.unsqueeze(1) * torch.nn.functional.softplus(cos)


def directionality_asym_loss(
    encoder_h: torch.Tensor,
    *,
    coeff: float,
    eligible: bool,
) -> dict[str, torch.Tensor]:
    """Maximize pairwise asymmetry on diam≤9; zero contribution otherwise.

    ``L = λ · (1 − asym)`` when eligible and asym is finite; else 0.
    """
    device = encoder_h.device
    zero = encoder_h.new_tensor(0.0)
    if not eligible or float(coeff) <= 0.0:
        return {
            "directionality_asym": zero,
            "directionality_asym_raw": zero,
            "directionality_asym_index": encoder_h.new_tensor(float("nan")),
        }

    influence = source_norm_cosine_influence(encoder_h)
    asym = pairwise_asymmetry_index(influence)
    if not torch.isfinite(asym):
        return {
            "directionality_asym": zero,
            "directionality_asym_raw": zero,
            "directionality_asym_index": asym.detach(),
        }

    # Maximize asym toward 1 (fully antisymmetric pairs).
    raw = (1.0 - asym).clamp_min(0.0)
    loss = float(coeff) * raw
    return {
        "directionality_asym": loss,
        "directionality_asym_raw": raw.detach(),
        "directionality_asym_index": asym.detach(),
    }


def filter_proteins_diam_le9(proteins: Iterable[dict]) -> list[dict]:
    """Keep only Stage A pass-group (diam≤9) entries."""
    out: list[dict] = []
    for prot in proteins:
        pid = str(prot.get("pdb_id") or "").strip().upper()
        if pid in DIAM_LE_9_PASS_GROUP:
            out.append(prot)
    return out
