"""
V6 Loss Function — Phase-Aware Combined Loss with Asymmetric Capacity Regularization
======================================================================================
Eidetix Bio | 2026-05

Combines all loss terms for v6 training with phase-aware coefficients:
  - cone_loss: Radial depth regression (→ RadialHead only)
  - cone_depth_anticollapse: dist0 spread penalty (→ app / shell probes)
  - shell_correlation: SASA + disc gradient alignment (probes 1 & 4)
  - neighborhood_consistency: Spatial attraction/repulsion (→ both heads)
  - angular_diversity: Penalize angular collapse (→ AngularHead only)
  - domain_separation_2d/3d: Push domain centroids apart (→ AngularHead only)
  - evidential: Uncertainty calibration
  - capacity_loss: Asymmetric capacity penalty (penalizes starvation only)
  - routing_entropy: Monitoring metric H(f̄) (not a loss term, just tracked)
  - routing_entropy_mean_residue: Optional sparsity term λ * mean_i H(p_i)

Phase-aware behavior:
  Phase 1: balance_coeff=0.1, no expert dropout → strong balance pressure
  Phase 2: balance_coeff=0.001, asymmetric capacity loss → experts specialize
  Phase 3: gate frozen, experts fine-tuned → routing locked
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from science.dtie.v5.gnn.model import (
    angular_diversity_loss,
    cone_loss_v5,
    domain_separation_loss_2d,
    domain_separation_loss_3d,
    evidential_regression_loss,
    neighborhood_consistency_loss,
)
from science.training.routing_metrics import (
    routing_load_ceiling_penalty,
    routing_load_floor_penalty,
)


def majority_committed_share_hinge(
    expert_weights: torch.Tensor,
    *,
    tau: float = 0.56,
    commit_thr: float = 0.60,
    min_committed: int = 20,
    dehydron: torch.Tensor | None = None,
    core_only: bool = False,
) -> torch.Tensor:
    """STE hinge on within-structure committed hard majority share.

    Forward value equals hard-assignment share of the majority expert among
    the eligible set. Gradients flow through soft weights on the stopgrad
    majority mask only.

    When ``core_only=True``, eligible = committed ∧ (dehydron == 0). Direct
    pressure on dehydron=1 residues is excluded (zero grad from this term);
    purity floors must still catch indirect dilution onto minority experts.

    Returns unscaled ``ReLU(share − tau)²`` (0 when |eligible| < min_committed
    or share ≤ tau).
    """
    w = expert_weights.float()
    if w.ndim != 2 or w.shape[0] == 0:
        return w.new_zeros(())
    n_experts = int(w.shape[-1])
    max_p = w.max(dim=-1).values
    commit = max_p >= float(commit_thr)
    if core_only:
        if dehydron is None:
            return w.new_zeros(())
        dh = dehydron.reshape(-1).float()[: w.shape[0]]
        eligible = commit & (dh <= 0.0)
    else:
        eligible = commit
    n_e = int(eligible.sum().item())
    if n_e < int(min_committed):
        return w.new_zeros(())
    hard = w.argmax(dim=-1)
    elig_idx = eligible.nonzero(as_tuple=False).view(-1)
    counts = torch.bincount(hard[elig_idx], minlength=n_experts).float()
    e_star = int(counts.argmax().item())
    maj = (eligible & (hard == e_star)).detach()
    # Straight-through: forward hard one-hot, backward soft weights.
    hard_oh = F.one_hot(hard, num_classes=n_experts).to(dtype=w.dtype)
    st = hard_oh + (w - w.detach())
    # Maj-restricted share of e*: forward == hard majority share among eligible.
    share = (st[:, e_star] * maj.float()).sum() / float(n_e)
    excess = torch.relu(share - float(tau))
    return excess * excess


def core_majority_committed_share_hinge(
    expert_weights: torch.Tensor,
    dehydron: torch.Tensor,
    *,
    tau: float = 0.56,
    commit_thr: float = 0.60,
    min_committed: int = 20,
) -> torch.Tensor:
    """Core-only STE majority hinge (dehydron=0 eligible set)."""
    return majority_committed_share_hinge(
        expert_weights,
        tau=tau,
        commit_thr=commit_thr,
        min_committed=min_committed,
        dehydron=dehydron,
        core_only=True,
    )


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable Pearson r; returns 0 when either vector has zero variance."""
    a = a.squeeze(-1) if a.dim() > 1 else a
    b = b.squeeze(-1) if b.dim() > 1 else b
    if a.numel() < 3:
        return torch.zeros((), device=a.device, dtype=a.dtype)
    ac = a - a.mean()
    bc = b - b.mean()
    denom = ac.norm() * bc.norm() + 1e-8
    return (ac * bc).sum() / denom


def v3_aleatoric_shaping_loss(
    aleatoric: torch.Tensor,
    target_dehydron: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    w_var_penalty: float = 2.8,
    w_aleatoric_hinge: float = 4.8,
    hinge_target: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """v3 var_penalty + aleatoric_hinge on regular residues, train_mask only (G4)."""
    ale = aleatoric.squeeze(-1) if aleatoric.dim() > 1 else aleatoric
    dehyd = target_dehydron.squeeze(-1) if target_dehydron.dim() > 1 else target_dehydron
    regularity = (1.0 - dehyd).clamp(min=0.0, max=1.0)
    mask = train_mask.float() * regularity
    denom = mask.sum().clamp(min=1.0)
    var_penalty = (mask * ale).sum() / denom
    hinge_t = max(float(hinge_target), 1e-6)
    aleatoric_hinge = (mask * torch.relu(ale - hinge_t).pow(2)).sum() / denom
    total = w_var_penalty * var_penalty + w_aleatoric_hinge * aleatoric_hinge
    return {
        "v3_aleatoric_shaping_total": total,
        "var_penalty": var_penalty,
        "aleatoric_hinge": aleatoric_hinge,
    }


def resolve_cone_target_depth(
    *,
    target_rho: torch.Tensor | None = None,
    target_dehydron: torch.Tensor | None = None,
    mode: str = "rho_wrap",
) -> torch.Tensor:
    """Map supervised labels → continuous radial depth target in [0, 1].

    Modes:
      - ``rho_wrap`` (legacy): high ρ → high depth
      - ``tau_dehydron_rim``: binary τ=1 → rim
      - ``rho_rim``: continuous underwrapping — low ρ → rim, high ρ → center
    """
    if mode == "tau_dehydron_rim":
        if target_dehydron is None:
            raise ValueError("target_dehydron required for tau_dehydron_rim cone mode")
        return target_dehydron.squeeze(-1).clamp(0.0, 1.0)
    if mode == "rho_rim":
        if target_rho is None:
            raise ValueError("target_rho required for rho_rim cone mode")
        # Invert legacy polarity: buried/high-ρ → center (low depth).
        return (1.0 - (target_rho.squeeze(-1) / 30.0).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    if target_rho is None:
        raise ValueError(f"target_rho required for cone mode {mode!r}")
    return (target_rho.squeeze(-1) / 30.0).clamp(0.0, 1.0)


def cone_alignment_loss(
    radial_depth: torch.Tensor,
    *,
    target_rho: torch.Tensor | None = None,
    target_dehydron: torch.Tensor | None = None,
    mode: str = "rho_wrap",
) -> torch.Tensor:
    """
    Radial cone supervision.

    rho_wrap (legacy): high ρ (buried wrap) → high depth via ρ/30.
    tau_dehydron_rim (MASTER cold): τ=1 (dehydron) → rim (high depth).
    rho_rim (feeler continuous): low ρ → rim, high ρ → center (no binary τ).
    """
    if mode == "rho_wrap":
        if target_rho is None:
            raise ValueError("target_rho required for rho_wrap cone mode")
        return cone_loss_v5(radial_depth, target_rho)

    target_depth = resolve_cone_target_depth(
        target_rho=target_rho,
        target_dehydron=target_dehydron,
        mode=mode,
    )
    pred = radial_depth.squeeze(-1)
    correlation = _pearson_corr(pred, target_depth)
    corr_loss = 1.0 - correlation
    depth_std = pred.std()
    variance_penalty = torch.relu(0.15 - depth_std) * 5.0
    return corr_loss + variance_penalty


def _zscore_batch(x: torch.Tensor) -> torch.Tensor:
    """Per-structure z-score (one protein per forward pass)."""
    x = x.squeeze(-1) if x.dim() > 1 else x
    s = x.std()
    if float(s) < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / s


def _ols_residual_detached(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Residual of y after OLS on z; regression coefficients detached from autograd."""
    y = y.squeeze(-1) if y.dim() > 1 else y
    z = z.squeeze(-1) if z.dim() > 1 else z
    n = y.numel()
    if n < 3:
        return torch.zeros_like(y)
    ones = torch.ones(n, 1, device=y.device, dtype=y.dtype)
    z_col = z.unsqueeze(-1)
    X = torch.cat([ones, z_col], dim=1)
    with torch.no_grad():
        beta = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution
    return y - (X @ beta).squeeze(-1)


def epi_ale_decorrelation_loss(
    epistemic: torch.Tensor,
    aleatoric: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Penalize Pearson correlation between epistemic and aleatoric (head coupling)."""
    device = epistemic.device
    dtype = epistemic.dtype
    epi = epistemic.squeeze(-1) if epistemic.dim() > 1 else epistemic
    ale = aleatoric.squeeze(-1) if aleatoric.dim() > 1 else aleatoric
    if epi.numel() < 3:
        z = torch.zeros((), device=device, dtype=dtype)
        return {"epi_ale_decorrelation": z, "r_epi_ale": z}
    r = _pearson_corr(epi, ale)
    return {"epi_ale_decorrelation": r**2, "r_epi_ale": r.detach()}


def epistemic_decoupling_loss(
    epistemic: torch.Tensor,
    cone_depth: torch.Tensor,
    sasa: torch.Tensor,
    b_factor_ca: torch.Tensor,
    b_factor_present: torch.Tensor | None = None,
    *,
    min_epi_std: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """
    Step 4 epistemic–depth decoupling (B-factor residual + SASA partial penalty).

    Depth partialling uses detached OLS (matches eval partial_corr definition).
    SASA penalty uses squared normalized covariance on depth residuals.
    """
    device = epistemic.device
    dtype = epistemic.dtype
    depth = cone_depth.squeeze(-1)
    epi = epistemic.squeeze(-1) if epistemic.dim() > 1 else epistemic
    sasa_flat = sasa.squeeze(-1) if sasa.dim() > 1 else sasa
    bf = b_factor_ca.squeeze(-1) if b_factor_ca.dim() > 1 else b_factor_ca

    if b_factor_present is not None:
        present = (
            b_factor_present.squeeze(-1).bool()
            if b_factor_present.dim() > 1
            else b_factor_present.bool()
        )
        if int(present.sum()) < 5:
            z = torch.zeros((), device=device, dtype=dtype)
            return {
                "epistemic_decoupling_total": z,
                "epistemic_bf_align": z,
                "epistemic_sasa_pen": z,
                "epistemic_anticollapse": z,
                "r_epi_bf_resid": z,
                "partial_epi_sasa_given_depth": z,
            }
        epi = epi[present]
        depth = depth[present]
        sasa_flat = sasa_flat[present]
        bf = bf[present]

    bf_z = _zscore_batch(bf)
    bf_resid = _ols_residual_detached(bf_z, depth)
    epi_resid = _ols_residual_detached(epi, depth)
    sasa_resid = _ols_residual_detached(sasa_flat, depth)

    r_epi_bf = _pearson_corr(epi, bf_resid)
    bf_align_loss = 1.0 - r_epi_bf

    std_epi = epi_resid.std()
    std_sasa = sasa_resid.std()
    if float(std_epi) < 1e-8 or float(std_sasa) < 1e-8:
        partial_epi_sasa = torch.zeros((), device=device, dtype=dtype)
        sasa_pen_loss = torch.zeros((), device=device, dtype=dtype)
    else:
        epi_c = epi_resid - epi_resid.mean()
        sasa_c = sasa_resid - sasa_resid.mean()
        partial_epi_sasa = (epi_c * sasa_c).sum() / (
            std_epi * std_sasa * max(epi_resid.numel() - 1, 1) + 1e-8
        )
        sasa_pen_loss = partial_epi_sasa**2

    epi_var_loss = torch.relu(min_epi_std - epi.std()) * 5.0

    return {
        "epistemic_decoupling_total": bf_align_loss + sasa_pen_loss + epi_var_loss,
        "epistemic_bf_align": bf_align_loss,
        "epistemic_sasa_pen": sasa_pen_loss,
        "epistemic_anticollapse": epi_var_loss,
        "r_epi_bf_resid": r_epi_bf.detach(),
        "partial_epi_sasa_given_depth": partial_epi_sasa.detach(),
    }


def cone_depth_anticollapse_loss(
    cone_depth: torch.Tensor,
    *,
    min_std: float = 0.02,
) -> torch.Tensor:
    """Penalize flat dist0 (cone_depth) — the signal the app and shell probes read."""
    depth = cone_depth.squeeze(-1)
    return torch.relu(min_std - depth.std()) * 10.0


def shell_correlation_topology_loss(
    cone_depth: torch.Tensor,
    hyp_proj_2d: torch.Tensor,
    *,
    proj_depth_weight: float = 1.0,
    disc_spread_weight: float = 0.5,
    disc_spread_min_std: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Disc–depth coupling without SASA (topology / MASTER cold lineage)."""
    depth = cone_depth.squeeze(-1)
    disc_r = hyp_proj_2d.norm(dim=-1)

    r_proj_depth = _pearson_corr(disc_r, depth)
    proj_depth_loss = 1.0 - r_proj_depth
    disc_std = disc_r.std()
    disc_spread_loss = torch.relu(disc_spread_min_std - disc_std) * 5.0

    total = proj_depth_weight * proj_depth_loss + disc_spread_weight * disc_spread_loss
    z = torch.zeros((), device=depth.device, dtype=depth.dtype)
    return {
        "shell_corr_total": total,
        "shell_corr_depth_sasa": z,
        "shell_corr_epi_sasa": z,
        "shell_corr_proj_depth": proj_depth_loss,
        "shell_corr_disc_spread": disc_spread_loss,
        "shell_corr_disc_sasa": z,
        "shell_r_depth_sasa": z,
        "shell_r_epi_sasa": z,
        "shell_r_proj_depth": r_proj_depth.detach(),
        "shell_r_disc_sasa": z,
        "disc_r_std": disc_std.detach(),
    }


def shell_correlation_loss(
    cone_depth: torch.Tensor,
    epistemic: torch.Tensor,
    sasa: torch.Tensor,
    hyp_proj_2d: torch.Tensor,
    *,
    depth_sasa_weight: float = 1.0,
    epi_sasa_weight: float = 0.5,
    proj_depth_weight: float = 1.0,
    disc_spread_weight: float = 0.5,
    disc_sasa_weight: float = 0.3,
    disc_spread_min_std: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    Maximize shell probe correlations (minimize 1 - r).

    Targets probe 1 (depth/epi vs SASA), probe 4 (|proj| vs depth),
    and disc spread / |proj| vs SASA for central-clustering on the 2D disc.
    """
    depth = cone_depth.squeeze(-1)
    epi = epistemic.squeeze(-1) if epistemic.dim() > 1 else epistemic
    sasa_flat = sasa.squeeze(-1) if sasa.dim() > 1 else sasa
    disc_r = hyp_proj_2d.norm(dim=-1)

    r_depth_sasa = _pearson_corr(depth, sasa_flat)
    r_epi_sasa = _pearson_corr(epi, sasa_flat)
    r_proj_depth = _pearson_corr(disc_r, depth)
    r_disc_sasa = _pearson_corr(disc_r, sasa_flat)

    depth_sasa_loss = 1.0 - r_depth_sasa
    epi_sasa_loss = 1.0 - r_epi_sasa
    proj_depth_loss = 1.0 - r_proj_depth
    disc_sasa_loss = 1.0 - r_disc_sasa
    disc_std = disc_r.std()
    disc_spread_loss = torch.relu(disc_spread_min_std - disc_std) * 5.0

    total = (
        depth_sasa_weight * depth_sasa_loss
        + epi_sasa_weight * epi_sasa_loss
        + proj_depth_weight * proj_depth_loss
        + disc_spread_weight * disc_spread_loss
        + disc_sasa_weight * disc_sasa_loss
    )
    return {
        "shell_corr_total": total,
        "shell_corr_depth_sasa": depth_sasa_loss,
        "shell_corr_epi_sasa": epi_sasa_loss,
        "shell_corr_proj_depth": proj_depth_loss,
        "shell_corr_disc_spread": disc_spread_loss,
        "shell_corr_disc_sasa": disc_sasa_loss,
        "shell_r_depth_sasa": r_depth_sasa.detach(),
        "shell_r_epi_sasa": r_epi_sasa.detach(),
        "shell_r_proj_depth": r_proj_depth.detach(),
        "shell_r_disc_sasa": r_disc_sasa.detach(),
        "disc_r_std": disc_std.detach(),
    }


def disc_depth_scale_loss(
    cone_depth: torch.Tensor,
    hyp_proj_2d: torch.Tensor,
    *,
    target_radius: float = 0.45,
) -> torch.Tensor:
    """
    Push |hyp_proj_2d| to match per-batch normalized cone_depth.

    Trains MobiusLinear(hyp_proj_head_2d) directly — angular_coeff does not
    reach this head because hyp_proj is computed from x_routed_hyp, not x_hyp.
    """
    depth = cone_depth.squeeze(-1)
    lo, hi = depth.min(), depth.max()
    if float(hi - lo) < 1e-6:
        depth_norm = torch.zeros_like(depth)
    else:
        depth_norm = (depth - lo) / (hi - lo)
    target_r = depth_norm * target_radius
    disc_r = hyp_proj_2d.norm(dim=-1)
    return F.mse_loss(disc_r, target_r)


def disc_occupancy_loss(
    hyp_proj_2d: torch.Tensor,
    *,
    min_sigma_ratio: float = 0.35,
    scale: float = 5.0,
    min_disc_r_mean: float = 0.05,
    collapse_scale: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """Penalize rank-1 disc streaks (centered σ₂/σ₁ floor on hyp_projections_2d)."""
    from science.training.disc_occupancy import disc_occupancy_loss as _occupancy_loss

    return _occupancy_loss(
        hyp_proj_2d,
        min_sigma_ratio=min_sigma_ratio,
        scale=scale,
        min_disc_r_mean=min_disc_r_mean,
        collapse_scale=collapse_scale,
    )


def disc_path_align_loss(
    hyp_proj_2d: torch.Tensor,
    legacy_teacher: torch.Tensor,
) -> torch.Tensor:
    """MSE new-path disc coords toward legacy post-routing teacher (detached)."""
    return F.mse_loss(hyp_proj_2d, legacy_teacher.detach())


def prototype_gram_logdet_hinge(
    gram_logdet: torch.Tensor,
    *,
    tau_logdet: float = -1.15,
) -> torch.Tensor:
    """Saturating full-bank Gram volume hinge: ReLU(τ − logdet)².

    Unit-row Gram logdet is bounded above near 0 (orthogonality). The hinge
    zeros once logdet ≥ τ (registered bank floor), so pressure does not keep
    pushing toward full orthogonality past the success criteria.
    """
    return torch.relu(
        float(tau_logdet) - gram_logdet
    ).pow(2)


def weighted_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-imbalanced BCE with optional per-residue mask."""
    if mask is not None and not mask.any():
        return logits.new_zeros(())
    if mask is not None:
        logits = logits[mask]
        targets = targets[mask]
    if logits.numel() == 0:
        return logits.new_zeros(())
    pos = targets.sum()
    neg = targets.numel() - pos
    pos_weight = (neg / pos.clamp(min=1.0)).clamp(max=50.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def _loss_cone_depth(output: Dict[str, Any]) -> torch.Tensor:
    """Prefer decoupled per-expert mixed depth when the model emits it."""
    return output.get("cone_depth_for_loss", output["cone_depth"])


def gosp_loss_v6(
    output: Dict[str, Any],
    target_rho: torch.Tensor,
    ca_coords: torch.Tensor,
    domain_labels: Optional[torch.Tensor] = None,
    sasa: Optional[torch.Tensor] = None,
    *,
    # Phase-aware coefficient (caller sets based on epoch)
    balance_coeff: float = 0.01,
    # Fixed loss coefficients
    cone_coeff: float = 0.15,
    neighborhood_coeff: float = 0.25,
    angular_coeff: float = 0.20,
    domain_sep_2d_coeff: float = 0.30,
    domain_sep_3d_coeff: float = 0.30,
    evidential_coeff: float = 0.001,
    cone_depth_anticollapse_coeff: float = 0.50,
    shell_corr_coeff: float = 0.25,
    cone_depth_min_std: float = 0.02,
    shell_corr_depth_sasa_weight: float = 1.0,
    shell_corr_epi_sasa_weight: float = 0.5,
    shell_corr_proj_depth_weight: float = 1.0,
    shell_corr_disc_spread_weight: float = 0.5,
    shell_corr_disc_sasa_weight: float = 0.3,
    disc_spread_min_std: float = 0.15,
    proj_violation_coeff: float = 2.0,
    disc_depth_scale_coeff: float = 0.0,
    disc_depth_scale_target: float = 0.45,
    disc_occupancy_coeff: float = 0.0,
    disc_occupancy_min_sigma_ratio: float = 0.35,
    disc_min_r_mean: float = 0.05,
    disc_r_collapse_scale: float = 10.0,
    core_radial_floor_coeff: float = 0.0,
    core_radial_floor_min_r: float = 0.15,
    disc_pc_repulsion_coeff: float = 0.0,
    disc_pc2_min_std: float = 0.08,
    disc_eff_rank_coeff: float = 0.0,
    disc_eff_rank_min: float = 1.6,
    disc_batch_diversity_coeff: float = 0.0,
    disc_batch_min_pairwise_dist: float = 0.035,
    rim_angular_repulsion_coeff: float = 0.0,
    rim_angular_min_r: float = 0.35,
    rim_angular_min_sep: float = 0.12,
    rim_angular_spatial_exempt: float = 8.0,
    rim_pc2_floor_coeff: float = 0.0,
    rim_pc2_min_r: float = 0.35,
    rim_pc2_min_std: float = 0.06,
    disc_angular_coverage_coeff: float = 0.0,
    disc_angular_coverage_min_r: float = 0.12,
    disc_angular_coverage_n_bins: int = 12,
    disc_angular_coverage_min_bin_frac: float = 0.40,
    disc_angular_coverage_temperature: float = 0.20,
    expert_angular_diversity_coeff: float = 0.0,
    expert_angular_max_R: float = 0.55,
    expert_angular_min_mean_sep: float = 0.55,
    expert_sector_recruit_coeff: float = 0.0,
    expert_sector_recruit_min_bin_frac: float = 0.25,
    geometric_angular_fidelity_coeff: float = 0.0,
    disc_path_align_coeff: float = 0.0,
    disc_thickness_floor_coeff: float = 0.0,
    disc_thickness_floor_min: float = 0.025,
    disc_origin_span_floor_coeff: float = 0.0,
    disc_origin_span_min_spread: float = 0.12,
    x_hyp_thickness_floor_coeff: float = 0.0,
    x_hyp_thickness_floor_min: float = 0.18,
    shell_floor_coeff: float = 0.0,
    shell_floor_min_r_depth_sasa: float = 0.60,
    epistemic_decoupling_coeff: float = 0.0,
    epi_ale_decorrelation_coeff: float = 0.0,
    epistemic_bf_align_coeff: float = 0.22,
    epistemic_sasa_pen_coeff: float = 0.246,
    epistemic_anticollapse_coeff: float = 0.05,
    epistemic_min_epi_std: float = 0.02,
    routing_load_floor_coeff: float = 0.0,
    routing_load_floor_min: float = 0.05,
    routing_load_ceiling_coeff: float = 0.0,
    routing_load_ceiling_max: float = 0.45,
    # Scheduled λ from stage_runner (warmup applied by caller; default off).
    routing_entropy_sparsity_coeff: float = 0.0,
    # Accepted from LossCoeffs.model_dump(); schedule lives in stage_runner.
    routing_entropy_sparsity_warmup_epochs: int = 8,
    prototype_repulsion_coeff: float = 0.0,
    prototype_repulsion_margin: float = 0.25,
    prototype_gram_logdet_coeff: float = 0.0,
    prototype_gram_logdet_tau: float = -1.15,
    majority_committed_share_coeff: float = 0.0,
    majority_committed_share_tau: float = 0.56,
    majority_committed_share_commit_thr: float = 0.60,
    majority_committed_share_min_n: int = 20,
    core_majority_committed_share_coeff: float = 0.0,
    core_majority_committed_share_tau: float = 0.56,
    core_majority_committed_share_commit_thr: float = 0.60,
    core_majority_committed_share_min_n: int = 20,
    directionality_asym_coeff: float = 0.0,
    directionality_eligible: bool = False,
    pocket_bce_coeff: float = 0.0,
    interface_bce_coeff: float = 0.0,
    leak_bce_coeff: float = 0.0,
    cone_target_mode: str = "rho_wrap",
    v3_aleatoric_shaping_coeff: float = 0.0,
    w_var_penalty: float = 2.8,
    w_aleatoric_hinge: float = 4.8,
    aleatoric_hinge_target: float = 1.0,
    aleatoric_shaping_train_mask: Optional[torch.Tensor] = None,
    target_pocket: Optional[torch.Tensor] = None,
    target_interface: Optional[torch.Tensor] = None,
    target_leak: Optional[torch.Tensor] = None,
    target_dehydron: Optional[torch.Tensor] = None,
    pocket_label_mask: Optional[torch.Tensor] = None,
    interface_label_mask: Optional[torch.Tensor] = None,
    leak_label_mask: Optional[torch.Tensor] = None,
    b_factor_ca: Optional[torch.Tensor] = None,
    b_factor_present: Optional[torch.Tensor] = None,
    topology_depth: bool = False,
    # Neighborhood consistency params
    spatial_cutoff: float = 8.0,
    attract_margin: float = 1.0,
    repel_margin: float = 2.5,
) -> Dict[str, Any]:
    """
    V6 combined training loss with asymmetric capacity regularization.

    Key differences from gosp_loss_v5:
      - Uses capacity_loss (asymmetric) instead of balance_loss (symmetric)
      - Tracks routing_entropy as a monitoring metric
      - Phase-aware balance_coeff controls capacity loss strength

    Args:
        output: Model forward pass output dict from GOSPConeMapperV6.
        target_rho: Target dehydron density [N].
        ca_coords: Cα coordinates [N, 3] for neighborhood consistency.
        domain_labels: Optional domain labels [N] for domain separation loss.
        balance_coeff: Phase-aware coefficient for capacity loss.
        cone_coeff: Weight for cone (radial depth) loss.
        neighborhood_coeff: Weight for neighborhood consistency loss.
        angular_coeff: Weight for angular diversity loss.
        domain_sep_2d_coeff: Weight for 2D domain separation loss.
        domain_sep_3d_coeff: Weight for 3D domain separation loss.
        evidential_coeff: Weight for evidential regression loss.
        spatial_cutoff: Cutoff distance for neighborhood consistency.
        attract_margin: Attraction margin for neighborhood consistency.
        repel_margin: Repulsion margin for neighborhood consistency.

    Returns:
        Dict with 'total' loss and all individual loss components plus routing_entropy.
    """
    device = output["x_hyp"].device

    # ── Evidential loss ───────────────────────────────────────────────────
    ev_loss = evidential_regression_loss(
        output["evidence"], target_rho, coeff=evidential_coeff
    )

    # ── Asymmetric capacity loss (replaces v5 symmetric balance_loss) ─────
    capacity_loss = output["capacity_loss"]
    routing_load_floor_raw = output.get("routing_load_floor")
    # Prefer phase coeff floor threshold over gate.min_usage (often stuck at 0.05).
    load = output.get("expert_load")
    if routing_load_floor_coeff > 0 and routing_load_floor_min > 0 and load is not None:
        routing_load_floor_raw = routing_load_floor_penalty(
            load.reshape(-1).float(),
            min_fraction=float(routing_load_floor_min),
        )
    routing_load_floor_loss = torch.tensor(0.0, device=device)
    if routing_load_floor_coeff > 0 and routing_load_floor_raw is not None:
        routing_load_floor_loss = routing_load_floor_coeff * routing_load_floor_raw

    routing_load_ceiling_loss = torch.tensor(0.0, device=device)
    if (
        routing_load_ceiling_coeff > 0
        and routing_load_ceiling_max > 0
        and load is not None
    ):
        routing_load_ceiling_loss = routing_load_ceiling_coeff * routing_load_ceiling_penalty(
            load.reshape(-1).float(),
            max_fraction=float(routing_load_ceiling_max),
        )

    # ── Prototype nearest-pair repulsion (hinge on current min pair) ──────
    prototype_repulsion_loss = torch.tensor(0.0, device=device)
    prototype_pair_min = output.get("prototype_pair_min_dist")
    if prototype_pair_min is None:
        audit = output.get("audit_trail") or {}
        if isinstance(audit, dict):
            prototype_pair_min = audit.get("prototype_pair_min_dist")
    if (
        prototype_repulsion_coeff > 0
        and prototype_pair_min is not None
        and torch.is_tensor(prototype_pair_min)
    ):
        prototype_repulsion_loss = prototype_repulsion_coeff * torch.relu(
            float(prototype_repulsion_margin) - prototype_pair_min
        )

    # ── Full-bank Gram logdet hinge (saturating; pre-reg GRAM_COND_*) ─────
    prototype_gram_hinge_loss = torch.tensor(0.0, device=device)
    prototype_gram_hinge_raw = torch.tensor(0.0, device=device)
    prototype_gram_logdet = output.get("prototype_gram_logdet")
    if prototype_gram_logdet is None:
        audit = output.get("audit_trail") or {}
        if isinstance(audit, dict):
            prototype_gram_logdet = audit.get("prototype_gram_logdet")
    if (
        prototype_gram_logdet_coeff > 0
        and prototype_gram_logdet is not None
        and torch.is_tensor(prototype_gram_logdet)
    ):
        prototype_gram_hinge_raw = prototype_gram_logdet_hinge(
            prototype_gram_logdet,
            tau_logdet=float(prototype_gram_logdet_tau),
        )
        prototype_gram_hinge_loss = (
            float(prototype_gram_logdet_coeff) * prototype_gram_hinge_raw
        )

    # ── Majority-conditional committed-share hinge (local monopole) ───────
    majority_share_loss = torch.tensor(0.0, device=device)
    majority_share_raw = torch.tensor(0.0, device=device)
    if majority_committed_share_coeff > 0:
        weights = output.get("expert_weights")
        if weights is not None and torch.is_tensor(weights):
            majority_share_raw = majority_committed_share_hinge(
                weights,
                tau=float(majority_committed_share_tau),
                commit_thr=float(majority_committed_share_commit_thr),
                min_committed=int(majority_committed_share_min_n),
            )
            majority_share_loss = (
                float(majority_committed_share_coeff) * majority_share_raw
            )

    # ── Core-only majority hinge (dh=0 eligible; no direct grad on dh=1) ──
    core_majority_share_loss = torch.tensor(0.0, device=device)
    core_majority_share_raw = torch.tensor(0.0, device=device)
    if core_majority_committed_share_coeff > 0:
        weights = output.get("expert_weights")
        if (
            weights is not None
            and torch.is_tensor(weights)
            and target_dehydron is not None
        ):
            dh = (
                target_dehydron.squeeze(-1)
                if target_dehydron.dim() > 1
                else target_dehydron
            )
            core_majority_share_raw = core_majority_committed_share_hinge(
                weights,
                dh,
                tau=float(core_majority_committed_share_tau),
                commit_thr=float(core_majority_committed_share_commit_thr),
                min_committed=int(core_majority_committed_share_min_n),
            )
            core_majority_share_loss = (
                float(core_majority_committed_share_coeff) * core_majority_share_raw
            )

    # ── Cone loss — flows through RadialHead only ─────────────────────────
    cone_loss = cone_alignment_loss(
        output["radial_features"],
        target_rho=target_rho,
        target_dehydron=target_dehydron,
        mode=cone_target_mode,
    )

    # ── Angular diversity — flows through AngularHead only ────────────────
    ang_loss = angular_diversity_loss(output["x_routed_hyp"])

    # ── Curvature ─────────────────────────────────────────────────────────
    c = output["audit_trail"]["curvature_value"]
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, device=device)

    # ── Neighborhood consistency — flows through both heads ───────────────
    nbr_loss = neighborhood_consistency_loss(
        x_hyp=output["x_routed_hyp"],
        ca_coords=ca_coords,
        c=c,
        spatial_cutoff=spatial_cutoff,
        attract_margin=attract_margin,
        repel_margin=repel_margin,
    )

    # ── Domain separation on 2D and 3D projections ────────────────────────
    dom_loss_2d = torch.tensor(0.0, device=device)
    dom_loss_3d = torch.tensor(0.0, device=device)
    if domain_labels is not None:
        if domain_sep_2d_coeff > 0:
            dom_loss_2d = domain_separation_loss_2d(
                hyp_proj=output["hyp_projections_2d"],
                domain_labels=domain_labels,
                c=c,
            )
        if domain_sep_3d_coeff > 0:
            dom_loss_3d = domain_separation_loss_3d(
                hyp_proj_3d=output["hyp_projections_3d"],
                domain_labels=domain_labels,
                c=c,
            )

    # ── dist0 anti-collapse + shell correlation (app / probe aligned) ─────
    anticollapse_loss = torch.tensor(0.0, device=device)
    shell_losses: Dict[str, torch.Tensor] = {}
    if cone_depth_anticollapse_coeff > 0 and "cone_depth" in output:
        anticollapse_loss = cone_depth_anticollapse_loss(
            _loss_cone_depth(output),
            min_std=cone_depth_min_std,
        )
    if shell_corr_coeff > 0 and "cone_depth" in output:
        if topology_depth:
            shell_losses = shell_correlation_topology_loss(
                _loss_cone_depth(output),
                output["hyp_projections_2d"],
                proj_depth_weight=shell_corr_proj_depth_weight,
                disc_spread_weight=shell_corr_disc_spread_weight,
                disc_spread_min_std=disc_spread_min_std,
            )
        elif sasa is not None:
            shell_losses = shell_correlation_loss(
                _loss_cone_depth(output),
                output["uncertainty"]["epistemic"],
                sasa,
                output["hyp_projections_2d"],
                depth_sasa_weight=shell_corr_depth_sasa_weight,
                epi_sasa_weight=shell_corr_epi_sasa_weight,
                proj_depth_weight=shell_corr_proj_depth_weight,
                disc_spread_weight=shell_corr_disc_spread_weight,
                disc_sasa_weight=shell_corr_disc_sasa_weight,
                disc_spread_min_std=disc_spread_min_std,
            )

    disc_scale_loss = torch.tensor(0.0, device=device)
    if (
        disc_depth_scale_coeff > 0
        and "cone_depth" in output
        and "hyp_projections_2d" in output
    ):
        disc_scale_loss = disc_depth_scale_loss(
            _loss_cone_depth(output),
            output["hyp_projections_2d"],
            target_radius=disc_depth_scale_target,
        )

    disc_occ_losses: Dict[str, torch.Tensor] = {}
    disc_occ_total = torch.tensor(0.0, device=device)
    if disc_occupancy_coeff > 0 and "hyp_projections_2d" in output:
        disc_occ_losses = disc_occupancy_loss(
            output["hyp_projections_2d"],
            min_sigma_ratio=disc_occupancy_min_sigma_ratio,
            min_disc_r_mean=disc_min_r_mean,
            collapse_scale=disc_r_collapse_scale,
        )
        disc_occ_total = disc_occ_losses["disc_occupancy"]

    core_radial_total = torch.tensor(0.0, device=device)
    core_radial_aux: Dict[str, torch.Tensor] = {}
    if core_radial_floor_coeff > 0 and "hyp_projections_2d" in output:
        from science.training.disc_occupancy import core_radial_floor_loss

        core_radial_aux = core_radial_floor_loss(
            output["hyp_projections_2d"],
            target_rho,
            target_tau=target_dehydron,
            min_r=core_radial_floor_min_r,
        )
        core_radial_total = core_radial_aux["core_radial_floor"]

    disc_pc_losses: Dict[str, torch.Tensor] = {}
    disc_pc_total = torch.tensor(0.0, device=device)
    if disc_pc_repulsion_coeff > 0 and "hyp_projections_2d" in output:
        from science.training.disc_occupancy import disc_pc_repulsion_loss

        disc_pc_losses = disc_pc_repulsion_loss(
            output["hyp_projections_2d"],
            min_pc2_std=disc_pc2_min_std,
            min_disc_r_mean=disc_min_r_mean,
        )
        disc_pc_total = disc_pc_losses["disc_pc_repulsion"]

    disc_eff_rank_losses: Dict[str, torch.Tensor] = {}
    disc_eff_rank_total = torch.tensor(0.0, device=device)
    if disc_eff_rank_coeff > 0 and "hyp_projections_2d" in output:
        from science.training.disc_occupancy import disc_eff_rank_loss

        disc_eff_rank_losses = disc_eff_rank_loss(
            output["hyp_projections_2d"],
            min_eff_rank=disc_eff_rank_min,
            min_disc_r_mean=disc_min_r_mean,
        )
        disc_eff_rank_total = disc_eff_rank_losses["disc_eff_rank"]

    disc_batch_losses: Dict[str, torch.Tensor] = {}
    disc_batch_total = torch.tensor(0.0, device=device)
    if disc_batch_diversity_coeff > 0 and "hyp_projections_2d" in output:
        from science.training.disc_occupancy import batch_diversity_repulsion_loss

        disc_batch_losses = batch_diversity_repulsion_loss(
            output["hyp_projections_2d"],
            min_pairwise_dist=disc_batch_min_pairwise_dist,
            min_disc_r_mean=disc_min_r_mean,
        )
        disc_batch_total = disc_batch_losses["disc_batch_diversity"]

    rim_fanout_losses: Dict[str, torch.Tensor] = {}
    rim_angular_total = torch.tensor(0.0, device=device)
    rim_pc2_total = torch.tensor(0.0, device=device)
    if (
        (rim_angular_repulsion_coeff > 0 or rim_pc2_floor_coeff > 0)
        and "hyp_projections_2d" in output
        and ca_coords is not None
    ):
        from science.training.rim_fanout import rim_angular_repulsion_loss, rim_pc2_floor_loss

        hyp2d = output["hyp_projections_2d"]
        if rim_angular_repulsion_coeff > 0:
            rim_fanout_losses.update(
                rim_angular_repulsion_loss(
                    hyp2d,
                    ca_coords,
                    min_r_rim=rim_angular_min_r,
                    min_angular_sep=rim_angular_min_sep,
                    spatial_exempt_cutoff=rim_angular_spatial_exempt,
                )
            )
            rim_angular_total = rim_fanout_losses["rim_angular_repulsion"]
        if rim_pc2_floor_coeff > 0:
            pc2_losses = rim_pc2_floor_loss(
                hyp2d,
                min_r_rim=rim_pc2_min_r,
                min_pc2_std=rim_pc2_min_std,
            )
            rim_fanout_losses.update(pc2_losses)
            rim_pc2_total = pc2_losses["rim_pc2_floor"]

    disc_coverage_losses: Dict[str, torch.Tensor] = {}
    disc_coverage_total = torch.tensor(0.0, device=device)
    if disc_angular_coverage_coeff > 0 and "hyp_projections_2d" in output:
        from science.training.disc_occupancy import disc_angular_coverage_loss

        disc_coverage_losses = disc_angular_coverage_loss(
            output["hyp_projections_2d"],
            min_r=disc_angular_coverage_min_r,
            n_bins=disc_angular_coverage_n_bins,
            min_bin_frac=disc_angular_coverage_min_bin_frac,
            temperature=disc_angular_coverage_temperature,
        )
        disc_coverage_total = disc_coverage_losses["disc_angular_coverage"]

    expert_ang_losses: Dict[str, torch.Tensor] = {}
    expert_ang_total = torch.tensor(0.0, device=device)
    expert_recruit_total = torch.tensor(0.0, device=device)
    if (
        (expert_angular_diversity_coeff > 0 or expert_sector_recruit_coeff > 0)
        and "hyp_projections_2d" in output
        and output.get("expert_weights") is not None
    ):
        from science.training.expert_angular import (
            expert_angular_diversity_loss,
            expert_sector_recruit_loss,
        )

        hyp2d = output["hyp_projections_2d"]
        ew = output["expert_weights"]
        if expert_angular_diversity_coeff > 0:
            expert_ang_losses.update(
                expert_angular_diversity_loss(
                    hyp2d,
                    ew,
                    min_r=disc_angular_coverage_min_r,
                    max_resultant_length=expert_angular_max_R,
                    min_mean_sep=expert_angular_min_mean_sep,
                )
            )
            expert_ang_total = expert_ang_losses["expert_angular_diversity"]
        if expert_sector_recruit_coeff > 0:
            recruit = expert_sector_recruit_loss(
                hyp2d,
                ew,
                min_r=disc_angular_coverage_min_r,
                n_bins=disc_angular_coverage_n_bins,
                temperature=disc_angular_coverage_temperature,
                global_min_bin_frac=disc_angular_coverage_min_bin_frac,
                expert_min_bin_frac=expert_sector_recruit_min_bin_frac,
            )
            expert_ang_losses.update(recruit)
            expert_recruit_total = recruit["expert_sector_recruit"]

    geom_fidelity_losses: Dict[str, torch.Tensor] = {}
    geom_fidelity_total = torch.tensor(0.0, device=device)
    if (
        geometric_angular_fidelity_coeff > 0
        and "hyp_projections_2d" in output
        and output.get("geom_theta_prior") is not None
    ):
        from science.dtie.v66.geometric_angular_prior import (
            geometric_angular_fidelity_loss,
        )

        geom_fidelity_losses = geometric_angular_fidelity_loss(
            output["hyp_projections_2d"],
            output["geom_theta_prior"],
        )
        geom_fidelity_total = geom_fidelity_losses["geometric_angular_fidelity"]

    disc_path_align_total = torch.tensor(0.0, device=device)
    if (
        disc_path_align_coeff > 0
        and "hyp_projections_2d" in output
        and output.get("hyp_projections_2d_legacy_teacher") is not None
    ):
        disc_path_align_total = disc_path_align_loss(
            output["hyp_projections_2d"],
            output["hyp_projections_2d_legacy_teacher"],
        )

    disc_thickness_losses: Dict[str, torch.Tensor] = {}
    disc_thickness_total = torch.tensor(0.0, device=device)
    if disc_thickness_floor_coeff > 0 and "hyp_projections_2d" in output:
        from science.training.disc_occupancy import disc_line_thickness_floor_loss

        hyp2d = output["hyp_projections_2d"]
        with torch.autocast(device_type=hyp2d.device.type, enabled=False):
            disc_thickness_losses = disc_line_thickness_floor_loss(
                hyp2d,
                min_thickness=disc_thickness_floor_min,
            )
        disc_thickness_total = disc_thickness_losses["disc_thickness_floor"]

    disc_span_losses: Dict[str, torch.Tensor] = {}
    disc_span_total = torch.tensor(0.0, device=device)
    if disc_origin_span_floor_coeff > 0 and "hyp_projections_2d" in output:
        from science.training.disc_occupancy import disc_origin_span_floor_loss

        disc_span_losses = disc_origin_span_floor_loss(
            output["hyp_projections_2d"],
            min_circular_spread=disc_origin_span_min_spread,
        )
        disc_span_total = disc_span_losses["disc_origin_span_floor"]

    x_hyp_thickness_losses: Dict[str, torch.Tensor] = {}
    x_hyp_thickness_total = torch.tensor(0.0, device=device)
    if x_hyp_thickness_floor_coeff > 0 and "x_hyp" in output:
        from science.training.disc_occupancy import ball_line_thickness_floor_loss

        x_hyp = output["x_hyp"]
        with torch.autocast(device_type=x_hyp.device.type, enabled=False):
            x_hyp_thickness_losses = ball_line_thickness_floor_loss(
                x_hyp,
                min_thickness=x_hyp_thickness_floor_min,
            )
        x_hyp_thickness_total = x_hyp_thickness_losses["x_hyp_thickness_floor"]

    shell_floor_loss = torch.tensor(0.0, device=device)
    if (
        not topology_depth
        and shell_floor_coeff > 0
        and sasa is not None
        and "cone_depth" in output
    ):
        depth = output["cone_depth"].squeeze(-1)
        r_ds = _pearson_corr(depth, sasa.squeeze(-1) if sasa.dim() > 1 else sasa)
        shell_floor_loss = torch.relu(shell_floor_min_r_depth_sasa - r_ds) * shell_floor_coeff

    epistemic_decoupling_losses: Dict[str, torch.Tensor] = {}
    epistemic_decoupling_total = torch.tensor(0.0, device=device)
    if (
        epistemic_decoupling_coeff > 0
        and b_factor_ca is not None
        and "cone_depth" in output
        and "uncertainty" in output
    ):
        epistemic_decoupling_losses = epistemic_decoupling_loss(
            output["uncertainty"]["epistemic"],
            output["cone_depth"],
            sasa if sasa is not None else torch.zeros_like(b_factor_ca.squeeze(-1)),
            b_factor_ca,
            b_factor_present,
            min_epi_std=epistemic_min_epi_std,
        )
        epistemic_decoupling_total = (
            epistemic_bf_align_coeff * epistemic_decoupling_losses["epistemic_bf_align"]
            + epistemic_sasa_pen_coeff * epistemic_decoupling_losses["epistemic_sasa_pen"]
            + epistemic_anticollapse_coeff * epistemic_decoupling_losses["epistemic_anticollapse"]
        )

    epi_ale_decorrelation_losses: Dict[str, torch.Tensor] = {}
    epi_ale_decorrelation_total = torch.tensor(0.0, device=device)
    if (
        epi_ale_decorrelation_coeff > 0
        and "uncertainty" in output
        and "epistemic" in output["uncertainty"]
        and "aleatoric" in output["uncertainty"]
    ):
        epi_ale_decorrelation_losses = epi_ale_decorrelation_loss(
            output["uncertainty"]["epistemic"],
            output["uncertainty"]["aleatoric"],
        )
        epi_ale_decorrelation_total = epi_ale_decorrelation_losses["epi_ale_decorrelation"]

    pocket_bce = torch.tensor(0.0, device=device)
    interface_bce = torch.tensor(0.0, device=device)
    leak_bce = torch.tensor(0.0, device=device)
    if pocket_bce_coeff > 0 and target_pocket is not None:
        tgt = target_pocket.squeeze(-1) if target_pocket.dim() > 1 else target_pocket
        pocket_bce = weighted_binary_cross_entropy(
            output["binding_logits_pocket"],
            tgt,
            mask=pocket_label_mask,
        )
    if interface_bce_coeff > 0 and target_interface is not None:
        tgt_i = target_interface.squeeze(-1) if target_interface.dim() > 1 else target_interface
        interface_bce = weighted_binary_cross_entropy(
            output["binding_logits_interface"],
            tgt_i,
            mask=interface_label_mask,
        )
    if leak_bce_coeff > 0 and target_leak is not None:
        tgt_l = target_leak.squeeze(-1) if target_leak.dim() > 1 else target_leak
        leak_bce = weighted_binary_cross_entropy(
            output["binding_logits_leak"],
            tgt_l,
            mask=leak_label_mask,
        )

    v3_shaping_total = torch.tensor(0.0, device=device)
    v3_shaping_losses: Dict[str, torch.Tensor] = {}
    if (
        v3_aleatoric_shaping_coeff > 0
        and target_dehydron is not None
        and aleatoric_shaping_train_mask is not None
        and "uncertainty" in output
        and "aleatoric" in output["uncertainty"]
    ):
        v3_shaping_losses = v3_aleatoric_shaping_loss(
            output["uncertainty"]["aleatoric"],
            target_dehydron,
            aleatoric_shaping_train_mask,
            w_var_penalty=w_var_penalty,
            w_aleatoric_hinge=w_aleatoric_hinge,
            hinge_target=aleatoric_hinge_target,
        )
        v3_shaping_total = v3_aleatoric_shaping_coeff * v3_shaping_losses[
            "v3_aleatoric_shaping_total"
        ]

    # Path 2 directionality (diam≤9 eligibility set by caller / train_loop).
    from science.dtie.common.directionality_objective import directionality_asym_loss

    encoder_h = output.get("encoder_h")
    if encoder_h is None or not torch.is_tensor(encoder_h):
        dir_terms = {
            "directionality_asym": torch.tensor(0.0, device=device),
            "directionality_asym_raw": torch.tensor(0.0, device=device),
            "directionality_asym_index": torch.tensor(float("nan"), device=device),
        }
    else:
        dir_terms = directionality_asym_loss(
            encoder_h,
            coeff=float(directionality_asym_coeff),
            eligible=bool(directionality_eligible),
        )
    directionality_asym_term = dir_terms["directionality_asym"]

    # ── Total loss ────────────────────────────────────────────────────────
    total = (
        (ev_loss if evidential_coeff > 0 else torch.tensor(0.0, device=device))
        + balance_coeff * capacity_loss
        + cone_coeff * cone_loss
        + neighborhood_coeff * nbr_loss
        + angular_coeff * ang_loss
        + domain_sep_2d_coeff * dom_loss_2d
        + domain_sep_3d_coeff * dom_loss_3d
        + cone_depth_anticollapse_coeff * anticollapse_loss
        + disc_depth_scale_coeff * disc_scale_loss
        + disc_occupancy_coeff * disc_occ_total
        + core_radial_floor_coeff * core_radial_total
        + disc_pc_repulsion_coeff * disc_pc_total
        + disc_eff_rank_coeff * disc_eff_rank_total
        + disc_batch_diversity_coeff * disc_batch_total
        + rim_angular_repulsion_coeff * rim_angular_total
        + rim_pc2_floor_coeff * rim_pc2_total
        + disc_angular_coverage_coeff * disc_coverage_total
        + expert_angular_diversity_coeff * expert_ang_total
        + expert_sector_recruit_coeff * expert_recruit_total
        + geometric_angular_fidelity_coeff * geom_fidelity_total
        + disc_path_align_coeff * disc_path_align_total
        + disc_thickness_floor_coeff * disc_thickness_total
        + disc_origin_span_floor_coeff * disc_span_total
        + x_hyp_thickness_floor_coeff * x_hyp_thickness_total
        + shell_floor_loss
        + epistemic_decoupling_coeff * epistemic_decoupling_total
        + epi_ale_decorrelation_coeff * epi_ale_decorrelation_total
        + routing_load_floor_loss
        + routing_load_ceiling_loss
        + prototype_repulsion_loss
        + prototype_gram_hinge_loss
        + majority_share_loss
        + core_majority_share_loss
        + directionality_asym_term
        + pocket_bce_coeff * pocket_bce
        + interface_bce_coeff * interface_bce
        + leak_bce_coeff * leak_bce
        + v3_shaping_total
    )
    if shell_losses:
        total = total + shell_corr_coeff * shell_losses["shell_corr_total"]

    # ── Projection safety penalties ───────────────────────────────────────
    hyp_norms_2d = output["hyp_projections_2d"].norm(dim=-1)
    hyp_norms_3d = output["hyp_projections_3d"].norm(dim=-1)
    proj_violation = (
        torch.relu(hyp_norms_2d - 0.99).mean()
        + torch.relu(hyp_norms_3d - 0.99).mean()
    )
    total = total + proj_violation_coeff * proj_violation

    # ── Routing entropy (H(f̄) monitor) + optional mean-residue sparsity ──
    routing_entropy = output["routing_entropy"]
    L_sparse = output.get("routing_entropy_mean_residue")
    sparse_coeff = float(routing_entropy_sparsity_coeff)
    if L_sparse is not None and sparse_coeff > 0:
        if not torch.is_tensor(L_sparse):
            L_sparse = torch.as_tensor(float(L_sparse), device=device)
        sparse_term = sparse_coeff * L_sparse
    else:
        sparse_term = torch.tensor(0.0, device=device)
    total = total + sparse_term

    result: Dict[str, Any] = {
        "total": total,
        "evidential": ev_loss,
        "capacity_loss": capacity_loss,
        "routing_load_floor": routing_load_floor_loss,
        "routing_load_ceiling": routing_load_ceiling_loss,
        "prototype_repulsion": prototype_repulsion_loss,
        "prototype_gram_logdet_hinge": prototype_gram_hinge_loss,
        "prototype_gram_logdet_hinge_raw": (
            float(prototype_gram_hinge_raw.detach())
            if torch.is_tensor(prototype_gram_hinge_raw)
            else float("nan")
        ),
        "prototype_gram_logdet": (
            float(prototype_gram_logdet.detach())
            if torch.is_tensor(prototype_gram_logdet)
            else float("nan")
        ),
        "majority_committed_share": majority_share_loss,
        "majority_committed_share_raw": (
            float(majority_share_raw.detach())
            if torch.is_tensor(majority_share_raw)
            else float("nan")
        ),
        "core_majority_committed_share": core_majority_share_loss,
        "core_majority_committed_share_raw": (
            float(core_majority_share_raw.detach())
            if torch.is_tensor(core_majority_share_raw)
            else float("nan")
        ),
        "directionality_asym": directionality_asym_term,
        "directionality_asym_raw": dir_terms["directionality_asym_raw"],
        "directionality_asym_index": dir_terms["directionality_asym_index"],
        "prototype_pair_min_dist": (
            float(prototype_pair_min.detach())
            if torch.is_tensor(prototype_pair_min)
            else float("nan")
        ),
        "cone_consistency": cone_loss,
        "cone_depth_anticollapse": anticollapse_loss,
        "disc_depth_scale": disc_scale_loss,
        "disc_occupancy": disc_occ_total,
        "core_radial_floor": core_radial_total,
        "disc_pc_repulsion": disc_pc_total,
        "disc_eff_rank": disc_eff_rank_total,
        "disc_batch_diversity": disc_batch_total,
        "rim_angular_repulsion": rim_angular_total,
        "rim_pc2_floor": rim_pc2_total,
        "disc_angular_coverage": disc_coverage_total,
        "expert_angular_diversity": expert_ang_total,
        "expert_sector_recruit": expert_recruit_total,
        "geometric_angular_fidelity": geom_fidelity_total,
        "disc_path_align": disc_path_align_total,
        "disc_thickness_floor": disc_thickness_total,
        "disc_origin_span_floor": disc_span_total,
        "x_hyp_thickness_floor": x_hyp_thickness_total,
        "shell_floor": shell_floor_loss,
        "epistemic_decoupling": epistemic_decoupling_total,
        "epi_ale_decorrelation": epi_ale_decorrelation_total,
        "neighborhood_consistency": nbr_loss,
        "angular_diversity": ang_loss,
        "domain_separation_2d": dom_loss_2d,
        "domain_separation_3d": dom_loss_3d,
        "projection_violation": proj_violation,
        "routing_entropy": routing_entropy,
        "routing_entropy_mean_residue": L_sparse,
        "routing_entropy_sparsity_loss": sparse_term,
        "routing_entropy_sparsity_coeff": sparse_coeff,
        "expert_load": output["expert_load"],
        "pocket_bce": pocket_bce,
        "interface_bce": interface_bce,
        "leak_bce": leak_bce,
        "v3_aleatoric_shaping": v3_shaping_total,
    }
    if core_radial_aux:
        result["core_radial_floor_weight_mean"] = core_radial_aux[
            "core_radial_floor_weight_mean"
        ]
        result["core_radial_floor_r_weighted"] = core_radial_aux[
            "core_radial_floor_r_weighted"
        ]
    if v3_shaping_losses:
        result["var_penalty"] = v3_shaping_losses["var_penalty"]
        result["aleatoric_hinge"] = v3_shaping_losses["aleatoric_hinge"]
    if shell_losses:
        result["shell_correlation"] = shell_losses["shell_corr_total"]
        result["shell_corr_depth_sasa"] = shell_losses["shell_corr_depth_sasa"]
        result["shell_corr_epi_sasa"] = shell_losses["shell_corr_epi_sasa"]
        result["shell_corr_proj_depth"] = shell_losses["shell_corr_proj_depth"]
        result["shell_corr_disc_spread"] = shell_losses["shell_corr_disc_spread"]
        result["shell_corr_disc_sasa"] = shell_losses["shell_corr_disc_sasa"]
        result["shell_r_depth_sasa"] = shell_losses["shell_r_depth_sasa"]
        result["shell_r_epi_sasa"] = shell_losses["shell_r_epi_sasa"]
        result["shell_r_proj_depth"] = shell_losses["shell_r_proj_depth"]
        result["shell_r_disc_sasa"] = shell_losses["shell_r_disc_sasa"]
        result["disc_r_std"] = shell_losses["disc_r_std"]
    if disc_occ_losses:
        result["disc_sigma2_sigma1"] = disc_occ_losses["disc_sigma2_sigma1"]
        result["disc_effective_rank"] = disc_occ_losses["disc_effective_rank"]
        result["disc_r_mean"] = disc_occ_losses.get("disc_r_mean")
        result["disc_r_collapse"] = disc_occ_losses.get("disc_r_collapse")
    if disc_pc_losses:
        result["disc_pc2_std"] = disc_pc_losses["disc_pc2_std"]
    if disc_thickness_losses:
        result["disc_line_thickness_rms"] = disc_thickness_losses["disc_line_thickness_rms"]
    if disc_span_losses:
        result["disc_origin_circular_spread"] = disc_span_losses["disc_origin_circular_spread"]
    if disc_coverage_losses:
        result["disc_angular_coverage_empty_bins"] = disc_coverage_losses[
            "disc_angular_coverage_empty_bins"
        ]
        result["disc_angular_coverage_min_mass"] = disc_coverage_losses[
            "disc_angular_coverage_min_mass"
        ]
    if expert_ang_losses:
        if "expert_angular_mean_R" in expert_ang_losses:
            result["expert_angular_mean_R"] = expert_ang_losses["expert_angular_mean_R"]
            result["expert_angular_min_sep"] = expert_ang_losses["expert_angular_min_sep"]
            result["expert_angular_concentration"] = expert_ang_losses[
                "expert_angular_concentration"
            ]
            result["expert_angular_separation"] = expert_ang_losses[
                "expert_angular_separation"
            ]
        if "expert_sector_empty_bins" in expert_ang_losses:
            result["expert_sector_empty_bins"] = expert_ang_losses[
                "expert_sector_empty_bins"
            ]
    if x_hyp_thickness_losses:
        result["x_hyp_line_thickness_rms"] = x_hyp_thickness_losses["x_hyp_line_thickness_rms"]
    if epistemic_decoupling_losses:
        result["epistemic_bf_align"] = epistemic_decoupling_losses["epistemic_bf_align"]
        result["epistemic_sasa_pen"] = epistemic_decoupling_losses["epistemic_sasa_pen"]
        result["epistemic_anticollapse"] = epistemic_decoupling_losses["epistemic_anticollapse"]
        result["r_epi_bf_resid"] = epistemic_decoupling_losses["r_epi_bf_resid"]
        result["partial_epi_sasa_given_depth"] = epistemic_decoupling_losses[
            "partial_epi_sasa_given_depth"
        ]
    if epi_ale_decorrelation_losses:
        result["r_epi_ale"] = epi_ale_decorrelation_losses["r_epi_ale"]
    return result


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    x = x.squeeze(-1) if x.dim() > 1 else x
    lo, hi = x.min(), x.max()
    if float(hi - lo) < 1e-6:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def v2_teacher_distill_loss(
    output: Dict[str, Any],
    teacher: Dict[str, torch.Tensor],
    sasa: torch.Tensor,
    *,
    depth_coeff: float = 0.20,
    epistemic_coeff: float = 0.10,
    shell_floor: float = 0.30,
) -> Dict[str, torch.Tensor]:
    """
  Distill v2 shell signals into v6: normalized cone depth + SASA-weighted epistemic.

  Teacher tensors (CPU): cone_depth_norm [N], epistemic [N], expert_id [N].
  """
    device = output["cone_depth"].device
    t_depth = teacher["cone_depth_norm"].to(device)
    t_epi = teacher["epistemic"].to(device)
    sasa_w = shell_floor + (1.0 - shell_floor) * sasa.squeeze(-1).clamp(0.0, 1.0)

    v6_depth = _normalize01(output["cone_depth"])
    depth_loss = F.mse_loss(v6_depth, t_depth)

    v6_epi = output["uncertainty"]["epistemic"].squeeze(-1)
    epi_loss = (sasa_w * (v6_epi - t_epi).pow(2)).mean()

    total = depth_coeff * depth_loss + epistemic_coeff * epi_loss
    return {
        "v2_teacher_total": total,
        "v2_teacher_depth": depth_loss,
        "v2_teacher_epistemic": epi_loss,
    }
