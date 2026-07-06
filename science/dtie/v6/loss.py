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
  - routing_entropy: Monitoring metric (not a loss term, just tracked)

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
    """
    if mode == "tau_dehydron_rim":
        if target_dehydron is None:
            raise ValueError("target_dehydron required for tau_dehydron_rim cone mode")
        target_depth = target_dehydron.squeeze(-1).clamp(0.0, 1.0)
    else:
        if target_rho is None:
            raise ValueError("target_rho required for rho_wrap cone mode")
        return cone_loss_v5(radial_depth, target_rho)

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
    disc_pc_repulsion_coeff: float = 0.0,
    disc_pc2_min_std: float = 0.08,
    disc_eff_rank_coeff: float = 0.0,
    disc_eff_rank_min: float = 1.6,
    disc_batch_diversity_coeff: float = 0.0,
    disc_batch_min_pairwise_dist: float = 0.035,
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
    pocket_bce_coeff: float = 0.0,
    interface_bce_coeff: float = 0.0,
    leak_bce_coeff: float = 0.0,
    cone_target_mode: str = "rho_wrap",
    target_pocket: Optional[torch.Tensor] = None,
    target_interface: Optional[torch.Tensor] = None,
    target_leak: Optional[torch.Tensor] = None,
    target_dehydron: Optional[torch.Tensor] = None,
    pocket_label_mask: Optional[torch.Tensor] = None,
    interface_label_mask: Optional[torch.Tensor] = None,
    leak_label_mask: Optional[torch.Tensor] = None,
    b_factor_ca: Optional[torch.Tensor] = None,
    b_factor_present: Optional[torch.Tensor] = None,
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
    routing_load_floor_loss = torch.tensor(0.0, device=device)
    if routing_load_floor_coeff > 0 and routing_load_floor_raw is not None:
        routing_load_floor_loss = routing_load_floor_coeff * routing_load_floor_raw

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
            output["cone_depth"],
            min_std=cone_depth_min_std,
        )
    if shell_corr_coeff > 0 and sasa is not None and "cone_depth" in output:
        shell_losses = shell_correlation_loss(
            output["cone_depth"],
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
            output["cone_depth"],
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
        shell_floor_coeff > 0
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
        + disc_pc_repulsion_coeff * disc_pc_total
        + disc_eff_rank_coeff * disc_eff_rank_total
        + disc_batch_diversity_coeff * disc_batch_total
        + disc_path_align_coeff * disc_path_align_total
        + disc_thickness_floor_coeff * disc_thickness_total
        + disc_origin_span_floor_coeff * disc_span_total
        + x_hyp_thickness_floor_coeff * x_hyp_thickness_total
        + shell_floor_loss
        + epistemic_decoupling_coeff * epistemic_decoupling_total
        + epi_ale_decorrelation_coeff * epi_ale_decorrelation_total
        + routing_load_floor_loss
        + pocket_bce_coeff * pocket_bce
        + interface_bce_coeff * interface_bce
        + leak_bce_coeff * leak_bce
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

    # ── Routing entropy (monitoring only, not a loss term) ────────────────
    routing_entropy = output["routing_entropy"]

    result: Dict[str, Any] = {
        "total": total,
        "evidential": ev_loss,
        "capacity_loss": capacity_loss,
        "routing_load_floor": routing_load_floor_loss,
        "cone_consistency": cone_loss,
        "cone_depth_anticollapse": anticollapse_loss,
        "disc_depth_scale": disc_scale_loss,
        "disc_occupancy": disc_occ_total,
        "disc_pc_repulsion": disc_pc_total,
        "disc_eff_rank": disc_eff_rank_total,
        "disc_batch_diversity": disc_batch_total,
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
        "expert_load": output["expert_load"],
        "pocket_bce": pocket_bce,
        "interface_bce": interface_bce,
        "leak_bce": leak_bce,
    }
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
