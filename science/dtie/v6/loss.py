"""
V6 Loss Function — Phase-Aware Combined Loss with Asymmetric Capacity Regularization
======================================================================================
Eidetix Bio | 2026-05

Combines all loss terms for v6 training with phase-aware coefficients:
  - cone_loss: Radial depth regression (→ RadialHead only)
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


def gosp_loss_v6(
    output: Dict[str, Any],
    target_rho: torch.Tensor,
    ca_coords: torch.Tensor,
    domain_labels: Optional[torch.Tensor] = None,
    *,
    # Phase-aware coefficient (caller sets based on epoch)
    balance_coeff: float = 0.01,
    # Fixed loss coefficients
    cone_coeff: float = 0.15,
    neighborhood_coeff: float = 0.25,
    angular_coeff: float = 0.20,
    domain_sep_2d_coeff: float = 0.30,
    domain_sep_3d_coeff: float = 0.30,
    evidential_coeff: float = 0.005,
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

    # ── Cone loss — flows through RadialHead only ─────────────────────────
    cone_loss = cone_loss_v5(output["radial_features"], target_rho)

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

    # ── Total loss ────────────────────────────────────────────────────────
    total = (
        ev_loss
        + balance_coeff * capacity_loss
        + cone_coeff * cone_loss
        + neighborhood_coeff * nbr_loss
        + angular_coeff * ang_loss
        + domain_sep_2d_coeff * dom_loss_2d
        + domain_sep_3d_coeff * dom_loss_3d
    )

    # ── Projection safety penalties ───────────────────────────────────────
    hyp_norms_2d = output["hyp_projections_2d"].norm(dim=-1)
    hyp_norms_3d = output["hyp_projections_3d"].norm(dim=-1)
    proj_violation = (
        torch.relu(hyp_norms_2d - 0.99).mean()
        + torch.relu(hyp_norms_3d - 0.99).mean()
    )
    total = total + 2.0 * proj_violation

    # ── Routing entropy (monitoring only, not a loss term) ────────────────
    routing_entropy = output["routing_entropy"]

    return {
        "total": total,
        "evidential": ev_loss,
        "capacity_loss": capacity_loss,
        "cone_consistency": cone_loss,
        "neighborhood_consistency": nbr_loss,
        "angular_diversity": ang_loss,
        "domain_separation_2d": dom_loss_2d,
        "domain_separation_3d": dom_loss_3d,
        "projection_violation": proj_violation,
        "routing_entropy": routing_entropy,
        "expert_load": output["expert_load"],
    }
