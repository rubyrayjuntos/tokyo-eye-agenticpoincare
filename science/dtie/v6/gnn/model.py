"""
Tokyo Eyes v6: Topologically-Routed MoE Specialization
=======================================================
Eidetix Bio | 2026-05

MODEL HISTORY
- version: v6
  date: 2026-05
  change: Achieve true expert specialization via expanded gate input,
    asymmetric capacity regularization, and expert dropout.
  before_state: v5 MoE gate trapped in high-entropy uniform state (~0.25 each)
    due to insufficient gate input variance and overpowering balance loss.
  changes:
    1. Expand gate input: [x_tangent, clustering, cone_depth, log_degree_norm,
       rho_norm, ss_onehot] (hidden + 7 dimensions)
    2. Asymmetric capacity loss: penalizes starvation (<5%) but permits dominance
    3. Expert dropout: randomly mask one expert during training (p=0.15)
    4. Capacity-aware soft routing: two-pass logit adjustment for overloaded experts
    5. Three-phase training schedule for reliable specialization convergence
  reasoning: Experts need high-variance discriminative features and freedom to
    specialize without being forced back to uniformity by the balance loss.

DESIGN CONTRACT:
  Inputs (unchanged from v5):
    x: [N, 4]         node features (rho, tau_flag, ss_type, sasa)
    edge_index: [2,E] graph connectivity
    edge_attr: [E,4]  (rel_x, rel_y, rel_z, distance)
    clustering: [N]   precomputed clustering coefficients

  V6 additional inputs (attached to PyG Data):
    degree: [N]       node degree from contact graph
    ss_onehot: [N,3]  one-hot secondary structure [H, E, C]
    rho: [N]          raw dehydron density for gate shortcut

  Outputs (v5 superset + v6 additions):
    All v5 outputs preserved, plus:
    capacity_loss: scalar           Asymmetric capacity penalty
    expert_load: Tensor[4]          Per-expert mean routing probability
    routing_entropy: scalar         H(expert_weights) for monitoring
    gate_features_used: list[str]   Audit: which features fed the gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import geoopt
from typing import Any, Dict, List, Tuple, Optional

from science.dtie.v5.gnn.model import (
    EquivariantConv,
    MobiusLinear,
    RadialHead,
    AngularHead,
    EvidentialHead,
    precompute_clustering,
    scan_tensors_for_invalid,
)


# ==================== 1. TOPOLOGICAL MoE GATE V6 ====================

class TopologicalMoEGateV6(nn.Module):
    """
    V6 MoE gate with enriched topological features and capacity-aware routing.

    Gate input: [x_tangent, clustering, cone_depth, log_degree_norm, rho_norm, ss_onehot]
    Total dimensions: hidden + 7

    Features:
    - Running statistics for degree/rho normalization (zero-mean, unit-variance)
    - Two-pass capacity-aware logit adjustment
    - Expert dropout during training (p=0.15)
    - Asymmetric capacity loss (penalizes starvation only)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        capacity_threshold: float = 0.4,
        expert_dropout_p: float = 0.15,
        min_usage: float = 0.05,
        topology_only: bool = False,
    ):
        super().__init__()
        # topology_only=True: gate routes ONLY on physics (8D: clustering, cone_depth,
        # log_degree, rho, ss_H, ss_E, ss_C, sasa_proxy). No x_tangent drowning.
        self.topology_only = topology_only
        if topology_only:
            gate_input_dim = 7  # clustering + cone_depth + log_degree + rho + ss_onehot[3]
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 16),
                nn.SiLU(),
                nn.Linear(16, num_experts),
            )
        else:
            # Original: hidden + 7
            gate_input_dim = hidden_dim + 7
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Linear(32, num_experts),
            )
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_threshold = capacity_threshold
        self.expert_dropout_p = expert_dropout_p
        self.min_usage = min_usage
        self.detach_gate_input = False  # Set True in Phase 3 to freeze routing boundaries
        self.use_gumbel = False  # Set True to enable Gumbel-Softmax hard routing
        self.temperature = 1.0  # Gumbel temperature (annealed during training)

        # Running statistics for degree/rho normalization
        self.register_buffer('degree_mean', torch.tensor(0.0))
        self.register_buffer('degree_var', torch.tensor(1.0))
        self.register_buffer('rho_mean', torch.tensor(0.0))
        self.register_buffer('rho_var', torch.tensor(1.0))
        self.register_buffer('num_updates', torch.tensor(0))

    def _update_running_stats(self, log_degree: torch.Tensor, rho: torch.Tensor) -> None:
        """Update running mean/variance using exponential moving average."""
        momentum = 0.1
        with torch.no_grad():
            batch_degree_mean = log_degree.mean()
            batch_degree_var = log_degree.var(unbiased=False)
            batch_rho_mean = rho.mean()
            batch_rho_var = rho.var(unbiased=False)

            if self.num_updates == 0:
                self.degree_mean.copy_(batch_degree_mean)
                self.degree_var.copy_(batch_degree_var)
                self.rho_mean.copy_(batch_rho_mean)
                self.rho_var.copy_(batch_rho_var)
            else:
                self.degree_mean.mul_(1 - momentum).add_(batch_degree_mean * momentum)
                self.degree_var.mul_(1 - momentum).add_(batch_degree_var * momentum)
                self.rho_mean.mul_(1 - momentum).add_(batch_rho_mean * momentum)
                self.rho_var.mul_(1 - momentum).add_(batch_rho_var * momentum)

            self.num_updates.add_(1)

    def forward(
        self,
        x_tangent: torch.Tensor,      # [N, hidden]
        clustering: torch.Tensor,      # [N]
        cone_depth: torch.Tensor,      # [N, 1]
        degree: torch.Tensor,          # [N] integer node degree
        rho: torch.Tensor,             # [N] dehydron density
        ss_onehot: torch.Tensor,       # [N, 3] one-hot secondary structure
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with capacity-aware routing and expert dropout.

        Returns:
            scores: [N, num_experts] routing probabilities
            capacity_loss: scalar asymmetric capacity penalty
        """
        # Log-transform degree (preserves ordering, compresses range)
        log_degree = torch.log1p(degree.float())

        # Update running stats during training
        if self.training:
            self._update_running_stats(log_degree, rho)

        # Normalize to zero-mean unit-variance using running statistics
        norm_degree = (log_degree - self.degree_mean) / (self.degree_var.sqrt() + 1e-8)
        norm_rho = (rho - self.rho_mean) / (self.rho_var.sqrt() + 1e-8)

        # Assemble gate input: [x_tangent, clustering, cone_depth, norm_degree, norm_rho, ss_onehot]
        # In Phase 3, detach x_tangent so backbone updates don't shift routing boundaries
        # In topology_only mode, x_tangent is excluded entirely (information bottleneck)
        if self.topology_only:
            gate_input = torch.cat([
                clustering.unsqueeze(-1),     # [N, 1]
                cone_depth.detach(),          # [N, 1]
                norm_degree.unsqueeze(-1),    # [N, 1]
                norm_rho.unsqueeze(-1),       # [N, 1]
                ss_onehot,                    # [N, 3]
            ], dim=-1)  # Total: [N, 7]
        else:
            gate_x_tangent = x_tangent.detach() if self.detach_gate_input else x_tangent
            gate_input = torch.cat([
                gate_x_tangent,               # [N, hidden]
                clustering.unsqueeze(-1),     # [N, 1]
                cone_depth.detach(),          # [N, 1]
                norm_degree.unsqueeze(-1),    # [N, 1]
                norm_rho.unsqueeze(-1),       # [N, 1]
                ss_onehot,                    # [N, 3]
            ], dim=-1)  # Total: [N, hidden + 7]

        # Compute raw logits
        raw_logits = self.gate_net(gate_input)  # [N, num_experts]

        # --- Two-pass capacity-aware logit adjustment ---
        # Pass 1: compute expected load from unpenalized softmax
        scores_initial = F.softmax(raw_logits, dim=-1)
        expected_load = scores_initial.mean(dim=0)  # [num_experts]

        # Pass 2: subtract quadratic penalty for overloaded experts
        overload_penalty = torch.relu(expected_load - self.capacity_threshold) ** 2
        adjusted_logits = raw_logits - 2.0 * overload_penalty.unsqueeze(0)

        # --- Expert dropout (training only) ---
        if self.training and torch.rand(1).item() < self.expert_dropout_p:
            drop_idx = torch.randint(0, self.num_experts, (1,)).item()
            adjusted_logits[:, drop_idx] = -1e9

        # --- Routing activation ---
        gumbel_active = self.use_gumbel and getattr(self, '_gumbel_active', True)
        if gumbel_active and self.training:
            # Gumbel-Softmax: hard=True gives one-hot forward, smooth gradient backward
            scores = F.gumbel_softmax(adjusted_logits, tau=self.temperature, hard=True, dim=-1)
        elif gumbel_active and not self.training:
            # Eval mode: deterministic argmax (no Gumbel noise)
            scores = torch.zeros_like(adjusted_logits)
            scores.scatter_(1, adjusted_logits.argmax(dim=-1, keepdim=True), 1.0)
        else:
            # Standard soft routing (Phase 1 or legacy mode)
            scores = F.softmax(adjusted_logits, dim=-1)

        # --- Asymmetric capacity loss ---
        # Use soft probabilities for capacity loss (not the hard one-hot)
        soft_scores = F.softmax(adjusted_logits, dim=-1)
        f = soft_scores.mean(dim=0)  # [num_experts]
        capacity_loss = torch.relu(self.min_usage - f).pow(2).sum()

        return scores, capacity_loss


# ==================== 2. MAIN TOKYO EYES v6 ====================

class GOSPConeMapperV6(nn.Module):
    """
    Tokyo Eyes v6 — Topologically-Routed MoE Specialization.

    Key architectural differences from v5:
      - MoE gate receives expanded input (hidden + 7 topological features)
      - Asymmetric capacity loss replaces symmetric balance loss
      - Expert dropout builds redundancy during training
      - Capacity-aware soft routing prevents expert overload

    Preserves from v5:
      - Decoupled RadialHead/AngularHead architecture
      - SE(3)-equivariant backbone
      - EvidentialHead for uncertainty
      - Dual 2D/3D hyperbolic projections
    """

    def __init__(
        self,
        node_dim: int = 4,
        hidden: int = 128,
        num_layers: int = 6,
        num_experts: int = 4,
        projection_dim: int = 64,
        hyp_proj_dim_2d: int = 2,
        hyp_proj_dim_3d: int = 3,
        depth_conditioning: bool = False,
        projection_audit_tolerance: float = 1e-7,
        aleatoric_logvar_min: float = -12.0,
        aleatoric_logvar_max: float = 6.0,
        epistemic_temp_scaling: float = 1.0,
        capacity_threshold: float = 0.4,
        expert_dropout_p: float = 0.15,
        min_usage: float = 0.05,
        topology_only_gate: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth_conditioning = depth_conditioning
        self.projection_audit_tolerance = projection_audit_tolerance

        # Node embedding
        self.node_emb = nn.Linear(node_dim, hidden)

        # SE(3) backbone
        self.convs = nn.ModuleList([
            EquivariantConv(hidden, irreps_hidden="32x0e + 8x1e")
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])

        # Learnable curvature
        self.log_c = nn.Parameter(torch.tensor(0.0))

        # Decoupled radial and angular heads (warm-startable from v5)
        self.radial_head = RadialHead(hidden)
        self.angular_head = AngularHead(hidden)

        # V6 NEW: Enriched MoE gate
        self.gate = TopologicalMoEGateV6(
            hidden_dim=hidden,
            num_experts=num_experts,
            capacity_threshold=capacity_threshold,
            expert_dropout_p=expert_dropout_p,
            min_usage=min_usage,
            topology_only=topology_only_gate,
        )

        # Experts — operate on tangent space (fresh initialization)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(num_experts)
        ])

        # Uncertainty head — hidden + 2 (tangent + depth + cone_width)
        self.uncertainty_head = EvidentialHead(
            hidden_dim=hidden,
            extra_input_dim=2,
            aleatoric_logvar_min=aleatoric_logvar_min,
            aleatoric_logvar_max=aleatoric_logvar_max,
            epistemic_temp_scaling=epistemic_temp_scaling,
        )

        # Euclidean scrubber projection (backward compat)
        self.projection_head = nn.Linear(hidden, projection_dim)

        # Dual hyperbolic projections — 2D disc AND 3D ball
        self.hyp_proj_head_2d = MobiusLinear(hidden, hyp_proj_dim_2d)
        self.hyp_proj_head_3d = MobiusLinear(hidden, hyp_proj_dim_3d)
        self.hyp_proj_dim_2d = hyp_proj_dim_2d
        self.hyp_proj_dim_3d = hyp_proj_dim_3d

    def _project_with_audit(
        self, x: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to ball interior with audit."""
        from geoopt.manifolds.stereographic import math as pmath
        x_proj = pmath.project(x, k=k)
        changed = (x_proj - x).abs().amax(dim=-1) > self.projection_audit_tolerance
        return x_proj, changed.sum(), changed.float().mean()

    @property
    def curvature(self) -> torch.Tensor:
        """Learnable positive curvature."""
        return F.softplus(self.log_c) + 1e-4

    def forward(self, data: Data) -> Dict[str, Any]:
        from geoopt.manifolds.stereographic import math as pmath

        # ── Step 1: Feature embedding ─────────────────────────────────────
        x = self.node_emb(data.x)

        # ── Step 2: Equivariant message passing ───────────────────────────
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, data.edge_index, data.edge_attr)
            x = F.silu(norm(x))
            x = x + x_res

        # ── Step 3: DECOUPLED Hyperbolic Lift (preserved from v5) ─────────
        c = self.curvature
        k = -c

        # Radial pathway — controls depth in hierarchy
        radial_depth = self.radial_head(x)  # [N, 1] positive

        # Angular pathway — controls direction/clustering
        angular_direction = self.angular_head(x)  # [N, hidden] unit vectors

        # Recombine into tangent vector
        tangent_vector = radial_depth * angular_direction  # [N, hidden]

        # Lift to Poincaré ball
        x_hyp = pmath.expmap0(tangent_vector, k=k)
        x_hyp, proj_count_s1, proj_frac_s1 = self._project_with_audit(x_hyp, k=k)

        # Compute depth and cone_width from the ball position
        depth = pmath.dist0(x_hyp, k=k, keepdim=True)  # [N, 1]
        cone_width = torch.exp(-depth)  # [N, 1]

        # ── Step 4: V6 MoE routing with enriched gate ─────────────────────
        x_tangent = pmath.logmap0(x_hyp, k=k)  # [N, hidden]

        # V6: pass expanded topological features to gate
        scores, capacity_loss = self.gate(
            x_tangent=x_tangent,
            clustering=data.clustering,
            cone_depth=depth,
            degree=data.degree,
            rho=data.rho,
            ss_onehot=data.ss_onehot,
        )

        expert_outputs = torch.stack(
            [expert(x_tangent) for expert in self.experts], dim=1
        )  # [N, num_experts, hidden]

        # Weighted combination in tangent space
        x_routed_tangent = torch.einsum("ne,neh->nh", scores, expert_outputs)

        # Re-lift to Poincaré ball
        x_routed_hyp = pmath.expmap0(x_routed_tangent, k=k)
        x_routed_hyp, proj_count_s2, proj_frac_s2 = self._project_with_audit(
            x_routed_hyp, k=k
        )

        # ── Step 5: Uncertainty ───────────────────────────────────────────
        x_routed_tangent_out = pmath.logmap0(x_routed_hyp, k=k)
        x_for_unc = torch.cat([x_routed_tangent_out, depth, cone_width], dim=-1)
        uncertainty, evidence = self.uncertainty_head(x_for_unc)

        # ── Step 6: Projections ───────────────────────────────────────────
        # 2D disc projection
        hyp_proj_2d = self.hyp_proj_head_2d(x_routed_hyp, c=c)
        hyp_proj_2d = pmath.project(hyp_proj_2d, k=k)
        hyp_norms_2d = hyp_proj_2d.norm(dim=-1, keepdim=True)
        hyp_proj_2d = hyp_proj_2d * torch.clamp(0.99 / (hyp_norms_2d + 1e-8), max=1.0)

        # 3D ball projection
        hyp_proj_3d = self.hyp_proj_head_3d(x_routed_hyp, c=c)
        hyp_proj_3d = pmath.project(hyp_proj_3d, k=k)
        hyp_norms_3d = hyp_proj_3d.norm(dim=-1, keepdim=True)
        hyp_proj_3d = hyp_proj_3d * torch.clamp(0.99 / (hyp_norms_3d + 1e-8), max=1.0)

        # Euclidean scrubber projection (backward compat)
        projections = self.projection_head(x_routed_tangent_out)
        if self.depth_conditioning:
            projections = projections * cone_width

        # ── V6 monitoring metrics ─────────────────────────────────────────
        expert_load = scores.mean(dim=0).detach()  # [num_experts]
        routing_entropy = -(scores.mean(dim=0) * torch.log(scores.mean(dim=0) + 1e-8)).sum()

        # ── Audit trail ───────────────────────────────────────────────────
        proj_count_total = (proj_count_s1 + proj_count_s2).detach()
        proj_frac_avg = ((proj_frac_s1 + proj_frac_s2) / 2.0).detach()

        audit_trail = {
            "version": "v6",
            "architecture": "topological_moe_specialization",
            "depth_metric": "hyperbolic_dist0",
            "depth_used_in": ["gate", "cone_loss", "uncertainty_head"],
            "radial_head_scale": self.radial_head.radial_scale.item(),
            "projection_applied_count": proj_count_total,
            "projection_applied_fraction": proj_frac_avg,
            "curvature_value": c.detach(),
            "gate_input_dim": self.gate.hidden_dim + 7,
            "affected_outputs": [
                "cone_depth", "cone_width", "expert_weights",
                "x_hyp", "x_routed_hyp", "hyp_projections_2d", "hyp_projections_3d",
                "uncertainty", "capacity_loss", "expert_load", "routing_entropy",
            ],
            "affected_losses": ["cone_consistency", "capacity_loss"],
            "tangent_space_used_for": ["gate", "experts", "euclidean_projection"],
            "ball_space_used_for": ["hyp_projections_2d", "hyp_projections_3d", "x_hyp_output"],
        }

        return {
            # Backward-compatible outputs (all v5 outputs preserved)
            "projections": projections,
            "uncertainty": uncertainty,
            "cone_depth": depth,
            "cone_width": cone_width,
            "expert_weights": scores,
            "balance_loss": capacity_loss,  # backward compat key
            "evidence": evidence,
            "audit_trail": audit_trail,
            "x_hyp": x_hyp,
            "x_routed_hyp": x_routed_hyp,
            "hyp_projections": hyp_proj_2d,  # backward compat alias
            "hyp_projections_2d": hyp_proj_2d,
            "hyp_projections_3d": hyp_proj_3d,
            "radial_features": radial_depth,
            "angular_features": angular_direction,
            # V6 NEW outputs
            "capacity_loss": capacity_loss,
            "expert_load": expert_load,
            "routing_entropy": routing_entropy,
            "gate_features_used": [
                "x_tangent", "clustering", "cone_depth",
                "log_degree_norm", "rho_norm", "ss_onehot",
            ],
        }
