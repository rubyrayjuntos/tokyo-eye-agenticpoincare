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

from science.dtie.common.poincare_conventions import rescale_tangent_before_expmap
from science.dtie.v6.gnn.hyperbolic_moe import (
    HyperbolicPrototypeGate,
    mobius_weighted_combine,
    project_ball,
    project_disc_2d,
    project_disc_2d_legacy,
)
from science.dtie.v5.gnn.model import (
    EquivariantConv,
    MobiusLinear,
    RadialHead,
    AngularHead,
    EvidentialHead,
    precompute_clustering,
    scan_tensors_for_invalid,
)
from science.dtie.v6.gnn.evidential import (
    DecoupledEvidentialHead,
    expand_coupled_uncertainty_state_dict,
    uncertainty_head_is_decoupled,
)
from science.training.routing_metrics import routing_load_floor_penalty


def resolve_message_passing_edges(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Return edges for EquivariantConv — explicit hyperbolic graph when attached.

    ``edge_index`` / ``edge_attr`` remain the Cα contact graph for provenance,
    resistance profiler, and clustering precompute. Message passing uses
    ``hyperbolic_edge_index`` / ``hyperbolic_edge_attr`` when ``hyperbolic_graph``.
    """
    if bool(getattr(data, "hyperbolic_graph", False)):
        hei = getattr(data, "hyperbolic_edge_index", None)
        hea = getattr(data, "hyperbolic_edge_attr", None)
        if hei is not None and hea is not None:
            return hei, hea
    return data.edge_index, data.edge_attr


# ==================== 1. TOPOLOGICAL MoE GATE V6 ====================

class TopologicalMoEGateV6(nn.Module):
    """
    V6 MoE gate with enriched topological features and capacity-aware routing.

    Gate input: [x_tangent, clustering, cone_depth, log_degree_norm, rho_norm, tau_flag, ss_onehot]
    Total dimensions: hidden + 8

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
        # log_degree, rho, tau, ss_H, ss_E, ss_C). No x_tangent drowning.
        self.topology_only = topology_only
        if topology_only:
            gate_input_dim = 8  # clustering + cone_depth + log_degree + rho + tau + ss_onehot[3]
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 16),
                nn.SiLU(),
                nn.Linear(16, num_experts),
            )
        else:
            # Original: hidden + 8
            gate_input_dim = hidden_dim + 8
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
        # Epoch-level hard bans (train-only); set by ExpertTimeoutController.
        self.expert_timeout_banned: list[int] = []

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
        tau_flag: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with capacity-aware routing and expert dropout.

        Returns:
            scores: [N, num_experts] routing probabilities
            capacity_loss: scalar asymmetric capacity penalty
            routing_load_floor: scalar min-expert share hinge (differentiable)
        """
        # Log-transform degree (preserves ordering, compresses range)
        log_degree = torch.log1p(degree.float())

        # Update running stats during training
        if self.training:
            self._update_running_stats(log_degree, rho)

        # Normalize to zero-mean unit-variance using running statistics
        norm_degree = (log_degree - self.degree_mean) / (self.degree_var.sqrt() + 1e-8)
        norm_rho = (rho - self.rho_mean) / (self.rho_var.sqrt() + 1e-8)
        if tau_flag is None:
            tau_feat = torch.zeros_like(clustering).unsqueeze(-1)
        else:
            tau_feat = tau_flag.unsqueeze(-1).float()

        # Assemble gate input: [x_tangent, clustering, cone_depth, norm_degree, norm_rho, tau, ss_onehot]
        # In Phase 3, detach x_tangent so backbone updates don't shift routing boundaries
        # In topology_only mode, x_tangent is excluded entirely (information bottleneck)
        if self.topology_only:
            gate_input = torch.cat([
                clustering.unsqueeze(-1),     # [N, 1]
                cone_depth.detach(),          # [N, 1]
                norm_degree.unsqueeze(-1),    # [N, 1]
                norm_rho.unsqueeze(-1),       # [N, 1]
                tau_feat,                     # [N, 1]
                ss_onehot,                    # [N, 3]
            ], dim=-1)  # Total: [N, 8]
        else:
            gate_x_tangent = x_tangent.detach() if self.detach_gate_input else x_tangent
            gate_input = torch.cat([
                gate_x_tangent,               # [N, hidden]
                clustering.unsqueeze(-1),     # [N, 1]
                cone_depth.detach(),          # [N, 1]
                norm_degree.unsqueeze(-1),    # [N, 1]
                norm_rho.unsqueeze(-1),       # [N, 1]
                tau_feat,                     # [N, 1]
                ss_onehot,                    # [N, 3]
            ], dim=-1)  # Total: [N, hidden + 8]

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

        # --- Hard expert timeout bans (training only; eval stays unmasked) ---
        if self.training and self.expert_timeout_banned:
            for ban_idx in self.expert_timeout_banned:
                if 0 <= int(ban_idx) < self.num_experts:
                    adjusted_logits[:, int(ban_idx)] = -1e9

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
        routing_load_floor = routing_load_floor_penalty(f, self.min_usage)

        return scores, capacity_loss, routing_load_floor


# ==================== 2. MAIN TOKYO EYES v6 ====================

def infer_v65_model_kwargs(
    state_dict: dict[str, torch.Tensor],
    architecture: dict[str, Any] | str | None = None,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer constructor kwargs from checkpoint payload."""
    arch: dict[str, Any] = {}
    if isinstance(architecture, dict):
        arch = architecture
    tc = training_config or {}
    hyperbolic_gate = arch.get("hyperbolic_gate")
    if hyperbolic_gate is None:
        hyperbolic_gate = any(
            k.startswith("gate.mobius1") or k.startswith("gate.prototype_bank")
            for k in state_dict
        )
    hyperbolic_expert_mix = bool(arch.get("hyperbolic_expert_mix", tc.get("hyperbolic_expert_mix", False)))
    topology_only = bool(arch.get("topology_only_gate", tc.get("topology_only_gate", False)))
    gate_disc_input = arch.get("gate_disc_input")
    if gate_disc_input is None:
        gate_disc_input = any(k.startswith("gate.gate_disc_proj") for k in state_dict)
    else:
        gate_disc_input = bool(gate_disc_input)
    deep_hyperbolic_gate = bool(arch.get("deep_hyperbolic_gate", tc.get("deep_hyperbolic_gate", False)))
    if not deep_hyperbolic_gate:
        deep_hyperbolic_gate = any(k.startswith("gate.mobius3") for k in state_dict)
    gate_disc_scale = float(arch.get("gate_disc_scale", tc.get("gate_disc_scale", 1.0)))
    gate_gumbel = bool(arch.get("gate_gumbel", tc.get("gate_gumbel", False)))
    gate_key = "gate.gate_net.0.weight"
    if hyperbolic_gate and not topology_only:
        topology_only = not any(k.startswith("gate.mobius1") for k in state_dict)
    elif not hyperbolic_gate and gate_key in state_dict:
        topology_only = int(state_dict[gate_key].shape[1]) == 8
    node_dim = int(arch.get("node_dim", tc.get("node_dim", 0)))
    if not node_dim:
        w = state_dict.get("node_emb.weight")
        node_dim = int(w.shape[1]) if w is not None else 4
    return {
        "node_dim": node_dim,
        "hidden": int(arch.get("hidden", 128)),
        "num_experts": int(arch.get("num_experts", 4)),
        "hyperbolic_gate": bool(hyperbolic_gate),
        "hyperbolic_expert_mix": hyperbolic_expert_mix,
        "topology_only_gate": topology_only,
        "gate_disc_input": bool(gate_disc_input),
        "deep_hyperbolic_gate": deep_hyperbolic_gate,
        "gate_disc_scale": gate_disc_scale,
        "gate_gumbel": gate_gumbel,
        "expert_depth_decouple": bool(
            arch.get("expert_depth_decouple", tc.get("expert_depth_decouple", False))
        ),
        "structure_gate": bool(arch.get("structure_gate", tc.get("structure_gate", False))),
        "legacy_disc_projection": infer_legacy_disc_projection_from_checkpoint(
            training_config=tc,
        ),
        "radial_angular_recombine": str(
            arch.get("radial_angular_recombine", tc.get("radial_angular_recombine", "multiply"))
        ),
        "disc_radial_source": resolve_disc_radial_source(
            arch.get("disc_radial_source", tc.get("disc_radial_source"))
        ),
        "disc_projection_path": str(
            arch.get(
                "disc_projection_path",
                tc.get(
                    "disc_projection_path",
                    "post_routing"
                    if infer_legacy_disc_projection_from_checkpoint(training_config=tc)
                    else "pre_routing",
                ),
            )
        ),
        "decoupled_uncertainty_heads": bool(
            arch.get(
                "decoupled_uncertainty_heads",
                tc.get(
                    "decoupled_uncertainty_heads",
                    uncertainty_head_is_decoupled(state_dict),
                ),
            )
        ),
        "uncertainty_from_backbone": bool(
            arch.get("uncertainty_from_backbone", tc.get("uncertainty_from_backbone", False))
        ),
    }


def resolve_disc_projection_path(
    *,
    legacy_disc_projection: bool = False,
    disc_projection_path: str | None = None,
) -> str:
    """Active 2D disc source: pre_routing (x_hyp) or post_routing (x_routed_hyp)."""
    if disc_projection_path is not None:
        path = str(disc_projection_path)
        if path not in ("pre_routing", "post_routing"):
            raise ValueError(f"disc_projection_path must be pre_routing or post_routing, got {path!r}")
        return path
    return "post_routing" if legacy_disc_projection else "pre_routing"


DISC_RADIAL_SOURCES = ("mobius", "radial_depth", "dist0_x_hyp")
# Authoritative-radius overrides: tier-1 save uses thickness + shell, not eff_rank / σ₂/σ₁.
DISC_RADIAL_OVERRIDE_SOURCES = ("radial_depth", "dist0_x_hyp")


def uses_disc_radial_override(disc_radial_source: str | None) -> bool:
    """True when 2D disc radius is decoupled from MobiusLinear magnitude."""
    return str(disc_radial_source or "mobius") in DISC_RADIAL_OVERRIDE_SOURCES


def resolve_disc_radial_source(disc_radial_source: str | None) -> str:
    """Validate pre-routing 2D radial authority mode (v5.5 Lever A)."""
    source = "mobius" if disc_radial_source is None else str(disc_radial_source)
    if source not in DISC_RADIAL_SOURCES:
        raise ValueError(
            f"disc_radial_source must be one of {DISC_RADIAL_SOURCES}, got {source!r}"
        )
    return source


def structural_ball_lift_from_disc(
    z_disc: torch.Tensor,
    hidden: int,
    c: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift frozen 2D structural disc positions to hidden-dim ball embeddings.

    Embeds ``z_disc`` in the first two tangent coordinates, zero-pads to
    ``hidden``, and applies ``expmap0`` so MoE/uncertainty operate on a ball
    point anchored to the structural SSOT layout.
    """
    from geoopt.manifolds.stereographic import math as pmath

    n = z_disc.shape[0]
    device, dtype = z_disc.device, z_disc.dtype
    z2 = z_disc[:, :2]
    tangent2 = pmath.logmap0(z2, k=k)
    tangent = torch.zeros(n, hidden, device=device, dtype=dtype)
    tangent[:, :2] = tangent2
    tangent = rescale_tangent_before_expmap(tangent, c)
    x_hyp = pmath.expmap0(tangent, k=k)
    depth = pmath.dist0(x_hyp, k=k, keepdim=True)
    return x_hyp, tangent, depth


def apply_disc_radial_override(
    hyp_proj_2d_raw: torch.Tensor,
    target_radius: torch.Tensor,
    *,
    k: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Decoupled 2D disc: MobiusLinear direction + authoritative radial depth."""
    from geoopt.manifolds.stereographic import math as pmath

    out_dtype = hyp_proj_2d_raw.dtype
    raw = hyp_proj_2d_raw.float()
    kf = k.float() if isinstance(k, torch.Tensor) else k
    radius = target_radius.float()
    if radius.dim() == 1:
        radius = radius.unsqueeze(-1)
    tangent_2d = pmath.logmap0(raw, k=kf)
    direction_2d = F.normalize(tangent_2d, p=2, dim=-1, eps=eps)
    tangent_2d_scaled = direction_2d * radius
    return pmath.expmap0(tangent_2d_scaled, k=kf).to(dtype=out_dtype)


def infer_legacy_disc_projection_from_checkpoint(
    *,
    training_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    override: bool | None = None,
) -> bool:
    """Default legacy=True for checkpoints saved before pre-routing disc path."""
    if override is not None:
        return bool(override)
    tc = training_config or {}
    if "legacy_disc_projection" in tc:
        return bool(tc["legacy_disc_projection"])
    if tc.get("disc_projection_source") == "pre_routing_x_hyp":
        return False
    if isinstance(metrics, dict) and metrics.get("disc_projection_source") == "pre_routing_x_hyp":
        return False
    if isinstance(metrics, dict) and "legacy_disc_projection" in metrics:
        return bool(metrics["legacy_disc_projection"])
    return True


class GOSPConeMapperV65(nn.Module):
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
        hyperbolic_gate: bool = True,
        hyperbolic_expert_mix: bool = False,
        gate_disc_input: bool = True,
        gate_disc_scale: float = 1.0,
        gate_gumbel: bool = False,
        deep_hyperbolic_gate: bool = False,
        expert_depth_decouple: bool = False,
        structure_gate: bool = False,
        legacy_disc_projection: bool = False,
        disc_projection_path: str | None = None,
        radial_angular_recombine: str = "multiply",
        disc_radial_source: str = "mobius",
        decoupled_uncertainty_heads: bool = False,
        uncertainty_from_backbone: bool = False,
        init_seed: int | None = None,
    ):
        super().__init__()
        self.hidden = hidden
        self.depth_conditioning = depth_conditioning
        self.projection_audit_tolerance = projection_audit_tolerance
        self.hyperbolic_gate = hyperbolic_gate
        self.hyperbolic_expert_mix = hyperbolic_expert_mix
        self.gate_disc_input = gate_disc_input
        self.gate_disc_scale = gate_disc_scale
        self.gate_gumbel = gate_gumbel
        self.deep_hyperbolic_gate = deep_hyperbolic_gate
        self.expert_depth_decouple = expert_depth_decouple
        self.structure_gate = structure_gate
        self.init_seed = init_seed
        # Softer than 0.99 — aggressive clamping collapses angular spread on the disc.
        self.disc_proj_softness = 0.95
        self.disc_projection_path = resolve_disc_projection_path(
            legacy_disc_projection=legacy_disc_projection,
            disc_projection_path=disc_projection_path,
        )
        self.legacy_disc_projection = self.disc_projection_path == "post_routing"
        self.radial_angular_recombine = radial_angular_recombine
        self.disc_radial_source = resolve_disc_radial_source(disc_radial_source)
        self.decoupled_uncertainty_heads = decoupled_uncertainty_heads
        self.uncertainty_from_backbone = uncertainty_from_backbone
        self.gate_mode = "hyperbolic" if hyperbolic_gate else "tangent_mlp"

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
        if radial_angular_recombine in ("mlp_fusion", "angular_lift"):
            self.radial_angular_fusion = nn.Sequential(
                nn.Linear(hidden + 1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            nn.init.zeros_(self.radial_angular_fusion[-1].weight)
            nn.init.zeros_(self.radial_angular_fusion[-1].bias)
        else:
            self.radial_angular_fusion = None

        # V6 MoE gate — hyperbolic prototype (default) or legacy tangent MLP
        if hyperbolic_gate:
            self.gate = HyperbolicPrototypeGate(
                hidden_dim=hidden,
                num_experts=num_experts,
                capacity_threshold=capacity_threshold,
                expert_dropout_p=expert_dropout_p,
                min_usage=min_usage,
                topology_only=topology_only_gate,
                use_disc_position=gate_disc_input,
                disc_feature_scale=gate_disc_scale,
                use_gumbel=gate_gumbel,
                deep_gate=deep_hyperbolic_gate,
                structure_gate=structure_gate,
                init_seed=init_seed,
            )
        else:
            self.gate = TopologicalMoEGateV6(
                hidden_dim=hidden,
                num_experts=num_experts,
                capacity_threshold=capacity_threshold,
                expert_dropout_p=expert_dropout_p,
                min_usage=min_usage,
                topology_only=topology_only_gate,
            )

        # Experts — tangent MLP (Stage 3 can mix outputs on the ball)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(num_experts)
        ])
        if expert_depth_decouple:
            init = torch.linspace(-0.20, 0.20, num_experts)
            self.expert_depth_bias = nn.Parameter(init.unsqueeze(-1))
        else:
            self.expert_depth_bias = None

        # Uncertainty head — hidden + 2 (tangent + depth + cone_width)
        head_cls = DecoupledEvidentialHead if decoupled_uncertainty_heads else EvidentialHead
        self.uncertainty_head = head_cls(
            hidden_dim=hidden,
            extra_input_dim=2,
            aleatoric_logvar_min=aleatoric_logvar_min,
            aleatoric_logvar_max=aleatoric_logvar_max,
            epistemic_temp_scaling=epistemic_temp_scaling,
        )

        # Euclidean scrubber projection (backward compat)
        self.projection_head = nn.Linear(hidden, projection_dim)

        # ResidueStage1/2 supervised heads (pocket + interface + source-leak BCE)
        self.binding_head_pocket = nn.Linear(hidden, 1)
        self.binding_head_interface = nn.Linear(hidden, 1)
        self.binding_head_leak = nn.Linear(hidden, 1)

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

    def _tangent_from_radial_angular(
        self,
        radial_depth: torch.Tensor,
        angular_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Cone multiply (baseline), residual fusion, or direction-only lift ablation."""
        if self.radial_angular_recombine == "angular_lift":
            base = angular_direction
            if self.radial_angular_fusion is not None:
                fused_in = torch.cat([radial_depth, angular_direction], dim=-1)
                return base + self.radial_angular_fusion(fused_in)
            return base
        cone = radial_depth * angular_direction
        if self.radial_angular_recombine != "mlp_fusion" or self.radial_angular_fusion is None:
            return cone
        fused_in = torch.cat([radial_depth, angular_direction], dim=-1)
        return cone + self.radial_angular_fusion(fused_in)

    def _pre_routing_disc_raw(
        self,
        x_hyp: torch.Tensor,
        *,
        radial_depth: torch.Tensor,
        depth: torch.Tensor,
        c: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """2D pre-routing disc before soft clamp; optional decoupled radial override."""
        hyp_proj_2d_pre_raw = self.hyp_proj_head_2d(x_hyp, c=c)
        if self.disc_radial_source == "mobius":
            return hyp_proj_2d_pre_raw
        if self.disc_radial_source == "radial_depth":
            target_radius = radial_depth
        else:
            target_radius = depth
        return apply_disc_radial_override(hyp_proj_2d_pre_raw, target_radius, k=k)

    def _post_routing_disc_raw(
        self,
        x_routed_hyp: torch.Tensor,
        *,
        depth_routed: torch.Tensor,
        c: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """2D post-routing disc before soft clamp; optional decoupled radial override.

        For both ``radial_depth`` and ``dist0_x_hyp`` modes the authoritative
        post-routing radius is ``depth_routed`` (dist0 of the MoE-combined ball
        point) so the 2D disc reflects expert-shifted topology, not pre-routed
        burial depth.
        """
        hyp_proj_2d_post_raw = self.hyp_proj_head_2d(x_routed_hyp, c=c)
        if self.disc_radial_source == "mobius":
            return hyp_proj_2d_post_raw
        return apply_disc_radial_override(hyp_proj_2d_post_raw, depth_routed, k=k)

    def forward(self, data: Data) -> Dict[str, Any]:
        from geoopt.manifolds.stereographic import math as pmath

        # ── Step 1: Feature embedding ─────────────────────────────────────
        x = self.node_emb(data.x)

        # ── Step 2: Equivariant message passing ───────────────────────────
        mp_edge_index, mp_edge_attr = resolve_message_passing_edges(data)
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, mp_edge_index, mp_edge_attr)
            x = F.silu(norm(x))
            x = x + x_res

        # ── Step 3: Hyperbolic lift (learned or structural SSOT) ────────
        c = self.curvature
        k = -c

        structural_frozen = bool(
            getattr(data, "structural_z_disc_frozen", False)
        ) and hasattr(data, "structural_z_disc") and data.structural_z_disc is not None

        if structural_frozen:
            structural_z = data.structural_z_disc.to(dtype=x.dtype, device=x.device)
            if structural_z.shape[0] != x.shape[0]:
                raise ValueError(
                    f"structural_z_disc rows ({structural_z.shape[0]}) "
                    f"!= graph nodes ({x.shape[0]})"
                )
            x_hyp, tangent_vector, radial_depth = structural_ball_lift_from_disc(
                structural_z, hidden=self.hidden, c=c, k=k
            )
            x_hyp, proj_count_s1, proj_frac_s1 = self._project_with_audit(x_hyp, k=k)
            angular_direction = self.angular_head(x)
        else:
            radial_depth = self.radial_head(x)
            angular_direction = self.angular_head(x)
            tangent_vector = self._tangent_from_radial_angular(radial_depth, angular_direction)
            tangent_vector = rescale_tangent_before_expmap(tangent_vector, c)
            x_hyp = pmath.expmap0(tangent_vector, k=k)
            x_hyp, proj_count_s1, proj_frac_s1 = self._project_with_audit(x_hyp, k=k)

        # Compute depth and cone_width from the ball position
        depth = pmath.dist0(x_hyp, k=k, keepdim=True)  # [N, 1]
        cone_width = torch.exp(-depth)  # [N, 1]
        cone_depth_for_loss = depth
        if self.expert_depth_bias is not None:
            # Per-expert depth offsets mixed by routing — breaks shared-depth collapse.
            expert_depths = depth.unsqueeze(1) + self.expert_depth_bias.view(1, -1, 1)

        disc_softness = self.disc_proj_softness
        gate_disc_kw: dict[str, torch.Tensor] = {}

        # Pre-routing disc (always computed for audit + optional gate enrichment)
        if structural_frozen:
            hyp_proj_2d_pre_raw = structural_z[:, :2]
        else:
            hyp_proj_2d_pre_raw = self._pre_routing_disc_raw(
                x_hyp,
                radial_depth=radial_depth,
                depth=depth,
                c=c,
                k=k,
            )
        hyp_proj_2d_pre, disc_r_pre = project_disc_2d(
            hyp_proj_2d_pre_raw, k=k, softness=disc_softness
        )
        if (
            self.disc_projection_path == "pre_routing"
            and self.hyperbolic_gate
            and getattr(self.gate, "use_disc_position", False)
        ):
            gate_disc_kw = {"disc_xy": hyp_proj_2d_pre, "disc_r": disc_r_pre}

        tau_flag = (
            data.x[:, 1]
            if data.x.size(1) > 1
            else torch.zeros(data.x.size(0), device=x_hyp.device, dtype=data.x.dtype)
        )

        # ── Step 4: V6 MoE routing ────────────────────────────────────────
        gate_audit: dict[str, Any] = {}
        routing_load_floor = torch.tensor(0.0, device=x_hyp.device)
        if self.hyperbolic_gate:
            scores, capacity_loss, gate_audit = self.gate(
                x_hyp=x_hyp,
                k=k,
                clustering=data.clustering,
                cone_depth=depth,
                degree=data.degree,
                rho=data.rho,
                ss_onehot=data.ss_onehot,
                tau_flag=tau_flag,
                **gate_disc_kw,
            )
            gate_features = [
                "x_hyp",
                "clustering",
                "cone_depth",
                "log_degree_norm",
                "rho_norm",
                "tau_flag",
                "ss_onehot",
            ]
            if getattr(self.gate, "use_disc_position", False):
                gate_features.extend(["gate_disc_xy", "gate_disc_r"])
            routing_load_floor = gate_audit.get("routing_load_floor", routing_load_floor)
        else:
            x_tangent = pmath.logmap0(x_hyp, k=k)
            scores, capacity_loss, routing_load_floor = self.gate(
                x_tangent=x_tangent,
                clustering=data.clustering,
                cone_depth=depth,
                degree=data.degree,
                rho=data.rho,
                ss_onehot=data.ss_onehot,
                tau_flag=tau_flag,
            )
            gate_features = [
                "x_tangent",
                "clustering",
                "cone_depth",
                "log_degree_norm",
                "rho_norm",
                "tau_flag",
                "ss_onehot",
            ]

        x_tangent = pmath.logmap0(x_hyp, k=k)
        expert_outputs = torch.stack(
            [expert(x_tangent) for expert in self.experts], dim=1
        )

        if self.hyperbolic_expert_mix:
            expert_hyp = torch.stack(
                [
                    pmath.project(
                        pmath.expmap0(
                            rescale_tangent_before_expmap(expert_outputs[:, e, :], c),
                            k=k,
                        ),
                        k=k,
                    )
                    for e in range(len(self.experts))
                ],
                dim=1,
            )
            x_routed_hyp, proj_count_s2, proj_frac_s2 = self._project_with_audit(
                mobius_weighted_combine(expert_hyp, scores, k=k), k=k
            )
        else:
            x_routed_tangent = torch.einsum("ne,neh->nh", scores, expert_outputs)
            x_routed_tangent = rescale_tangent_before_expmap(x_routed_tangent, c)
            x_routed_hyp, proj_count_s2, proj_frac_s2 = self._project_with_audit(
                pmath.expmap0(x_routed_tangent, k=k), k=k
            )

        depth_routed = pmath.dist0(x_routed_hyp, k=k, keepdim=True)
        if self.expert_depth_bias is not None:
            cone_depth_for_loss = (scores.unsqueeze(-1) * expert_depths).sum(dim=1)
            depth_routed = cone_depth_for_loss
        x_routed_tangent_out = pmath.logmap0(x_routed_hyp, k=k)
        unc_source = x_tangent if self.uncertainty_from_backbone else x_routed_tangent_out
        x_for_unc = torch.cat([unc_source, depth, cone_width], dim=-1)
        uncertainty, evidence = self.uncertainty_head(x_for_unc)

        # ── Step 6: Projections (path-aware; always emit pre + post for audit) ──
        hyp_proj_2d_post_raw = self._post_routing_disc_raw(
            x_routed_hyp,
            depth_routed=depth_routed,
            c=c,
            k=k,
        )
        hyp_proj_2d_post, _ = project_disc_2d(
            hyp_proj_2d_post_raw, k=k, softness=disc_softness
        )
        with torch.no_grad():
            hyp_proj_2d_post_legacy, _ = project_disc_2d_legacy(hyp_proj_2d_post_raw, k=k)

        if structural_frozen:
            hyp_proj_2d = hyp_proj_2d_pre
            disc_projection_source = "structural_ssot_frozen"
            hyp_proj_3d_raw = self.hyp_proj_head_3d(x_routed_hyp, c=c)
            hyp_proj_3d, _ = project_ball(hyp_proj_3d_raw, k=k, softness=disc_softness)
        elif self.disc_projection_path == "pre_routing":
            hyp_proj_2d = hyp_proj_2d_pre
            disc_projection_source = "pre_routing_x_hyp"
            hyp_proj_3d_raw = self.hyp_proj_head_3d(x_routed_hyp, c=c)
            hyp_proj_3d, _ = project_ball(hyp_proj_3d_raw, k=k, softness=disc_softness)
        else:
            hyp_proj_2d = hyp_proj_2d_post_legacy
            disc_projection_source = "legacy_post_routing_x_routed_hyp"
            hyp_proj_3d_raw = self.hyp_proj_head_3d(x_routed_hyp, c=c)
            hyp_proj_3d, _ = project_disc_2d_legacy(hyp_proj_3d_raw, k=k)

        hyp_proj_2d_routed = hyp_proj_2d_post
        hyp_proj_2d_legacy_teacher = hyp_proj_2d_post_legacy

        # Euclidean scrubber projection (backward compat)
        projections = self.projection_head(x_routed_tangent_out)
        if self.depth_conditioning:
            projections = projections * cone_width

        binding_logits_pocket = self.binding_head_pocket(x_routed_tangent_out).squeeze(-1)
        binding_logits_interface = self.binding_head_interface(x_routed_tangent_out).squeeze(-1)
        binding_logits_leak = self.binding_head_leak(x_routed_tangent_out).squeeze(-1)

        # ── V6 monitoring metrics ─────────────────────────────────────────
        expert_load = scores.mean(dim=0).detach()  # [num_experts]
        routing_entropy = -(scores.mean(dim=0) * torch.log(scores.mean(dim=0) + 1e-8)).sum()

        # ── Audit trail ───────────────────────────────────────────────────
        proj_count_total = (proj_count_s1 + proj_count_s2).detach()
        proj_frac_avg = ((proj_frac_s1 + proj_frac_s2) / 2.0).detach()

        audit_trail = {
            "version": "v6",
            "architecture": "hyperbolic_prototype_moe" if self.hyperbolic_gate else "topological_moe_specialization",
            "hyperbolic_gate": self.hyperbolic_gate,
            "hyperbolic_expert_mix": self.hyperbolic_expert_mix,
            "gate_disc_input": getattr(self, "gate_disc_input", False),
            "deep_hyperbolic_gate": getattr(self, "deep_hyperbolic_gate", False),
            "disc_projection_source": disc_projection_source,
            "structural_disc_frozen": structural_frozen,
            "structural_disc_layout": (
                getattr(data, "structural_disc_layout", None) if structural_frozen else None
            ),
            "disc_projection_path": self.disc_projection_path,
            "disc_path_used": self.disc_projection_path,
            "gate_mode": self.gate_mode,
            "legacy_disc_projection": self.legacy_disc_projection,
            "radial_angular_recombine": self.radial_angular_recombine,
            "disc_radial_source": self.disc_radial_source,
            "disc_proj_softness": disc_softness if not self.legacy_disc_projection else 0.0,
            "depth_metric": "hyperbolic_dist0",
            "depth_used_in": ["gate", "cone_loss", "uncertainty_head", "cone_depth_routed"],
            "uncertainty_input": "backbone" if self.uncertainty_from_backbone else "routed",
            "radial_head_scale": self.radial_head.radial_scale.item(),
            "projection_applied_count": proj_count_total,
            "projection_applied_fraction": proj_frac_avg,
            "curvature_value": c.detach(),
            "gate_input_dim": (
                (HyperbolicPrototypeGate.TOPO_DIM + HyperbolicPrototypeGate.DISC_DIM)
                if getattr(self.gate, "use_disc_position", False)
                else (
                    HyperbolicPrototypeGate.TOPO_DIM
                    if getattr(self.gate, "topology_only", False)
                    else self.hidden + 7
                )
            ),
            "affected_outputs": [
                "cone_depth", "cone_depth_routed", "cone_width", "expert_weights",
                "x_hyp", "x_routed_hyp", "hyp_projections_2d", "hyp_projections_3d",
                "uncertainty", "capacity_loss", "expert_load", "routing_entropy",
            ],
            "affected_losses": ["cone_consistency", "capacity_loss"],
            "tangent_space_used_for": ["experts", "euclidean_projection"],
            "ball_space_used_for": [
                "gate",
                "hyp_projections_2d",
                "hyp_projections_3d",
                "x_hyp_output",
                "x_routed_hyp",
            ] if self.hyperbolic_gate else [
                "hyp_projections_2d",
                "hyp_projections_3d",
                "x_hyp_output",
            ],
            **gate_audit,
        }

        return {
            # Backward-compatible outputs (all v5 outputs preserved)
            "projections": projections,
            "uncertainty": uncertainty,
            "cone_depth": depth,
            "cone_depth_for_loss": cone_depth_for_loss,
            "cone_depth_routed": depth_routed,
            "cone_width": cone_width,
            "expert_weights": scores,
            "balance_loss": capacity_loss,  # backward compat key
            "evidence": evidence,
            "audit_trail": audit_trail,
            "x_hyp": x_hyp,
            "x_routed_hyp": x_routed_hyp,
            "hyp_projections": hyp_proj_2d,  # backward compat alias
            "hyp_projections_2d": hyp_proj_2d,
            "hyp_projections_2d_pre": hyp_proj_2d_pre,
            "hyp_projections_2d_post": hyp_proj_2d_post,
            "hyp_projections_2d_routed": hyp_proj_2d_routed,
            "hyp_projections_2d_legacy_teacher": hyp_proj_2d_legacy_teacher,
            "disc_path_used": self.disc_projection_path,
            "gate_mode": self.gate_mode,
            "hyp_projections_3d": hyp_proj_3d,
            "radial_features": radial_depth,
            "angular_features": angular_direction,
            # V6 NEW outputs
            "capacity_loss": capacity_loss,
            "routing_load_floor": routing_load_floor,
            "expert_load": expert_load,
            "routing_entropy": routing_entropy,
            "gate_features_used": gate_features,
            "binding_logits_pocket": binding_logits_pocket,
            "binding_logits_interface": binding_logits_interface,
            "binding_logits_leak": binding_logits_leak,
        }


def _checkpoint_num_experts(state_dict: dict[str, torch.Tensor]) -> int | None:
    for key in ("gate.expert_bias", "gate.prototype_bank.prototype_tangent"):
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def _expand_expert_axis0(old: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Copy checkpoint expert rows into a wider model tensor; new rows keep template init."""
    if old.shape[0] >= template.shape[0]:
        return old
    if old.shape[1:] != template.shape[1:]:
        return template
    expanded = template.clone()
    expanded[: old.shape[0]] = old
    return expanded


def adapt_checkpoint_expert_count(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Widen MoE tensors when resuming a 4-expert checkpoint into a 6-expert model."""
    old_e = _checkpoint_num_experts(state_dict)
    new_e = _checkpoint_num_experts(model_state)
    if old_e is None or new_e is None or old_e >= new_e:
        return dict(state_dict)

    adapted: dict[str, torch.Tensor] = {}
    for key, old_t in state_dict.items():
        if key not in model_state:
            adapted[key] = old_t
            continue
        new_t = model_state[key]
        if key.startswith("experts."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) >= old_e:
                continue
        if (
            old_t.ndim >= 1
            and new_t.ndim >= 1
            and old_t.shape[0] == old_e
            and new_t.shape[0] == new_e
            and old_t.shape[1:] == new_t.shape[1:]
        ):
            adapted[key] = _expand_expert_axis0(old_t, new_t)
        elif old_t.shape == new_t.shape:
            adapted[key] = old_t
    return adapted


def load_v65_state_dict(
    model: GOSPConeMapperV65,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """Load weights tolerating gate topo expansion and barcode node_emb widen."""
    from science.dtie.v6.gnn.model import adapt_checkpoint_node_emb_width

    adapted = dict(state_dict)
    if getattr(model, "decoupled_uncertainty_heads", False):
        adapted = expand_coupled_uncertainty_state_dict(adapted)
    topo_key = "gate.topo_encoder.0.weight"
    model_state = model.state_dict()
    if topo_key in adapted and topo_key in model_state:
        old_w = adapted[topo_key]
        new_w = model_state[topo_key]
        if old_w.shape != new_w.shape and old_w.shape[0] == new_w.shape[0] and old_w.shape[1] < new_w.shape[1]:
            expanded = new_w.clone()
            expanded[:, : old_w.shape[1]] = old_w
            adapted[topo_key] = expanded
    adapted = adapt_checkpoint_node_emb_width(adapted, model_state)
    adapted = adapt_checkpoint_expert_count(adapted, model_state)
    incompatible = model.load_state_dict(adapted, strict=False)
    missing = list(getattr(incompatible, "missing_keys", incompatible[0] if isinstance(incompatible, tuple) else []))
    unexpected = list(getattr(incompatible, "unexpected_keys", incompatible[1] if isinstance(incompatible, tuple) else []))
    return missing, unexpected


def verify_v65_checkpoint(
    checkpoint_path: str | None = None,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Verify checkpoint loads into the in-repo GOSPConeMapperV65 architecture.

    Used by health checks and CI to catch architecture/weight drift before ingest.
    """
    from science.contracts.model_registry import get_production_checkpoint_path, resolve_checkpoint_file

    logical = checkpoint_path or get_production_checkpoint_path()
    resolved = resolve_checkpoint_file(logical)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"V6.5 checkpoint not found: {logical}")

    # Import gate module first — missing file fails fast with ImportError.
    from science.dtie.v6.gnn import hyperbolic_moe  # noqa: F401

    checkpoint_data = torch.load(resolved, map_location=device, weights_only=False)
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
        architecture = checkpoint_data.get("architecture")
        training_config = checkpoint_data.get("training_config")
    else:
        state_dict = checkpoint_data
        architecture = training_config = None

    kwargs = infer_v65_model_kwargs(state_dict, architecture, training_config)
    node_dim = int(kwargs.pop("node_dim", 4))
    model = GOSPConeMapperV65(node_dim=node_dim, **kwargs)
    missing, unexpected = load_v65_state_dict(model, state_dict)

    return {
        "path": logical,
        "resolved_path": str(resolved),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "load_ok": not missing,
        "hyperbolic_moe_ok": True,
        "deep_hyperbolic_gate": bool(kwargs.get("deep_hyperbolic_gate")),
        "gate_disc_scale": float(kwargs.get("gate_disc_scale", 1.0)),
        "hyperbolic_gate": bool(kwargs.get("hyperbolic_gate")),
        "hyperbolic_expert_mix": bool(kwargs.get("hyperbolic_expert_mix")),
        "mobius3_in_checkpoint": any(k.startswith("gate.mobius3") for k in state_dict),
    }
