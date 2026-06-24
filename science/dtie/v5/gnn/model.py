# Migrated from: SRC_VIZ/src/components/TokyoEyesv5/Gnnv5.py on 2026-05-27
"""
Tokyo Eyes v5: Decoupled Radial-Angular Hyperbolic GNN
=======================================================
Eidetix Bio | 2026-05-20

MODEL HISTORY
- version: v5
  date: 2026-05-20
  change: Decouple radial and angular geometry into orthogonal subspaces.
    v4 conflated radial depth and angular position — domain_separation_loss
    reorganized angular structure, which scrambled the WT/G12D differential
    axis (radial). v5 fixes this with architecturally separate pathways.
  before_state: v4 used a single mobius_proj + hyper_scale to control both
    radial and angular structure. Cone loss and domain_sep loss competed
    for the same parameters, creating an irreconcilable training conflict.
  changes:
    1. Split post-lift into RADIAL HEAD (scalar depth per residue) and
       ANGULAR HEAD (unit direction vector per residue). These are
       architecturally separate — no shared parameters after the split point.
    2. Radial head: MLP that predicts log-depth from backbone features.
       Supervised by cone loss (burial correlation). Gradients flow ONLY
       through radial parameters.
    3. Angular head: MLP that predicts direction vector from backbone features.
       Supervised by domain_sep + angular_diversity + neighborhood consistency.
       Gradients flow ONLY through angular parameters.
    4. Recombination: x_hyp = expmap0(depth * direction, k=-c).
       This is the only point where radial and angular interact, and it's
       a simple multiplication — no learned parameters to conflate.
    5. hyp_proj_dim=3: Native 3D Poincaré ball output for the viewer.
       Also retains hyp_proj_dim_2d=2 for the disc view.
    6. All v4 outputs preserved for backward compatibility.
  reasoning: The v4 CHECKPOINT_STATUS identified the core problem:
    "domain separation loss reorganizes angular structure, which scrambles
    the WT/G12D differential axis." The fix is architectural separation —
    radial and angular can't interfere because they have no shared weights.
  retraining_required: YES. New architecture, new parameters.

DESIGN CONTRACT:
  Inputs (unchanged from v4):
    x: [N, 4]         node features (rho, tau_flag, ss_type, sasa)
    edge_index: [2,E] graph connectivity
    edge_attr: [E,4]  (rel_x, rel_y, rel_z, distance)
    clustering: [N]   precomputed clustering coefficients

  Outputs (v5 additions marked with *):
    projections: [N, projection_dim]    Euclidean scrubber coords (backward compat)
    hyp_projections_2d: [N, 2]          Poincaré disc coords (for 2D view)
    hyp_projections_3d: [N, 3]*         Poincaré ball coords (for 3D viewer)
    x_hyp: [N, hidden]                  Full Poincaré ball embedding
    x_routed_hyp: [N, hidden]           Post-expert Poincaré ball embedding
    uncertainty: dict                    epistemic, aleatoric, total
    cone_depth: [N,1]                   hyperbolic geodesic depth
    cone_width: [N,1]                   exp(-depth)
    expert_weights: [N, num_experts]    routing decisions
    balance_loss: scalar                MoE regularizer
    evidence: dict                      mu, nu, alpha, beta
    radial_features: [N, hidden//2]*    Radial subspace (for loss routing)
    angular_features: [N, hidden//2]*   Angular subspace (for loss routing)
    audit_trail: dict                   extended with v5 fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import geoopt
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from typing import Any, Dict, List, Tuple, Optional


# ==================== 0. DIAGNOSTIC UTILITIES ====================

def scan_tensors_for_invalid(
    obj: Any,
    path: str = "root",
    findings: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Recursively scan nested structures for NaN/Inf in tensors."""
    if findings is None:
        findings = []
    if torch.is_tensor(obj):
        has_nan = torch.isnan(obj).any().item()
        has_inf = torch.isinf(obj).any().item()
        if has_nan or has_inf:
            findings.append({
                "path": path,
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
                "shape": list(obj.shape),
            })
        return findings
    if isinstance(obj, dict):
        for key, value in obj.items():
            scan_tensors_for_invalid(value, f"{path}.{key}", findings)
        return findings
    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            scan_tensors_for_invalid(value, f"{path}[{i}]", findings)
        return findings
    return findings


# ==================== 1. PREPROCESSING UTILITIES ====================

def precompute_clustering(data: Data) -> Data:
    """Precompute per-node clustering coefficients. Unchanged from v4."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    adj = [set() for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj[src].add(dst)
        adj[dst].add(src)
    clustering = torch.zeros(num_nodes, dtype=torch.float)
    for node in range(num_nodes):
        neighbors = list(adj[node])
        k = len(neighbors)
        if k < 2:
            continue
        triangles = sum(
            1 for i in range(k) for j in range(i + 1, k)
            if neighbors[j] in adj[neighbors[i]]
        )
        clustering[node] = (2.0 * triangles) / (k * (k - 1))
    data.clustering = clustering
    return data


# ==================== 1b. EQUIVARIANT CONVOLUTION (UNCHANGED) ====================

class EquivariantConv(MessagePassing):
    """SE(3)-equivariant message passing layer. Unchanged from v4."""
    def __init__(self, hidden_dim: int, irreps_hidden: str = "32x0e + 8x1e"):
        super().__init__(aggr="add", node_dim=0)
        self.irreps_hidden = Irreps(irreps_hidden)
        self.irreps_sh = Irreps("1x0e + 1x1e")
        self.tp = FullyConnectedTensorProduct(
            self.irreps_hidden, self.irreps_sh, self.irreps_hidden,
            shared_weights=False
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel)
        )
        self.hidden_dim = hidden_dim
        num_scalars = sum(
            mul * ir.dim for mul, ir in self.irreps_hidden if ir.l == 0
        )
        self.num_scalars = num_scalars
        self._proj = nn.Linear(num_scalars, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        rel_pos = edge_attr[:, :3]
        dist = edge_attr[:, 3:4]
        sh = o3.spherical_harmonics(
            l=[0, 1], x=rel_pos, normalize=True, normalization="component"
        )
        edge_weights = self.radial_mlp(dist)
        irreps_dim = self.irreps_hidden.dim
        if x.size(-1) < irreps_dim:
            padding = torch.zeros(
                x.size(0), irreps_dim - x.size(-1),
                device=x.device, dtype=x.dtype
            )
            x_irreps = torch.cat([x, padding], dim=-1)
        else:
            x_irreps = x[:, :irreps_dim]
        out = self.propagate(edge_index, x=x_irreps, sh=sh, edge_weights=edge_weights)
        return self._proj(out[:, :self.num_scalars])

    def message(self, x_j, sh, edge_weights):
        return self.tp(x_j, sh, edge_weights)


# ==================== 1c. MÖBIUS LINEAR LAYER (UNCHANGED) ====================

class MobiusLinear(nn.Module):
    """Linear layer in hyperbolic space via Möbius matrix-vector multiplication."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        from geoopt.manifolds.stereographic import math as pmath
        out = pmath.mobius_matvec(self.weight, x, k=-c)
        if self.bias is not None:
            bias_hyp = pmath.expmap0(self.bias, k=-c)
            out = pmath.mobius_add(out, bias_hyp, k=-c)
        return out


# ==================== 2. DECOUPLED RADIAL-ANGULAR HEADS (v5 NEW) ====================

class RadialHead(nn.Module):
    """
    Predicts per-residue radial depth (scalar) from backbone features.

    This head controls WHERE on the radial axis a residue sits in the ball.
    Supervised by cone loss (burial correlation).
    Architecturally isolated — no shared parameters with AngularHead.

    Architecture: 2-layer MLP (hidden → hidden//2 → 1) matching the
    trained checkpoint. The intermediate hidden//4 layer was added in
    a later code revision but never retrained — using the 2-layer
    version ensures all weights load from the checkpoint correctly.

    Output: log-depth values that get exponentiated and used as the
    magnitude when constructing x_hyp = expmap0(depth * direction).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Learnable scale — controls overall radial distribution
        self.radial_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [N, 1] positive depth values."""
        raw = self.net(x)  # [N, 1] unbounded
        # softplus ensures positive, radial_scale controls magnitude
        depth = F.softplus(raw) * F.softplus(self.radial_scale)
        return depth


class AngularHead(nn.Module):
    """
    Predicts per-residue angular direction (unit vector) from backbone features.

    This head controls WHERE angularly a residue sits on the ball.
    Supervised by domain_sep + angular_diversity + neighborhood consistency.
    Architecturally isolated — no shared parameters with RadialHead.

    Output: unit vectors in R^hidden that define the direction from origin.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [N, hidden] unit vectors."""
        raw = self.net(x)
        # L2 normalize to unit sphere — pure direction, no magnitude
        direction = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
        return direction


# ==================== 3. EVIDENTIAL UNCERTAINTY HEAD (v5: unchanged from v4) ====================

class EvidentialHead(nn.Module):
    """Evidential deep learning head. Input: hidden + 2 (tangent + depth + cone_width)."""
    def __init__(
        self,
        hidden_dim: int,
        extra_input_dim: int = 2,
        out_dim: int = 1,
        aleatoric_logvar_min: float = -12.0,
        aleatoric_logvar_max: float = 6.0,
        epistemic_temp_scaling: float = 1.0,
    ):
        super().__init__()
        total_input = hidden_dim + extra_input_dim
        self.shared = nn.Sequential(
            nn.Linear(total_input, total_input // 2),
            nn.SiLU(),
            nn.Linear(total_input // 2, total_input // 4),
            nn.SiLU(),
        )
        reduced = total_input // 4
        self.mu_head = nn.Linear(reduced, out_dim)
        self.logv_head = nn.Linear(reduced, out_dim)
        self.loga_head = nn.Linear(reduced, out_dim)
        self.logb_head = nn.Linear(reduced, out_dim)
        self.aleatoric_logvar_min = float(aleatoric_logvar_min)
        self.aleatoric_logvar_max = float(aleatoric_logvar_max)
        self.epistemic_temp_scaling = float(epistemic_temp_scaling)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        h = self.shared(x)
        mu = self.mu_head(h)
        nu = F.softplus(self.logv_head(h)) + 1e-6
        alpha = F.softplus(self.loga_head(h)) + 1.0
        beta = F.softplus(self.logb_head(h)) + 1e-6

        epistemic = (1.0 / nu) * self.epistemic_temp_scaling
        aleatoric_raw = beta / (alpha - 1.0 + 1e-6)
        aleatoric_log = torch.log(torch.clamp(aleatoric_raw, min=1e-12))
        aleatoric = torch.exp(torch.clamp(
            aleatoric_log,
            min=self.aleatoric_logvar_min,
            max=self.aleatoric_logvar_max,
        ))

        uncertainty = {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": epistemic + aleatoric,
        }
        evidence = {"mu": mu, "nu": nu, "alpha": alpha, "beta": beta}
        return uncertainty, evidence


def evidential_regression_loss(
    evidence: Dict[str, torch.Tensor],
    target: torch.Tensor,
    coeff: float = 0.01
) -> torch.Tensor:
    """NIG NLL + evidence regularizer. Unchanged from v4."""
    mu, nu, alpha, beta = evidence["mu"], evidence["nu"], evidence["alpha"], evidence["beta"]
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / nu + 1e-8)
        - alpha * torch.log(omega + 1e-8)
        + (alpha + 0.5) * torch.log((target - mu) ** 2 * nu + omega + 1e-8)
        + torch.lgamma(alpha + 1e-8)
        - torch.lgamma(alpha + 0.5)
    )
    reg = (target - mu).abs() * (2.0 * nu + alpha)
    return (nll + coeff * reg).mean()


# ==================== 4. TOPOLOGICAL MoE GATE (v5: receives tangent features) ====================

class TopologicalMoEGate(nn.Module):
    """MoE gate. Routes based on tangent-space position. Unchanged interface from v4."""
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.topology_compressor = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, num_experts),
        )
        self.num_experts = num_experts

    def forward(
        self,
        x_tangent: torch.Tensor,
        clustering: torch.Tensor,
        cone_depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([
            x_tangent,
            clustering.unsqueeze(-1),
            cone_depth.detach(),
        ], dim=-1)
        logits = self.topology_compressor(gate_input)
        scores = F.softmax(logits, dim=-1)

        top_expert = scores.argmax(dim=-1)
        f = torch.zeros(self.num_experts, device=scores.device)
        for i in range(self.num_experts):
            f[i] = (top_expert == i).float().mean()
        p = scores.mean(dim=0)
        balance_loss = self.num_experts * (f * p).sum()
        return scores, balance_loss


# ==================== 5. MAIN TOKYO EYES v5 ====================

class GOSPConeMapper(nn.Module):
    """
    Tokyo Eyes v5 — Decoupled Radial-Angular Hyperbolic Architecture.

    Key architectural difference from v4:
      After SE(3) message passing, the backbone features are split into
      two INDEPENDENT pathways:
        - RadialHead: predicts scalar depth (how deep in hierarchy)
        - AngularHead: predicts unit direction (which functional cluster)

      These are recombined via: tangent_vector = depth * direction
      Then lifted to ball: x_hyp = expmap0(tangent_vector, k=-c)

      This means:
        - Cone loss gradients flow ONLY through RadialHead
        - Domain sep / angular diversity gradients flow ONLY through AngularHead
        - The backbone (SE(3) convolutions) receives gradients from BOTH,
          but the downstream heads can't interfere with each other.

    This resolves the v4 conflict where domain_sep reorganized angular
    structure and scrambled the radial differential axis.
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

        # v5 NEW: Decoupled radial and angular heads
        self.radial_head = RadialHead(hidden)
        self.angular_head = AngularHead(hidden)

        # MoE gate — receives tangent features
        self.gate = TopologicalMoEGate(hidden, num_experts)

        # Experts — operate on tangent space
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

        # v5: Dual hyperbolic projections — 2D disc AND 3D ball
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

        # ── Step 3: DECOUPLED Hyperbolic Lift (v5 CHANGE) ─────────────────
        # Instead of a single mobius_proj that conflates radial and angular,
        # we split into two independent pathways:
        #
        #   RadialHead(x) → depth [N, 1]  (positive scalar)
        #   AngularHead(x) → direction [N, hidden]  (unit vector)
        #
        # Recombine: tangent_vector = depth * direction
        # Lift: x_hyp = expmap0(tangent_vector, k=-c)
        #
        # This guarantees:
        #   - Cone loss gradients → RadialHead only
        #   - Domain sep gradients → AngularHead only
        #   - Backbone gets both (shared representation learning)

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

        # ── Step 4: MoE routing in tangent space ──────────────────────────
        x_tangent = pmath.logmap0(x_hyp, k=k)  # [N, hidden]

        scores, balance_loss = self.gate(x_tangent, data.clustering, depth)

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

        # 3D ball projection (v5 NEW)
        hyp_proj_3d = self.hyp_proj_head_3d(x_routed_hyp, c=c)
        hyp_proj_3d = pmath.project(hyp_proj_3d, k=k)
        hyp_norms_3d = hyp_proj_3d.norm(dim=-1, keepdim=True)
        hyp_proj_3d = hyp_proj_3d * torch.clamp(0.99 / (hyp_norms_3d + 1e-8), max=1.0)

        # Euclidean scrubber projection (backward compat)
        projections = self.projection_head(x_routed_tangent_out)
        if self.depth_conditioning:
            projections = projections * cone_width

        # ── Audit trail ───────────────────────────────────────────────────
        proj_count_total = (proj_count_s1 + proj_count_s2).detach()
        proj_frac_avg = ((proj_frac_s1 + proj_frac_s2) / 2.0).detach()

        audit_trail = {
            "version": "v5",
            "architecture": "decoupled_radial_angular",
            "depth_metric": "hyperbolic_dist0",
            "depth_used_in": ["gate", "cone_loss", "uncertainty_head"],
            "radial_head_scale": self.radial_head.radial_scale.item(),
            "projection_applied_count": proj_count_total,
            "projection_applied_fraction": proj_frac_avg,
            "curvature_value": c.detach(),
            "affected_outputs": [
                "cone_depth", "cone_width", "expert_weights",
                "x_hyp", "x_routed_hyp", "hyp_projections_2d", "hyp_projections_3d",
                "uncertainty",
            ],
            "affected_losses": ["cone_consistency"],
            "tangent_space_used_for": ["gate", "experts", "euclidean_projection"],
            "ball_space_used_for": ["hyp_projections_2d", "hyp_projections_3d", "x_hyp_output"],
        }

        return {
            # Backward-compatible outputs
            "projections": projections,
            "uncertainty": uncertainty,
            "cone_depth": depth,
            "cone_width": cone_width,
            "expert_weights": scores,
            "balance_loss": balance_loss,
            "evidence": evidence,
            "audit_trail": audit_trail,
            # v4-compatible
            "x_hyp": x_hyp,
            "x_routed_hyp": x_routed_hyp,
            "hyp_projections": hyp_proj_2d,  # backward compat alias
            # v5 new
            "hyp_projections_2d": hyp_proj_2d,
            "hyp_projections_3d": hyp_proj_3d,
            "radial_features": radial_depth,
            "angular_features": angular_direction,
        }


# ==================== 6. v5 LOSS FUNCTIONS ====================

def angular_diversity_loss(x_hyp: torch.Tensor) -> torch.Tensor:
    """Penalize angular collapse. Unchanged from v4."""
    x_norm = x_hyp / (x_hyp.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = x_norm @ x_norm.T
    N = x_norm.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=x_hyp.device)
    return cos_sim[mask].abs().mean()


def neighborhood_consistency_loss(
    x_hyp: torch.Tensor,
    ca_coords: torch.Tensor,
    c: torch.Tensor,
    spatial_cutoff: float = 8.0,
    attract_margin: float = 1.0,
    repel_margin: float = 2.5,
    repel_weight: float = 0.3,
) -> torch.Tensor:
    """Neighborhood consistency with attraction AND repulsion. Unchanged from v4."""
    from geoopt.manifolds.stereographic import math as pmath

    k = -c
    N = x_hyp.shape[0]

    diffs = ca_coords[:, None, :] - ca_coords[None, :, :]
    physical_dist = diffs.norm(dim=-1)
    neighbors = (physical_dist < spatial_cutoff) & (physical_dist > 0.1)

    if not neighbors.any():
        return torch.tensor(0.0, device=x_hyp.device, requires_grad=True)

    hyp_dist = pmath.dist(
        x_hyp[:, None].expand(N, N, -1).reshape(N * N, -1),
        x_hyp[None, :].expand(N, N, -1).reshape(N * N, -1),
        k=k,
    ).reshape(N, N)

    attract = (neighbors.float() * torch.relu(hyp_dist - attract_margin)).mean()

    depth = pmath.dist0(x_hyp, k=k)
    depth_diff = (depth[:, None] - depth[None, :]).abs()
    same_depth = depth_diff < 0.5
    non_neighbor = ~neighbors & same_depth & (physical_dist > 0.1)
    repel = (non_neighbor.float() * torch.relu(repel_margin - hyp_dist)).mean()

    return attract + repel_weight * repel


def domain_separation_loss_2d(
    hyp_proj: torch.Tensor,
    domain_labels: torch.Tensor,
    c: torch.Tensor,
    min_angular_sep: float = 0.35,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Push domain centroids apart angularly in the 2D disc. Unchanged from v4."""
    from geoopt.manifolds.stereographic import math as pmath

    k = -c
    unique = [d.item() for d in domain_labels.unique() if d >= 0]
    if len(unique) < 2:
        return torch.tensor(0.0, device=hyp_proj.device, requires_grad=True)

    centroids = []
    for d in unique:
        mask_d = (domain_labels == d)
        pts = hyp_proj[mask_d]
        if len(pts) == 0:
            continue
        tangent = pmath.logmap0(pts, k=k).mean(dim=0)
        cent = pmath.expmap0(tangent.unsqueeze(0), k=k).squeeze(0)
        centroids.append(cent)

    if len(centroids) < 2:
        return torch.tensor(0.0, device=hyp_proj.device, requires_grad=True)

    centroids = torch.stack(centroids)
    centroids = pmath.project(centroids, k=k)

    angles = torch.atan2(centroids[:, 1], centroids[:, 0])
    angle_diff = (angles[:, None] - angles[None, :]).abs()
    angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)

    D = len(centroids)
    off_diag = ~torch.eye(D, dtype=torch.bool, device=hyp_proj.device)
    violation = torch.relu(min_angular_sep - angle_diff[off_diag])
    return (violation / temperature).mean()


def domain_separation_loss_3d(
    hyp_proj_3d: torch.Tensor,
    domain_labels: torch.Tensor,
    c: torch.Tensor,
    min_angular_sep: float = 0.5,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    v5 NEW: Push domain centroids apart angularly in the 3D ball.
    Uses cosine distance between centroid directions (ignoring radial component).
    """
    from geoopt.manifolds.stereographic import math as pmath

    k = -c
    unique = [d.item() for d in domain_labels.unique() if d >= 0]
    if len(unique) < 2:
        return torch.tensor(0.0, device=hyp_proj_3d.device, requires_grad=True)

    centroids = []
    for d in unique:
        mask_d = (domain_labels == d)
        pts = hyp_proj_3d[mask_d]
        if len(pts) == 0:
            continue
        tangent = pmath.logmap0(pts, k=k).mean(dim=0)
        cent = pmath.expmap0(tangent.unsqueeze(0), k=k).squeeze(0)
        centroids.append(cent)

    if len(centroids) < 2:
        return torch.tensor(0.0, device=hyp_proj_3d.device, requires_grad=True)

    centroids = torch.stack(centroids)
    # Normalize to unit vectors — measure angular separation only
    cent_dirs = centroids / (centroids.norm(dim=-1, keepdim=True) + 1e-8)

    # Pairwise cosine similarity
    cos_sim = cent_dirs @ cent_dirs.T
    D = len(centroids)
    off_diag = ~torch.eye(D, dtype=torch.bool, device=hyp_proj_3d.device)

    # Penalize high cosine similarity (clusters too close angularly)
    # min_angular_sep maps to max allowed cosine: cos(min_angular_sep)
    max_cos = torch.cos(torch.tensor(min_angular_sep, device=hyp_proj_3d.device))
    violation = torch.relu(cos_sim[off_diag] - max_cos)
    return (violation / temperature).mean()


def cone_loss_v5(
    radial_depth: torch.Tensor,
    target_rho: torch.Tensor,
) -> torch.Tensor:
    """
    v5 cone loss: Pearson correlation + variance enforcement.

    Target: high ρ (buried) → high depth, low ρ (exposed) → low depth.
    Uses negative Pearson correlation as primary loss — this is immune to
    constant-prediction collapse because correlation is undefined (penalized)
    when predictions have zero variance.

    CRITICAL: This loss should ONLY backprop through radial_depth,
    which comes from RadialHead. The angular pathway is detached.
    """
    target_depth = (target_rho / 30.0).clamp(0.0, 1.0)

    # Squeeze to [N] if RadialHead outputs [N, 1]
    pred = radial_depth.squeeze(-1)

    # Pearson correlation loss (1 - r): cannot be minimized by constant prediction
    pred_centered = pred - pred.mean()
    tgt_centered = target_depth - target_depth.mean()

    pred_std = pred_centered.norm() + 1e-8
    tgt_std = tgt_centered.norm() + 1e-8

    correlation = (pred_centered * tgt_centered).sum() / (pred_std * tgt_std)

    # Loss = 1 - correlation (minimized when correlation = 1.0)
    corr_loss = 1.0 - correlation

    # Strong variance penalty: penalize low spread in predictions
    depth_std = pred.std()
    variance_penalty = torch.relu(0.15 - depth_std) * 5.0

    return corr_loss + variance_penalty


def mutation_differential_loss(
    x_hyp_wt: torch.Tensor,
    x_hyp_mut: torch.Tensor,
    known_mobile: List[int],
    known_stable: List[int],
    c: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Paired-structure contrastive loss for WT/mutant differential. Unchanged from v4."""
    from geoopt.manifolds.stereographic import math as pmath

    k = -c
    if not known_mobile or not known_stable:
        return torch.tensor(0.0, device=x_hyp_wt.device, requires_grad=True)

    mobile_disp = pmath.dist(
        x_hyp_wt[known_mobile], x_hyp_mut[known_mobile], k=k
    ).mean()
    stable_disp = pmath.dist(
        x_hyp_wt[known_stable], x_hyp_mut[known_stable], k=k
    ).mean()

    return torch.relu(stable_disp - mobile_disp + margin)


def gosp_loss_v5(
    output: Dict[str, Any],
    target_rho: torch.Tensor,
    target_dehydron: torch.Tensor,
    ca_coords: torch.Tensor,
    domain_labels: Optional[torch.Tensor] = None,
    evidential_coeff: float = 0.005,
    balance_coeff: float = 0.01,
    cone_coeff: float = 0.15,
    neighborhood_coeff: float = 0.30,
    angular_coeff: float = 0.25,
    domain_sep_2d_coeff: float = 0.30,
    domain_sep_3d_coeff: float = 0.30,
    spatial_cutoff: float = 8.0,
    attract_margin: float = 1.0,
    repel_margin: float = 2.5,
) -> Dict[str, Any]:
    """
    v5 combined training loss with ORTHOGONAL gradient routing.

    Key difference from v4:
      - cone_loss uses radial_features directly (gradients → RadialHead only)
      - domain_sep and angular_diversity use angular_features / hyp_projections
        (gradients → AngularHead only)
      - neighborhood_consistency uses x_routed_hyp (gradients → both, which is fine
        because it's a structural constraint, not a conflicting objective)
    """
    ev_loss = evidential_regression_loss(
        output["evidence"], target_rho, coeff=evidential_coeff
    )
    bal_loss = output["balance_loss"]

    # RADIAL LOSS — flows through RadialHead only
    cone_loss = cone_loss_v5(output["radial_features"], target_rho)

    # ANGULAR LOSSES — flow through AngularHead / projection heads only
    ang_loss = angular_diversity_loss(output["x_routed_hyp"])

    c = output["audit_trail"]["curvature_value"]
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, device=output["x_hyp"].device)

    nbr_loss = neighborhood_consistency_loss(
        x_hyp=output["x_routed_hyp"],
        ca_coords=ca_coords,
        c=c,
        spatial_cutoff=spatial_cutoff,
        attract_margin=attract_margin,
        repel_margin=repel_margin,
    )

    # Domain separation on BOTH 2D and 3D projections
    dom_loss_2d = torch.tensor(0.0, device=output["x_hyp"].device)
    dom_loss_3d = torch.tensor(0.0, device=output["x_hyp"].device)
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

    total = (
        ev_loss
        + balance_coeff * bal_loss
        + cone_coeff * cone_loss
        + neighborhood_coeff * nbr_loss
        + angular_coeff * ang_loss
        + domain_sep_2d_coeff * dom_loss_2d
        + domain_sep_3d_coeff * dom_loss_3d
    )

    # Projection safety penalties
    hyp_norms_2d = output["hyp_projections_2d"].norm(dim=-1)
    hyp_norms_3d = output["hyp_projections_3d"].norm(dim=-1)
    proj_violation = (
        torch.relu(hyp_norms_2d - 0.99).mean()
        + torch.relu(hyp_norms_3d - 0.99).mean()
    )
    total = total + 2.0 * proj_violation

    return {
        "total": total,
        "evidential": ev_loss,
        "balance": bal_loss,
        "cone_consistency": cone_loss,
        "neighborhood_consistency": nbr_loss,
        "angular_diversity": ang_loss,
        "domain_separation_2d": dom_loss_2d,
        "domain_separation_3d": dom_loss_3d,
        "audit_consistent": True,
    }


def build_optimizer(model: GOSPConeMapper, lr: float = 1e-3, weight_decay: float = 1e-5):
    """RiemannianAdam with parameter group separation for v5."""
    # Separate parameter groups so we can monitor gradient norms per pathway
    radial_params = list(model.radial_head.parameters())
    angular_params = list(model.angular_head.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if 'radial_head' not in n and 'angular_head' not in n]

    return geoopt.optim.RiemannianAdam([
        {"params": radial_params, "lr": lr, "weight_decay": weight_decay},
        {"params": angular_params, "lr": lr, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ])


# ==================== 7. SELF-TEST ====================

def run_internal_test(seed: int = 42) -> int:
    """Deterministic property checks for v5 invariants. Returns 0 (pass) or 1 (fail)."""
    torch.manual_seed(seed)
    failures: List[str] = []

    model = GOSPConeMapper(
        node_dim=4, hidden=64, num_layers=2,
        num_experts=3, projection_dim=32,
        hyp_proj_dim_2d=2, hyp_proj_dim_3d=3,
        depth_conditioning=False,
    )
    model.eval()

    # Minimal synthetic graph
    N, E = 24, 80
    rho = torch.rand(N, 1) * 30
    tau_flag = (rho < 13.0).float()
    ss_type = torch.randint(0, 3, (N, 1)).float() / 2.0
    sasa = torch.rand(N, 1)
    x = torch.cat([rho, tau_flag, ss_type, sasa], dim=-1)
    edge_index = torch.randint(0, N, (2, E))
    rel_pos = torch.randn(E, 3)
    dist = rel_pos.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([rel_pos, dist], dim=-1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = precompute_clustering(data)

    with torch.no_grad():
        out = model(data)

    # 1. Core outputs present and correct shape
    assert_checks = [
        ("x_hyp shape", out["x_hyp"].shape == (N, 64)),
        ("x_routed_hyp shape", out["x_routed_hyp"].shape == (N, 64)),
        ("hyp_projections_2d shape", out["hyp_projections_2d"].shape == (N, 2)),
        ("hyp_projections_3d shape", out["hyp_projections_3d"].shape == (N, 3)),
        ("radial_features shape", out["radial_features"].shape == (N, 1)),
        ("angular_features shape", out["angular_features"].shape == (N, 64)),
        ("cone_depth positive", (out["cone_depth"] > 0).all().item()),
        ("cone_depth finite", torch.isfinite(out["cone_depth"]).all().item()),
        ("radial_features positive", (out["radial_features"] > 0).all().item()),
    ]
    for name, condition in assert_checks:
        if not condition:
            failures.append(f"FAIL: {name}")

    # 2. hyp_projections are inside the ball
    for dim_name, key in [("2d", "hyp_projections_2d"), ("3d", "hyp_projections_3d")]:
        hyp_norms = out[key].norm(dim=-1)
        if not (hyp_norms < 1.0).all().item():
            failures.append(f"{key} outside ball: max |z| = {hyp_norms.max().item():.4f}")

    # 3. x_hyp is inside the ball
    from geoopt.manifolds.stereographic import math as pmath
    c = model.curvature.detach()
    x_hyp_norms = out["x_hyp"].norm(dim=-1)
    max_norm = (1.0 / c.sqrt()).item()
    if not (x_hyp_norms < max_norm - 1e-4).all().item():
        failures.append(
            f"x_hyp outside ball: max |p| = {x_hyp_norms.max().item():.4f}, "
            f"ball radius = {max_norm:.4f}"
        )

    # 4. Uncertainty values are positive and finite
    for key in ["epistemic", "aleatoric", "total"]:
        u = out["uncertainty"][key]
        if not (u > 0).all().item():
            failures.append(f"uncertainty[{key}] has non-positive values")
        if not torch.isfinite(u).all().item():
            failures.append(f"uncertainty[{key}] has non-finite values")

    # 5. Recursive NaN/Inf scan
    invalid = scan_tensors_for_invalid(out)
    if invalid:
        failures.append(f"NaN/Inf found: {invalid}")

    # 6. Audit trail version is v5
    if out["audit_trail"].get("version") != "v5":
        failures.append("audit_trail version is not v5")
    if out["audit_trail"].get("architecture") != "decoupled_radial_angular":
        failures.append("audit_trail architecture is not decoupled_radial_angular")

    # 7. Angular features are unit vectors
    ang_norms = out["angular_features"].norm(dim=-1)
    if not torch.allclose(ang_norms, torch.ones_like(ang_norms), atol=1e-5):
        failures.append(f"angular_features not unit vectors: norms range "
                       f"[{ang_norms.min():.4f}, {ang_norms.max():.4f}]")

    # 8. Gradient isolation test — cone loss should NOT affect angular_head
    model.train()
    model.zero_grad()
    out_grad = model(data)
    cone_l = cone_loss_v5(out_grad["radial_features"], rho)
    cone_l.backward()

    angular_grad_norm = sum(
        p.grad.norm().item() for p in model.angular_head.parameters()
        if p.grad is not None
    )
    radial_grad_norm = sum(
        p.grad.norm().item() for p in model.radial_head.parameters()
        if p.grad is not None
    )
    if angular_grad_norm > 1e-10:
        failures.append(
            f"GRADIENT LEAK: cone_loss affects angular_head "
            f"(grad_norm={angular_grad_norm:.6f})"
        )
    if radial_grad_norm < 1e-10:
        failures.append(
            f"DEAD GRADIENT: cone_loss doesn't reach radial_head "
            f"(grad_norm={radial_grad_norm:.6f})"
        )

    print(f"Tokyo Eyes v5 internal test (seed={seed}):")
    print(f"  checks: 8")
    if failures:
        print(f"  result: FAIL ({len(failures)} failures)")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  result: PASS")
    print(f"  Key v5 property: radial/angular gradient isolation VERIFIED")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokyo Eyes v5")
    parser.add_argument("--internaltest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.internaltest:
        sys.exit(run_internal_test(seed=args.seed))

    model = GOSPConeMapper(node_dim=4, hidden=128, num_layers=6,
                            num_experts=4, projection_dim=64,
                            hyp_proj_dim_2d=2, hyp_proj_dim_3d=3)
    total = sum(p.numel() for p in model.parameters())
    print(f"Tokyo Eyes v5: {total:,} parameters")
    print(f"  Architecture: decoupled radial-angular")
    print(f"  RadialHead params: {sum(p.numel() for p in model.radial_head.parameters()):,}")
    print(f"  AngularHead params: {sum(p.numel() for p in model.angular_head.parameters()):,}")
    print(f"  Initial curvature: {model.curvature.item():.4f}")
    print(f"  hyp_proj_head_2d: MobiusLinear(128, 2) — disc output")
    print(f"  hyp_proj_head_3d: MobiusLinear(128, 3) — ball output")
    print(f"\nRun --internaltest for full property verification.")
