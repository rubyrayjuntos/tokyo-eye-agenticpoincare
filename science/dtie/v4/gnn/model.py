# Migrated from: SRC_VIZ/src/components/Gnnv4.py on 2026-05-27
"""
Tokyo Eyes v4: Geometric Ontological State Protein Graph Neural Network
========================================================================
Eidetix Bio | Ray + Codex | v4

MODEL HISTORY
- version: v4
  date: 2026-05-13
  change: Keep computation in hyperbolic space from Step 3 onward.
    v3 entered the Poincaré ball, extracted one scalar (cone_depth),
    then immediately returned to Euclidean for MoE routing, expert
    computation, uncertainty, and projections. v4 keeps everything
    in the ball or its tangent space through to the output.
  before_state: v3 gate and experts received Euclidean x (post-message-passing,
    pre-hyperbolic-lift). Uncertainty received Euclidean x_routed.
    Projections were Euclidean nn.Linear output.
    x_hyp was internal only — never exposed as an output.
  changes:
    1. Gate and experts receive x_tangent = logmap0(x_hyp) — tangent
       space at the origin, which is Euclidean but carries hyperbolic
       position information (direction and magnitude from origin).
    2. Expert outputs are combined in tangent space (weighted sum),
       then re-lifted to the ball via expmap0 → x_routed_hyp.
    3. Uncertainty head receives cat([logmap0(x_routed_hyp), depth,
       cone_width]) — tangent features + radial geometry. Input dim
       is hidden + 2.
    4. Projection head: MobiusLinear(hidden, hyp_proj_dim=2) produces
       native 2D Poincaré disc coordinates. The original Euclidean
       projection head (nn.Linear → projection_dim=64) is kept for
       scrubber/ranking backward compatibility.
    5. x_hyp and x_routed_hyp are exposed in forward() output dict.
  reasoning: The witness complex (Phase 3) should operate on hyperbolic
    distances in the GNN's learned conformational hierarchy space. The
    uncertainty decomposition should reflect position in that hierarchy,
    not position in an arbitrary Euclidean latent space. The 2D disc
    projection should be a native hyperbolic map, not a post-hoc PCA.
  retraining_required: YES. Expert and uncertainty head input shapes
    changed. Existing robust_experts.pt checkpoint is incompatible.
  who: Ray + Claude

CHECKPOINT NOTE:
  robust_experts.pt was trained with v3 (Euclidean experts, hidden-dim
  uncertainty head). It cannot be loaded into v4 with strict=True.
  Retrain from scratch using this architecture. Input contract is
  preserved: node_dim=4 (rho, tau_flag, ss_type, sasa).

DESIGN CONTRACT (unchanged from v3):
  Inputs:
    x: [N, 4]         node features (rho, tau_flag, ss_type, sasa)
    edge_index: [2,E] graph connectivity
    edge_attr: [E,4]  (rel_x, rel_y, rel_z, distance)
    clustering: [N]   precomputed clustering coefficients

  Outputs (v4 additions marked with *):
    projections: [N, projection_dim]    Euclidean scrubber coords (backward compat)
    hyp_projections: [N, hyp_proj_dim]* True Poincaré disc coords (new)
    x_hyp: [N, hidden]*                Full Poincaré ball embedding (new)
    x_routed_hyp: [N, hidden]*         Post-expert Poincaré ball embedding (new)
    uncertainty: dict epistemic, aleatoric, total (now hyperbolic-aware)
    cone_depth: [N,1]                   hyperbolic geodesic depth (unchanged)
    cone_width: [N,1]                   exp(-depth) (unchanged)
    expert_weights: [N, num_experts]    routing decisions (unchanged)
    balance_loss: scalar                MoE regularizer (unchanged)
    evidence: dict mu, nu, alpha, beta  (unchanged interface, new inputs)
    audit_trail: dict                   extended with v4 fields
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
    """
    Precompute per-node clustering coefficients.
    Unchanged from v3. Runs once during data loading.
    """
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
    """
    SE(3)-equivariant message passing layer. Unchanged from v3.
    """
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
    """
    Linear layer in hyperbolic space via Möbius matrix-vector multiplication.
    Unchanged from v3. Used for both the internal mobius_proj and the new
    hyp_proj_head that outputs 2D disc coordinates.
    """
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


# ==================== 2. EVIDENTIAL UNCERTAINTY HEAD (v4: hyperbolic-aware) ====================

class EvidentialHead(nn.Module):
    """
    Evidential deep learning head. v4 change: accepts hidden + extra_input_dim.

    In v3, this received Euclidean x_routed (shape [N, hidden]).
    In v4, it receives cat([logmap0(x_routed_hyp), depth, cone_width])
    so input_dim = hidden + 2. The extra 2 dimensions give the head
    explicit knowledge of where in the conformational hierarchy the
    residue sits (depth = how deep, cone_width = void width at that depth).

    This makes the uncertainty geometrically grounded:
      - A residue deep in the hierarchy with high aleatoric uncertainty
        is a genuine void in a well-constrained structural region.
      - A shallow residue with high epistemic uncertainty is a training
        gap near the conformational surface.
    """
    def __init__(
        self,
        hidden_dim: int,
        extra_input_dim: int = 2,  # depth + cone_width in v4
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
    """NIG NLL + evidence regularizer. Unchanged from v3."""
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


# ==================== 3. TOPOLOGICAL MoE GATE (v4: receives tangent features) ====================

class TopologicalMoEGate(nn.Module):
    """
    MoE gate. v4 change: input x is now x_tangent = logmap0(x_hyp), not
    the post-message-passing Euclidean x. The gate now routes based on
    hyperbolic position (via tangent space projection) rather than
    Euclidean feature position. Interface is identical.
    """
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
        x_tangent: torch.Tensor,  # v4: logmap0(x_hyp), not Euclidean x
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

        # Switch Transformer balance loss (Fedus et al. 2021)
        top_expert = scores.argmax(dim=-1)
        f = torch.zeros(self.num_experts, device=scores.device)
        for i in range(self.num_experts):
            f[i] = (top_expert == i).float().mean()
        p = scores.mean(dim=0)
        balance_loss = self.num_experts * (f * p).sum()
        return scores, balance_loss


# ==================== 4. MAIN TOKYO EYES v4 ====================

class GOSPConeMapper(nn.Module):
    """
    Tokyo Eyes v4 — Full Hyperbolic Architecture.

    Steps 1–2 are identical to v3. Step 3 is identical (hyperbolic lift).
    Steps 4–6 are rewritten to stay in hyperbolic/tangent space.

    Step 4 (v4): Gate and experts receive x_tangent = logmap0(x_hyp).
      Experts are standard Euclidean MLPs operating on tangent space.
      Expert outputs are combined in tangent space (weighted sum),
      then re-lifted to the ball: x_routed_hyp = expmap0(x_routed_tangent).

    Step 5 (v4): Uncertainty head receives
      cat([logmap0(x_routed_hyp), depth, cone_width]).
      This grounds uncertainty in the hyperbolic hierarchy.

    Step 6 (v4): Two projection heads.
      hyp_proj_head: MobiusLinear(hidden, hyp_proj_dim) → native disc coords.
      projection_head: nn.Linear(hidden, projection_dim) → Euclidean scrubber
        (applied to logmap0(x_routed_hyp) for backward compatibility).

    Output additions vs v3:
      x_hyp: [N, hidden]         Poincaré ball embedding after Step 3.
      x_routed_hyp: [N, hidden]  Poincaré ball embedding after Step 4.
      hyp_projections: [N, 2]    Native disc coordinates (Step 6).
    """

    def __init__(
        self,
        node_dim: int = 4,
        hidden: int = 128,
        num_layers: int = 6,
        num_experts: int = 4,
        projection_dim: int = 64,
        hyp_proj_dim: int = 2,
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

        # Node embedding — 4-dim input contract preserved from v3.
        self.node_emb = nn.Linear(node_dim, hidden)

        # SE(3) backbone — unchanged from v3.
        self.convs = nn.ModuleList([
            EquivariantConv(hidden, irreps_hidden="32x0e + 8x1e")
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])

        # Learnable curvature — unchanged from v3.
        self.log_c = nn.Parameter(torch.tensor(0.0))

        # Learnable hyperbolic scale — controls how far into the ball points land.
        # Multiplies tangent vectors BEFORE expmap0. At init=0.5, points land at
        # moderate radius. The optimizer can push this down (toward origin) or up
        # (toward boundary) with clean gradients — no tanh saturation.
        self.hyper_scale = nn.Parameter(torch.tensor(0.5))

        # Hyperbolic lift — unchanged from v3.
        self.hyper_lift = nn.Linear(hidden, hidden)
        self.mobius_proj = MobiusLinear(hidden, hidden)

        # MoE gate — receives tangent features in v4 (same interface).
        self.gate = TopologicalMoEGate(hidden, num_experts)

        # Experts — operate on tangent space in v4.
        # Architecture is identical to v3; the input has changed semantically.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(num_experts)
        ])

        # Uncertainty head — v4: takes hidden + 2 (tangent + depth + cone_width).
        self.uncertainty_head = EvidentialHead(
            hidden_dim=hidden,
            extra_input_dim=2,
            aleatoric_logvar_min=aleatoric_logvar_min,
            aleatoric_logvar_max=aleatoric_logvar_max,
            epistemic_temp_scaling=epistemic_temp_scaling,
        )

        # Euclidean scrubber projection — unchanged interface, new input.
        # Applied to logmap0(x_routed_hyp) for backward compatibility.
        self.projection_head = nn.Linear(hidden, projection_dim)

        # v4 NEW: Native hyperbolic disc projection.
        # MobiusLinear maps hidden-dim ball → hyp_proj_dim-dim ball.
        # Output is true Poincaré disc coordinates when hyp_proj_dim=2.
        self.hyp_proj_head = MobiusLinear(hidden, hyp_proj_dim)
        self.hyp_proj_dim = hyp_proj_dim

    def _project_with_audit(
        self, x: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to ball interior with audit. Unchanged from v3."""
        from geoopt.manifolds.stereographic import math as pmath
        x_proj = pmath.project(x, k=k)
        changed = (x_proj - x).abs().amax(dim=-1) > self.projection_audit_tolerance
        return x_proj, changed.sum(), changed.float().mean()

    @property
    def curvature(self) -> torch.Tensor:
        """Learnable positive curvature. Unchanged from v3."""
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

        # ── Step 3: Hyperbolic cone projection ────────────────────────────
        # v4.3: hyper_scale controls radial distribution.
        # The previous approach (hyper_lift → expmap0 → mobius_proj) produced
        # boundary-shell collapse because mobius_proj is unconstrained and pushes
        # all points to the same radius. The fix:
        #   1. hyper_lift produces tangent vectors
        #   2. L2-normalize to unit VECTOR norm (not layer_norm which gives √dim)
        #   3. softplus(hyper_scale) sets the tangent vector magnitude
        #   4. expmap0 maps to ball at controlled radius
        # This gives the optimizer a DIRECT lever on radius via hyper_scale,
        # while the backbone controls DIRECTION (angular structure).
        # mobius_proj adds hyperbolic nonlinearity; its output is also
        # normalized+scaled to prevent boundary collapse.
        c = self.curvature
        k = -c

        x_pre = self.hyper_lift(x)
        # L2 normalize to unit vectors, then scale controls radius
        x_pre = x_pre / (x_pre.norm(dim=-1, keepdim=True) + 1e-8)
        x_pre = x_pre * F.softplus(self.hyper_scale)
        x_hyp = pmath.expmap0(x_pre, k=k)
        x_hyp, proj_count_s1, proj_frac_s1 = self._project_with_audit(x_hyp, k=k)

        # Möbius projection adds hyperbolic mixing (direction-dependent transform).
        # It creates radial variance (different directions → different radii).
        # Re-scale by MEAN norm (not per-vector) to preserve relative radial
        # differences while controlling overall radius via hyper_scale.
        x_hyp = self.mobius_proj(x_hyp, c=c)
        x_hyp, proj_count_s2, proj_frac_s2 = self._project_with_audit(x_hyp, k=k)
        x_tangent_post = pmath.logmap0(x_hyp, k=k)
        mean_norm = x_tangent_post.norm(dim=-1, keepdim=True).mean()
        x_tangent_post = x_tangent_post / (mean_norm + 1e-8) * F.softplus(self.hyper_scale)
        x_hyp = pmath.expmap0(x_tangent_post, k=k)
        x_hyp = pmath.project(x_hyp, k=k)

        depth = pmath.dist0(x_hyp, k=k, keepdim=True)   # [N, 1] hyperbolic
        cone_width = torch.exp(-depth)                    # [N, 1]

        # ── Step 4: MoE routing in tangent space ──────────────────────────
        # v4 CHANGE: gate and experts receive tangent-space representation
        # of x_hyp, not Euclidean x.
        #
        # logmap0 maps a point in the ball back to the tangent space at
        # the origin. The result is Euclidean (valid for linear layers)
        # but encodes hyperbolic position: direction = angular position
        # in hierarchy, magnitude ∝ hyperbolic distance from origin.
        x_tangent = pmath.logmap0(x_hyp, k=k)  # [N, hidden]

        scores, balance_loss = self.gate(x_tangent, data.clustering, depth)

        # Experts operate on tangent-space features.
        expert_outputs = torch.stack(
            [expert(x_tangent) for expert in self.experts], dim=1
        )  # [N, num_experts, hidden]

        # Weighted combination in tangent space (Fréchet mean approximation).
        # This is the correct first-order approximation to the hyperbolic
        # weighted mean — average in tangent space, then re-lift to ball.
        # Direct Euclidean einsum on ball points would give a point outside
        # the ball; tangent-space averaging followed by expmap is stable.
        x_routed_tangent = torch.einsum("ne,neh->nh", scores, expert_outputs)

        # Re-lift to Poincaré ball.
        x_routed_hyp = pmath.expmap0(x_routed_tangent, k=k)
        x_routed_hyp, proj_count_s3, proj_frac_s3 = self._project_with_audit(
            x_routed_hyp, k=k
        )

        # ── Step 5: Uncertainty in hyperbolic space ───────────────────────
        # v4 CHANGE: uncertainty head receives cat of:
        #   logmap0(x_routed_hyp) — where in hierarchy (tangent coords)
        #   depth                 — radial distance from origin
        #   cone_width            — probability void width at this depth
        #
        # This grounds the NIG decomposition in hyperbolic geometry:
        # epistemic uncertainty reflects position uncertainty in the ball,
        # aleatoric uncertainty reflects genuine openness at that depth.
        x_routed_tangent_out = pmath.logmap0(x_routed_hyp, k=k)
        x_for_unc = torch.cat([x_routed_tangent_out, depth, cone_width], dim=-1)
        uncertainty, evidence = self.uncertainty_head(x_for_unc)

        # ── Step 6: Projections ───────────────────────────────────────────
        # v4 NEW: Native hyperbolic disc projection.
        # MobiusLinear maps x_routed_hyp (hidden-dim ball) to hyp_proj_dim-dim ball.
        # When hyp_proj_dim=2, output coordinates are valid Poincaré disc coords.
        hyp_projections = self.hyp_proj_head(x_routed_hyp, c=c)
        hyp_projections = pmath.project(hyp_projections, k=k)  # enforce ball
        # Extra safety clamp — hard guarantee |z| < 0.99
        hyp_norms = hyp_projections.norm(dim=-1, keepdim=True)
        hyp_projections = hyp_projections * torch.clamp(0.99 / (hyp_norms + 1e-8), max=1.0)

        # Euclidean scrubber projection — backward compatible.
        # Applied to tangent-space features for stability.
        projections = self.projection_head(x_routed_tangent_out)
        if self.depth_conditioning:
            projections = projections * cone_width

        # ── Audit trail ───────────────────────────────────────────────────
        proj_count_total = (proj_count_s1 + proj_count_s2 + proj_count_s3).detach()
        proj_frac_avg = ((proj_frac_s1 + proj_frac_s2 + proj_frac_s3) / 3.0).detach()

        depth_used_in = ["gate", "cone_loss", "uncertainty_head"]
        if self.depth_conditioning:
            depth_used_in.append("projection")

        audit_trail = {
            "version": "v4",
            "depth_metric": "hyperbolic_dist0",
            "depth_used_in": depth_used_in,
            "depth_conditioning_enabled": self.depth_conditioning,
            "projection_applied_count": proj_count_total,
            "projection_applied_fraction": proj_frac_avg,
            "projection_stage_counts": {
                "after_expmap0": proj_count_s1.detach(),
                "after_mobius": proj_count_s2.detach(),
                "after_expert_relift": proj_count_s3.detach(),
            },
            "curvature_value": c.detach(),
            "affected_outputs": [
                "cone_depth", "cone_width", "expert_weights",
                "x_hyp", "x_routed_hyp", "hyp_projections",
                "uncertainty",
            ],
            "affected_losses": ["cone_consistency"],
            "tangent_space_used_for": ["gate", "experts", "euclidean_projection"],
            "ball_space_used_for": ["hyp_projections", "x_hyp_output"],
        }

        return {
            # v3-compatible outputs
            "projections": projections,          # [N, projection_dim] Euclidean
            "uncertainty": uncertainty,           # dict: epistemic, aleatoric, total
            "cone_depth": depth,                 # [N, 1] hyperbolic geodesic
            "cone_width": cone_width,            # [N, 1]
            "expert_weights": scores,            # [N, num_experts]
            "balance_loss": balance_loss,        # scalar
            "evidence": evidence,                # dict: mu, nu, alpha, beta
            "audit_trail": audit_trail,
            # v4 new outputs
            "x_hyp": x_hyp,                     # [N, hidden] Poincaré ball (Step 3)
            "x_routed_hyp": x_routed_hyp,       # [N, hidden] Poincaré ball (Step 4)
            "hyp_projections": hyp_projections,  # [N, hyp_proj_dim] disc coords
        }


# ==================== 5. TRAINING UTILITIES ====================

def neighborhood_consistency_loss(
    x_hyp: torch.Tensor,
    ca_coords: torch.Tensor,
    c: torch.Tensor,
    spatial_cutoff: float = 8.0,
    attract_margin: float = 1.0,
    repel_margin: float = 2.5,
    repel_weight: float = 0.3,
) -> torch.Tensor:
    """
    Neighborhood consistency with attraction AND repulsion.

    Attraction: physical neighbors (within spatial_cutoff Å) should be
    close in the ball (hyp_dist < attract_margin).

    Repulsion: non-neighbors at the same hierarchical depth should be
    pushed apart (hyp_dist > repel_margin). "Same depth" = within 0.5
    hyperbolic units radially. This prevents the 1D solution from
    satisfying the constraint cheaply — sequential neighbors on a line
    are close, but non-neighbors at the same depth are also close on
    a line. The repulsive term forces angular separation.

    Parameters
    ----------
    x_hyp : [N, hidden]
        Poincaré ball positions.
    ca_coords : [N, 3]
        Cα Cartesian coordinates in Ångströms.
    c : scalar tensor
        Learned curvature (positive).
    spatial_cutoff : float
        Physical neighbor threshold in Å.
    attract_margin : float
        Max allowed hyperbolic distance for physical neighbors.
    repel_margin : float
        Min required hyperbolic distance for non-neighbors at same depth.
    repel_weight : float
        Relative weight of repulsive vs attractive term.
    """
    from geoopt.manifolds.stereographic import math as pmath

    k = -c
    N = x_hyp.shape[0]

    # Physical distances
    diffs = ca_coords[:, None, :] - ca_coords[None, :, :]
    physical_dist = diffs.norm(dim=-1)
    neighbors = (physical_dist < spatial_cutoff) & (physical_dist > 0.1)

    if not neighbors.any():
        return torch.tensor(0.0, device=x_hyp.device, requires_grad=True)

    # Hyperbolic pairwise distances (full N×N for repulsion)
    hyp_dist = pmath.dist(
        x_hyp[:, None].expand(N, N, -1).reshape(N * N, -1),
        x_hyp[None, :].expand(N, N, -1).reshape(N * N, -1),
        k=k,
    ).reshape(N, N)

    # ATTRACTIVE: physical neighbors should be close in ball
    attract = (neighbors.float() * torch.relu(hyp_dist - attract_margin)).mean()

    # REPULSIVE: non-neighbors at similar depth should be apart
    depth = pmath.dist0(x_hyp, k=k)  # [N]
    depth_diff = (depth[:, None] - depth[None, :]).abs()
    same_depth = depth_diff < 0.5
    non_neighbor = ~neighbors & same_depth & (physical_dist > 0.1)
    repel = (non_neighbor.float() * torch.relu(repel_margin - hyp_dist)).mean()

    return attract + repel_weight * repel


def angular_diversity_loss(x_hyp: torch.Tensor) -> torch.Tensor:
    """
    Penalize angular collapse — the primary 1D attractor breaker.

    When all points lie on a 1D line through the ball, their pairwise
    cosine similarities are ±1 (mean |cos| ≈ 0.9–1.0).
    When points fill the ball angularly, mean |cos| → 0 (for high-dim).

    This directly measures and penalizes the 1D collapse without
    requiring domain labels or community structure.

    Raw magnitude: ~0.8–0.95 when collapsed, ~0.1–0.3 when diverse.
    At coeff=0.2 this produces 0.16–0.19 loss, competing with cone_coeff=0.1.
    """
    # Normalize to unit vectors — strip radial information
    x_norm = x_hyp / (x_hyp.norm(dim=-1, keepdim=True) + 1e-8)

    # Pairwise cosine similarity matrix
    cos_sim = x_norm @ x_norm.T  # [N, N]

    # Penalize: minimize mean |cos_sim| off-diagonal
    N = x_norm.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=x_hyp.device)
    return cos_sim[mask].abs().mean()


def radial_hierarchy_loss(
    x_hyp: torch.Tensor,
    target_rho: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize radial collapse — forces points to use the INTERIOR of the ball,
    not just the boundary shell.

    The model currently projects all points to |p|≈1/√c (ball boundary).
    This loss has two components:

    1. Radial spread penalty: penalize when std(norms) < threshold.
       Forces the model to place some points near the origin and others
       near the boundary.

    2. Radial-burial correlation: buried residues (low ρ) should be
       CLOSER to the origin (deeper in the hierarchy), exposed residues
       (high ρ) should be near the boundary. This is the opposite of
       what cone_depth measures (dist0 = distance FROM origin), so we
       want: low ρ → low |p|, high ρ → high |p|.

       Wait — the design says deep = buried = far from origin (high dist0).
       So: low ρ → high |p| (near boundary), high ρ → low |p| (near origin).
       Actually the design contract says:
         depth = dist0(x_hyp) = hyperbolic distance from origin
         low ρ (buried) → HIGH depth → far from origin → high |p|
         high ρ (exposed) → LOW depth → near origin → low |p|

       So the correlation should be: ρ negatively correlates with |p|.
       Low ρ → high norm. High ρ → low norm.

    This directly addresses the boundary-shell collapse by requiring
    the model to use the full radial range of the ball.
    """
    norms = x_hyp.norm(dim=-1)  # [N]
    ball_radius = (1.0 / torch.sqrt(c)) - 1e-4

    # Component 1: Radial spread — penalize when all norms are the same
    norm_std = norms.std()
    target_std = 0.15 * ball_radius  # want ~15% of ball radius as spread
    spread_penalty = torch.relu(target_std - norm_std)

    # Component 2: Radial-burial ordering
    # Normalize ρ to [0, 1]: 0 = buried, 1 = exposed
    rho_norm = (target_rho.squeeze() / 30.0).clamp(0.0, 1.0)
    # Normalize norms to [0, 1]
    norm_norm = norms / (ball_radius + 1e-8)

    # We want: buried (low rho_norm) → high norm_norm (near boundary)
    #          exposed (high rho_norm) → low norm_norm (near origin)
    # So target_norm = 1 - rho_norm (inverted)
    target_norm = 1.0 - rho_norm
    ordering_loss = F.mse_loss(norm_norm, target_norm)

    return spread_penalty + 0.5 * ordering_loss


def domain_separation_loss_2d(
    hyp_proj: torch.Tensor,
    domain_labels: torch.Tensor,
    c: torch.Tensor,
    min_angular_sep: float = 0.35,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Push domain centroids apart angularly in the 2D disc.
    Safeguarded version with projection safety and temperature scaling.
    """
    from geoopt.manifolds.stereographic import math as pmath

    k = -c
    unique = [d.item() for d in domain_labels.unique() if d >= 0]
    if len(unique) < 2:
        return torch.tensor(0.0, device=hyp_proj.device, requires_grad=True)

    centroids = []
    for d in unique:
        mask = (domain_labels == d)
        pts = hyp_proj[mask]
        if len(pts) == 0:
            continue
        tangent = pmath.logmap0(pts, k=k).mean(dim=0)
        cent = pmath.expmap0(tangent.unsqueeze(0), k=k).squeeze(0)
        centroids.append(cent)

    if len(centroids) < 2:
        return torch.tensor(0.0, device=hyp_proj.device, requires_grad=True)

    centroids = torch.stack(centroids)
    centroids = pmath.project(centroids, k=k)  # safety

    angles = torch.atan2(centroids[:, 1], centroids[:, 0])
    angle_diff = (angles[:, None] - angles[None, :]).abs()
    angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)

    D = len(centroids)
    off_diag = ~torch.eye(D, dtype=torch.bool, device=hyp_proj.device)
    violation = torch.relu(min_angular_sep - angle_diff[off_diag])
    return (violation / temperature).mean()


def mutation_differential_loss(
    x_hyp_wt: torch.Tensor,
    x_hyp_mut: torch.Tensor,
    known_mobile: List[int],
    known_stable: List[int],
    c: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Paired-structure contrastive loss for WT/mutant differential.

    Supervises the known biology: Switch-I residues (29-36) should show
    large hyperbolic displacement between WT and G12D, while structurally
    stable residues (core helices) should show small displacement.

    Parameters
    ----------
    x_hyp_wt : [N, hidden] — WT embedding in Poincaré ball
    x_hyp_mut : [M, hidden] — Mutant embedding (aligned by residue index)
    known_mobile : list of int — indices of residues expected to move (Switch-I)
    known_stable : list of int — indices of residues expected to stay put
    c : scalar tensor — curvature
    margin : float — minimum gap between mobile and stable displacements
    """
    from geoopt.manifolds.stereographic import math as pmath

    k = -c

    if not known_mobile or not known_stable:
        return torch.tensor(0.0, device=x_hyp_wt.device, requires_grad=True)

    # Mobile residues should have large hyperbolic displacement
    mobile_disp = pmath.dist(
        x_hyp_wt[known_mobile], x_hyp_mut[known_mobile], k=k
    ).mean()

    # Stable residues should have small displacement
    stable_disp = pmath.dist(
        x_hyp_wt[known_stable], x_hyp_mut[known_stable], k=k
    ).mean()

    # Maximize separation: mobile_disp should exceed stable_disp by margin
    return torch.relu(stable_disp - mobile_disp + margin)


def gosp_loss(
    output: Dict[str, Any],
    target_rho: torch.Tensor,
    target_dehydron: torch.Tensor,
    ca_coords: torch.Tensor,
    domain_labels: Optional[torch.Tensor] = None,
    evidential_coeff: float = 0.01,
    balance_coeff: float = 0.01,
    cone_coeff: float = 0.1,
    neighborhood_coeff: float = 0.3,
    angular_coeff: float = 0.2,
    domain_sep_coeff: float = 0.15,
    spatial_cutoff: float = 8.0,
    attract_margin: float = 1.0,
    repel_margin: float = 2.5,
) -> Dict[str, Any]:
    """
    Combined training loss — v4 with cone regression + angular diversity +
    neighborhood consistency + domain separation.

    Cone loss v4.1: Direct regression WITHOUT sigmoid.
    The previous sigmoid(depth/3) approach saturated at depth≈7.1, producing
    a constant prediction of ~0.91 with zero gradient. The fix:
      1. Use raw cone_depth directly (it's always positive from dist0)
      2. Target is normalized ρ (0→1 scale)
      3. Normalize predicted depth to same scale via mean/std matching
      4. Add variance penalty to explicitly prevent constant collapse
    """
    ev_loss = evidential_regression_loss(
        output["evidence"], target_rho, coeff=evidential_coeff
    )
    bal_loss = output["balance_loss"]

    # Cone loss v4.2: direct regression without sigmoid saturation
    # Target: high ρ (buried, well-wrapped) → high depth (deep in ball)
    #         low ρ (dehydron, exposed) → low depth (shallow, near rim)
    # This matches the dehydron geometry: dehydrons are at the rim,
    # buried core residues are deep in the conformational hierarchy.
    target_depth = (target_rho / 30.0).clamp(0.0, 1.0)

    # Predicted: normalize cone_depth to [0,1] via min-max within batch
    # This avoids sigmoid saturation entirely — the model just needs to
    # produce RELATIVE ordering, and the normalization handles scale.
    raw_depth = output["cone_depth"]  # [N, 1], always positive
    depth_min = raw_depth.min()
    depth_max = raw_depth.max()
    depth_range = depth_max - depth_min + 1e-6
    predicted_depth = (raw_depth - depth_min) / depth_range  # [0, 1]

    # MSE on normalized depth
    cone_loss = F.mse_loss(predicted_depth, target_depth)

    # Variance penalty: if std(cone_depth) < threshold, add penalty
    # This directly punishes the constant-prediction attractor.
    # Target std of normalized depth ≈ 0.25 (healthy spread over [0,1])
    depth_std = predicted_depth.std()
    variance_penalty = torch.relu(0.20 - depth_std)  # kicks in when std < 0.20
    cone_loss = cone_loss + 0.5 * variance_penalty

    # Angular diversity loss — breaks 1D collapse
    ang_loss = angular_diversity_loss(output["x_routed_hyp"])

    # Neighborhood consistency with repulsion
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

    # Domain separation on 2D disc projections (if labels provided)
    dom_loss = torch.tensor(0.0, device=output["x_hyp"].device)
    if domain_labels is not None and domain_sep_coeff > 0:
        dom_loss = domain_separation_loss_2d(
            hyp_proj=output["hyp_projections"],
            domain_labels=domain_labels,
            c=c,
        )

    total = (
        ev_loss
        + balance_coeff * bal_loss
        + cone_coeff * cone_loss
        + neighborhood_coeff * nbr_loss
        + angular_coeff * ang_loss
        + domain_sep_coeff * dom_loss
    )

    # Projection safety penalty — strong enforcement of ball constraint
    hyp_norms = output["hyp_projections"].norm(dim=-1)
    proj_violation = torch.relu(hyp_norms - 0.99).mean()
    total = total + 2.0 * proj_violation

    audit_consistent = True
    audit = output.get("audit_trail", {})
    if isinstance(audit, dict):
        required = {"depth_metric", "depth_used_in", "affected_losses"}
        audit_consistent = required.issubset(set(audit.keys()))
        if audit_consistent:
            depth_used_in = set(audit.get("depth_used_in", []))
            affected_losses = set(audit.get("affected_losses", []))
            audit_consistent = (
                "cone_loss" in depth_used_in
                and "cone_consistency" in affected_losses
            )

    return {
        "total": total,
        "evidential": ev_loss,
        "balance": bal_loss,
        "cone_consistency": cone_loss,
        "neighborhood_consistency": nbr_loss,
        "angular_diversity": ang_loss,
        "domain_separation": dom_loss,
        "audit_consistent": audit_consistent,
    }


def build_optimizer(model: GOSPConeMapper, lr: float = 1e-3, weight_decay: float = 1e-5):
    """RiemannianAdam — unchanged from v3."""
    return geoopt.optim.RiemannianAdam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )


# ==================== 6. SELF-TEST ====================

def run_internal_test(seed: int = 42) -> int:
    """
    Deterministic property checks for v4 invariants.
    Returns 0 (pass) or 1 (fail).
    """
    torch.manual_seed(seed)
    failures: List[str] = []

    model = GOSPConeMapper(
        node_dim=4, hidden=64, num_layers=2,
        num_experts=3, projection_dim=32, hyp_proj_dim=2,
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
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = precompute_clustering(data)

    with torch.no_grad():
        out = model(data)

    # 1. Core outputs present and correct shape
    assert_checks = [
        ("x_hyp shape", out["x_hyp"].shape == (N, 64)),
        ("x_routed_hyp shape", out["x_routed_hyp"].shape == (N, 64)),
        ("hyp_projections shape", out["hyp_projections"].shape == (N, 2)),
        ("cone_depth positive", (out["cone_depth"] > 0).all().item()),
        ("cone_depth finite", torch.isfinite(out["cone_depth"]).all().item()),
    ]
    for name, condition in assert_checks:
        if not condition:
            failures.append(f"FAIL: {name}")

    # 2. hyp_projections are inside the Poincaré ball (|z| < 1)
    hyp_norms = out["hyp_projections"].norm(dim=-1)
    if not (hyp_norms < 1.0).all().item():
        failures.append(
            f"hyp_projections outside ball: max |z| = {hyp_norms.max().item():.4f}"
        )

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

    # 5. Recursive NaN/Inf scan on all outputs
    invalid = scan_tensors_for_invalid(out)
    if invalid:
        failures.append(f"NaN/Inf found: {invalid}")

    # 6. Audit trail version is v4
    if out["audit_trail"].get("version") != "v4":
        failures.append("audit_trail version is not v4")

    # 7. Hyperbolic distance monotonicity (geometry sanity check)
    k = -c
    direction = torch.randn(8, 64)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-9)
    scales = [0.1, 0.5, 1.0, 2.0]
    means = []
    with torch.no_grad():
        for s in scales:
            pts = pmath.expmap0(direction * s, k=k)
            pts = pmath.project(pts, k=k)
            d = pmath.dist0(pts, k=k)
            means.append(d.mean().item())
    if not (means[0] < means[1] < means[2] < means[3]):
        failures.append(f"dist0 monotonicity failed: {means}")

    # 8. Angular diversity loss computes without error and has expected range
    ang_loss = angular_diversity_loss(out["x_routed_hyp"])
    if not torch.isfinite(ang_loss):
        failures.append(f"angular_diversity_loss is not finite: {ang_loss.item()}")
    # Note: at initialization (random weights) angular collapse is expected.
    # This check just verifies the function runs. The post-training check
    # (mean |cosine| < 0.85) is a training criterion, not an architecture test.

    print(f"Tokyo Eyes v4 internal test (seed={seed}):")
    print(f"  checks: 8")
    if failures:
        print(f"  result: FAIL ({len(failures)} failures)")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  result: PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokyo Eyes v4")
    parser.add_argument("--internaltest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.internaltest:
        sys.exit(run_internal_test(seed=args.seed))

    # Quick smoke test
    model = GOSPConeMapper(node_dim=4, hidden=128, num_layers=6,
                            num_experts=4, projection_dim=64, hyp_proj_dim=2)
    total = sum(p.numel() for p in model.parameters())
    print(f"Tokyo Eyes v4: {total:,} parameters")
    print(f"  log_c learnable: {any(n == 'log_c' for n,_ in model.named_parameters())}")
    print(f"  Initial curvature: {model.curvature.item():.4f}")
    print(f"  hyp_proj_head: MobiusLinear({128}, {2}) — native disc output")
    print(f"  uncertainty_head input: hidden + 2 = {128 + 2}")
    print("\nRun --internaltest for full property verification.")
