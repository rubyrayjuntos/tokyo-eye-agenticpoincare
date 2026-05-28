# Migrated from: SRC_DEM/DTIE_GNN_ORCHESTRATION/Gnnv3.py on 2026-05-27
﻿"""
Tokyo Eyes v3: Geometric Ontological State Protein Graph Neural Network
========================================================================
Eidetix Bio | Ray + Codex | v3


MODEL HISTORY
- version: v3
 date: 2026-03-04
 change: Replace Euclidean cone depth with hyperbolic geodesic depth (`dist0`),
   disable implicit depth weighting by default, add explicit depth-conditioning
   flag, and attach a full audit trail for depth-dependent downstream behavior.
 before_state: v2 used Euclidean norm for cone depth and always scaled
   projections by cone width; boundary projection was safety-only with no
   visibility into how often clamping happened.
 reasoning: Preserve Tokyo Eyes invariance claims while exposing all
   stabilization effects so downstream outcomes are auditable.
 who: Ray+Codex


Architecture mapping:
 - Equivariant message passing  -> learns dehydron relational grammar (topology invariant)
 - Hyperbolic cone embedding    -> conformational funnel (rim=MA/dehydron, tip=folded)
 - Topological MoE routing      -> Type A (stable) vs Type B (hinge-proximal) specialists
 - Evidential uncertainty       -> distinguishes MA regions from training gaps
 - Projection head + scrubber   -> navigable conformational state space


Inputs:
 - Model forward input (`torch_geometric.data.Data`):
   - `x`: [N, 4], node features as [rho, tau_flag, ss_type, sasa]
   - `edge_index`: [2, E], graph connectivity
   - `edge_attr`: [E, 4], edge features as [rel_x, rel_y, rel_z, distance]
   - `clustering`: [N], per-node clustering coefficients
 - Loss inputs:
   - `target_rho`: [N, 1], wrapping-density targets
   - `target_dehydron`: [N, 1], dehydron labels


Outputs:
 - Forward dictionary:
   - `projections`: [N, projection_dim]
   - `uncertainty`: dict with `epistemic`, `aleatoric`, `total`
   - `cone_depth`: [N, 1], hyperbolic geodesic depth (`dist0`)
   - `cone_width`: [N, 1], derived as `exp(-cone_depth)`
   - `expert_weights`: [N, num_experts]
   - `balance_loss`: scalar tensor
   - `evidence`: dict with `mu`, `nu`, `alpha`, `beta`
   - `audit_trail`: depth/projection provenance and diagnostics
 - Loss dictionary:
   - `total`, `evidential`, `balance`, `cone_consistency`, `audit_consistent`
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
   """
   Recursively scan nested structures for NaN/Inf in tensors.


   Returns a list of findings with path-level diagnostics.
   """
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
   Precompute per-node clustering coefficients and attach to data object.
  
   Clustering coefficient = (triangles through node) / (possible triangles).
   High clustering → structurally embedded (Type A proxy).
   Low clustering  → exposed / hinge-proximal (Type B proxy).
  
   Runs once during data loading, NOT in the forward pass.
  
   For proteins with dense local connectivity (~20-30 neighbors per
   residue at 10Å cutoff), this is O(k²) per node — manageable for
   single proteins but should be vectorized for batch processing.
   """
   num_nodes = data.num_nodes
   edge_index = data.edge_index


   # Build adjacency set per node
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
           clustering[node] = 0.0
           continue
       # Count triangles
       triangles = 0
       for i in range(k):
           for j in range(i + 1, k):
               if neighbors[j] in adj[neighbors[i]]:
                   triangles += 1
       clustering[node] = (2.0 * triangles) / (k * (k - 1))


   data.clustering = clustering
   return data




# ==================== 1. EQUIVARIANT CONVOLUTION ====================


class EquivariantConv(MessagePassing):
   """
   SE(3)-equivariant message passing layer.
  
   Uses e3nn tensor products to process both scalar (0e) and vector (1e)
   features. Relative position vectors between residues provide the
   geometric context — this is how the network learns the relational
   grammar between dehydrons that is conserved across cone dimensions.
  
   Architecture:
     - Messages are computed via FullyConnectedTensorProduct of
       neighbor features ⊗ edge spherical harmonics
     - Edge distances modulate the tensor product via a radial MLP
     - Output preserves SE(3) equivariance: rotations of the protein
       produce corresponding rotations of the vector features
   """


   def __init__(self, hidden_dim: int, irreps_hidden: str = "32x0e + 8x1e"):
       super().__init__(aggr="add", node_dim=0)


       self.irreps_hidden = Irreps(irreps_hidden)
       self.irreps_sh = Irreps("1x0e + 1x1e")  # l=0 and l=1 spherical harmonics


       # Tensor product: node_features ⊗ spherical_harmonics → node_features
       self.tp = FullyConnectedTensorProduct(
           self.irreps_hidden,     # input irreps (neighbor features)
           self.irreps_sh,         # edge irreps (direction)
           self.irreps_hidden,     # output irreps
           shared_weights=False    # weights come from radial MLP
       )


       # Radial MLP: distance scalar → tensor product weights
       # Distance is rotation-invariant, so this preserves equivariance
       self.radial_mlp = nn.Sequential(
           nn.Linear(1, 64),
           nn.SiLU(),
           nn.Linear(64, 64),
           nn.SiLU(),
           nn.Linear(64, self.tp.weight_numel)
       )


       # Hidden dim for the scalar-only pathway used by downstream layers
       self.hidden_dim = hidden_dim


       # v2->v3 carryover: _proj is initialized in __init__.
       # Previously created dynamically in forward() via hasattr check.
       # That meant _proj was invisible to model.parameters(), never
       # received gradients, and the entire SE(3) backbone's contribution
       # was projected through permanently random weights.
       num_scalars = 0
       for mul, ir in self.irreps_hidden:
           if ir.l == 0:
               num_scalars += mul * ir.dim
           else:
               break
       self.num_scalars = num_scalars
       self._proj = nn.Linear(num_scalars, hidden_dim)


   def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
               edge_attr: torch.Tensor) -> torch.Tensor:
       """
       Args:
           x: Node features [N, hidden_dim] (scalar representation)
           edge_index: [2, E] connectivity
           edge_attr: [E, 4] — (rel_x, rel_y, rel_z, distance)
      
       Returns:
           Updated node features [N, hidden_dim]
       """
       # Decompose edge attributes
       rel_pos = edge_attr[:, :3]   # relative position vectors
       dist = edge_attr[:, 3:4]     # scalar distance


       # Compute spherical harmonics from relative positions (equivariant)
       sh = o3.spherical_harmonics(
           l=[0, 1],
           x=rel_pos,
           normalize=True,
           normalization="component"
       )


       # Radial weights from distance (invariant)
       edge_weights = self.radial_mlp(dist)


       # Lift scalar node features into irreps format for tensor product
       # Pad with zeros for the vector (1e) channels
       irreps_dim = self.irreps_hidden.dim
       if x.size(-1) < irreps_dim:
           padding = torch.zeros(
               x.size(0), irreps_dim - x.size(-1),
               device=x.device, dtype=x.dtype
           )
           x_irreps = torch.cat([x, padding], dim=-1)
       else:
           x_irreps = x[:, :irreps_dim]


       # Message passing with tensor product
       out = self.propagate(
           edge_index, x=x_irreps, sh=sh, edge_weights=edge_weights
       )


       # Project back to scalar hidden dim (extract 0e components)
       out_scalar = out[:, :self.num_scalars]
       return self._proj(out_scalar)


   def message(self, x_j: torch.Tensor, sh: torch.Tensor,
               edge_weights: torch.Tensor) -> torch.Tensor:
       """Compute equivariant messages via tensor product."""
       return self.tp(x_j, sh, edge_weights)




# ==================== 1b. MÖBIUS LINEAR LAYER ====================


class MobiusLinear(nn.Module):
   """
   Linear layer in hyperbolic space via Möbius matrix-vector multiplication.


   Unlike a standard nn.Linear followed by expmap0, this operates natively
   in the Poincaré ball — distances and angles are preserved correctly,
   especially near the ball boundary where Euclidean projections distort
   representations of deeply folded / stable regions.


   Forward signature takes curvature c explicitly so it can track the
   learnable curvature from the Tokyo Eyes cone mapper.
   """


   def __init__(self, in_features: int, out_features: int, bias: bool = True):
       super().__init__()
       self.weight = nn.Parameter(torch.empty(out_features, in_features))
       self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
       nn.init.xavier_uniform_(self.weight)


   def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
       from geoopt.manifolds.stereographic import math as pmath
       # Möbius matrix-vector multiplication (geoopt)
       out = pmath.mobius_matvec(self.weight, x, k=-c)
       if self.bias is not None:
           bias_hyp = pmath.expmap0(self.bias, k=-c)
           out = pmath.mobius_add(out, bias_hyp, k=-c)
       return out




# ==================== 2. EVIDENTIAL UNCERTAINTY HEAD ====================


class EvidentialHead(nn.Module):
   """
   Evidential deep learning head for per-node uncertainty estimation.
  
   Outputs parameters of a Normal-Inverse-Gamma (NIG) distribution:
     μ  — predicted value (rho or dehydron score)
     ν  — precision of the mean estimate (pseudo-observations)
     α  — shape of the inverse-gamma on variance
     β  — scale of the inverse-gamma on variance
  
   Two uncertainty types, mapping directly to Tokyo Eyes:
     Epistemic (knowledge uncertainty):  high when ν is low
       → Training gap: network hasn't seen this topology
       → Divergence + low coverage = CAUTION
    
     Aleatoric (data uncertainty):  high when β/α is high 
       → Genuine MA region: probability space is intrinsically open
       → Divergence + high coverage = INVESTIGATE (real biological signal)
  
   This distinction is critical: a dehydron with high aleatoric uncertainty
   is a genuine MA void (functional). A site with high epistemic uncertainty
   is a training gap (needs more data). The drug discovery implications
   are completely different.
  
   Loss follows Amini et al. 2020 (Deep Evidential Regression) — the
   Type II maximum likelihood form, deliberately omitting digamma terms
   from the full Bayesian treatment for training stability.
   """


   def __init__(self, hidden_dim: int, out_dim: int = 1):
       super().__init__()
       self.shared = nn.Sequential(
           nn.Linear(hidden_dim, hidden_dim // 2),
           nn.SiLU(),
           nn.Linear(hidden_dim // 2, hidden_dim // 4),
           nn.SiLU(),
       )
       reduced = hidden_dim // 4


       self.mu_head = nn.Linear(reduced, out_dim)       # mean prediction
       self.logv_head = nn.Linear(reduced, out_dim)     # log(ν) — ensures ν > 0
       self.loga_head = nn.Linear(reduced, out_dim)     # log(α - 1) — ensures α > 1
       self.logb_head = nn.Linear(reduced, out_dim)     # log(β) — ensures β > 0


   def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
       """
       Returns:
           uncertainty: dict with 'epistemic', 'aleatoric', 'total' per node
           evidence: dict with NIG parameters (mu, nu, alpha, beta)
       """
       h = self.shared(x)


       mu = self.mu_head(h)
       nu = F.softplus(self.logv_head(h)) + 1e-6      # ν > 0
       alpha = F.softplus(self.loga_head(h)) + 1.0     # α > 1
       beta = F.softplus(self.logb_head(h)) + 1e-6     # β > 0


       # Epistemic uncertainty: inversely proportional to evidence (ν)
       # Low ν = few pseudo-observations = network unsure = training gap
       epistemic = 1.0 / nu


       # Aleatoric uncertainty: expected variance under the NIG posterior
       # High β/α = intrinsically noisy = genuine MA / probability space openness
       aleatoric = beta / (alpha - 1.0 + 1e-6)


       uncertainty = {
           "epistemic": epistemic,     # Training gap signal
           "aleatoric": aleatoric,     # Genuine MA signal
           "total": epistemic + aleatoric,
       }


       evidence = {
           "mu": mu, "nu": nu, "alpha": alpha, "beta": beta
       }


       return uncertainty, evidence




def evidential_regression_loss(
   evidence: Dict[str, torch.Tensor],
   target: torch.Tensor,
   coeff: float = 0.01
) -> torch.Tensor:
   """
   NIG negative log-likelihood + evidence regularizer.
  
   The regularizer (scaled by coeff) penalizes evidence on incorrect
   predictions, preventing the network from being confidently wrong.
  
   Args:
       evidence: dict with mu, nu, alpha, beta from EvidentialHead
       target: ground truth values [N, 1]
       coeff: regularization weight (start small, ~0.01)
   """
   mu, nu, alpha, beta = evidence["mu"], evidence["nu"], evidence["alpha"], evidence["beta"]
   omega = 2.0 * beta * (1.0 + nu)


   # NIG negative log-likelihood
   nll = (
       0.5 * torch.log(torch.pi / nu + 1e-8)
       - alpha * torch.log(omega + 1e-8)
       + (alpha + 0.5) * torch.log((target - mu) ** 2 * nu + omega + 1e-8)
       + torch.lgamma(alpha + 1e-8)
       - torch.lgamma(alpha + 0.5)
   )


   # Evidence regularizer: penalize evidence (nu) when prediction is wrong
   reg = (target - mu).abs() * (2.0 * nu + alpha)


   return (nll + coeff * reg).mean()




# ==================== 3. TOPOLOGICAL MoE GATE ====================


class TopologicalMoEGate(nn.Module):
   """
   Mixture-of-Experts gate that routes based on topological context.
  
   Inputs:
     - Node features (learned representation)
     - Precomputed clustering coefficient (structural stability proxy)
     - Cone depth (from hyperbolic projection — where in the funnel)
  
   Routing logic (learned, not hardcoded):
     High clustering + deep cone  → stable structure experts (Type A)
     Low clustering + shallow cone → hinge/dynamic experts (Type B)
    
   The gate learns to specialize experts for different interaction scales,
   mirroring the MI-MoE architecture from the Tokyo Eyes design doc.
   """


   def __init__(self, hidden_dim: int, num_experts: int):
       super().__init__()
       # +2 for clustering coefficient and cone depth
       self.topology_compressor = nn.Sequential(
           nn.Linear(hidden_dim + 2, 64),
           nn.SiLU(),
           nn.Linear(64, 32),
           nn.SiLU(),
           nn.Linear(32, num_experts),
       )
       self.num_experts = num_experts


   def forward(
       self, x: torch.Tensor, clustering: torch.Tensor, cone_depth: torch.Tensor
   ) -> Tuple[torch.Tensor, torch.Tensor]:
       """
       Args:
           x: Node features [N, hidden_dim]
           clustering: Precomputed clustering coefficients [N]
           cone_depth: Hyperbolic norm / cone position [N, 1]
      
       Returns:
           scores: Expert weights [N, num_experts] (soft routing)
           balance_loss: Auxiliary loss encouraging even expert usage
       """
       gate_input = torch.cat([
           x,
           clustering.unsqueeze(-1),
           cone_depth.detach(),  # Detach: cone depth informs routing but routing doesn't backprop into cone
       ], dim=-1)


       logits = self.topology_compressor(gate_input)
       scores = F.softmax(logits, dim=-1)


       # Switch Transformer balance loss.
       # Previous version computed (avg_score * uniform_fraction).sum() * num_experts
       # which simplifies to avg_score.sum() = 1.0 always (softmax rows sum to 1,
       # mean of rows still sums to 1). Zero gradient — no regularization at all.
       #
       # Correct formulation (Fedus et al. 2021, Switch Transformers):
       #   f_i = fraction of nodes where expert i has the highest gate score
       #   p_i = mean gate probability for expert i across all nodes
       #   balance_loss = num_experts * sum(f_i * p_i)
       #
       # This penalizes correlation between dispatch frequency and gate
       # probability — if an expert is both frequently selected AND given
       # high probability, the loss increases, pushing toward uniform dispatch.
       top_expert = scores.argmax(dim=-1)  # [N] — which expert "wins" each node
       f = torch.zeros(self.num_experts, device=scores.device)
       for i in range(self.num_experts):
           f[i] = (top_expert == i).float().mean()
       p = scores.mean(dim=0)  # [num_experts] — average gate probability
       balance_loss = self.num_experts * (f * p).sum()


       return scores, balance_loss




# ==================== 4. MAIN TOKYO EYES HYPERBOLIC MAPPER ====================


class GOSPConeMapper(nn.Module):
   """
   Tokyo Eyes Graph Neural Network — Full Architecture
  
   Maps protein structure to conformational funnel coordinates:
  
   1. Embed node features (rho, tau_flag, ss_type, sasa)
   2. SE(3)-equivariant message passing learns relational grammar
      (the topology invariant conserved across cone dimensions)
   3. Hyperbolic projection maps to conformational funnel position
      (rim = MA/dehydron, tip = folded/stable)
   4. MoE routing specializes processing by structural context
      (Type A stable vs Type B hinge-proximal)
   5. Evidential uncertainty separates genuine MA from training gaps
   6. Projection head outputs scrubber-navigable state space coords
   """


   def __init__(
       self,
       node_dim: int = 4,
       hidden: int = 128,
       num_layers: int = 6,
       num_experts: int = 4,
       projection_dim: int = 64,
       depth_conditioning: bool = False,
       projection_audit_tolerance: float = 1e-7,
       extended_node_dim: int = 0,
   ):
       super().__init__()
       self.hidden = hidden
       self.depth_conditioning = depth_conditioning
       self.projection_audit_tolerance = projection_audit_tolerance

       # Feature projection: maps extended features down to node_dim
       # so the rest of the model (trained with node_dim=4) still works.
       # When extended_node_dim > 0, a Linear(extended_node_dim, node_dim)
       # layer is inserted before node_emb. Initialized as identity on
       # the first node_dim dims so the old checkpoint produces identical
       # output until fine-tuned.
       self.extended_node_dim = extended_node_dim
       if extended_node_dim > 0 and extended_node_dim != node_dim:
           self.feature_proj = nn.Linear(extended_node_dim, node_dim)
           # Initialize: identity on first node_dim dims, zeros on the rest
           with torch.no_grad():
               self.feature_proj.weight.zero_()
               self.feature_proj.bias.zero_()
               eye_size = min(node_dim, extended_node_dim)
               self.feature_proj.weight[:eye_size, :eye_size] = torch.eye(eye_size)
       else:
           self.feature_proj = None

       self.node_emb = nn.Linear(node_dim, hidden)


       # SE(3) Backbone: learns relational grammar between dehydrons
       self.convs = nn.ModuleList([
           EquivariantConv(hidden, irreps_hidden="32x0e + 8x1e")
           for _ in range(num_layers)
       ])
       self.norms = nn.ModuleList([
           nn.LayerNorm(hidden) for _ in range(num_layers)
       ])


       # Learnable positive curvature parameter.
       # softplus(0.0) ≈ 0.693, close to c=1.0 init.  softplus is preferred
       # over exp because its gradient doesn't explode for large values.
       self.log_c = nn.Parameter(torch.tensor(0.0))


       # Hyperbolic-native Möbius linear layer.
       # Euclidean pre-projection lifts hidden features before expmap0,
       # then MobiusLinear transforms natively within the Poincaré ball.
       self.hyper_lift = nn.Linear(hidden, hidden)
       self.mobius_proj = MobiusLinear(hidden, hidden)


       # MoE: topology-aware expert routing
       self.gate = TopologicalMoEGate(hidden, num_experts)
       self.experts = nn.ModuleList([
           nn.Sequential(
               nn.Linear(hidden, hidden),
               nn.SiLU(),
               nn.Linear(hidden, hidden),
           )
           for _ in range(num_experts)
       ])


       # Evidential uncertainty head
       self.uncertainty_head = EvidentialHead(hidden)


       # Scrubber projection: conformational state space coordinates
       self.projection_head = nn.Linear(hidden, projection_dim)


   def _project_with_audit(
       self, x: torch.Tensor, k: torch.Tensor
   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       """
       Project to the manifold interior and count which rows changed.
       """
       from geoopt.manifolds.stereographic import math as pmath
       x_proj = pmath.project(x, k=k)
       changed = (x_proj - x).abs().amax(dim=-1) > self.projection_audit_tolerance
       changed_count = changed.sum()
       changed_fraction = changed.float().mean()
       return x_proj, changed_count, changed_fraction


   @property
   def curvature(self) -> torch.Tensor:
       """Learnable positive curvature via softplus transform. Always > 1e-4."""
       return F.softplus(self.log_c) + 1e-4


   def forward(self, data: Data) -> Dict[str, Any]:
       """
       Args:
           data: PyG Data object with:
               - x: [N, 4] node features (rho, tau_flag, ss_type, sasa)
               - edge_index: [2, E] connectivity
               - edge_attr: [E, 4] (rel_x, rel_y, rel_z, distance)
               - clustering: [N] precomputed clustering coefficients
      
       Returns:
           Dictionary with model outputs, uncertainty/evidence, and audit trail.
       """
       # Step 1: Feature embedding.
       # If extended features are provided (node_dim > 4), project down first.
       node_features = data.x
       if self.feature_proj is not None:
           node_features = self.feature_proj(node_features)
       x = self.node_emb(node_features)


       # Step 2: Relational grammar phase (equivariant message passing).
       # Residual connections reduce oversmoothing across stacked layers.
       for conv, norm in zip(self.convs, self.norms):
           x_res = x
           x = conv(x, data.edge_index, data.edge_attr)
           x = F.silu(norm(x))
           x = x + x_res  # Residual path preserves per-node identity.


       # Step 3: Hyperbolic cone projection (conformational funnel).
       # Use pmath directly so autograd flows through learnable curvature c.
       from geoopt.manifolds.stereographic import math as pmath
       c = self.curvature
       k = -c  # PoincareBall convention: negative curvature k = -c.


       # Lift to ball, then transform within it
       x_pre = self.hyper_lift(x)
       x_hyp = pmath.expmap0(x_pre, k=k)


       # Projection is a numerical safety rail and is explicitly audited.
       x_hyp, proj_count_stage1, proj_fraction_stage1 = self._project_with_audit(x_hyp, k=k)


       # Native hyperbolic linear transform.
       x_hyp = self.mobius_proj(x_hyp, c=c)
       x_hyp, proj_count_stage2, proj_fraction_stage2 = self._project_with_audit(x_hyp, k=k)


       # Cone depth is hyperbolic geodesic distance from the origin.
       depth = pmath.dist0(x_hyp, k=k, keepdim=True)


       # Cone width is the derived probability-space width at this depth.
       cone_width = torch.exp(-depth)


       # Step 4: Topological MoE routing.
       # Gate sees learned features, structural stability, and cone depth.
       scores, balance_loss = self.gate(x, data.clustering, depth)


       # Expert computation.
       expert_outputs = torch.stack(
           [expert(x) for expert in self.experts], dim=1
       )  # [N, num_experts, hidden]


       # Weighted expert combination.
       x_routed = torch.einsum("ne,neh->nh", scores, expert_outputs)


       # Step 5: Evidential uncertainty decomposition.
       uncertainty, evidence = self.uncertainty_head(x_routed)


       # Step 6: Scrubber projections.
       # Depth conditioning is explicit and disabled by default.
       projections = self.projection_head(x_routed)
       if self.depth_conditioning:
           projections = projections * cone_width


       projection_applied_count = (proj_count_stage1 + proj_count_stage2).detach()
       projection_applied_fraction = ((proj_fraction_stage1 + proj_fraction_stage2) / 2.0).detach()
       depth_used_in = ["gate", "cone_loss"]
       if self.depth_conditioning:
           depth_used_in.append("projection")


       affected_outputs = ["cone_depth", "cone_width", "expert_weights"]
       if self.depth_conditioning:
           affected_outputs.append("projections")


       audit_trail = {
           "depth_metric": "hyperbolic_dist0",
           "depth_used_in": depth_used_in,
           "depth_conditioning_enabled": self.depth_conditioning,
           "projection_applied_count": projection_applied_count,
           "projection_applied_fraction": projection_applied_fraction,
           "projection_stage_counts": {
               "after_expmap0": proj_count_stage1.detach(),
               "after_mobius": proj_count_stage2.detach(),
           },
           "projection_stage_fractions": {
               "after_expmap0": proj_fraction_stage1.detach(),
               "after_mobius": proj_fraction_stage2.detach(),
           },
           "projection_tolerance": self.projection_audit_tolerance,
           "curvature_value": c.detach(),
           "affected_outputs": affected_outputs,
           "affected_losses": ["cone_consistency"],
       }


       return {
           "projections": projections,       # [N, projection_dim] — scrubber coords
           "uncertainty": uncertainty,        # dict: epistemic, aleatoric, total
           "cone_depth": depth,              # [N, 1] — funnel position
           "cone_width": cone_width,         # [N, 1] — probability space width
           "expert_weights": scores,         # [N, num_experts] — routing decisions
           "balance_loss": balance_loss,     # scalar — MoE regularizer
           "evidence": evidence,             # dict: mu, nu, alpha, beta — for loss
           "audit_trail": audit_trail,       # dict: depth/projection impact metadata
       }




# ==================== 5. TRAINING UTILITIES ====================


def gosp_loss(
   output: Dict[str, Any],
   target_rho: torch.Tensor,
   target_dehydron: torch.Tensor,
   evidential_coeff: float = 0.01,
   balance_coeff: float = 0.01,
   cone_coeff: float = 0.1,
) -> Dict[str, Any]:
   """
   Combined Tokyo Eyes training loss.
  
   Components:
     1. Evidential regression loss on rho prediction
     2. MoE load balancing regularizer
     3. Cone depth consistency: dehydrons should be near rim,
        protected H-bonds should be deep
  
   Args:
       output: dict from GOSPConeMapper.forward()
       target_rho: ground truth wrapping density [N, 1]
       target_dehydron: binary labels [N, 1] — 1 if rho < tau
       evidential_coeff: weight on evidential regularizer
       balance_coeff: weight on MoE balance loss
       cone_coeff: weight on cone consistency loss
   """
   # Component 1: Evidential regression loss (primary).
   ev_loss = evidential_regression_loss(
       output["evidence"], target_rho, coeff=evidential_coeff
   )


   # Component 2: MoE balance loss.
   bal_loss = output["balance_loss"]


   # Component 3: Cone consistency loss.
   # Dehydrons (target_dehydron=1) should have LOW depth (near rim)
   # Protected H-bonds (target_dehydron=0) should have HIGH depth (deep)
   # Binary cross-entropy on (1 - depth_sigmoid) vs dehydron label
   depth_normalized = torch.sigmoid(output["cone_depth"])
   cone_loss = F.binary_cross_entropy(
       1.0 - depth_normalized, target_dehydron
   )


   total = ev_loss + balance_coeff * bal_loss + cone_coeff * cone_loss


   audit_consistent = True
   audit = output.get("audit_trail")
   if isinstance(audit, dict):
       required = {"depth_metric", "depth_used_in", "affected_losses"}
       audit_consistent = required.issubset(set(audit.keys()))
       if audit_consistent:
           depth_used_in = set(audit.get("depth_used_in", []))
           affected_losses = set(audit.get("affected_losses", []))
           audit_consistent = ("cone_loss" in depth_used_in) and ("cone_consistency" in affected_losses)


   return {
       "total": total,
       "evidential": ev_loss,
       "balance": bal_loss,
       "cone_consistency": cone_loss,
       "audit_consistent": audit_consistent,
   }




def build_optimizer(model: GOSPConeMapper, lr: float = 1e-3, weight_decay: float = 1e-5):
   """
   Build optimizer with Riemannian gradient handling.


   v3 update: RiemannianAdam.
   Curvature is now learnable (log_c parameter), so we use
   geoopt.optim.RiemannianAdam which correctly handles both
   Euclidean and manifold parameters in a single optimizer.
   For standard parameters it behaves identically to Adam.


   Returns:
       geoopt.optim.RiemannianAdam optimizer
   """
   return geoopt.optim.RiemannianAdam(
       model.parameters(),
       lr=lr,
       weight_decay=weight_decay,
   )




# ==================== 6. EXAMPLE USAGE ====================


def create_example_data(num_nodes: int = 50, num_edges: int = 200) -> Data:
   """Create synthetic protein graph for testing."""
   # Node features: rho, tau_flag, ss_type, sasa
   rho = torch.rand(num_nodes, 1) * 30          # wrapping density 0-30
   tau_flag = (rho < 13.0).float()               # dehydron flag (tau=13.0)
   ss_type = torch.randint(0, 3, (num_nodes, 1)).float() / 2.0  # normalized
   sasa = torch.rand(num_nodes, 1)               # solvent accessibility


   x = torch.cat([rho, tau_flag, ss_type, sasa], dim=-1)  # [N, 4]


   # Random edges (in practice: distance-based cutoff on C-alpha)
   edge_index = torch.randint(0, num_nodes, (2, num_edges))


   # Edge features: relative position + distance
   rel_pos = torch.randn(num_edges, 3)
   dist = rel_pos.norm(dim=-1, keepdim=True)
   edge_attr = torch.cat([rel_pos, dist], dim=-1)  # [E, 4]


   data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


   # Precompute clustering (happens once, not in forward pass)
   data = precompute_clustering(data)


   return data




def run_internal_test(seed: int = 2) -> int:
   """
   Lightweight deterministic self-diagnostic for core invariants.


   Returns:
       process-style exit code (0 pass, 1 fail)
   """
   torch.manual_seed(seed)
   failures: List[str] = []


   # 1) Depth finite and non-collapsed.
   model = GOSPConeMapper(hidden=64, num_layers=2, num_experts=3, depth_conditioning=False)
   model.eval()
   data = create_example_data(num_nodes=48, num_edges=180)


   with torch.no_grad():
       output = model(data)


   depth = output["cone_depth"]
   if not torch.isfinite(depth).all():
       failures.append("cone_depth has NaN/Inf")
   if not (depth > 0).all():
       failures.append("cone_depth should be strictly positive under dist0")
   if depth.std().item() <= 1e-4:
       failures.append("cone_depth collapsed (std <= 1e-4)")


   # 2) Hyperbolic depth monotonicity under tangent scaling.
   from geoopt.manifolds.stereographic import math as pmath
   c = torch.tensor(0.8)
   k = -c
   direction = torch.randn(16, 8)
   direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-9)
   scales = [0.2, 0.6, 1.0, 1.4]
   means = []
   for s in scales:
       x = pmath.expmap0(direction * s, k=k)
       d = pmath.dist0(x, k=k)
       means.append(float(d.mean().item()))
   if not (means[0] < means[1] < means[2] < means[3]):
       failures.append(f"dist0 monotonicity failed: means={means}")


   # 3) Audit trail completeness.
   audit = output.get("audit_trail", {})
   required_fields = {
       "depth_metric",
       "depth_used_in",
       "depth_conditioning_enabled",
       "projection_applied_count",
       "projection_applied_fraction",
       "curvature_value",
       "affected_outputs",
       "affected_losses",
   }
   if not required_fields.issubset(set(audit.keys())):
       failures.append("audit_trail missing required fields")
   if audit.get("depth_metric") != "hyperbolic_dist0":
       failures.append("depth_metric is not hyperbolic_dist0")


   # 4) Recursive NaN/Inf scan.
   invalid = scan_tensors_for_invalid(output)
   if invalid:
       failures.append(f"recursive invalid scan found issues: {invalid}")


   # 5) Explicit depth-conditioning policy check.
   model_on = GOSPConeMapper(hidden=64, num_layers=2, num_experts=3, depth_conditioning=True)
   model_on.load_state_dict(model.state_dict())
   model_on.eval()
   with torch.no_grad():
       out_off = output
       out_on = model_on(data)
   expected = out_off["projections"] * out_on["cone_width"]
   if not torch.allclose(out_on["projections"], expected, atol=1e-5, rtol=1e-5):
       failures.append("depth_conditioning=True did not scale projections by cone_width")
   if "projection" in out_off["audit_trail"]["depth_used_in"]:
       failures.append("depth_conditioning=False unexpectedly marked projection as depth-affected")
   if "projection" not in out_on["audit_trail"]["depth_used_in"]:
       failures.append("depth_conditioning=True missing projection in depth_used_in")


   print("Tokyo Eyes v3 internal test:")
   print(f"  seed: {seed}")
   print(f"  checks: 5")
   if failures:
       print(f"  result: FAIL ({len(failures)} failures)")
       for failure in failures:
           print(f"    - {failure}")
       return 1
   print("  result: PASS")
   return 0




def run_example() -> int:
   print("Tokyo Eyes v3: Initializing...")
   print("  v3 updates: hyperbolic depth, explicit depth conditioning, projection audit trail\n")


   # Create model
   model = GOSPConeMapper(
       node_dim=4,
       hidden=128,
       num_layers=6,
       num_experts=4,
       projection_dim=64,
   )


   # Count parameters
   total_params = sum(p.numel() for p in model.parameters())
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print(f"  Total parameters:     {total_params:,}")
   print(f"  Trainable parameters: {trainable_params:,}")


   # Verify _proj is registered.
   proj_params = sum(p.numel() for name, p in model.named_parameters() if '_proj' in name)
   print(f"  _proj parameters:     {proj_params:,} (was 0 in v1 — invisible to optimizer)")


   # Verify log_c is registered.
   log_c_found = any(name == 'log_c' for name, _ in model.named_parameters())
   print(f"  log_c learnable:      {log_c_found}")
   print(f"  Initial curvature:    {model.curvature.item():.4f}")


   # Verify MobiusLinear parameters are registered.
   mobius_params = sum(p.numel() for name, p in model.named_parameters() if 'mobius_proj' in name)
   print(f"  Mobius proj params:   {mobius_params:,}")


   # Create synthetic data
   data = create_example_data(num_nodes=50, num_edges=200)
   print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
   print(f"  Clustering range: [{data.clustering.min():.3f}, {data.clustering.max():.3f}]")


   # Forward pass
   output = model(data)


   print("\nForward pass outputs:")
   print(f"  projections:    {output['projections'].shape}")
   print(f"  cone_depth:     {output['cone_depth'].shape}")
   print(f"  cone_width:     {output['cone_width'].shape}")
   print(f"  expert_weights: {output['expert_weights'].shape}")
   print(f"  epistemic unc:  {output['uncertainty']['epistemic'].shape}")
   print(f"  aleatoric unc:  {output['uncertainty']['aleatoric'].shape}")
   print(f"  balance_loss:   {output['balance_loss'].item():.4f}")
   print(f"  depth metric:   {output['audit_trail']['depth_metric']}")
   print(f"  proj clamp #:   {int(output['audit_trail']['projection_applied_count'].item())}")


   # Verify no NaN/Inf recursively across nested output
   invalid_findings = scan_tensors_for_invalid(output)
   print(f"  NaN/Inf check:  {'FAIL' if invalid_findings else 'PASS'}")


   # Example loss computation
   target_rho = data.x[:, 0:1]
   target_dehydron = (data.x[:, 0:1] < 13.0).float()


   losses = gosp_loss(output, target_rho, target_dehydron)
   print(f"\nLosses:")
   print(f"  total:            {losses['total'].item():.4f}")
   print(f"  evidential:       {losses['evidential'].item():.4f}")
   print(f"  balance:          {losses['balance'].item():.4f}")
   print(f"  cone_consistency: {losses['cone_consistency'].item():.4f}")


   # Build optimizer
   optimizer = build_optimizer(model)
   print(f"\nOptimizer: RiemannianAdam")
   print(f"  Parameter groups: {len(optimizer.param_groups)}")


   # Verify curvature gets gradients
   optimizer.zero_grad()
   losses['total'].backward()
   log_c_grad = model.log_c.grad
   print(f"  log_c gradient:   {log_c_grad.item():.6f} (nonzero = curvature is learning)")


   if invalid_findings:
       print("\nInvalid tensor findings:")
       for finding in invalid_findings:
           print(f"  - {finding}")


   print("\nTokyo Eyes v3: Ready for training.")
   return 0




if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Tokyo Eyes v3 model utility")
   parser.add_argument(
       "--internaltest",
       action="store_true",
       help="run lightweight deterministic internal property checks and exit",
   )
   parser.add_argument(
       "--seed",
       type=int,
       default=2,
       help="random seed for --internaltest (default: 2)",
   )
   args = parser.parse_args()


   if args.internaltest:
       sys.exit(run_internal_test(seed=args.seed))
   sys.exit(run_example())