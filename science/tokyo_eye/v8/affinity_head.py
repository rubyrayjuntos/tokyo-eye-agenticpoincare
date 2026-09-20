"""Pocket-gated hyperbolic affinity head (Sprint 10 + 10.1 joint R6)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8.attention import (
    GyroOrthogonalMap,
    einstein_midpoint,
    exp_map_zero,
    gyroscalar_mul,
    mobius_add,
    poincare_dist,
    project_to_ball,
    tau_relative_residual_step,
)
from science.tokyo_eye.v8.ligand_interface import LIGAND_FEAT_DIM, unique_r6_contacts


class PocketGatedAffinityHead(nn.Module):
    """Dehydron/mech/rim-gated Einstein pool → readout FFN → scalar (−log K)."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        gate_hidden: int = 32,
        ffn_hidden: int = 64,
        c: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.c = float(c)
        self.eps = float(eps)
        # Gate from manifold invariants only (no Linear on ball ambient coords).
        self.score = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.prototype = nn.Parameter(0.05 * torch.randn(hidden_dim))
        self.readout_scale = nn.Parameter(torch.ones(()))
        self.readout_bias = nn.Parameter(torch.zeros(()))

    def pocket_weights(
        self,
        z_hyp: torch.Tensor,
        *,
        mechanism_score: torch.Tensor,
        dehydron_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``w`` with shape ``[N, 1]``, ``softmax`` over nodes (dim=0)."""
        if z_hyp.ndim != 2:
            raise ValueError("z_hyp must be [N, d]")
        n = z_hyp.shape[0]
        r = torch.linalg.vector_norm(z_hyp, dim=-1, keepdim=True)
        mech = torch.sigmoid(mechanism_score).reshape(n, 1)
        dehyd = dehydron_labels.reshape(n, 1).to(dtype=z_hyp.dtype)
        feat = torch.cat([r, mech, dehyd], dim=-1)
        scores = self.score(feat)
        w = F.softmax(scores, dim=0)
        return w

    def pool_graph(
        self,
        z_hyp: torch.Tensor,
        *,
        mechanism_score: torch.Tensor,
        dehydron_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = project_to_ball(z_hyp, c=self.c, eps=self.eps)
        w = self.pocket_weights(
            z, mechanism_score=mechanism_score, dehydron_labels=dehydron_labels
        )
        # Einstein / Klein barycenter (pure-hyp); not tangent Euclidean mean.
        z_graph = einstein_midpoint(z, w.squeeze(-1), c=self.c, eps=self.eps)
        return z_graph, w

    def forward(
        self,
        z_hyp: torch.Tensor,
        *,
        mechanism_score: torch.Tensor,
        dehydron_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_graph, w = self.pool_graph(
            z_hyp,
            mechanism_score=mechanism_score,
            dehydron_labels=dehydron_labels,
        )
        proto = exp_map_zero(self.prototype.unsqueeze(0), c=self.c, eps=self.eps)
        d = poincare_dist(z_graph.unsqueeze(0), proto, c=self.c, eps=self.eps)
        pred = (-self.readout_scale * d + self.readout_bias).squeeze(-1)
        return {
            "affinity_pred": pred,
            "z_graph": z_graph,
            "pocket_weights": w,
        }


class JointPocketAffinityHead(nn.Module):
    """R6 asymmetric cross-attn (Q=z_hyp, K/V=ligand) + pocket-gated pool."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        ligand_feat_dim: int = LIGAND_FEAT_DIM,
        attn_dim: int | None = None,
        gate_hidden: int = 32,
        ffn_hidden: int = 64,
        c: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ligand_feat_dim = int(ligand_feat_dim)
        self.attn_dim = int(attn_dim if attn_dim is not None else hidden_dim)
        self.c = float(c)
        self.eps = float(eps)
        self.lig_in = nn.Linear(self.ligand_feat_dim, self.hidden_dim)
        self.R_q = GyroOrthogonalMap(self.hidden_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.attn_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.attn_dim)
        self.pocket = PocketGatedAffinityHead(
            hidden_dim,
            gate_hidden=gate_hidden,
            ffn_hidden=ffn_hidden,
            c=c,
            eps=eps,
        )
        # Undamped z⊕msgs matches attn/MoE residual inflation class.
        self.residual_logit = nn.Parameter(torch.tensor(-1.0986122886681098))

    def r6_messages(
        self,
        z_hyp: torch.Tensor,
        lig_feat: torch.Tensor,
        edge_index_r6: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Sparse res←lig messages; empty R6 short-circuits to zeros (no softmax)."""
        n = z_hyp.shape[0]
        m = z_hyp.new_zeros(n, self.hidden_dim)
        if (
            edge_index_r6 is None
            or edge_index_r6.numel() == 0
            or lig_feat is None
            or lig_feat.numel() == 0
        ):
            return m, True

        ei_np = unique_r6_contacts(edge_index_r6.detach().cpu().numpy())
        if ei_np.shape[1] == 0:
            return m, True
        ei = torch.as_tensor(ei_np, device=z_hyp.device, dtype=torch.long)
        res_idx = ei[0]
        lig_idx = ei[1]

        z = project_to_ball(z_hyp, c=self.c, eps=self.eps)
        q = self.R_q(z, c=self.c, eps=self.eps)
        ell = F.silu(self.lig_in(lig_feat))
        k_ball = exp_map_zero(self.W_k(ell), c=self.c, eps=self.eps)
        v_ball = exp_map_zero(self.W_v(ell), c=self.c, eps=self.eps)

        unique_res = torch.unique(res_idx)
        for ri in unique_res.tolist():
            mask = res_idx == int(ri)
            neigh = lig_idx[mask]
            if neigh.numel() == 0:
                continue
            d = poincare_dist(
                q[int(ri)].unsqueeze(0).expand(neigh.numel(), -1),
                k_ball[neigh],
                c=self.c,
                eps=self.eps,
            )
            alpha = F.softmax(-d, dim=0)
            m[int(ri)] = einstein_midpoint(v_ball[neigh], alpha, c=self.c, eps=self.eps)
        return m, False

    def forward(
        self,
        z_hyp: torch.Tensor,
        *,
        mechanism_score: torch.Tensor,
        dehydron_labels: torch.Tensor,
        lig_feat: torch.Tensor,
        edge_index_r6: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if z_hyp.ndim != 2:
            raise ValueError("z_hyp must be [N, d]")
        msgs, r6_empty = self.r6_messages(z_hyp, lig_feat, edge_index_r6)
        z = project_to_ball(z_hyp, c=self.c, eps=self.eps)
        r = torch.linalg.vector_norm(z, dim=-1)
        step = tau_relative_residual_step(
            self.residual_logit, tau_ceiling=0.70, radius=r
        )
        msgs_step = gyroscalar_mul(step, msgs, c=self.c, eps=self.eps)
        z_fuse = mobius_add(z, msgs_step, c=self.c, eps=self.eps)
        out = self.pocket(
            z_fuse,
            mechanism_score=mechanism_score,
            dehydron_labels=dehydron_labels,
        )
        out["r6_empty"] = torch.tensor(
            1.0 if r6_empty else 0.0, device=z_hyp.device, dtype=z_hyp.dtype
        )
        out["r6_messages"] = msgs
        return out


__all__ = ["JointPocketAffinityHead", "PocketGatedAffinityHead"]
