"""Pocket-gated hyperbolic affinity head (Sprint 10 + 10.1 joint R6)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8.attention import exp_map_zero, log_map_zero, project_to_ball
from science.tokyo_eye.v8.ligand_interface import LIGAND_FEAT_DIM, unique_r6_contacts


class PocketGatedAffinityHead(nn.Module):
    """Dehydron/mech/rim-gated tangent pool → FFN → scalar (−log K)."""

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
        self.h_inv = nn.Linear(hidden_dim, gate_hidden)
        self.score = nn.Sequential(
            nn.Linear(gate_hidden + 3, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, ffn_hidden),
            nn.SiLU(),
            nn.Linear(ffn_hidden, 1),
        )

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
        h = F.silu(self.h_inv(z_hyp))
        feat = torch.cat([h, r, mech, dehyd], dim=-1)
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
        u = log_map_zero(z, c=self.c, eps=self.eps)
        t_pool = torch.sum(w * u, dim=0)
        z_graph = exp_map_zero(t_pool.unsqueeze(0), c=self.c, eps=self.eps).squeeze(0)
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
        tangent = log_map_zero(z_graph.unsqueeze(0), c=self.c, eps=self.eps).squeeze(0)
        pred = self.ffn(tangent).squeeze(-1)
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
        self.W_q = nn.Linear(self.hidden_dim, self.attn_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.attn_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.attn_dim)
        self.msg_proj = nn.Linear(self.attn_dim, self.hidden_dim)
        self.pocket = PocketGatedAffinityHead(
            hidden_dim,
            gate_hidden=gate_hidden,
            ffn_hidden=ffn_hidden,
            c=c,
            eps=eps,
        )

    def r6_messages(
        self,
        z_hyp: torch.Tensor,
        lig_feat: torch.Tensor,
        edge_index_r6: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Sparse res←lig messages; empty R6 short-circuits to zeros (no softmax)."""
        n = z_hyp.shape[0]
        m = z_hyp.new_zeros(n, self.attn_dim)
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

        ell = F.silu(self.lig_in(lig_feat))
        q = self.W_q(z_hyp)
        k = self.W_k(ell)
        v = self.W_v(ell)
        scale = 1.0 / math.sqrt(float(self.attn_dim))

        unique_res = torch.unique(res_idx)
        for ri in unique_res.tolist():
            mask = res_idx == int(ri)
            neigh = lig_idx[mask]
            if neigh.numel() == 0:
                continue
            scores = (q[int(ri)] * k[neigh]).sum(dim=-1) * scale
            alpha = F.softmax(scores, dim=0)
            m[int(ri)] = (alpha.unsqueeze(-1) * v[neigh]).sum(dim=0)
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
        z_fuse = z_hyp + self.msg_proj(msgs)
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
