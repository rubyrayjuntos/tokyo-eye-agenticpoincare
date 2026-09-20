"""Sprint 7: live SE(3)-lite backbone binds MPtrj bank into forward grads."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from science.tokyo_eye.v8.equiformer_frontend import (
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    apply_weight_map,
    load_weight_map,
)
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints/v8/pretrained/equiformer_v3_baseline.pt"


def _tiny_batch(n: int = 8):
    x = torch.randn(n, 3)
    # chain edges
    src = torch.arange(n - 1)
    dst = src + 1
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    et = torch.zeros(ei.shape[1], dtype=torch.long)
    return x, ei, et


@pytest.mark.skipif(
    not CKPT.is_file() or CKPT.stat().st_size < 1_000_000,
    reason="Equiformer MPtrj ckpt missing",
)
def test_live_backbone_grads_reach_atom_embed() -> None:
    cfg = load_weight_map()
    fe = StubEquiformerFrontend(
        in_dim=3, scalar_dim=128, vector_dim=3, live_backbone=True
    )
    apply_weight_map(fe, CKPT, cfg)
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=128, vector_dim=3, hidden_dim=128, num_attn_layers=2, num_sdrp_classes=5
    )
    system = TokyoEyeV8WithFrontend(fe, spine)
    x, ei, et = _tiny_batch()
    out = system(x, ei, et, tau_ceiling=0.9)
    loss = out["mechanism_score"].sum() + out["z_hyp"].sum()
    loss.backward()
    g = fe.backbone.atom_embed.weight.grad
    assert g is not None
    assert float(g.abs().sum()) > 0.0
    assert fe.backbone.blocks[0].attn.source.weight.grad is not None


def test_freeze_backbone_uses_stub_no_bank_grad() -> None:
    fe = StubEquiformerFrontend(
        in_dim=3, scalar_dim=32, vector_dim=3, live_backbone=False
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=32, vector_dim=3, hidden_dim=32, num_attn_layers=2, num_sdrp_classes=5
    )
    system = TokyoEyeV8WithFrontend(fe, spine)
    x, ei, et = _tiny_batch()
    out = system(x, ei, et, tau_ceiling=0.9)
    out["mechanism_score"].sum().backward()
    assert fe.backbone.atom_embed.weight.grad is None
    assert fe.scalar_trunk[0].weight.grad is not None
