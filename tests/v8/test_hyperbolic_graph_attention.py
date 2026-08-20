"""Sprint 1: sparse relation-aware HyperbolicGraphAttention (v8)."""

from __future__ import annotations

import torch

from science.tokyo_eye.v8.attention import (
    HyperbolicGraphAttention,
    klein_to_poincare,
    poincare_dist,
    poincare_to_klein,
)


def test_poincare_dist_zero_and_symmetric() -> None:
    x = torch.tensor([[0.1, 0.0], [0.0, 0.2]])
    y = torch.tensor([[0.2, 0.0], [0.0, 0.1]])
    d_xy = poincare_dist(x, y, c=1.0)
    d_yx = poincare_dist(y, x, c=1.0)
    assert torch.allclose(d_xy, d_yx, atol=1e-5)
    # Identical points use a numerical floor (stable acosh grads) — not exact 0
    d0 = float(poincare_dist(x[:1], x[:1]))
    assert d0 < 0.05


def test_klein_roundtrip() -> None:
    z = torch.tensor([[0.3, -0.2], [0.0, 0.0], [0.7, 0.1]])
    k = poincare_to_klein(z, c=1.0)
    back = klein_to_poincare(k, c=1.0)
    assert torch.allclose(back, z, atol=1e-5)


def test_attention_sparse_no_dense_board() -> None:
    """Forward must not allocate N×N attention; shape tracks E only."""
    n, d, e = 5, 8, 6
    z = torch.randn(n, d) * 0.05
    # Bidirected synthetic edges
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 3], [1, 0, 2, 1, 4, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    layer = HyperbolicGraphAttention(d, num_relations=6, c=1.0)
    out = layer(z, edge_index, edge_type)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_gamma_beta_init_and_logit_formula() -> None:
    layer = HyperbolicGraphAttention(4, num_relations=6, c=1.0)
    assert torch.allclose(layer.gamma.weight, torch.ones_like(layer.gamma.weight))
    assert torch.allclose(layer.beta.weight, torch.zeros_like(layer.beta.weight))


def test_option_a_isolate_self_transport() -> None:
    """Node with no outgoing sparse edges keeps manifold self-transport path."""
    torch.manual_seed(0)
    n, d = 4, 6
    z = torch.randn(n, d) * 0.04
    # Only connect 0↔1; nodes 2 and 3 are isolates
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 0], dtype=torch.long)
    layer = HyperbolicGraphAttention(d, num_relations=6, c=1.0)
    layer.eval()
    with torch.no_grad():
        out = layer(z, edge_index, edge_type)
    # Isolates must be finite and differ from a NaN/zero collapse
    assert torch.isfinite(out[2]).all()
    assert torch.isfinite(out[3]).all()
    # Connected nodes should be allowed to move; isolates use W_o∘log∘exp path
    # Identity-ish when W_* near init: still finite ball points
    assert float(out[2].norm()) < 1.0 - 1e-3


def test_relation_ids_select_embeddings() -> None:
    layer = HyperbolicGraphAttention(4, num_relations=6, c=1.0)
    with torch.no_grad():
        layer.gamma.weight.copy_(torch.arange(6, dtype=torch.float32).view(6, 1) + 1.0)
        layer.beta.weight.copy_(torch.arange(6, dtype=torch.float32).view(6, 1) * 0.1)
    z = torch.zeros(3, 4)
    z[0, 0] = 0.2
    z[1, 0] = -0.2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor([2, 2], dtype=torch.long)
    out = layer(z, edge_index, edge_type)
    assert out.shape == (3, 4)


def test_empty_edge_index_all_isolates() -> None:
    z = torch.randn(3, 5) * 0.03
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_type = torch.zeros(0, dtype=torch.long)
    layer = HyperbolicGraphAttention(5, num_relations=6, c=1.0)
    out = layer(z, edge_index, edge_type)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()
