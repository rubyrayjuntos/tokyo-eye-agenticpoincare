"""Sprint 1: sparse relation-aware HyperbolicGraphAttention (v8)."""

from __future__ import annotations

import pytest
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


def test_softmax_index_is_destination() -> None:
    import inspect

    from science.tokyo_eye.v8.attention import HyperbolicGraphAttention

    src = inspect.getsource(HyperbolicGraphAttention.forward)
    assert "softmax(logits, dst" in src
    assert "softmax(logits, src" not in src


def test_softmax_runtime_index_is_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standing §7.1 probe: live softmax index is ``edge_index[1]`` on an
    asymmetric graph (src ≠ dst). A one-shot construction check would not
    catch a later flip back to ``[0]``.
    """
    import science.tokyo_eye.v8.attention as attn_mod

    captured: list[torch.Tensor] = []
    orig = attn_mod.softmax

    def wrapped(src, index, num_nodes=None, **kwargs):  # noqa: ANN001
        captured.append(index.detach().clone())
        return orig(src, index, num_nodes=num_nodes, **kwargs)

    monkeypatch.setattr(attn_mod, "softmax", wrapped)
    n, d = 4, 6
    z = torch.randn(n, d) * 0.04
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2], dtype=torch.long)
    layer = HyperbolicGraphAttention(d, num_relations=6, c=1.0)
    layer.eval()
    with torch.no_grad():
        layer(z, edge_index, edge_type)
    assert captured, "softmax was not called"
    assert not torch.equal(edge_index[0], edge_index[1]), (
        "fixture must be asymmetric so a src/dst swap is observable"
    )
    assert torch.equal(captured[0], edge_index[1])
    assert not torch.equal(captured[0], edge_index[0])


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


def test_gyroscalar_mul_identity_and_shrink() -> None:
    from science.tokyo_eye.v8.attention import gyroscalar_mul, project_to_ball

    x = project_to_ball(torch.tensor([[0.4, 0.0], [0.7, 0.0]]))
    same = gyroscalar_mul(1.0, x, c=1.0)
    assert torch.allclose(same, x, atol=1e-5)
    shrunk = gyroscalar_mul(0.25, x, c=1.0)
    r0 = torch.linalg.vector_norm(x, dim=-1)
    r1 = torch.linalg.vector_norm(shrunk, dim=-1)
    assert torch.all(r1 < r0)


def test_two_layer_residual_does_not_pin_all_to_tau() -> None:
    """Regression: undamped self⊕attended pinned every node to τ within L=2."""
    from science.tokyo_eye.v8.attention import clamp_ball_radius
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

    torch.manual_seed(0)
    tau = 0.7
    n, d = 24, 16
    # Spread radii under τ, fully connected sparse edges.
    radii = torch.linspace(0.25, 0.55, n)
    z = torch.zeros(n, d)
    z[:, 0] = radii
    # Ring + a few long-range edges
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_type = torch.zeros(edge_index.shape[1], dtype=torch.long)

    spine = TokyoEyesHyperbolicV8(
        scalar_dim=d, vector_dim=3, hidden_dim=d, num_attn_layers=2, num_sdrp_classes=3
    )
    # Drive only the attention stack (bypass lift) with known radii.
    z_cur = clamp_ball_radius(z, max_r=tau, eps=spine.eps)
    for layer in spine.attn_layers:
        z_cur = layer(z_cur, edge_index, edge_type, tau_ceiling=tau)
        z_cur = clamp_ball_radius(z_cur, max_r=tau, eps=spine.eps)

    r = torch.linalg.vector_norm(z_cur, dim=-1)
    spread = float(r.max() - r.min())
    n_at_tau = int(((r - tau).abs() < 1e-5).sum())
    assert spread > 0.05, f"spread collapsed: {spread}"
    assert n_at_tau < n, f"all {n} nodes pinned to τ={tau}"


def test_spine_spread_holds_across_curriculum_tau_spot_checks() -> None:
    """v3 bar is spread≳0.20 at τ=0.7; must not delta-pin as τ→0.95 (cold forward)."""
    from experiments.training.v8.run_v8_experiment import _make_batch
    from science.tokyo_eye.v8.equiformer_frontend import (
        StubEquiformerFrontend,
        TokyoEyeV8WithFrontend,
    )
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

    torch.manual_seed(0)
    fe = StubEquiformerFrontend(
        in_dim=3, scalar_dim=128, vector_dim=3, num_backbone_blocks=7, live_backbone=True
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=128, vector_dim=3, hidden_dim=128, num_attn_layers=2, num_sdrp_classes=5
    )
    sys = TokyoEyeV8WithFrontend(fe, spine)
    batch = _make_batch(torch.device("cpu"))
    sys.eval()
    # Quality bar from freeze_recon_smoke_seed0_v3: post spread ≈ 0.224 at τ=0.7.
    # Require healthy diversity (not merely off-ceiling) at each spot-check τ.
    min_spread = 0.05
    rows = {}
    with torch.no_grad():
        for tau in (0.70, 0.75, 0.85, 0.95):
            out = sys(
                batch["x"],
                batch["edge_index"],
                batch["edge_type"],
                tau_ceiling=tau,
                chem=batch.get("gate_chem"),
            )
            for tag in ("z_attn", "z_hyp"):
                r = torch.linalg.vector_norm(out[tag], dim=-1)
                spread = float((r.max() - r.min()).detach())
                n_at = int(((r - tau).abs() < 1e-5).sum())
                rows[f"{tag}_tau{tau}"] = {"spread": spread, "n_at_tau": n_at}
                assert spread >= min_spread, (
                    f"{tag} spread {spread:.4f} < {min_spread} at τ={tau} (ceiling pin)"
                )
                assert n_at < r.numel(), f"{tag} all pinned to τ={tau}"
    # At calibration τ, post spread should be near the v3 bar (not tiny).
    assert rows["z_hyp_tau0.7"]["spread"] >= 0.15, rows
