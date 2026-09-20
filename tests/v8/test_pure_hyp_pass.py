"""Property tests for trunk-wide pure_hyp_pass (addendum §2.4).

Known-bad must fail and known-good must pass *before* this check gates
any train/promote stamp. These tests are that proof.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from science.tokyo_eye.v8.attention import (
    GyroOrthogonalMap,
    exp_map_zero,
    log_map_zero,
    project_to_ball,
)
from science.tokyo_eye.v8.moe import GyroExpert, TopologyAwareHardMoE
from science.tokyo_eye.v8.pure_hyp_pass import (
    _ORTH_ATOL,
    _gram_inf,
    _is_origin_isometry_matrix,
    scan_forward,
)


def _ball_points(n: int = 6, d: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    z = torch.randn(n, d) * 0.05
    return project_to_ball(z, c=1.0, eps=1e-5)


class _KnownGoodGyro(nn.Module):
    """Pure gyro action on the ball — no Linear on manifold points."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.g = GyroOrthogonalMap(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        return self.g(z, c=1.0, eps=1e-5)


class _KnownBadLinearExpert(nn.Module):
    """Tangent-Linear expert — the pre-rebuild MoE failure mode."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.expert = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        return project_to_ball(self.expert(z), c=1.0, eps=1e-5)


class _KnownBadSandwich(nn.Module):
    """Classic exp₀(W · log₀(z)) geometry substitute."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        t = log_map_zero(z, c=1.0)
        t = self.lin(t)
        return exp_map_zero(t, c=1.0)


def test_known_good_gyro_passes() -> None:
    model = _KnownGoodGyro()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert report.passed, report.as_dict()
    assert report.violations == []


def test_known_bad_linear_on_ball_fails() -> None:
    model = _KnownBadLinearExpert()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "linear_on_ball" in kinds, report.as_dict()


def test_near_origin_linear_on_ball_is_known_limitation() -> None:
    """Radii ≤ _BALL_MIN_SIGNAL (1e-4) are not classified as ball — gap.

    Documents the numeric heuristic's near-origin false negative: a real
    Linear-on-ball violation at r≈1e-5 currently slips through. Do not
    treat a pass here as proof of purity right after a tiny lift.
    """
    from science.tokyo_eye.v8.pure_hyp_pass import (
        _BALL_MIN_SIGNAL,
        _is_open_ball_batch,
    )

    model = _KnownBadLinearExpert()
    # Uniform tiny radii well below the signal floor.
    z = torch.randn(6, 8)
    z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-12)
    z = z * 1e-5
    assert float(torch.linalg.vector_norm(z, dim=-1).max()) < _BALL_MIN_SIGNAL
    assert not _is_open_ball_batch(z)
    report = scan_forward(model, lambda: model(z))
    # Known limitation: scanner does not catch this batch.
    assert report.passed, report.as_dict()


def test_known_bad_log0_linear_exp0_sandwich_fails() -> None:
    model = _KnownBadSandwich()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "linear_on_log0" in kinds or "log0_linear_exp0_sandwich" in kinds, (
        report.as_dict()
    )


def test_live_moe_experts_pass_gyro() -> None:
    """Rebuilt gyro experts must clear the isolated scanner."""
    torch.manual_seed(1)
    moe = TopologyAwareHardMoE(dim=8, gate_hidden=4, temperature=1.0)
    moe.eval()
    z = _ball_points(n=7, d=8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    report = scan_forward(moe, lambda: moe(z, edge_index))
    assert report.passed, report.as_dict()


def test_attention_isolated_pure_hyp_pass() -> None:
    from science.tokyo_eye.v8.attention import HyperbolicGraphAttention

    torch.manual_seed(0)
    layer = HyperbolicGraphAttention(8, num_relations=6, c=1.0)
    layer.eval()
    z = _ball_points(n=6, d=8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_type = torch.zeros(4, dtype=torch.long)
    report = scan_forward(layer, lambda: layer(z, edge_index, edge_type))
    assert report.passed, report.as_dict()
    linears = [
        n for n, m in layer.named_modules() if isinstance(m, nn.Linear)
    ]
    assert linears == []


class _KnownBadTangentQKV(nn.Module):
    """exp₀(W · log₀(z)) QKV — the forbidden §7.2 substitute."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.w_q = nn.Linear(dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        return exp_map_zero(self.w_q(log_map_zero(z, c=1.0)), c=1.0)


def test_attention_known_bad_tangent_qkv_fails_isolated_scan() -> None:
    """Isolated scanner must fail before it is allowed to pass live attention."""
    model = _KnownBadTangentQKV()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "linear_on_log0" in kinds or "log0_linear_exp0_sandwich" in kinds, (
        report.as_dict()
    )


def test_spine_trunk_pure_hyp_pass() -> None:
    from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

    torch.manual_seed(0)
    model = TokyoEyesHyperbolicV8(
        scalar_dim=8, vector_dim=3, hidden_dim=8, num_attn_layers=2, num_sdrp_classes=3
    )
    model.eval()
    n = 6
    s = torch.randn(n, 8)
    v = torch.randn(n, 3) * 0.1
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_type = torch.zeros(4, dtype=torch.long)
    chem = torch.rand(n, 7)
    report = scan_forward(
        model, lambda: model(s, v, edge_index, edge_type, chem=chem)
    )
    assert report.passed, report.as_dict()


class _KnownBadFunctionalLinear(nn.Module):
    """Bare ``F.linear`` on ball points — no ``nn.Linear`` child."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim) * 0.5)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        return project_to_ball(F.linear(z, self.weight, self.bias), c=1.0, eps=1e-5)


class _KnownBadUnconstrainedMatmul(nn.Module):
    """``z @ W`` with a free matrix — the relabeled-Linear failure mode."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        return project_to_ball(z @ self.weight, c=1.0, eps=1e-5)


def test_known_bad_functional_linear_on_ball_fails() -> None:
    model = _KnownBadFunctionalLinear()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "linear_on_ball" in kinds, report.as_dict()
    assert not any(isinstance(m, nn.Linear) for m in model.modules())


def test_known_bad_unconstrained_matmul_on_ball_fails() -> None:
    model = _KnownBadUnconstrainedMatmul()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "euclidean_matmul_on_ball" in kinds, report.as_dict()


def test_gyro_orthogonal_map_is_so_d_not_relabeled_linear() -> None:
    """QKV gyro map must be an origin-fixing SO(d) isometry, not nn.Linear."""
    assert not issubclass(GyroOrthogonalMap, nn.Linear)
    torch.manual_seed(2)
    g = GyroOrthogonalMap(8)
    r = g.rotation()
    eye = torch.eye(8)
    assert torch.allclose(r.T @ r, eye, atol=1e-4)
    assert float(torch.det(r).detach()) > 0.0
    z = _ball_points()
    origin = torch.zeros(1, 8)
    out0 = g(origin, c=1.0, eps=1e-5)
    assert torch.allclose(out0, origin, atol=1e-5)
    out = g(z, c=1.0, eps=1e-5)
    r_in = torch.linalg.vector_norm(z, dim=-1)
    r_out = torch.linalg.vector_norm(out, dim=-1)
    assert torch.allclose(r_in, r_out, atol=1e-4)


def test_filename_grep_is_not_this_module() -> None:
    """Guardrail: the scanner must not be a string allowlist of two files."""
    import inspect

    from science.tokyo_eye.v8 import pure_hyp_pass as php

    src = inspect.getsource(php.scan_forward)
    assert "affinity_head.py" not in src
    assert "attention.py" not in src
    install = inspect.getsource(php.PureHypTracer.install)
    # Live Module tree plus functional-op patches — not named_children-only.
    assert "named_modules" in install
    tracer_src = inspect.getsource(php.PureHypTracer)
    assert "F.linear" in tracer_src
    assert "matmul" in tracer_src
    wrap_qr = inspect.getsource(php.PureHypTracer._wrap_qr)
    assert "_skip_linalg" in wrap_qr
    assert "named_modules" not in wrap_qr
    assert "torch.linalg" in inspect.getsource(php.PureHypTracer.install)


def _sheared_identity(dim: int, delta: float) -> torch.Tensor:
    r = torch.eye(dim)
    r[0, 1] = float(delta)
    return r


def test_orthogonality_tolerance_has_an_edge() -> None:
    """``atol=1e-4`` is not exact equality: 1e-4-class residual passes, 1e-3 shear fails."""
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert _is_origin_isometry_matrix(q)
    assert _gram_inf(q) < 1e-6

    near = _sheared_identity(8, 5e-5)
    err = _gram_inf(near)
    assert 1e-7 < err < 1e-4
    assert _is_origin_isometry_matrix(near, atol=1e-4)
    assert not _is_origin_isometry_matrix(near, atol=1e-7)
    assert _is_origin_isometry_matrix(near)
    assert _ORTH_ATOL == 1e-4

    skew = _sheared_identity(8, 5e-3)
    assert _gram_inf(skew) > _ORTH_ATOL
    assert not _is_origin_isometry_matrix(skew)


class _FixedRightMap(nn.Module):
    def __init__(self, r: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("r", r)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        return project_to_ball(z @ self.r, c=1.0, eps=1e-5)


def test_smuggled_shear_fails_isometry_scan() -> None:
    z = _ball_points()
    near = _FixedRightMap(_sheared_identity(8, 5e-5))
    assert scan_forward(near, lambda: near(z)).passed
    skew = _FixedRightMap(_sheared_identity(8, 5e-3))
    report = scan_forward(skew, lambda: skew(z))
    assert not report.passed
    assert "euclidean_matmul_on_ball" in {v.kind for v in report.violations}


def test_gyro_qr_is_fresh_each_forward() -> None:
    """R is re-QR'd from the raw Parameter every call — no cached rotation."""
    import inspect

    g = GyroOrthogonalMap(8)
    r1 = g.rotation().detach().clone()
    with torch.no_grad():
        g.weight.add_(torch.randn_like(g.weight))
    r2 = g.rotation().detach().clone()
    assert not torch.allclose(r1, r2, atol=1e-4)
    assert _is_origin_isometry_matrix(r1)
    assert _is_origin_isometry_matrix(r2)
    fwd = inspect.getsource(GyroOrthogonalMap.forward)
    assert "self.rotation()" in fwd
    ctor = inspect.getsource(GyroOrthogonalMap.__init__)
    assert "register_buffer" not in ctor


def test_qr_skip_does_not_mask_ball_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Euclidean op inside the live ``torch.linalg.qr`` window must still flag."""
    orig_qr = torch.linalg.qr
    box: dict[str, torch.Tensor] = {}

    def sneaky_qr(a: torch.Tensor, *args: object, **kwargs: object):  # noqa: ANN001
        box["sneak"] = F.linear(box["z"], box["W"])
        return orig_qr(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", sneaky_qr)

    class _GyroPlusSneak(nn.Module):
        def __init__(self, dim: int = 8) -> None:
            super().__init__()
            self.g = GyroOrthogonalMap(dim)
            self.W = nn.Parameter(torch.eye(dim) * 0.4)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            box["z"] = z
            box["W"] = self.W
            return self.g(z, c=1.0, eps=1e-5)

    model = _GyroPlusSneak()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed, report.as_dict()
    assert "linear_on_ball" in {v.kind for v in report.violations}


class _KnownBadSoftGyroMix(nn.Module):
    """Valid gyro experts, Euclidean convex combination of their ball outputs."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.e0 = GyroExpert(dim)
        self.e1 = GyroExpert(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        a = self.e0(z, c=1.0, eps=1e-5)
        b = self.e1(z, c=1.0, eps=1e-5)
        return project_to_ball(0.4 * a + 0.6 * b, c=1.0, eps=1e-5)


def test_known_bad_naive_weighted_gyro_mix_fails() -> None:
    model = _KnownBadSoftGyroMix()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    assert "euclidean_mix_on_ball" in {v.kind for v in report.violations}, report.as_dict()


def test_soft_moe_routing_mix_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Soft blending of expert ball outputs is the veto, not the gyro internals."""

    def _soft_route(
        _self: TopologyAwareHardMoE, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing = F.softmax(logits, dim=-1)
        return routing, routing.new_zeros(routing.shape[0])

    monkeypatch.setattr(TopologyAwareHardMoE, "_route", _soft_route)
    torch.manual_seed(1)
    moe = TopologyAwareHardMoE(dim=8, gate_hidden=4, temperature=1.0)
    moe.eval()
    z = _ball_points(n=7, d=8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    report = scan_forward(moe, lambda: moe(z, edge_index))
    assert not report.passed
    assert "euclidean_mix_on_ball" in {v.kind for v in report.violations}, report.as_dict()


class _HistoricalLinearOnBallMoE(nn.Module):
    """Literal pre-rebuild experts: ``nn.Linear`` applied to ball points."""

    def __init__(self, dim: int = 8, n_exp: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_exp)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        stacked = torch.stack([expert(z) for expert in self.experts], dim=1)
        routing = F.one_hot(
            torch.zeros(z.shape[0], dtype=torch.long),
            num_classes=len(self.experts),
        ).to(dtype=z.dtype)
        return project_to_ball(
            (routing.unsqueeze(-1) * stacked).sum(dim=1), c=1.0, eps=1e-5
        )


def test_historical_linear_on_ball_moe_experts_fail() -> None:
    """Closing regression: the audit's original MoE counterexample still fails."""
    model = _HistoricalLinearOnBallMoE()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    assert "linear_on_ball" in {v.kind for v in report.violations}, report.as_dict()


class _KnownBadTop1ThenConfidenceScale(nn.Module):
    """Hard-select one gyro expert, then ``* softmax confidence`` in ambient coords.

    The original MoE pattern: ``out[mask] = expert_out * max_weights[mask].unsqueeze(-1)``.
    """

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.expert = GyroExpert(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = project_to_ball(z, c=1.0, eps=1e-5)
        expert_outputs = self.expert(z, c=1.0, eps=1e-5)
        mask = torch.ones(z.shape[0], dtype=torch.bool)
        max_weights = torch.full((z.shape[0],), 0.87, dtype=z.dtype)
        out_tokens = torch.zeros_like(expert_outputs)
        out_tokens[mask] = expert_outputs * max_weights[mask].unsqueeze(-1)
        return project_to_ball(out_tokens, c=1.0, eps=1e-5)


def test_known_bad_top1_then_confidence_scale_fails() -> None:
    model = _KnownBadTop1ThenConfidenceScale()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "euclidean_scale_on_ball" in kinds, report.as_dict()
    assert "euclidean_mix_on_ball" not in kinds or "euclidean_scale_on_ball" in kinds


def test_hard_01_select_is_not_a_confidence_scale() -> None:
    """``1·z`` / ``0·z`` one-hot masks must not be scored as a scale."""

    class _HardSelect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.g = GyroOrthogonalMap(8)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            z = project_to_ball(z, c=1.0, eps=1e-5)
            y = self.g(z, c=1.0, eps=1e-5)
            return project_to_ball(y * 1.0, c=1.0, eps=1e-5)

    model = _HardSelect()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert report.passed, report.as_dict()


def _unlisted_ambient_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """New closed form nobody added to the manifold-identity allowlist."""
    return 0.5 * a + 0.5 * b


def test_unlisted_closed_form_is_scanned_not_exempt() -> None:
    from science.tokyo_eye.v8.pure_hyp_pass import MANIFOLD_FORMULA_NAMES

    assert "unlisted_ambient_mean" not in MANIFOLD_FORMULA_NAMES
    assert _unlisted_ambient_mean.__name__ not in MANIFOLD_FORMULA_NAMES
    assert _unlisted_ambient_mean.__code__ not in {
        getattr(GyroOrthogonalMap, "__code__", None)
    }

    class _UnlistedMix(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e0 = GyroExpert(8)
            self.e1 = GyroExpert(8)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            z = project_to_ball(z, c=1.0, eps=1e-5)
            a = self.e0(z, c=1.0, eps=1e-5)
            b = self.e1(z, c=1.0, eps=1e-5)
            return project_to_ball(_unlisted_ambient_mean(a, b), c=1.0, eps=1e-5)

    model = _UnlistedMix()
    z = _ball_points()
    report = scan_forward(model, lambda: model(z))
    assert not report.passed
    kinds = {v.kind for v in report.violations}
    assert "euclidean_mix_on_ball" in kinds or "euclidean_scale_on_ball" in kinds, (
        report.as_dict()
    )


def test_pre_lift_scope_matches_trailing_projector() -> None:
    """Full-spine wrap names the lift ``spine.projector`` (trailing segment)."""
    from science.tokyo_eye.v8.pure_hyp_pass import _is_pre_lift_scope

    assert _is_pre_lift_scope("projector")
    assert _is_pre_lift_scope("spine.projector")
    assert _is_pre_lift_scope("spine.projector.radial_mlp")
    assert _is_pre_lift_scope("TokyoEyeV8WithFrontend.projector")
    assert not _is_pre_lift_scope("moe")
    assert not _is_pre_lift_scope("spine.moe.experts.0")


def test_nested_radial_projector_passes_scan() -> None:
    """Reproduce the full-spine nesting that false-positived before the amend."""
    from science.tokyo_eye.v8.projector import RadialAngularProjector

    class _Spine(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projector = RadialAngularProjector(scalar_dim=4, vector_dim=3, hidden_dim=8)

        def forward(self, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return self.projector(s, v, tau_ceiling=0.7)

    torch.manual_seed(0)
    model = _Spine()
    s = torch.randn(5, 4)
    v = torch.randn(5, 3)  # [N, vector_dim] — matches Angular Linear last-dim
    report = scan_forward(model, lambda: model(s, v))
    assert report.passed, report.as_dict()
