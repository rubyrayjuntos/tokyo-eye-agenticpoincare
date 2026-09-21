"""A-grad-logger: logged bucket grads = train-step multi-structure backward.

Acceptance from tokyo_eye_equ_grad_telemetry_next_card_design.json §1:
logged bucket_grad_l2 equals the post-accumulation train backward, not an
extra hold[0]-only probe.
"""

from __future__ import annotations

from pathlib import Path

import torch

from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
    GumbelTemperatureSchedule,
)
from science.tokyo_eye.v8.equiformer_frontend import (
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
)
from science.tokyo_eye.v8.grad_reachability import SPINE_NZ_BUCKETS
from science.tokyo_eye.v8.heads import sdrp_cross_entropy
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from scripts.wrap1_zhyp_g_fit import (
    SDRP_COEFF,
    _bucket_grad_l2_map,
    _euc_skip_share,
    _step0_spine_ok,
    run_step_sdrp_only,
)

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "wrap1_zhyp_g_fit.py"


def _tiny_system(device: torch.device) -> TokyoEyeV8WithFrontend:
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=8,
        vector_dim=3,
        num_backbone_blocks=2,
        live_backbone=False,
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=8,
        vector_dim=3,
        hidden_dim=8,
        num_attn_layers=2,
        num_sdrp_classes=5,
        num_relations=6,
        gate_hidden=8,
    )
    system = TokyoEyeV8WithFrontend(frontend, spine).to(device)
    system.set_moe_mode("ablated")
    system.set_moe_temperature(1.0)
    system.set_moe_explore_epsilon(0.0)
    return system


def _toy_batch(device: torch.device, *, seed: int, n: int = 8) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    y = torch.randint(0, 5, (n,), generator=g)
    return {
        "x": x.to(device),
        "edge_index": edge_index.to(device),
        "edge_type": edge_type.to(device),
        "sdrp_target": y.to(device),
    }


def _manual_multi_structure_bucket(
    system: TokyoEyeV8WithFrontend,
    batches: list[dict[str, torch.Tensor]],
    *,
    tau: float,
) -> dict[str, float]:
    """Same mean-over-n SDRP backward as run_step / step0 (no clip/step)."""
    system.train()
    system.zero_grad(set_to_none=True)
    n = len(batches)
    for batch in batches:
        out = system(
            batch["x"],
            batch["edge_index"],
            batch["edge_type"],
            tau_ceiling=tau,
        )
        loss = (
            float(SDRP_COEFF)
            * sdrp_cross_entropy(out["sdrp_logits"], batch["sdrp_target"])
        ) / float(n)
        loss.backward()
    return _bucket_grad_l2_map(system)


def test_source_no_hold0_probe_for_midrun_telemetry() -> None:
    src = RUNNER.read_text(encoding="utf-8")
    assert "probe_batch = train_batches[0]" not in src
    assert "bucket_grad_l2_by_name(model)" not in src
    assert 'grad_source": "train_backward_multi_structure"' in src
    assert 'metrics.get("bucket_grad_l2")' in src
    assert "euc_skip_share" in src
    assert "A-grad-logger-train-backward-align" in src


def test_single_structure_probe_differs_from_multi_structure() -> None:
    """Documents the measurement bug class decision 1 fixes."""
    device = torch.device("cpu")
    system = _tiny_system(device)
    b0 = _toy_batch(device, seed=0)
    b1 = _toy_batch(device, seed=1)
    tau = 0.70
    single = _manual_multi_structure_bucket(system, [b0], tau=tau)
    multi = _manual_multi_structure_bucket(system, [b0, b1], tau=tau)
    assert any(
        abs(float(single.get(b, 0.0)) - float(multi.get(b, 0.0))) > 1e-8
        for b in SPINE_NZ_BUCKETS
    ), "single-structure probe matched multi-structure — fixture too weak"


def test_run_step_logged_bucket_equals_train_backward() -> None:
    """Acceptance: returned bucket_grad_l2 == pre-clip multi-structure backward."""
    device = torch.device("cpu")
    tau_start = 0.70
    batches = [_toy_batch(device, seed=10), _toy_batch(device, seed=11)]

    torch.manual_seed(123)
    system_a = _tiny_system(device)
    sd = {k: v.detach().clone() for k, v in system_a.state_dict().items()}
    system_b = _tiny_system(device)
    system_b.load_state_dict(sd)

    expected = _manual_multi_structure_bucket(system_a, batches, tau=tau_start)

    opt = torch.optim.SGD(system_b.parameters(), lr=0.0)
    radius = CurriculumRadiusController(
        tau_start=tau_start, tau_end=0.90, total_epochs=10
    )
    gumbel = GumbelTemperatureSchedule(
        tau_start=1.0, tau_end=0.5, total_epochs=10, schedule="linear"
    )
    metrics = run_step_sdrp_only(
        system_b,
        opt,
        batches,
        epoch=0,
        radius=radius,
        gumbel=gumbel,
        explore_epsilon=0.0,
        max_grad_norm=1.0,
        sdrp_coeff=SDRP_COEFF,
    )
    assert metrics["nan_abort"] == 0.0
    got = metrics["bucket_grad_l2"]
    assert set(SPINE_NZ_BUCKETS).issubset(got.keys())
    for b in list(SPINE_NZ_BUCKETS) + ["sdrp_head", "euc_skip"]:
        if b not in expected and b not in got:
            continue
        assert abs(float(got.get(b, 0.0)) - float(expected.get(b, 0.0))) < 1e-5, (
            f"bucket {b}: got={got.get(b)} expected={expected.get(b)}"
        )
    share = float(metrics["euc_skip_share"])
    expected_share = _euc_skip_share(got)
    if share != share:  # NaN
        assert expected_share != expected_share
    else:
        assert abs(share - expected_share) < 1e-9


def test_step0_uses_full_train_batches() -> None:
    device = torch.device("cpu")
    system = _tiny_system(device)
    batches = [_toy_batch(device, seed=20), _toy_batch(device, seed=21)]
    step0 = _step0_spine_ok(system, batches, tau=0.70)
    assert step0["n_train_structures"] == 2
    assert step0["ok"] is True
    assert all(step0["statuses"][b] == "NZ" for b in SPINE_NZ_BUCKETS)
    assert all(p.grad is None for _, p in system.spine.named_parameters())
