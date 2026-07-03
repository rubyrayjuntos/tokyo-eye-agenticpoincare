"""Property tests for MoE routing collapse metrics (P_ROUTING_01)."""

from __future__ import annotations

import math

import pytest
import torch

from science.training.mlflow_governance import stage_a_gate_passed
from science.training.routing_metrics import (
    effective_experts,
    min_routing_fraction,
)


def test_effective_experts_uniform_vs_collapsed() -> None:
    n = 8
    uniform = torch.ones(n) / n
    collapsed = torch.zeros(n)
    collapsed[0] = 1.0

    eff_u = effective_experts(uniform)
    eff_c = effective_experts(collapsed)

    assert abs(eff_u - n) < 0.1
    assert abs(eff_c - 1.0) < 0.1
    assert eff_u != eff_c


def test_mean_routing_fraction_is_constant_under_collapse() -> None:
    """Mean fraction is always 1/N — it cannot detect collapse."""
    n = 4
    uniform = torch.ones(n) / n
    collapsed = torch.zeros(n)
    collapsed[0] = 1.0
    assert abs(uniform.mean().item() - collapsed.mean().item()) < 1e-9


def test_min_routing_fraction_detects_collapse() -> None:
    n = 4
    uniform = torch.ones(n) / n
    collapsed = torch.zeros(n)
    collapsed[0] = 1.0
    assert min_routing_fraction(uniform) > 0.2
    assert min_routing_fraction(collapsed) == pytest.approx(0.0)


def test_stage_gate_uses_effective_experts_not_mean_fraction() -> None:
    health = {
        "disc_sigma2_sigma1_mean": 0.665,
        "probe_r_depth_sasa": 0.73,
    }
    # Healthy routing: ~3.9 effective experts (entropy ~1.37 over 4 experts)
    losses_good = {
        "effective_experts": 3.9,
        "effective_experts_min": 3.5,
        "min_routing_fraction": 0.18,
        "per_fold_loss.3_40_50_300": 1.0,
        "per_fold_loss.3_80_20_20": 1.1,
    }
    assert stage_a_gate_passed(health, losses_good) == 1

    # Old bug: mean fraction ~0.25 always — would fail [3, 4.5] even when healthy
    losses_fraction_only = {
        "expert_load_0": 0.25,
        "expert_load_1": 0.25,
        "expert_load_2": 0.25,
        "expert_load_3": 0.25,
        "per_fold_loss.3_40_50_300": 1.0,
        "per_fold_loss.3_80_20_20": 1.1,
    }
    assert stage_a_gate_passed(health, losses_fraction_only) == 0

    # Collapsed routing: effective → 1
    losses_collapsed = {
        "effective_experts": 1.0,
        "effective_experts_min": 1.0,
        "min_routing_fraction": 0.0,
    }
    assert stage_a_gate_passed(health, losses_collapsed) == 0


def test_four_expert_lever_a_routing_entropy_band() -> None:
    """routing_entropy ≈ 1.38 (lever_a) → effective ≈ 4, inside Stage A band."""
    h = 1.3785733779271443
    eff = math.exp(h)
    assert 3.0 <= eff <= 4.5


@pytest.mark.integration
def test_collapsed_checkpoint_trips_stage_gate() -> None:
    """Belt-and-suspenders: archived collapsed routing must fail the gate (can say no)."""
    from pathlib import Path

    from experiments.training.v6.assess_checkpoint import load_v6_model
    from experiments.training.v6.corpus import load_training_proteins
    from experiments.training.v6.train_loop import attach_v6_features, measure_geometry_health
    from science.training.mlflow_governance import stage_a_gate_passed

    collapsed_ckpt = Path("checkpoints/v6/runs/full_hyp_moe_test/v6_best.pt")
    healthy_ckpt = Path("checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt")
    if not collapsed_ckpt.is_file() or not healthy_ckpt.is_file():
        pytest.skip("archived checkpoints not present in workspace")

    manifest = Path("manifests/v6_corpus_disc_target.json")
    proteins, _ = load_training_proteins(Path("/tmp/dtie_pdb_cache"), manifest, max_proteins=3, max_residues=600)

    def _routing_losses(model: torch.nn.Module) -> dict[str, float]:
        effs, min_fracs, route_hs = [], [], []
        with torch.no_grad():
            for prot in proteins:
                data = attach_v6_features(prot["data"].to("cpu"))
                out = model(data)
                load = out["expert_load"].cpu()
                effs.append(effective_experts(load))
                min_fracs.append(min_routing_fraction(load))
                route_hs.append(float(out["routing_entropy"].cpu()))
        return {
            "effective_experts": float(sum(effs) / len(effs)),
            "effective_experts_min": float(min(effs)),
            "min_routing_fraction": float(min(min_fracs)),
            "routing_entropy": float(sum(route_hs) / len(route_hs)),
            "per_fold_loss.3_40_50_300": 1.0,
            "per_fold_loss.3_80_20_20": 1.0,
        }

    healthy = load_v6_model(healthy_ckpt, "cpu")
    collapsed = load_v6_model(collapsed_ckpt, "cpu")
    health_h = measure_geometry_health(healthy, proteins, "cpu")
    health_c = measure_geometry_health(collapsed, proteins, "cpu")
    losses_h = _routing_losses(healthy)
    losses_c = _routing_losses(collapsed)

    eff_h = losses_h["effective_experts"]
    eff_c = losses_c["effective_experts"]
    assert eff_h > eff_c, f"expected healthy ({eff_h:.2f}) > collapsed ({eff_c:.2f})"
    assert eff_h >= 3.0
    assert eff_c < 3.0, f"collapsed effective_experts={eff_c:.2f} should be below gate floor"
    assert stage_a_gate_passed(health_h, losses_h) == 1
    assert stage_a_gate_passed(health_c, losses_c) == 0
