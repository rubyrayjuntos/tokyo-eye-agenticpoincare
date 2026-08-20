"""Prototype nearest-pair repulsion loss + PROTO_SEP ladder scoring."""

from __future__ import annotations

import pytest
import torch

from experiments.diagnostics.prototype_repulsion_epoch import score_proto_sep_ladder
from science.dtie.v6.loss import gosp_loss_v6
from science.training.config import (
    apply_prototype_nearest_pair_repulsion,
    apply_v66_feeler_phases,
)


def _minimal_output(*, pair_min: torch.Tensor) -> dict:
    n, e = 8, 4
    return {
        "x_hyp": torch.randn(n, 2) * 0.1,
        "x_routed_hyp": torch.randn(n, 2) * 0.1,
        "radial_features": torch.linspace(0.1, 0.9, n).unsqueeze(1),
        "cone_depth": torch.linspace(0.5, 2.0, n).unsqueeze(1),
        "hyp_projections_2d": torch.randn(n, 2) * 0.3,
        "hyp_projections_3d": torch.randn(n, 3) * 0.3,
        "capacity_loss": torch.tensor(0.0),
        "routing_entropy": torch.tensor(1.0),
        "expert_load": torch.ones(e) * 0.25,
        "evidence": {
            "mu": torch.zeros(n),
            "nu": torch.ones(n),
            "alpha": torch.ones(n) * 2.0,
            "beta": torch.ones(n),
        },
        "uncertainty": {"epistemic": torch.linspace(0.2, 1.0, n).unsqueeze(1)},
        "audit_trail": {"curvature_value": -1.0},
        "prototype_pair_min_dist": pair_min,
    }


_ZERO_GEOM = dict(
    evidential_coeff=0.0,
    balance_coeff=0.0,
    cone_coeff=0.0,
    neighborhood_coeff=0.0,
    angular_coeff=0.0,
    domain_sep_2d_coeff=0.0,
    domain_sep_3d_coeff=0.0,
    cone_depth_anticollapse_coeff=0.0,
    shell_corr_coeff=0.0,
    proj_violation_coeff=0.0,
    epistemic_bf_align_coeff=0.0,
    epistemic_sasa_pen_coeff=0.0,
    epistemic_anticollapse_coeff=0.0,
)


def test_prototype_repulsion_hinge_pushes_when_below_margin() -> None:
    pair_min = torch.tensor(0.05, requires_grad=True)
    losses = gosp_loss_v6(
        _minimal_output(pair_min=pair_min),
        target_rho=torch.rand(8),
        ca_coords=torch.randn(8, 3),
        prototype_repulsion_coeff=1.0,
        prototype_repulsion_margin=0.25,
        **_ZERO_GEOM,
    )
    assert float(losses["prototype_repulsion"]) == pytest.approx(0.20, abs=1e-5)
    losses["total"].backward()
    assert pair_min.grad is not None
    assert float(pair_min.grad) < 0  # increasing min dist lowers loss


def test_prototype_repulsion_zero_when_above_margin() -> None:
    pair_min = torch.tensor(0.40, requires_grad=True)
    losses = gosp_loss_v6(
        _minimal_output(pair_min=pair_min),
        target_rho=torch.rand(8),
        ca_coords=torch.randn(8, 3),
        prototype_repulsion_coeff=1.0,
        prototype_repulsion_margin=0.25,
        **_ZERO_GEOM,
    )
    assert float(losses["prototype_repulsion"]) == 0.0


def test_apply_prototype_repulsion_patches_all_phases() -> None:
    phases = apply_v66_feeler_phases(epochs=20)
    patched = apply_prototype_nearest_pair_repulsion(phases, coeff=1.0, margin=0.25)
    assert all(p.coeffs.prototype_repulsion_coeff == 1.0 for p in patched)
    assert all(p.coeffs.prototype_repulsion_margin == 0.25 for p in patched)


def test_score_proto_sep_l1_and_fails() -> None:
    l1_report = {
        "historical_twin_hyp_dist": 0.18,
        "nearest_pair_hyp_dist": 0.18,
        "nearest_pair": [1, 3],
        "mean_soft_on_rival_twin": 0.25,
        "degree_swap_mean_abs_delta_logit_gap": 0.06,
        "other_hyp_dist_min": 0.20,
        "twin_over_other_mean": 0.30,
        "frac_max_p_ge_0_60": 0.0,
        "soft_load_max": 0.30,
        "effective_experts_soft": 3.8,
    }
    scored = score_proto_sep_ladder(l1_report)
    assert scored["l1"] is True
    assert scored["verdict"] == "PROTO_SEP_L1"

    dead = dict(l1_report)
    dead["degree_swap_mean_abs_delta_logit_gap"] = 0.01
    assert score_proto_sep_ladder(dead)["verdict"] == "DISTANCE_WITHOUT_SENSITIVITY"

    collat = dict(l1_report)
    collat["other_hyp_dist_min"] = 0.02
    assert score_proto_sep_ladder(collat)["verdict"] == "COLLATERAL_DAMAGE"


def test_score_stack_ladder_win_and_dist_range_fail() -> None:
    from experiments.diagnostics.prototype_repulsion_epoch import score_stack_ladder

    win = {
        "nearest_pair_hyp_dist": 0.28,
        "degree_swap_mean_abs_delta_logit_gap": 0.20,
        "mean_soft_on_rival_twin": 0.18,
        "other_hyp_dist_min": 0.25,
        "frac_max_p_ge_0_60": 0.08,
        "effective_softplus": 6.5,
        "mean_dist_range": 0.30,
        "mean_dist_cv": 0.05,
        "best_tau_mean_contrast": 0.55,
    }
    assert score_stack_ladder(win)["verdict"] == "STACK_WIN_L1"

    kill = dict(win)
    kill["mean_dist_range"] = 0.10
    kill["frac_max_p_ge_0_60"] = 0.0
    assert score_stack_ladder(kill)["verdict"] == "DIST_RANGE_NULLIFIES"
    assert "DIST_RANGE_KILLS_SCALE" in score_stack_ladder(kill)["fails"]


def test_routing_balance_soft_can_hide_hard_concentration() -> None:
    """Corpus soft near-uniform while committed hard mass funnels to one expert."""
    import numpy as np

    from experiments.diagnostics.prototype_repulsion_epoch import (
        routing_balance_from_weight_rows,
    )

    # Protein A: all residues hard-commit to expert 2 (peaky).
    peak = np.array([[0.05, 0.05, 0.85, 0.05]] * 100, dtype=float)
    # Protein B: soft-flat, uncommitted (max-p ~0.28).
    flat = np.array([[0.28, 0.24, 0.24, 0.24]] * 100, dtype=float)
    # Soft means average toward uniform; committed set is 100% expert 2.
    out = routing_balance_from_weight_rows(
        [peak, flat], pdb_ids=["PEAK", "FLAT"], commit_thr=0.60
    )
    assert out["soft_load_max"] < 0.56
    assert out["n_committed"] == 100
    assert out["committed_hard_share_max"] == pytest.approx(1.0)
    assert out["committed_hard_share"][2] == pytest.approx(1.0)
    assert out["hard_share_max"] == pytest.approx(0.5)  # 100/200
    assert out["per_structure_soft_max"] == pytest.approx(0.85)
    assert out["per_structure_committed_hard_max"] == pytest.approx(1.0)
    assert out["per_structure"][0]["pdb_id"] == "PEAK"
    assert out["per_structure"][0]["committed_hard_share_max"] == pytest.approx(1.0)


def test_routing_balance_uniform_hard_and_committed() -> None:
    import numpy as np

    from experiments.diagnostics.prototype_repulsion_epoch import (
        routing_balance_from_weight_rows,
    )

    rows = []
    for e in range(4):
        w = np.full((50, 4), 0.05, dtype=float)
        w[:, e] = 0.85
        rows.append(w)
    out = routing_balance_from_weight_rows(rows, pdb_ids=[f"E{e}" for e in range(4)])
    assert out["hard_share_max"] == pytest.approx(0.25)
    assert out["committed_hard_share_max"] == pytest.approx(0.25)
    assert out["effective_experts_hard"] == pytest.approx(4.0, abs=1e-6)
    assert out["per_structure_soft_max"] == pytest.approx(0.85)
    assert out["per_structure_soft_min"] == pytest.approx(0.05)
    assert out["n_proteins_with_committed"] == 4
    assert out["committed_protein_share_max"] == pytest.approx(0.25)


def test_score_committed_distribution_floors() -> None:
    from experiments.diagnostics.prototype_repulsion_epoch import (
        score_committed_distribution,
    )

    pass_report = {
        "n_committed": 500,
        "n_proteins_with_committed": 12,
        "n_proteins_total": 12,
        "committed_protein_share_max": 0.17,
        "committed_hard_share_max": 0.40,
        "per_structure_committed_hard_max": 0.50,
    }
    scored = score_committed_distribution(pass_report)
    assert scored["passes"] is True
    assert scored["verdict"] == "COMMITTED_DISTRIBUTION_PASS"

    concentrated = dict(pass_report)
    concentrated["committed_protein_share_max"] = 0.50
    concentrated["n_proteins_with_committed"] = 3
    assert (
        score_committed_distribution(concentrated)["verdict"]
        == "COMMITTED_PROTEIN_CONCENTRATED"
    )

    monopole = dict(pass_report)
    monopole["committed_hard_share_max"] = 0.60
    assert (
        score_committed_distribution(monopole)["verdict"]
        == "COMMITTED_HARD_CONCENTRATED"
    )

    local_mono = dict(pass_report)
    local_mono["per_structure_committed_hard_max"] = 0.90
    assert (
        score_committed_distribution(local_mono)["verdict"]
        == "COMMITTED_HARD_CONCENTRATED"
    )
    assert "PER_STRUCTURE_COMMITTED_HARD_MONOPOLE" in score_committed_distribution(
        local_mono
    )["fails"]


def test_committed_majority_partition_and_purity_floors() -> None:
    import numpy as np

    from experiments.diagnostics.prototype_repulsion_epoch import (
        committed_majority_partition,
        score_dehydron_partition_purity,
        score_majority_conditional_ladder,
    )

    # 80 committed on e0 (dehydron=0), 20 on e1 (dehydron=1)
    w = np.zeros((100, 4), dtype=float)
    w[:80, 0] = 0.70
    w[:80, 1:] = 0.10
    w[80:, 1] = 0.70
    w[80:, 0] = 0.10
    w[80:, 2:] = 0.10
    dh = np.zeros(100, dtype=float)
    dh[80:] = 1.0
    part = committed_majority_partition(w, dh)
    assert part["majority_expert"] == 0
    assert part["majority_share"] == pytest.approx(0.80)
    assert part["majority_dehydron_rate"] == pytest.approx(0.0)
    assert part["minority_dehydron_rate"] == pytest.approx(1.0)

    pure_rows = [
        {
            "pdb_id": "1F88",
            "n_committed": 100,
            "n_minority": 20,
            "majority_dehydron_rate": 0.0,
            "minority_dehydron_rate": 1.0,
            "corpus_dehydron_frac": 0.30,  # core-majority structure
        }
    ]
    pure = score_dehydron_partition_purity(pure_rows)
    assert pure["passes"] is True
    assert pure["verdict"] == "DEHYDRON_PARTITION_PURE"
    assert pure["floor_mode"] == "relative"

    # Dehydron-majority corpus + dehydron-majority committed = PASS under relative.
    dh_maj_rows = [
        {
            "pdb_id": "1LYZ",
            "n_committed": 100,
            "n_minority": 20,
            "majority_dehydron_rate": 1.0,
            "minority_dehydron_rate": 0.0,
            "corpus_dehydron_frac": 0.68,
        }
    ]
    assert score_dehydron_partition_purity(dh_maj_rows)["passes"] is True
    # Same committed rates fail under legacy fixed floor.
    assert (
        score_dehydron_partition_purity(
            dh_maj_rows, floor_mode="fixed_core_majority"
        )["passes"]
        is False
    )

    blurred_rows = [
        {
            "pdb_id": "1F88",
            "n_committed": 100,
            "n_minority": 20,
            "majority_dehydron_rate": 0.08,
            "minority_dehydron_rate": 0.85,
            "corpus_dehydron_frac": 0.30,
        }
    ]
    blurred = score_dehydron_partition_purity(blurred_rows)
    assert blurred["passes"] is False
    assert blurred["verdict"] == "DEHYDRON_PARTITION_BLURRED"
    assert "DEHYDRON_PARTITION_BLURRED" in blurred["fails"]

    win = score_majority_conditional_ladder(
        {
            "frac_max_p_ge_0_60": 0.50,
            "per_structure_committed_hard_max": 0.50,
            "best_tau_mean_contrast": 0.90,
            "committed_distribution": {"passes": True},
            "dehydron_partition_purity": {"passes": True},
        }
    )
    assert win["verdict"] == "MAJORITY_SPLIT_WIN"

    no_move = score_majority_conditional_ladder(
        {
            "frac_max_p_ge_0_60": 0.50,
            "per_structure_committed_hard_max": 0.74,
            "best_tau_mean_contrast": 0.90,
            "committed_distribution": {"passes": False},
            "dehydron_partition_purity": {"passes": True},
        }
    )
    assert no_move["verdict"] == "MAJORITY_SPLIT_COEFF_INCONCLUSIVE"

    scramble = score_majority_conditional_ladder(
        {
            "frac_max_p_ge_0_60": 0.50,
            "per_structure_committed_hard_max": 0.50,
            "best_tau_mean_contrast": 0.90,
            "committed_distribution": {"passes": True},
            "dehydron_partition_purity": {"passes": False},
        }
    )
    assert scramble["verdict"] == "AXIS_SCRAMBLED_BY_DIVERSITY"


def test_majority_committed_share_hinge_matches_hard_share() -> None:
    from science.dtie.v6.loss import majority_committed_share_hinge

    n, e = 40, 4
    logits = torch.zeros(n, e, requires_grad=True)
    # 28 residues strongly prefer e0, 12 prefer e1 — committed peaky (share 0.70)
    with torch.no_grad():
        logits[:28, 0] = 5.0
        logits[28:, 1] = 5.0
    w = torch.softmax(logits, dim=-1)
    raw = majority_committed_share_hinge(w, tau=0.56, commit_thr=0.60, min_committed=20)
    assert float(raw) == pytest.approx(0.0196, abs=1e-4)
    out = _minimal_output(pair_min=torch.tensor(0.40))
    # resize evidence/features to n
    out["expert_weights"] = w
    out["x_hyp"] = torch.randn(n, 2) * 0.1
    out["x_routed_hyp"] = torch.randn(n, 2) * 0.1
    out["radial_features"] = torch.linspace(0.1, 0.9, n).unsqueeze(1)
    out["cone_depth"] = torch.linspace(0.5, 2.0, n).unsqueeze(1)
    out["hyp_projections_2d"] = torch.randn(n, 2) * 0.3
    out["hyp_projections_3d"] = torch.randn(n, 3) * 0.3
    out["evidence"] = {
        "mu": torch.zeros(n),
        "nu": torch.ones(n),
        "alpha": torch.ones(n) * 2.0,
        "beta": torch.ones(n),
    }
    out["uncertainty"] = {"epistemic": torch.linspace(0.2, 1.0, n).unsqueeze(1)}
    losses = gosp_loss_v6(
        out,
        target_rho=torch.rand(n),
        ca_coords=torch.randn(n, 3),
        majority_committed_share_coeff=0.5,
        majority_committed_share_tau=0.56,
        **_ZERO_GEOM,
    )
    assert float(losses["majority_committed_share"]) == pytest.approx(0.5 * 0.0196, abs=1e-4)
    losses["total"].backward()
    assert logits.grad is not None
    # dL/dw_e*>0 on maj → through softmax dL/dz_e*>0; descent lowers majority logits
    assert float(logits.grad[:28, 0].mean()) > 0


def test_majority_committed_share_hinge_zero_when_balanced() -> None:
    from science.dtie.v6.loss import majority_committed_share_hinge

    n, e = 80, 4
    w = torch.zeros(n, e)
    for i in range(e):
        w[i * 20 : (i + 1) * 20, i] = 0.85
        for j in range(e):
            if j != i:
                w[i * 20 : (i + 1) * 20, j] = 0.05
    raw = majority_committed_share_hinge(w, tau=0.56, min_committed=20)
    assert float(raw) == 0.0


def test_core_majority_hinge_zero_grad_on_dehydron_ones() -> None:
    from science.dtie.v6.loss import core_majority_committed_share_hinge

    n, e = 60, 4
    logits = torch.zeros(n, e, requires_grad=True)
    dh = torch.zeros(n)
    # 40 core on e0, 20 dehydron on e1
    with torch.no_grad():
        logits[:40, 0] = 5.0
        logits[40:, 1] = 5.0
    dh[40:] = 1.0
    w = torch.softmax(logits, dim=-1)
    raw = core_majority_committed_share_hinge(w, dh, tau=0.56, min_committed=20)
    # core share 40/40 = 1.0 among C0 → (1-0.56)^2
    assert float(raw) == pytest.approx(0.1936, abs=1e-4)
    raw.backward()
    # dh=1 residues: no direct grad path through this term's maj mask
    assert float(logits.grad[40:, :].abs().sum()) == pytest.approx(0.0, abs=1e-8)
    assert float(logits.grad[:40, 0].mean()) > 0


def test_core_majority_hinge_balanced_core_noop() -> None:
    from science.dtie.v6.loss import core_majority_committed_share_hinge

    n, e = 80, 4
    w = torch.zeros(n, e)
    dh = torch.zeros(n)
    # 40 core split evenly e0/e1; 40 dehydron all on e2 (ignored by core hinge)
    for i in range(2):
        w[i * 20 : (i + 1) * 20, i] = 0.85
        for j in range(e):
            if j != i:
                w[i * 20 : (i + 1) * 20, j] = 0.05
    w[40:, 2] = 0.85
    for j in (0, 1, 3):
        w[40:, j] = 0.05
    dh[40:] = 1.0
    raw = core_majority_committed_share_hinge(w, dh, tau=0.56, min_committed=20)
    assert float(raw) == 0.0


def test_score_core_majority_conditional_ladder() -> None:
    from experiments.diagnostics.prototype_repulsion_epoch import (
        score_core_majority_conditional_ladder,
    )

    win = score_core_majority_conditional_ladder(
        {
            "frac_max_p_ge_0_60": 0.50,
            "per_structure_committed_hard_max": 0.50,
            "best_tau_mean_contrast": 0.90,
            "committed_distribution": {"passes": True},
            "dehydron_partition_purity": {"passes": True},
        }
    )
    assert win["verdict"] == "CORE_MAJORITY_SPLIT_WIN"
    assert "eligible_min_committed" in win["floors"]
    assert win["floors"]["eligible_min_committed"] == 20
    assert win["floors"]["eligible_min_minority"] == 5
