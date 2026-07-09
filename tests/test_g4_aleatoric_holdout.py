"""Tests for G4 aleatoric shaping holdout masks and reports."""

from __future__ import annotations

import numpy as np
import torch

from science.training.aleatoric_shaping_holdout import (
    G4_DEFAULT_HOLDOUT_MODE,
    aleatoric_shaping_holdout_masks,
    aleatoric_shaping_holdout_masks_torch,
    corpus_protein_holdout_ids,
    holdout_mask_metadata,
)
from science.dtie.v6.loss import v3_aleatoric_shaping_loss
from science.training.evidential_validation import (
    G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN,
    G4_TRANSFER_RATIO_FULL_REL_LIFT_MIN,
    aleatoric_dehydron_stratification_report,
    g4_aleatoric_shaping_holdout_report,
)


def test_holdout_masks_partition_residues() -> None:
    n = 40
    dehyd = np.array([0.0] * 20 + [1.0] * 20)
    ids = [f"r{i}" for i in range(n)]
    train, hold = aleatoric_shaping_holdout_masks(
        dehyd,
        ids,
        pdb_id="1ABC",
        holdout_fraction=0.25,
        holdout_mode="residue_stratified",
    )
    assert train.shape == (n,)
    assert hold.shape == (n,)
    assert not np.any(train & hold)
    assert np.all(train | hold)
    assert hold.sum() >= 4


def test_protein_holdout_is_all_or_nothing() -> None:
    dehyd = np.array([0.0, 1.0, 0.0, 1.0])
    ids = ["a", "b", "c", "d"]
    holdout_proteins = frozenset({"1ABC"})
    train, hold = aleatoric_shaping_holdout_masks(
        dehyd,
        ids,
        pdb_id="1ABC",
        holdout_mode="protein",
        corpus_holdout_proteins=holdout_proteins,
    )
    assert hold.all() and not train.any()
    train2, hold2 = aleatoric_shaping_holdout_masks(
        dehyd,
        ids,
        pdb_id="1XYZ",
        holdout_mode="protein",
        corpus_holdout_proteins=holdout_proteins,
    )
    assert train2.all() and not hold2.any()


def test_corpus_protein_holdout_stable() -> None:
    ids = ["1MBN", "1LYZ", "1BG1", "4OBE", "1TIM", "2SHP"]
    a = corpus_protein_holdout_ids(ids, holdout_fraction=0.20, seed=42)
    b = corpus_protein_holdout_ids(ids, holdout_fraction=0.20, seed=42)
    assert a == b
    assert 0 < len(a) < len(ids)


def test_v3_shaping_zero_on_holdout_mask() -> None:
    ale = torch.tensor([0.5, 1.2, 0.8, 1.5], dtype=torch.float32)
    dehyd = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    train_mask = torch.tensor([True, False, True, False])
    out = v3_aleatoric_shaping_loss(ale, dehyd, train_mask, w_var_penalty=1.0, w_aleatoric_hinge=1.0)
    assert float(out["var_penalty"]) > 0
    hold_only = v3_aleatoric_shaping_loss(
        ale,
        dehyd,
        torch.zeros_like(train_mask),
        w_var_penalty=1.0,
        w_aleatoric_hinge=1.0,
    )
    assert float(hold_only["var_penalty"]) == 0.0


def test_g4_report_transfer_ratio_gate() -> None:
    rows = []
    for i in range(200):
        near_tau = abs(float(i % 20) - 10) <= 1
        is_hold = i % 5 == 0
        ale = (0.20 if near_tau else 0.08) if is_hold else (0.18 if near_tau else 0.06)
        rows.append(
            {
                "rho": float(12.0 + (i % 20)),
                "aleatoric": ale,
                "ale_shaping_holdout": is_hold,
                "holdout_granularity": "protein",
                "structure_id": "1MBN" if is_hold else "1LYZ",
            }
        )
    report = g4_aleatoric_shaping_holdout_report(
        rows,
        holdout_metadata=holdout_mask_metadata(rows),
    )
    assert report["gate"] == "G4"
    assert "relative_lift_transfer_ratio" in report
    assert report["thresholds_frozen"]["transfer_ratio_min"] == G4_HOLDOUT_RELATIVE_LIFT_TRANSFER_MIN
    assert "r_ale_rho_marginal" in report


def test_g4_transfer_ratio_not_evaluable_when_full_corpus_flat() -> None:
    """Sub-threshold full corpus must not show spurious transfer/memorization passes."""
    rows = []
    for i in range(200):
        rows.append(
            {
                "rho": float(12.0 + (i % 20)),
                "aleatoric": 1.76 + 0.001 * (i % 3),
                "tau_flag": 1.0 if i % 4 == 0 else 0.0,
                "ale_shaping_holdout": i % 5 == 0,
            }
        )
    report = g4_aleatoric_shaping_holdout_report(rows)
    assert report["ok"] is False
    assert report["relative_lift_transfer_status"] == "not_evaluable_sub_threshold"
    assert report["relative_lift_transfer_evaluable"] is False
    assert report["mask_memorization_status"] == "vacuous_full_p8_fail"
    assert report["rule_summary"]["transfer_ratio"] == "not_evaluable"
    assert report["rule_summary"]["memorization"] == "vacuous_full_p8_fail"


def test_g4_transfer_ratio_evaluable_when_full_informative() -> None:
    from science.dtie.common.residue_features import TAU

    rows = []
    for i in range(200):
        rho = float(TAU + (i % 20) - 10)
        near_tau = abs(rho - TAU) <= 1
        is_hold = i % 5 == 0
        ale = (0.40 if near_tau else 0.10) if is_hold else (0.38 if near_tau else 0.08)
        rows.append(
            {
                "rho": rho,
                "aleatoric": ale,
                "epistemic": 2.0 + 0.01 * i,
                "tau_flag": 1.0 if i % 4 == 0 else 0.0,
                "ale_shaping_holdout": is_hold,
            }
        )
    report = g4_aleatoric_shaping_holdout_report(rows)
    assert report["p8_full_corpus"]["informative_aleatoric"] is True
    assert report["p8_full_corpus"]["aleatoric_tau_lift_relative"] >= G4_TRANSFER_RATIO_FULL_REL_LIFT_MIN
    assert report["relative_lift_transfer_evaluable"] is True
    assert report["mask_memorization_evaluable"] is True


def test_dehydron_stratification_detects_global_collapse() -> None:
    rows = [
        {"tau_flag": 1.0, "aleatoric": 1.76},
        {"tau_flag": 0.0, "aleatoric": 1.75},
        {"tau_flag": 1.0, "aleatoric": 1.77},
        {"tau_flag": 0.0, "aleatoric": 1.76},
    ]
    report = aleatoric_dehydron_stratification_report(rows)
    assert report["dehydron_also_collapsed"] is True
    assert report["regular_also_collapsed"] is True
    assert "global penalty" in report["interpretation"] or "mask not gating" in report["interpretation"]


def test_population_separation_frozen_criterion() -> None:
    from science.training.evidential_validation import (
        G4_POPULATION_GAP_STD_MULT_MIN,
        aleatoric_population_separation_report,
    )

    rows = []
    for i in range(120):
        is_dehyd = i % 2 == 0
        ale = 0.12 if is_dehyd else 0.11
        rows.append({"tau_flag": 1.0 if is_dehyd else 0.0, "aleatoric": ale, "rho": 13.0})
    fail = aleatoric_population_separation_report(rows)
    assert fail["ok"] is False

    rows_pass = []
    for i in range(120):
        is_dehyd = i % 2 == 0
        ale = 0.35 if is_dehyd else 0.05
        rows_pass.append({"tau_flag": 1.0 if is_dehyd else 0.0, "aleatoric": ale, "rho": 13.0})
    ok = aleatoric_population_separation_report(rows_pass)
    assert ok["gap_over_pooled_std"] >= G4_POPULATION_GAP_STD_MULT_MIN
    assert ok["global_informative_aleatoric"] is True
    assert ok["ok"] is True


def test_p4_v3_phase_config_respects_w_var_penalty() -> None:
    from science.training.config import p4_v3_aleatoric_shaping_phase_config

    cfg = p4_v3_aleatoric_shaping_phase_config(w_var_penalty=0.3)
    assert cfg.coeffs.w_var_penalty == 0.3


def test_p4_g4_shaping_only_isolation_zeros_competing_losses() -> None:
    from science.training.config import p4_g4_shaping_only_isolation_phase_config

    cfg = p4_g4_shaping_only_isolation_phase_config(w_var_penalty=0.3)
    assert cfg.coeffs.v3_aleatoric_shaping_coeff == 1.0
    assert cfg.coeffs.w_var_penalty == 0.3
    assert cfg.coeffs.evidential_coeff == 0.0
    assert cfg.coeffs.epi_ale_decorrelation_coeff == 0.0
    assert cfg.coeffs.epistemic_decoupling_coeff == 0.0
    assert cfg.coeffs.epistemic_anticollapse_coeff == 0.0


def test_p4_g4_ale_only_unshaped_freezes_epi_and_shaping() -> None:
    from science.training.config import p4_g4_ale_only_unshaped_phase_config

    cfg = p4_g4_ale_only_unshaped_phase_config()
    assert cfg.aleatoric_only_train is True
    assert cfg.coeffs.v3_aleatoric_shaping_coeff == 0.0
    assert cfg.coeffs.evidential_coeff == 0.01
    assert cfg.coeffs.epi_ale_decorrelation_coeff == 0.0
    assert cfg.coeffs.epistemic_decoupling_coeff == 0.0


def test_torch_masks_match_numpy() -> None:
    dehyd = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    ids = ["a", "b", "c", "d"]
    train_np, hold_np = aleatoric_shaping_holdout_masks(
        dehyd.numpy(),
        ids,
        pdb_id="X",
        holdout_fraction=0.5,
        holdout_mode="residue_stratified",
    )
    train_t, hold_t = aleatoric_shaping_holdout_masks_torch(
        dehyd,
        ids,
        pdb_id="X",
        holdout_fraction=0.5,
        holdout_mode="residue_stratified",
    )
    assert train_t.cpu().numpy().tolist() == train_np.tolist()
    assert hold_t.cpu().numpy().tolist() == hold_np.tolist()


def test_default_holdout_mode_is_protein() -> None:
    assert G4_DEFAULT_HOLDOUT_MODE == "protein"


def test_holdout_corpus_contrast_z_scores() -> None:
    from science.training.aleatoric_shaping_holdout import holdout_corpus_contrast

    proteins = []
    for i, pdb in enumerate(["1MBN", "1LYZ", "1BG1", "1F88"]):
        n = 50 + i * 10
        dehyd = np.zeros(n)
        dehyd[::4] = 1.0
        proteins.append(
            {
                "pdb_id": pdb,
                "fold_id": f"fold.{i}",
                "gene": pdb,
                "n_residues": n,
                "target_dehydron": torch.tensor(dehyd, dtype=torch.float32).unsqueeze(1),
                "target_rho": torch.linspace(8.0, 20.0, n).unsqueeze(1),
                "data": type("D", (), {"x": torch.stack([torch.linspace(8.0, 20.0, n), torch.zeros(n), torch.full((n,), 0.1)], dim=1)})(),
            }
        )
    contrast = holdout_corpus_contrast(proteins, frozenset({"1F88"}), holdout_seed=42)
    assert contrast["n_holdout_proteins"] == 1
    detail = contrast["holdout_details"][0]
    assert detail["pdb_id"] == "1F88"
    assert "dehydron_fraction_z_vs_corpus" in detail["z_vs_corpus"]
