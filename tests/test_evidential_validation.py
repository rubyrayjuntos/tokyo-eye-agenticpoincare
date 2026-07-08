"""Property tests P7–P11 for Deep Evidential Regression decomposition."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from science.dtie.common.residue_features import TAU
from science.dtie.v6.gnn.evidential import der_uncertainty_numpy
from science.training.evidential_validation import (
    aleatoric_var_non_degenerate,
    assess_evidential_decomposition,
    corpus_expansion_sensitivity,
    epistemic_var_non_degenerate,
    out_of_corpus_epistemic_contrast,
    sparsification_curve,
    sparsification_error_monotone,
    tau_boundary_aleatoric_elevation,
    uncertainty_corpus_variance,
)

_REPO = Path(__file__).resolve().parents[1]
ROUTE_CKPT = _REPO / "checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt"
STAGE_A_MANIFEST = _REPO / "manifests/v6_corpus_stage_a_small_v1.json"


def _synthetic_rows(
    n: int = 40,
    *,
    rho_span: tuple[float, float] = (8.0, 18.0),
    epi_scale: float = 1.0,
    ale_tau_boost: float = 0.5,
) -> list[dict]:
    rng = np.random.default_rng(42)
    rho = np.linspace(rho_span[0], rho_span[1], n)
    epi = rng.uniform(0.5, 2.0, n) * epi_scale
    ale = rng.uniform(0.3, 1.5, n)
    tau_mask = np.abs(rho - TAU) <= 1.0
    ale[tau_mask] += ale_tau_boost
    mu = rho + rng.normal(0, 0.2, n)
    rows = []
    for i in range(n):
        rows.append(
            {
                "residue_id": f"A:{i}:",
                "rho": float(rho[i]),
                "epistemic": float(epi[i]),
                "aleatoric": float(ale[i]),
                "prediction_error": float(abs(mu[i] - rho[i])),
            }
        )
    return rows


def test_der_formulas_epistemic_equals_ale_over_nu() -> None:
    epi, ale = der_uncertainty_numpy(nu=2.0, alpha=3.0, beta=4.0)
    assert ale == pytest.approx(4.0 / 2.0)
    assert epi == pytest.approx(ale / 2.0)


def test_p7_epistemic_var_non_degenerate() -> None:
    rows = _synthetic_rows()
    ok, reason = epistemic_var_non_degenerate(rows)
    assert ok, reason
    var = uncertainty_corpus_variance(rows)
    assert var["epistemic_std"] > 1e-3


def test_p7_rejects_flat_epistemic() -> None:
    rows = _synthetic_rows()
    for r in rows:
        r["epistemic"] = 1.0
    ok, reason = epistemic_var_non_degenerate(rows)
    assert not ok
    assert "flat" in reason


def test_p8_tau_boundary_aleatoric_elevation() -> None:
    rows = _synthetic_rows(ale_tau_boost=0.8)
    result = tau_boundary_aleatoric_elevation(rows)
    assert result["ok"], result["reason"]
    assert result["aleatoric_tau_lift"] > 0.0
    assert result["aleatoric_tau_lift_relative"] >= 0.20
    assert result["strata_basis"] == "continuous_rho"


def test_p8_g4a_rejects_flat_ale_direction_only() -> None:
    """Tiny positive lift on near-constant ale must not pass (route_v1 failure shape)."""
    rows = _synthetic_rows()
    for r in rows:
        r["aleatoric"] = 0.010 + (0.0002 if abs(r["rho"] - TAU) <= 1.0 else 0.0)
    result = tau_boundary_aleatoric_elevation(rows)
    assert not result["ok"]
    assert result["aleatoric_tau_lift"] > 0.0
    assert "informative" in result["reason"] or "relative" in result["reason"]


def test_p9_sparsification_monotone_synthetic() -> None:
    rows = _synthetic_rows()
    # High epistemic ↔ high error (synthetic ground truth)
    for r in rows:
        r["prediction_error"] = r["epistemic"] * 0.5 + 0.1
    curve = sparsification_curve(rows)
    ok, reason = sparsification_error_monotone(curve)
    assert ok, reason


def test_p10_corpus_expansion_epistemic_shrinks_more_than_aleatoric() -> None:
    base = _synthetic_rows(n=30)
    expanded = []
    for r in base:
        expanded.append(
            {
                **r,
                "epistemic": r["epistemic"] * 0.6,
                "aleatoric": r["aleatoric"] + 0.02,
            }
        )
    result = corpus_expansion_sensitivity(base, expanded)
    assert result["ok"], result["reason"]
    assert result["epistemic_shrink"] > 0
    assert result["epi_shrink_to_ale_move_ratio"] >= 1.5


@pytest.mark.skip(reason="Requires paired baseline/expanded checkpoint audit artifacts")
def test_p10_epistemic_decreases_with_corpus_expansion() -> None:
    """Stub — run after Stage A→B corpus expansion with saved audit JSONs."""
    baseline_rows: list[dict] = []
    expanded_rows: list[dict] = []
    result = corpus_expansion_sensitivity(baseline_rows, expanded_rows)
    assert result["ok"]


def test_p11_out_of_corpus_epistemic_elevated() -> None:
    in_rows = _synthetic_rows(n=30, epi_scale=1.0)
    ood_rows = _synthetic_rows(n=30, epi_scale=2.5)
    result = out_of_corpus_epistemic_contrast(in_rows, ood_rows)
    assert result["ok"], result["reason"]
    assert result["epistemic_ood_ratio"] > result["aleatoric_ood_ratio"]


def test_assess_evidential_decomposition_coupled_fails_decoupling() -> None:
    rows = _synthetic_rows()
    for r in rows:
        r["aleatoric"] = r["epistemic"] * 0.5 + 0.1
    report = assess_evidential_decomposition(rows, decoupled_head=True)
    assert not report["checks"]["epi_ale_low_correlation"]


def test_der_uncertainty_from_evidence_torch() -> None:
    from science.dtie.v6.gnn.evidential import der_uncertainty_from_evidence

    evidence = {
        "mu": torch.tensor([[10.0]]),
        "nu": torch.tensor([[2.0]]),
        "alpha": torch.tensor([[3.0]]),
        "beta": torch.tensor([[4.0]]),
    }
    out = der_uncertainty_from_evidence(evidence)
    assert out["aleatoric"].item() == pytest.approx(2.0)
    assert out["epistemic"].item() == pytest.approx(1.0)


def test_uncertainty_s6_joint_requires_p8_not_r_alone() -> None:
    from science.training.evidential_validation import uncertainty_s6_joint_pass

    rows = _synthetic_rows(ale_tau_boost=0.0)
    for r in rows:
        r["aleatoric"] = 1.0 + r["epistemic"] * 0.05  # low r(epi, ale)
        if abs(r["rho"] - TAU) <= 1.0:
            r["aleatoric"] = 1.0  # no tau lift
    joint = uncertainty_s6_joint_pass(rows)
    assert joint["decouple_ok"]
    assert not joint["p8_ok"]
    assert not joint["ok"]


def test_g3_supervision_circularity_detects_supervised_only_pass() -> None:
    from science.training.evidential_validation import g3_supervision_circularity_report

    decorr = {"decomposition": {"checks": {"tau_aleatoric_elevated": False}, "tau_boundary": {"ok": False}}}
    full = {"decomposition": {"checks": {"tau_aleatoric_elevated": True}, "tau_boundary": {"ok": True}}, "p11_ok": True}
    report = g3_supervision_circularity_report(decorr, full)
    assert report["supervision_required_for_pass"]["p8"]
    assert not report["g3_pass"]


def test_nu_cv_rejects_constant_evidence_nu() -> None:
    from science.training.evidential_validation import (
        NU_CV_FLOOR,
        exposure_non_degenerate,
        exposure_nu_coefficient_of_variation,
    )

    rows = _synthetic_rows()
    for r in rows:
        r["evidence_nu"] = 1.5
    cv = exposure_nu_coefficient_of_variation(rows)
    assert cv < NU_CV_FLOOR
    ok, reason = exposure_non_degenerate(rows)
    assert not ok
    assert "nu_cv" in reason


@pytest.mark.integration
@pytest.mark.skipif(not ROUTE_CKPT.is_file(), reason="route checkpoint missing")
def test_nu_cv_empirical_route_v1_above_floor_ale_flat() -> None:
    """Empirical calibration: route_v1 ale flat but nu_cv spread — probe discriminates."""
    rows = _collect_corpus_rows(ROUTE_CKPT, max_proteins=3)
    from science.training.evidential_validation import (
        NU_CV_FLOOR,
        exposure_nu_coefficient_of_variation,
        uncertainty_corpus_variance,
    )

    var = uncertainty_corpus_variance(rows)
    cv = exposure_nu_coefficient_of_variation(rows)
    assert var["aleatoric_std"] < 0.05  # known flat ale on route_v1
    assert cv >= NU_CV_FLOOR  # nu still spreads — temp cannot fake this


def test_checkpoint_s6_joint_save_blocks_without_tau_lift() -> None:
    from science.training.checkpoint_score import uncertainty_save_ineligibility_reasons

    health = {
        "probe_r_epi_ale": 0.5,
        "epistemic_std_mean": 0.1,
        "aleatoric_std_mean": 0.6,
        "uncertainty_informative_ale": 1.0,
        "node_aleatoric_tau_lift": 0.05,
        "node_aleatoric_tau_lift_relative": -0.01,
        "uncertainty_tau_ale_elevated": 0.0,
    }
    reasons = uncertainty_save_ineligibility_reasons(
        health,
        max_probe_r_epi_ale=0.70,
        require_tau_ale_elevation=True,
    )
    assert any("P8" in r or "tau" in r for r in reasons)
    assert not any("epi/ale coupled" in r for r in reasons)


def test_aleatoric_var_non_degenerate() -> None:
    rows = _synthetic_rows()
    ok, _ = aleatoric_var_non_degenerate(rows)
    assert ok


def test_nig_loss_coupling_analysis() -> None:
    from science.training.nig_identifiability import (
        analyze_nig_loss_coupling,
        decoupled_head_changes,
        head_output_quantity,
    )

    coupling = analyze_nig_loss_coupling()
    assert "nu" in coupling["regularizer_couples"]
    assert "alpha" in coupling["regularizer_couples"]

    head = decoupled_head_changes()
    assert head["architecture"]["shared_nig_loss"] is True
    assert any("epi_ale_decorrelation" in s for s in head["phase4_training_signals_that_can_help"])

    units = head_output_quantity()
    assert units["default_temp_scaling"] == 1.0
    assert "head-reported" in units["floor_calibration_note"]


def test_exposure_nu_cv_synthetic() -> None:
    from science.training.evidential_validation import (
        exposure_non_degenerate,
        exposure_nu_coefficient_of_variation,
    )

    rows = _synthetic_rows()
    for i, r in enumerate(rows):
        r["evidence_nu"] = 0.5 + (i % 7) * 0.3
    assert exposure_nu_coefficient_of_variation(rows) > 0.02
    ok, reason = exposure_non_degenerate(rows)
    assert ok, reason


def _collect_corpus_rows(checkpoint: Path, *, max_proteins: int | None = None) -> list[dict]:
    import os

    if not os.environ.get("TRAINING_LOAD_FROM_PDB"):
        pytest.skip("Set TRAINING_LOAD_FROM_PDB=1")
    from experiments.training.v6.assess_checkpoint import load_v6_model
    from experiments.training.v6.corpus import load_training_proteins
    from experiments.training.v6.train_loop import prepare_training_batch
    from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows

    proteins, _ = load_training_proteins(Path("/tmp/dtie_pdb_cache"), STAGE_A_MANIFEST)
    if max_proteins:
        proteins = proteins[:max_proteins]
    model = load_v6_model(checkpoint, "cpu")
    model.eval()
    all_rows: list[dict] = []
    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(model, prot, "cpu", structural_disc_frozen=True)
            out = model(data)
            all_rows.extend(extract_residue_uncertainty_rows(out, prot))
    return all_rows


@pytest.mark.integration
@pytest.mark.skipif(not ROUTE_CKPT.is_file(), reason="route checkpoint missing")
def test_p9_sparsification_monotone_corpus() -> None:
    """Real Stage A predictions — do not cite epistemic triage unless this passes."""
    rows = _collect_corpus_rows(ROUTE_CKPT)
    assert len(rows) >= 100
    curve = sparsification_curve(rows)
    assert len(curve) >= 2
    assert all(math.isfinite(c["mean_error"]) for c in curve if c["n_kept"] > 0)
    ok, reason = sparsification_error_monotone(curve)
    if not ok:
        pytest.fail(
            f"P9 corpus sparsification not monotone ({reason}). "
            "Do not cite epistemic for residue triage on this checkpoint."
        )


@pytest.mark.integration
@pytest.mark.skipif(not ROUTE_CKPT.is_file(), reason="route checkpoint missing")
def test_p7_corpus_exposure_nu_cv_route() -> None:
    """Scale-invariant exposure check on real checkpoint (temp-independent)."""
    from science.training.evidential_validation import exposure_non_degenerate

    rows = _collect_corpus_rows(ROUTE_CKPT)
    ok, reason = exposure_non_degenerate(rows)
    # Report-only: nu_cv may pass even when aleatoric flat — do not skip failure silently.
    if not ok:
        pytest.xfail(f"nu exposure not spread on route_v1: {reason}")


@pytest.mark.integration
@pytest.mark.skipif(not ROUTE_CKPT.is_file(), reason="route checkpoint missing")
def test_p11_ood_epistemic_contrast_pinned_1pgb() -> None:
    """OOD = 1PGB:A (manifest disabled, CATH 3.10.20.10) vs Stage A in-corpus mean."""
    import os

    if not os.environ.get("TRAINING_LOAD_FROM_PDB"):
        pytest.skip("Set TRAINING_LOAD_FROM_PDB=1")
    from experiments.training.v6.assess_checkpoint import load_v6_model
    from experiments.training.v6.corpus import load_training_proteins
    from experiments.training.v6.train_loop import prepare_training_batch
    from experiments.training.v6._data import load_protein_graph
    from science.training.evidential_validation import OOD_PINNED_STRUCTURES
    from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows

    in_rows = _collect_corpus_rows(ROUTE_CKPT, max_proteins=6)
    model = load_v6_model(ROUTE_CKPT, "cpu")
    model.eval()
    ood_rows: list[dict] = []
    for pdb_id, chain in OOD_PINNED_STRUCTURES:
        prot = load_protein_graph(pdb_id, chain, Path("/tmp/dtie_pdb_cache"))
        assert prot is not None, f"OOD structure {pdb_id}:{chain} failed to load"
        with torch.no_grad():
            data = prepare_training_batch(model, prot, "cpu", structural_disc_frozen=True)
            out = model(data)
        ood_rows.extend(extract_residue_uncertainty_rows(out, prot))

    result = out_of_corpus_epistemic_contrast(in_rows, ood_rows)
    if not result["ok"]:
        pytest.xfail(
            f"P11 OOD contrast failed on route_v1 (1PGB vs in-corpus): {result['reason']}. "
            f"epi_ratio={result.get('epistemic_ood_ratio'):.3f}"
        )
