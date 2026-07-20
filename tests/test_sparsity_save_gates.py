"""Sparsity-aware checkpoint save gates (mean residue-H band; no H(f̄) ceiling)."""

from __future__ import annotations

from science.training.checkpoint_score import (
    LOAD_ENTROPY_DIVERSITY_ABORT,
    MEAN_RESIDUE_H_SAVE_MAX,
    MEAN_RESIDUE_H_SAVE_MIN,
    score_checkpoint,
    sparsity_routing_save_ineligibility_reasons,
)


def _healthy_geom() -> dict[str, float]:
    return {
        "proj_frac_mean": 0.0,
        "cone_range_mean": 1.2,
        "probe_r_proj_depth": 0.99,
        "probe_r_depth_tau": 0.80,
        "disc_sigma2_sigma1_mean": 0.85,
        "disc_r_std_mean": 0.15,
        "disc_effective_rank_mean": 1.7,
        "disc_line_thickness_rms_mean": 0.05,
    }


def _base_losses(**extra: float) -> dict[str, float]:
    out = {
        "total": 0.1,
        "routing_entropy": 1.35,  # above legacy 1.21 ceiling
        "routing_entropy_mean_residue": 0.64,
        "expert_starvation_count": 0.0,
        "expert_load_0": 0.22,
        "expert_load_1": 0.27,
        "expert_load_2": 0.28,
        "expert_load_3": 0.23,
    }
    out.update(extra)
    return out


def test_legacy_mode_still_blocks_high_batch_H() -> None:
    bad = score_checkpoint(
        _healthy_geom(),
        _base_losses(),
        phase=12,
    )
    assert bad.eligible is False
    assert any("routing_H" in r for r in bad.reasons)


def test_sparsity_mode_allows_high_batch_H_in_band() -> None:
    good = score_checkpoint(
        _healthy_geom(),
        _base_losses(),
        phase=12,
        routing_entropy_mean_residue_min_save=MEAN_RESIDUE_H_SAVE_MIN,
        routing_entropy_mean_residue_max_save=MEAN_RESIDUE_H_SAVE_MAX,
        routing_entropy_mean_residue_final3=0.64,
        max_eval_routing_fraction_save=0.45,
        inference_routing={"max_routing_fraction": 0.29, "min_routing_fraction": 0.10},
    )
    assert good.eligible is True


def test_sparsity_mode_rejects_mean_H_above_band() -> None:
    bad = score_checkpoint(
        _healthy_geom(),
        _base_losses(routing_entropy_mean_residue=1.05),
        phase=12,
        routing_entropy_mean_residue_min_save=0.50,
        routing_entropy_mean_residue_max_save=0.90,
        inference_routing={"max_routing_fraction": 0.30},
    )
    assert bad.eligible is False
    assert any("mean_residue_H" in r for r in bad.reasons)


def test_sparsity_mode_rejects_max_share() -> None:
    reasons = sparsity_routing_save_ineligibility_reasons(
        _base_losses(),
        inference_routing={"max_routing_fraction": 0.46},
        max_soft_share=0.45,
    )
    assert any("max_share" in r for r in reasons)


def test_sparsity_mode_rejects_final3_out_of_band() -> None:
    bad = score_checkpoint(
        _healthy_geom(),
        _base_losses(routing_entropy_mean_residue=0.64),
        phase=12,
        routing_entropy_mean_residue_min_save=0.50,
        routing_entropy_mean_residue_max_save=0.90,
        routing_entropy_mean_residue_final3=0.95,
        inference_routing={"max_routing_fraction": 0.30},
    )
    assert bad.eligible is False
    assert any("final3" in r for r in bad.reasons)


def test_diversity_abort_constant() -> None:
    assert LOAD_ENTROPY_DIVERSITY_ABORT == 0.80
