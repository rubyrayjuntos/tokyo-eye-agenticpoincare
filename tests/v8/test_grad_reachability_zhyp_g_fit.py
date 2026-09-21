"""z_hyp G_fit decision 5 — gradient reachability fixture (toy batch, no PDB).

Must PASS on CPU and on CUDA before the 400-step card may seal / run.
See data/gates/tokyo_eye_equ_wrap1_zhyp_g_fit_prereg.json decisions_locked.5_*.
"""

from __future__ import annotations

import pytest
import torch

from science.tokyo_eye.v8.grad_reachability import (
    assert_dehydron_alone_spine_unreachable,
    assert_sdrp_live_spine_reachable,
    backward_bucket_table,
    bucket_status,
    forward_losses,
    make_toy_spine,
)


def _run_sdrp_live_card_total(device: torch.device) -> dict:
    model, batch = make_toy_spine(device=device, seed=0)
    losses = forward_losses(
        model,
        batch,
        sdrp_coeff=0.1,
        dehydron_coeff=0.0,
        margin_coeff=0.0,
    )
    stats = backward_bucket_table(model, losses["CARD_TOTAL"])
    assert_sdrp_live_spine_reachable(stats, context=f"device={device}")
    return stats


def test_sdrp_live_spine_reachable_cpu() -> None:
    stats = _run_sdrp_live_card_total(torch.device("cpu"))
    # euc_skip is in DualSpaceSDRPHead via h_euc — may be NZ; not a spine fail.
    assert bucket_status(stats["euc_skip"]) in ("NZ", "ZERO")


def test_sdrp_ce_alone_matches_card_total_graph() -> None:
    """SDRP CE alone must light the same spine buckets as CARD_TOTAL."""
    model, batch = make_toy_spine(device=torch.device("cpu"), seed=1)
    losses = forward_losses(
        model, batch, sdrp_coeff=0.1, dehydron_coeff=0.0, margin_coeff=0.0
    )
    stats = backward_bucket_table(model, losses["SDRP_CE"])
    assert_sdrp_live_spine_reachable(stats, context="SDRP_CE_alone")


def test_dehydron_bce_alone_does_not_reach_spine() -> None:
    """Negative control — euc_skip OPEN_WIRING (grad_attribution_check)."""
    model, batch = make_toy_spine(device=torch.device("cpu"), seed=2)
    losses = forward_losses(
        model, batch, sdrp_coeff=0.1, dehydron_coeff=0.0, margin_coeff=0.0
    )
    stats = backward_bucket_table(model, losses["BCE_dehydron"])
    assert_dehydron_alone_spine_unreachable(stats, context="BCE_alone")
    assert bucket_status(stats["mechanism_head"]) == "NZ"
    assert bucket_status(stats["euc_skip"]) == "NZ"


def test_zero_coeff_via_multiply_would_pollute_graph() -> None:
    """Document why the card omits disabled terms instead of ``0.0 * loss``.

    If dehydron is multiplied by 0.0 but left in the expression, mechanism_head
    enters the graph with ZERO grads — the failure mode the card forbids.
    """
    model, batch = make_toy_spine(device=torch.device("cpu"), seed=3)
    losses = forward_losses(
        model, batch, sdrp_coeff=0.1, dehydron_coeff=0.0, margin_coeff=0.0
    )
    # Pollute: include 0.0 * BCE explicitly.
    polluted = float(0.1) * losses["SDRP_CE"] + float(0.0) * losses["BCE_dehydron"]
    stats = backward_bucket_table(model, polluted)
    # Spine still NZ, but mechanism_head is no longer cleanly NONE.
    assert bucket_status(stats["attn_layers"]) == "NZ"
    assert bucket_status(stats["mechanism_head"]) != "NONE"
    assert stats["mechanism_head"]["zero"] + stats["mechanism_head"]["nz"] > 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for seal blocker #2"
)
def test_sdrp_live_spine_reachable_cuda() -> None:
    stats = _run_sdrp_live_card_total(torch.device("cuda"))
    assert stats["sdrp_head"]["grad_l2"] > 0.0
