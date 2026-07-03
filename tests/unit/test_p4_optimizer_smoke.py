"""Phase 4 stage 1 optimizer smoke gate (integration)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"


@pytest.mark.integration
def test_p4_optimizer_smoke_gate() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip(f"warm-start checkpoint missing: {CHECKPOINT}")

    from experiments.training.v6.p4_optimizer_smoke import run_smoke

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_smoke(
        checkpoint=CHECKPOINT,
        pdb_dir=Path("/tmp/dtie_pdb_cache"),
        manifest=REPO / "manifests/v6_corpus_disc_target.json",
        device=device,
    )
    assert results["min_param_delta"] > 1e-6
    assert results["loss_epoch1"] < results["loss_epoch0"]
    assert abs(results["r_epi_bf_resid_epoch1"] - 0.040) > 1e-4
    assert results["delta_r_epi_bf_resid"] > -0.010
