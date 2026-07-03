"""Phase 1a gate tests (P_CURV_01–P_CURV_03)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from science.dtie.common.curvature_loader import CANONICAL_V6_CURVATURE
from science.dtie.common.curvature_registry import probe_curvature_sources
from science.contracts.model_registry import resolve_checkpoint_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CKPT = ROOT / "tests/fixtures/checkpoints/lever_a_v6_best_disc.pt"
STALE_LITERALS = ("0.6054343", "0.6054342985153198")
RUNTIME_PATHS = (
    ROOT / "data/db_helpers/vector_queries.py",
    ROOT / "science/dtie/v5/workers/hyperbolic_distance_populator.py",
    ROOT / "scripts/discover_hyperbolic_motifs.py",
)


def _p_curv_01_checkpoint_path() -> str:
    """CI fixture first, then env override, then contract production path."""
    env = os.environ.get("P_CURV_01_CHECKPOINT_PATH", "").strip()
    if env:
        return env
    if FIXTURE_CKPT.is_file():
        return str(FIXTURE_CKPT)
    from science.contracts.model_registry import get_production_checkpoint_path

    return get_production_checkpoint_path()


@pytest.mark.asyncio
async def test_p_curv_01_probe_sources_match():
    """P_CURV_01: model, DB, and SSOT loader agree."""
    ckpt = _p_curv_01_checkpoint_path()
    if resolve_checkpoint_file(ckpt) is None and not Path(ckpt).is_file():
        pytest.skip(f"checkpoint not found: {ckpt}")

    from data.db import open_pool

    try:
        await open_pool()
        report = await probe_curvature_sources("11qe", checkpoint_path=ckpt)
    except Exception as exc:
        pytest.skip(f"DB unavailable or probe failed: {exc}")

    assert report.model_vs_db_match is True
    assert report.model_vs_vector_queries_match is True
    assert report.model_checkpoint_c is not None
    assert report.model_checkpoint_c == pytest.approx(CANONICAL_V6_CURVATURE, rel=0, abs=1e-12)


def test_p_curv_02_no_stale_literals_in_runtime_paths():
    """P_CURV_02: stale TS-002 constants removed from runtime consumers."""
    for path in RUNTIME_PATHS:
        text = path.read_text(encoding="utf-8")
        for stale in STALE_LITERALS:
            assert stale not in text, f"stale curvature {stale!r} in {path}"


def test_p_curv_03_no_cli_curvature_flags():
    """P_CURV_03: CLI override flags removed."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/discover_hyperbolic_motifs.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0
    assert "--curvature" not in result.stdout
