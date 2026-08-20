"""Physics-only scoring contract for the corpus Investigation audit."""

from __future__ import annotations

import pytest

from experiments.diagnostics.investigation_audit_corpus12 import (
    SCORING_PATH,
    summarize_structure,
)


def _row(
    key: str,
    *,
    physics: float | None,
    evidential: float,
    depth: float,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "key": key,
        "investigation": evidential,
        "cone_depth": depth,
        "expert": 0,
    }
    if physics is not None:
        row["physics_investigation"] = physics
    return row


def test_audit_ranks_and_partitions_by_physics_investigation() -> None:
    rows = [
        _row("A:1", physics=0.9, evidential=0.0, depth=4.0),
        _row("A:2", physics=0.8, evidential=0.1, depth=3.0),
        _row("A:3", physics=0.2, evidential=0.8, depth=1.0),
        _row("A:4", physics=0.1, evidential=0.9, depth=0.0),
    ]

    summary = summarize_structure(rows, "4obe")

    assert SCORING_PATH == "physics_investigation"
    assert summary["scoring_path"] == "physics_investigation"
    assert summary["top_investigation"][0]["key"] == "A:1"
    assert summary["mean_depth_high_investigation"] > summary[
        "mean_depth_low_investigation"
    ]


def test_audit_refuses_evidential_only_rows() -> None:
    rows = [_row("A:1", physics=None, evidential=0.9, depth=2.0)]

    with pytest.raises(ValueError, match="physics_investigation"):
        summarize_structure(rows, "4obe")
