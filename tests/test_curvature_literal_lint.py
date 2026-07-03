"""Curvature literal lint gate — no TS-002 hardcoded fallbacks in production code."""

from __future__ import annotations

import pytest

from science.dtie.common.curvature_literal_lint import scan_repository, scan_text


def test_repository_has_no_curvature_literal_violations() -> None:
    violations = scan_repository()
    assert violations == [], "\n".join(
        f"{v.path}:{v.line_no} [{v.rule}] {v.line}" for v in violations
    )


@pytest.mark.parametrize(
    "line",
    [
        "curvature = 1.0",
        "curvature_c: float = 1.0",
        'snapshot.get("curvature", 1.0)',
        "curvature or 1.0",
        'rows[0].get("curvature") or 1.0',
        "curvature ?? 1.0",
    ],
)
def test_scanner_detects_banned_patterns(line: str) -> None:
    hits = scan_text("synthetic.py", line)
    assert hits, f"expected violation for: {line!r}"


def test_scanner_allows_poincare_distance_math() -> None:
    line = "den_u = 1.0 - curvature_c * norm_u_sq"
    assert scan_text("data/db_helpers/vector_queries.py", line) == []


def test_scanner_allows_explicit_allow_comment() -> None:
    line = "curvature = 1.0  # curvature-lint: allow"
    assert scan_text("synthetic.py", line) == []
