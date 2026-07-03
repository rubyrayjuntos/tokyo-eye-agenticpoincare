"""Learned-curvature read helpers — never substitute TS-002 numeric defaults."""

from __future__ import annotations

import math
from typing import Any


class MissingLearnedCurvatureError(ValueError):
    """Raised when governed data lacks a positive learned curvature."""


def require_learned_curvature(
    value: float | None,
    *,
    context: str = "",
) -> float:
    """Return a positive finite curvature or raise."""
    if value is None:
        suffix = f" ({context})" if context else ""
        raise MissingLearnedCurvatureError(f"missing learned curvature{suffix}")
    curvature = float(value)
    if not math.isfinite(curvature) or curvature <= 0:
        suffix = f" ({context})" if context else ""
        raise MissingLearnedCurvatureError(f"invalid learned curvature: {value!r}{suffix}")
    return curvature


def learned_curvature_from_row(
    row: dict[str, Any],
    *,
    key: str = "curvature",
    context: str = "",
) -> float:
    return require_learned_curvature(row.get(key), context=context or f"row[{key!r}]")


def learned_curvature_from_rows(
    rows: list[dict[str, Any]],
    *,
    key: str = "curvature",
    context: str = "",
) -> float:
    if not rows:
        raise MissingLearnedCurvatureError(
            f"no rows available to resolve learned curvature{': ' + context if context else ''}"
        )
    for row in rows:
        raw = row.get(key)
        if raw is not None:
            return require_learned_curvature(raw, context=context or f"rows[{key!r}]")
    raise MissingLearnedCurvatureError(
        f"embedding rows missing learned curvature ({key}){': ' + context if context else ''}"
    )
