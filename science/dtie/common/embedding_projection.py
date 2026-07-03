"""Parse and validate governed hyperbolic embedding projections at read boundaries."""

from __future__ import annotations

import math
from typing import Any

from science.dtie.common.poincare_conventions import ball_radius

# Interior margin matching motif-discovery ball checks.
_BALL_INTERIOR_EPS = 1e-10


class MalformedHyperbolicCoordinateError(ValueError):
    """Raised when embedding projections cannot be read as a valid in-ball disc point."""


def validate_disc_point_xy(
    x: float,
    y: float,
    *,
    curvature: float,
) -> tuple[float, float]:
    """Validate a Poincaré disc point; raise if non-finite or outside the ball."""
    if curvature <= 0 or not math.isfinite(curvature):
        raise MalformedHyperbolicCoordinateError(
            f"invalid curvature for disc validation: {curvature!r}"
        )
    if not math.isfinite(x) or not math.isfinite(y):
        raise MalformedHyperbolicCoordinateError(
            f"non-finite hyperbolic coordinate: ({x!r}, {y!r})"
        )
    max_sq = ball_radius(curvature) ** 2 - _BALL_INTERIOR_EPS
    sq_norm = x * x + y * y
    if sq_norm >= max_sq:
        raise MalformedHyperbolicCoordinateError(
            f"hyperbolic coordinate outside Poincaré ball: ({x}, {y}), "
            f"sq_norm={sq_norm}, max_sq={max_sq}"
        )
    return x, y


def _parse_float_pair(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) < 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None
    return None


def _extract_xy_from_row(row: dict[str, Any]) -> tuple[float, float] | None:
    pair = _parse_float_pair(row.get("hyp_projection_2d"))
    if pair is not None:
        return pair

    proj = row.get("hyp_projections")
    if isinstance(proj, dict):
        for key_x, key_y in (("x", "y"), ("u", "v"), ("0", "1")):
            if key_x in proj and key_y in proj:
                try:
                    return float(proj[key_x]), float(proj[key_y])
                except (TypeError, ValueError):
                    return None
        values: list[float] = []
        for value in proj.values():
            if isinstance(value, (int, float)):
                values.append(float(value))
        if len(values) >= 2:
            return values[0], values[1]
    elif isinstance(proj, list) and len(proj) >= 2:
        try:
            return float(proj[0]), float(proj[1])
        except (TypeError, ValueError):
            return None
    return None


def parse_embedding_projection_xy(
    row: dict[str, Any],
    *,
    curvature: float | None = None,
) -> tuple[float, float]:
    """Extract and validate Poincaré disc x/y from governed embedding columns.

    Missing, unparseable, non-finite, or out-of-ball coordinates raise
    ``MalformedHyperbolicCoordinateError``. This function never substitutes
    the disc origin as a silent fallback.
    """
    resolved_curvature = curvature
    if resolved_curvature is None:
        raw_c = row.get("curvature")
        if raw_c is not None:
            try:
                resolved_curvature = float(raw_c)
            except (TypeError, ValueError):
                resolved_curvature = None
    if resolved_curvature is None:
        raise MalformedHyperbolicCoordinateError(
            "missing curvature required to validate hyperbolic projection"
        )

    pair = _extract_xy_from_row(row)
    if pair is None:
        raise MalformedHyperbolicCoordinateError(
            "missing or unparseable hyperbolic projection columns"
        )
    return validate_disc_point_xy(pair[0], pair[1], curvature=resolved_curvature)
