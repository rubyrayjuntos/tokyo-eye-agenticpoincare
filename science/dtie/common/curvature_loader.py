"""Single SSOT read path for pinned hyperbolic curvature (Phase 1a).

Reads embedding_space.curvature (+ optional curvature_hash verification).
Secrets Manager cross-check is Phase 1b — not implemented here.
"""

from __future__ import annotations

import hashlib
from typing import Any

from data.db import DBAdapter, get_connection

# Production v6 hyperbolic space (embedding_space.name / space_id suffix).
V6_HYP_SPACE_NAME = "gospconemapper_v6_hyp128"
V6_HYP_SPACE_ID = "space_gospconemapper_v6_hyp128"

# Pin from lever_a_clean_slate_v1/v6_best_disc.pt — ckpt['curvature'] at full precision.
CANONICAL_V6_CURVATURE = 0.7026273608207703


class CurvatureSSOTError(RuntimeError):
    """Raised when curvature cannot be loaded or fails hash verification."""


def normalize_space_key(space: str) -> str:
    """Map space_id or name to embedding_space.name."""
    key = space.strip()
    if key.startswith("space_"):
        return key[len("space_") :]
    return key


def curvature_hash_for(c: float) -> str:
    """SHA256 hex digest matching migration 051 population (Python repr)."""
    return hashlib.sha256(repr(float(c)).encode("ascii")).hexdigest()


def _verify_hash(curvature: float, stored_hash: str | None) -> None:
    if not stored_hash:
        return
    expected = curvature_hash_for(curvature)
    if stored_hash.lower() != expected.lower():
        raise CurvatureSSOTError(
            "curvature hash mismatch for embedding_space row "
            f"(stored={stored_hash[:12]}… expected={expected[:12]}…)"
        )


async def _fetch_curvature_row(db: DBAdapter, space_key: str) -> dict[str, Any]:
    return await db.fetch_one(
        """
        SELECT curvature, curvature_hash, name, space_id
        FROM embedding_space
        WHERE name = :key OR space_id = :key OR space_id = :prefixed
        LIMIT 1
        """,
        {
            "key": space_key,
            "prefixed": f"space_{space_key}",
        },
    )


async def get_curvature(
    space_name: str,
    *,
    db: Any | None = None,
) -> float:
    """Load pinned curvature for a hyperbolic embedding space.

    Args:
        space_name: embedding_space.name (e.g. gospconemapper_v6_hyp128) or space_id.
        db: Optional DBAdapter; opens a short-lived pool connection when omitted.

    Raises:
        CurvatureSSOTError: row missing, NULL curvature, or hash mismatch.
    """
    key = normalize_space_key(space_name)
    if db is not None:
        row = await _fetch_curvature_row(db, key)
    else:
        async with get_connection() as conn:
            row = await _fetch_curvature_row(DBAdapter(conn), key)

    if not row:
        raise CurvatureSSOTError(f"embedding_space not found for key {key!r}")
    curvature = row.get("curvature")
    if curvature is None:
        raise CurvatureSSOTError(
            f"embedding_space.curvature is NULL for {row.get('name') or key!r}"
        )
    c = float(curvature)
    _verify_hash(c, row.get("curvature_hash"))
    return c


async def get_curvature_for_space_id(
    space_id: str,
    *,
    db: Any | None = None,
) -> float:
    """Convenience wrapper for callers that hold space_id instead of name."""
    return await get_curvature(space_id, db=db)
