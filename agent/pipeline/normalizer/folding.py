"""Normalize LERP folding path metadata into fact_folding_path.

Write path:
    folding path metadata  →  fact_folding_path  (one row per path)

The actual frame data (coordinates for each interpolation step) lives in GCS.
Only the path metadata is written to Cloud SQL; the GCS URI is stored as a
pointer so downstream consumers can fetch frames on demand.

Schema columns (from 002_schema_extensions.sql §fact_folding_path):
    path_id, structure_id, method, num_frames, num_atoms,
    gcs_uri, source_type, computed_at
"""
import uuid
from typing import Optional

import sqlalchemy as sa


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_INSERT = sa.text("""
    INSERT INTO fact_folding_path
        (path_id, structure_id, method,
         num_frames, num_atoms, gcs_uri, source_type)
    VALUES
        (:path_id, :structure_id, :method,
         :num_frames, :num_atoms, :gcs_uri, :source_type)
    ON CONFLICT (path_id) DO NOTHING
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize_folding_path(
    conn,
    structure_id: str,
    gcs_uri: str,
    num_frames: int,
    num_atoms: int,
    method: str = "lerp",
    source_type: str = "deterministic",
) -> str:
    """Write folding path metadata to fact_folding_path.

    Parameters
    ----------
    conn:
        Async SQLAlchemy connection (or any object with an ``execute``
        coroutine).
    structure_id:
        UUID v4 identifying the parent structure in dim_structure.
    gcs_uri:
        GCS URI where the frame data blob is stored (e.g.
        ``"gs://gosp-folding/paths/<path_id>.npy"``).
    num_frames:
        Number of interpolation frames in the path.
    num_atoms:
        Number of atoms per frame.
    method:
        Interpolation method — ``"lerp"`` (default) or ``"slerp"``.
    source_type:
        Provenance tag — ``"deterministic"`` for physics-kernel LERP paths.

    Returns
    -------
    str
        The generated ``path_id`` (UUID v4 string) for the inserted row.
    """
    path_id = str(uuid.uuid4())

    await conn.execute(_INSERT, {
        "path_id":      path_id,
        "structure_id": structure_id,
        "method":       method,
        "num_frames":   num_frames,
        "num_atoms":    num_atoms,
        "gcs_uri":      gcs_uri,
        "source_type":  source_type,
    })

    return path_id
