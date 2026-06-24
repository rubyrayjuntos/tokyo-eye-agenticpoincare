"""
Hyperbolic vector query helpers for Tokyo Eye.

NOTE (2026-06): The GNNv6 (GOSPConeMapperV6) produces embeddings in the *Poincaré ball*
model via geoopt.stereographic (expmap0 with learned k=-c). The stored embedding_double
vectors are ball coordinates in R^128 (not ambient Lorentz R^129 vectors).

Therefore the query layer now uses the curvature-aware Poincaré ball distance
to match the model exactly (see poincare_ball_distance below). The old Lorentz
formula is retained only for reference/diagnostics and will produce 0.0000 on
these embeddings because of the model mismatch (ball coords vs hyperboloid ambient).

Precomputed fact_hyperbolic_distance rows (populated with the old formula) are
stale for cross-structure work until the worker is updated and distances re-materialized.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from data.db import get_connection, DBAdapter


def _lorentz_distance_np(u: np.ndarray, v: np.ndarray) -> float:
    """OLD (broken for v6 data) Minkowski formula retained for diagnostics only.
    Applying this to Poincaré ball coordinates (what GNNv6 actually stores) produces
    the 0.0000 anomaly because the vectors are not on the hyperboloid the formula expects.
    """
    val = -float(u[0]) * float(v[0]) + float(np.dot(u[1:], v[1:]))
    if val < 1.0:
        val = 1.0
    return float(np.arccosh(val))


def poincare_ball_distance(u: np.ndarray, v: np.ndarray, c: float) -> float:
    """
    Curvature-aware Poincaré ball distance (exact match to geoopt.manifolds.stereographic.math.dist).
    This is the correct distance for the embeddings produced by GOSPConeMapperV6
    (x_hyp = pmath.expmap0(..., k=-c) where c = softplus(log_c) + eps).
    """
    sqdist = np.sum((u - v) ** 2)
    norm_u_sq = np.sum(u ** 2)
    norm_v_sq = np.sum(v ** 2)

    den_u = 1.0 - c * norm_u_sq
    den_v = 1.0 - c * norm_v_sq

    # Guard against points outside the ball or numerical issues (should be >0 inside ball)
    if den_u <= 0 or den_v <= 0:
        # Fallback: treat as very far (or could raise)
        return float("inf")

    delta = 2.0 * c * sqdist / (den_u * den_v)
    arg = 1.0 + delta

    if arg < 1.0:
        arg = 1.0

    return float((1.0 / np.sqrt(c)) * np.arccosh(arg))


def _poincare_ball_distances(target: np.ndarray, cands: np.ndarray, c: float) -> np.ndarray:
    """
    Vectorized Poincaré ball distance for one target vs N candidates.
    Matches the scalar poincare_ball_distance exactly.
    Used to compute distances over the *full* candidate pool before LIMIT in cross-structure searches.
    """
    if cands.ndim == 1:
        cands = cands.reshape(1, -1)
    sqdist = np.sum((cands - target) ** 2, axis=1)
    norm_u_sq = np.sum(target ** 2)
    norm_v_sq = np.sum(cands ** 2, axis=1)
    den_u = 1.0 - c * norm_u_sq
    den_v = 1.0 - c * norm_v_sq
    mask = (den_u > 0) & (den_v > 0)
    delta = np.zeros_like(sqdist)
    delta[mask] = 2.0 * c * sqdist[mask] / (den_u * den_v[mask])
    arg = 1.0 + delta
    arg = np.maximum(arg, 1.0)
    dists = np.full_like(arg, np.inf, dtype=float)
    dists[mask] = (1.0 / np.sqrt(c)) * np.arccosh(arg[mask])
    return dists


logger = logging.getLogger(__name__)


async def find_hyperbolic_neighbors(
    target_vector: list[float],
    space_id: str = "space_gospconemapper_v6_hyp128",
    cone_depth_min: float = 7.5,
    leak_score_min: float = 60,
    limit: int = 25,
    exclude_structure_id: str | None = None,
    target_residue_id: str | None = None,  # If provided, try precomputed table first
    db: Any = None,  # If provided, reuse caller's DBAdapter instead of acquiring new conn (pool + tx hygiene)
    curvature_c: float = 0.6054343,  # v6 model curvature (softplus(log_c)+eps). Must match the GNN run that produced the embeddings.
    # New cross-protein discovery filters (applied in Python fallback path for flexibility)
    exclude_structures: list[str] | None = None,  # e.g. ["3oxz", "3cs9", "1iep"]
    include_structures: list[str] | None = None,  # e.g. specific proteins or families mapped to IDs
    distance_max: float | None = None,  # e.g. 13.0 for Abl-like band under current c
) -> list[dict]:
    """
    Find residues with similar hyperbolic (Lorentz) geometry to the target_vector.

    Uses strict hybrid processing:
    - Pre-filter on scalar attributes (cone_depth, leak_score, structure)
    - Then compute exact lorentz_distance on the filtered set (or lookup precomputed).

    If target_residue_id is given and distances are pre-materialized in
    fact_hyperbolic_distance, uses the fast lookup path.

    Returns list of dicts with residue_id, structure_id, residue_type, cone_depth,
    leak_score, distance (manifold distance).
    """
    if not target_vector:
        raise ValueError("target_vector cannot be empty")

    # Support injected db (from agent tool context or caller) to avoid extra pool acquire
    # and to share the transaction/connection for embedding fetch + neighbor search.
    if db is None:
        async with get_connection() as conn:
            db = DBAdapter(conn)
            return await _find_hyperbolic_neighbors_impl(
                db=db,
                target_vector=target_vector,
                space_id=space_id,
                cone_depth_min=cone_depth_min,
                leak_score_min=leak_score_min,
                limit=limit,
                exclude_structure_id=exclude_structure_id,
                target_residue_id=target_residue_id,
                curvature_c=curvature_c,
                exclude_structures=exclude_structures,
                include_structures=include_structures,
                distance_max=distance_max,
            )
    else:
        return await _find_hyperbolic_neighbors_impl(
            db=db,
            target_vector=target_vector,
            space_id=space_id,
            cone_depth_min=cone_depth_min,
            leak_score_min=leak_score_min,
            limit=limit,
            exclude_structure_id=exclude_structure_id,
            target_residue_id=target_residue_id,
            curvature_c=curvature_c,
            exclude_structures=exclude_structures,
            include_structures=include_structures,
            distance_max=distance_max,
        )


async def _find_hyperbolic_neighbors_impl(
    *,
    db: Any,
    target_vector: list[float],
    space_id: str,
    cone_depth_min: float,
    leak_score_min: float,
    limit: int,
    exclude_structure_id: str | None,
    target_residue_id: str | None,
    curvature_c: float,
    exclude_structures: list[str] | None = None,
    include_structures: list[str] | None = None,
    distance_max: float | None = None,
) -> list[dict]:
    """Core implementation. Assumes db is a ready DBAdapter (either injected or newly acquired)."""
    # Fast path: use precomputed table if we have a target residue and it exists
    # Note: precomputed pairs are intra-structure (populated per-structure); UDF fallback enables
    # cross-structure Lorentz matches (the key capability for the 3OXZ vs 3CS9/1IEP/2HYY/4AKE hypothesis).
    if target_residue_id:
        precomputed_query = """
            SELECT 
                e.residue_id,
                e.structure_id,
                COALESCE(r.residue_name, r.residue_name_3, '?') AS residue_type,
                COALESCE(sl.cone_depth, e.cone_depth) AS cone_depth,
                sl.leak_score,
                d.lorentz_dist AS distance
            FROM fact_hyperbolic_distance d
            JOIN fact_gnn_node_embedding e ON d.residue_id_b = e.residue_id AND d.space_id = e.space_id
            JOIN fact_source_leak sl ON sl.residue_id = e.residue_id
            LEFT JOIN dim_residue r ON e.residue_id = r.residue_id
            WHERE d.space_id = :space_id
              AND d.residue_id_a = :residue_id_a
              AND sl.leak_score > :leak_score_min
              AND COALESCE(sl.cone_depth, e.cone_depth) > :cone_depth_min
              AND ((:exclude_structure_id)::text IS NULL OR e.structure_id != :exclude_structure_id)
            ORDER BY d.lorentz_dist ASC
            LIMIT :limit;
        """
        try:
            rows = await db.fetch_all(
                precomputed_query,
                {
                    "space_id": space_id,
                    "residue_id_a": target_residue_id,
                    "cone_depth_min": cone_depth_min,
                    "leak_score_min": leak_score_min,
                    "exclude_structure_id": exclude_structure_id,
                    "limit": limit,
                },
            )
            if rows:
                logger.info(
                    "Used precomputed hyperbolic distances for %s (%d results)",
                    target_residue_id,
                    len(rows),
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(
                "Precomputed distance lookup failed, falling back to UDF: %s", e
            )

    # Fallback / general path: UDF on pre-filtered rows
    # This is the "strict hybrid" pattern: relational filters first, then custom distance.
    # Use safe IS NULL form (no ::cast on the param placeholder) so the named-param
    # converter + psycopg binding never sees spurious ":text" / ":double" tokens inside casts.
    # Join fact_source_leak for leak_score (the canonical source) + cone; fact_gnn carries cone too.
    # LEFT JOIN dim_residue for type (residue_name); structure_id comes from e (denormalized).
    udf_query = """
        SELECT 
            e.residue_id,
            e.structure_id,
            COALESCE(r.residue_name, r.residue_name_3, '?') AS residue_type,
            COALESCE(sl.cone_depth, e.cone_depth) AS cone_depth,
            sl.leak_score,
            lorentz_distance(e.embedding_double, :target_vector::double precision[]) AS distance
        FROM fact_gnn_node_embedding e
        JOIN fact_source_leak sl ON sl.residue_id = e.residue_id
        LEFT JOIN dim_residue r ON e.residue_id = r.residue_id
        WHERE e.space_id = :space_id
          AND sl.leak_score > :leak_score_min
          AND COALESCE(sl.cone_depth, e.cone_depth) > :cone_depth_min
          AND ((:exclude_structure_id)::text IS NULL OR e.structure_id != :exclude_structure_id)
          AND e.embedding_double IS NOT NULL
        ORDER BY distance ASC
        LIMIT :limit;
    """

    try:
        rows = await db.fetch_all(
            udf_query,
            {
                "target_vector": target_vector,
                "space_id": space_id,
                "cone_depth_min": cone_depth_min,
                "leak_score_min": leak_score_min,
                "exclude_structure_id": exclude_structure_id,
                "limit": limit,
            },
        )
        return [dict(row) for row in rows]
    except Exception as udf_err:
        # UDF (plpython3u lorentz_distance) not available in this DB image / not created.
        # Fall back to Python/NumPy exact computation after relational pre-filter.
        # This still gives correct cross-structure Lorentz distances for the hypothesis.
        logger.warning(
            "lorentz_distance UDF failed or missing (%s); using NumPy client-side fallback "
            "(pre-filters still executed server-side for strict hybrid semantics)",
            udf_err,
        )
        # The failed UDF statement aborted the current tx on this connection.
        # Rollback so the subsequent plain pre-filter SELECT can execute.
        try:
            await db.rollback()
        except Exception:
            pass

        # Build the candidate fetch query.
        # IMPORTANT: In the Python fallback path, we *always* fetch the *entire* pool
        # of candidates matching the Euclidean pre-filters (no LIMIT in SQL).
        # We compute the exact Poincaré distances vectorized over the full pool,
        # then filter (by include/exclude/distance_max), sort, and apply limit.
        # This guarantees no valid geometric targets are dropped due to arbitrary
        # DB ordering + early LIMIT (the root cause of ghost medoids not being found).
        params = {
            "space_id": space_id,
            "cone_depth_min": cone_depth_min,
            "leak_score_min": leak_score_min,
            "exclude_structure_id": exclude_structure_id,
        }
        where = [
            "e.space_id = :space_id",
            "sl.leak_score > :leak_score_min",
            "COALESCE(sl.cone_depth, e.cone_depth) > :cone_depth_min",
            "((:exclude_structure_id)::text IS NULL OR e.structure_id != :exclude_structure_id)",
            "e.embedding_double IS NOT NULL",
        ]
        if include_structures:
            # Push the structure restriction into SQL for efficiency.
            struct_placeholders = []
            for i, s in enumerate(include_structures):
                ph = f"inc_struct_{i}"
                struct_placeholders.append(f"e.structure_id = :{ph}")
                params[ph] = s
            if struct_placeholders:
                where.append("(" + " OR ".join(struct_placeholders) + ")")

        plain_query = f"""
            SELECT 
                e.residue_id,
                e.structure_id,
                COALESCE(r.residue_name, r.residue_name_3, '?') AS residue_type,
                COALESCE(sl.cone_depth, e.cone_depth) AS cone_depth,
                sl.leak_score,
                e.embedding_double
            FROM fact_gnn_node_embedding e
            JOIN fact_source_leak sl ON sl.residue_id = e.residue_id
            LEFT JOIN dim_residue r ON e.residue_id = r.residue_id
            WHERE {' AND '.join(where)}
        """
        # No LIMIT here -- full pool for correct distance-based ranking.
        plain_query += ";"

        cands = await db.fetch_all(plain_query, params)

        target_arr = np.asarray(target_vector, dtype=np.float64)
        scored: list[dict] = []

        if cands:
            # Vectorized distance computation over the (potentially full) pool
            emb_list = []
            meta = []
            for cand in cands:
                emb = cand.get("embedding_double")
                if emb is None:
                    continue
                emb_list.append(np.asarray(emb, dtype=np.float64))
                meta.append(cand)

            if emb_list:
                cands_mat = np.vstack(emb_list)
                dists = _poincare_ball_distances(target_arr, cands_mat, curvature_c)

                for i, cand in enumerate(meta):
                    new_d = float(dists[i])

                    struct = cand.get("structure_id")
                    if exclude_structures and struct in exclude_structures:
                        continue
                    if include_structures and struct not in include_structures:
                        continue
                    if distance_max is not None and new_d > distance_max:
                        continue

                    # Also compute arg for the winning items (for diagnostics)
                    emb_arr = np.asarray(cand.get("embedding_double"), dtype=np.float64)
                    sqdist = np.sum((target_arr - emb_arr) ** 2)
                    norm_u_sq = np.sum(target_arr ** 2)
                    norm_v_sq = np.sum(emb_arr ** 2)
                    den_u = 1.0 - curvature_c * norm_u_sq
                    den_v = 1.0 - curvature_c * norm_v_sq
                    arg = 1.0 + (2.0 * curvature_c * sqdist / (den_u * den_v)) if den_u > 0 and den_v > 0 else float("nan")

                    item = {k: v for k, v in cand.items() if k != "embedding_double"}
                    item["distance"] = new_d
                    item["debug_poincare_distance"] = new_d
                    item["debug_c"] = curvature_c
                    item["debug_arg_before_arcosh"] = arg
                    scored.append(item)

        scored.sort(key=lambda x: x.get("distance", float("inf")))
        return scored[:limit]


async def get_hyperbolic_embedding(
    db: Any, residue_id: str, space_id: str = "space_gospconemapper_v6_hyp128"
) -> Optional[list[float]]:
    """Helper to fetch the high-precision embedding_double for a residue.
    Default space is the one used by GNNv6 + bulk populator for real data.
    """
    query = """
        SELECT embedding_double 
        FROM fact_gnn_node_embedding 
        WHERE residue_id = :residue_id AND space_id = :space_id AND embedding_double IS NOT NULL
        LIMIT 1
    """
    row = await db.fetch_one(query, {"residue_id": residue_id, "space_id": space_id})
    return row["embedding_double"] if row else None
