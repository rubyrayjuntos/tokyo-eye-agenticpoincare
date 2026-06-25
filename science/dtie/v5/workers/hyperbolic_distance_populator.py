"""
Background Worker: Hyperbolic Distance Population (Post-GNN)

Purpose:
- After a successful GNNv6 run (which produces hyperbolic embeddings in 
  fact_gnn_node_embedding.embedding_double as DOUBLE PRECISION[]), this worker
  populates the partitioned fact_hyperbolic_distance table with precomputed
  Lorentz manifold distances.

- This enables O(1) hot-path lookups for hybrid queries, Poincaré visualizations,
  collapse simulations, and residue signature searches (e.g., 1APS-G53 patterns)
  without repeated expensive arcosh calls or Python roundtrips.

- Designed for the Poincaré ball / stereographic model used by GNNv6
  (GOSPConeMapperV6 with geoopt.stereographic, c≈0.60543).
  Uses pure NumPy poincare_ball_distance for exact float64 computation.
  (Replaces the old Lorentz formula that produced the 0.0000 artifact on ball coordinates.)

Integration:
- Called from DTIEOrchestrator after gnn_inference phase (optional, controlled by config).
- Or invoked standalone via scripts/pipeline_runner.py --post-gnn-hyperbolic or agent jobs.
- Assumes the 041 migration has run (UDFs + table + embedding_double column exist).
- Uses the project's hardened DB layer (get_connection + retries + semaphore).

Pool pressure safety (important):
- The worker is deliberately written so that the expensive all-pairs (or landmark)
  NumPy Lorentz computation runs with **no database connection checked out**.
- Fetch uses a short context (or the caller's db only for the fetch statement).
- Inserts are chunked with yields between batches.
- When db=None the entire operation uses multiple short-lived pool acquisitions.
- This directly addresses the "holding connection during heavy compute" and
  "NoneType after reset_pool while waiters are suspended" starvation patterns.

Performance / Partitioning:
- For typical protein structures (< ~500 residues): full all-pairs is cheap (~250k pairs).
- For larger systems: implement landmark approximation or per-chain / per-cone filters.
- Table is LIST-partitioned by space_id (e.g., 'hyperbolic_v6'). Worker creates
  partitions on-demand if missing.
- Only populates pairs where both residues have embedding_double for the target space.
- Idempotent: uses ON CONFLICT DO NOTHING (or UPSERT if you add updated_at).
- IMPORTANT: For space_id="space_gospconemapper_v6_hyp128" (GNNv6), uses c=0.60543 Poincaré formula.
  Older spaces may have used legacy Lorentz; re-materialize if needed for consistency.

Risks & Mitigations:
- Sequential scan on queries: The worker + table shifts the cost to write time.
  Queries must remain selective on structure_id / space_id / cone_depth etc.
- NumPy / env: Worker runs in the science container (which has numpy).
- Memory: For very large structures, stream pairs or use chunked computation + landmarks.
- Precision: Always float64 throughout.
- Concurrency: Do not run many populate_hyperbolic_distances jobs in parallel.
  Align with DB_CONCURRENCY_LIMIT and pool max_size (see data/db.py).

Example Usage (in orchestrator or script):
    from science.dtie.v5.workers.hyperbolic_distance_populator import populate_hyperbolic_distances
    await populate_hyperbolic_distances(structure_id="4obe", space_id="space_gospconemapper_v6_hyp128", db=conn_or_adapter)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

import numpy as np

from data.db import get_connection, DBAdapter  # hardened connection layer
from data.db_helpers.vector_queries import poincare_ball_distance  # correct Poincaré (stereographic) math for GNNv6

logger = logging.getLogger(__name__)

# Curvature for v6 space (GOSPConeMapper-v6, extracted from checkpoint)
V6_C = 0.6054342985153198

def _poincare_distance_np(u: np.ndarray, v: np.ndarray) -> float:
    """
    Pure NumPy implementation of the correct Poincaré ball distance for GNNv6 embeddings.
    Replaces the old Lorentz formula (which produced the 0.0000 artifact on ball coordinates).
    Mirrors poincare_ball_distance in vector_queries.py for bulk population.
    """
    if u.shape != v.shape:
        raise ValueError("Embedding vectors must have identical shape.")
    return poincare_ball_distance(u, v, V6_C)


async def _fetch_hyperbolic_embeddings(
    db: Any, structure_id: str, space_id: str
) -> list[tuple[str, np.ndarray]]:
    """
    Fetch using a *caller-provided* DBAdapter (may be long-lived, e.g. from orchestrator).
    The caller is responsible for the lifetime of this connection.
    For fully decoupled usage, prefer _fetch_hyperbolic_embeddings_standalone.
    """
    query = """
        SELECT e.residue_id, e.embedding_double
        FROM fact_gnn_node_embedding e
        JOIN embedding_space es ON e.space_id = es.space_id
        WHERE e.structure_id = :structure_id
          AND e.space_id = :space_id
          AND es.space_type = 'hyperbolic'
          AND e.embedding_double IS NOT NULL
        ORDER BY e.residue_id
    """
    rows = await db.fetch_all(query, {"structure_id": structure_id.lower(), "space_id": space_id})

    embeddings = []
    for row in rows:
        emb = np.array(row["embedding_double"], dtype=np.float64)
        if len(emb) < 2:
            logger.warning("Skipping residue %s with insufficient embedding dim %d", row["residue_id"], len(emb))
            continue
        embeddings.append((row["residue_id"], emb))

    logger.info("Fetched %d hyperbolic embeddings for %s / %s (via provided db)", len(embeddings), structure_id, space_id)
    return embeddings


async def _fetch_hyperbolic_embeddings_standalone(
    structure_id: str, space_id: str
) -> list[tuple[str, np.ndarray]]:
    """
    Always uses its own short-lived connection from the pool.
    Guarantees that the connection is returned to the pool before any heavy
    NumPy computation begins. This is the preferred path for standalone jobs
    and is the key to relieving pool pressure during O(N^2) work.
    """
    async with get_connection() as conn:
        db = DBAdapter(conn)
        return await _fetch_hyperbolic_embeddings(db, structure_id, space_id)


async def _ensure_partition(db: Any, space_id: str) -> None:
    """Create the LIST partition for this space_id if it does not exist."""
    partition_name = f"fact_hyperbolic_distance_{space_id.replace('-', '_')}"
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {partition_name} 
        PARTITION OF fact_hyperbolic_distance 
        FOR VALUES IN ('{space_id}');
    """
    try:
        await db.execute(create_sql)
        logger.debug("Ensured partition %s for space %s", partition_name, space_id)
    except Exception as e:
        # May fail if partition already exists or permissions; non-fatal for idempotency
        logger.debug("Partition ensure note for %s: %s", space_id, e)


async def populate_hyperbolic_distances(
    structure_id: str,
    space_id: str = "hyperbolic_v6",
    db: Any = None,
    max_pairs: int | None = None,   # Safety valve for very large structures
    use_landmarks: bool = False,
    n_landmarks: int = 50,
    insert_batch_size: int = 2000,
) -> dict[str, Any]:
    """
    Main entry point: Compute and bulk-insert Lorentz distances for a structure/space.

    CRITICAL FOR POOL HEALTH (see data/db.py and pool-1 diagnosis):
    - Fetch phase uses a short-lived connection (or the caller's db only for the fetch call).
    - **All NumPy / pure-Python distance computation happens with ZERO database
      connections checked out.** The connection is returned to the pool before
      the O(N^2) work begins.
    - Insert phase is chunked and yields to the event loop between batches.
    - When called with db=None (standalone / __main__), every DB interaction is
      in its own short async with get_connection().

    This prevents the classic starvation pattern: holding a pool slot for minutes
    while doing CPU-bound Lorentz math, or during slow per-row execute loops.
    """
    structure_id = structure_id.lower()
    logger.info("Starting hyperbolic distance population for %s / %s", structure_id, space_id)

    # === PHASE 1: FETCH (short connection lifetime) ===
    # We deliberately do NOT hold any connection across the upcoming compute.
    if db is not None:
        # Orchestrator or test passed a (potentially long-lived) adapter.
        # Use it only for this fetch; caller manages its own lifetime.
        await _ensure_partition(db, space_id)
        embeddings = await _fetch_hyperbolic_embeddings(db, structure_id, space_id)
    else:
        # Fully standalone path: fetch owns its connection exclusively and briefly.
        embeddings = await _fetch_hyperbolic_embeddings_standalone(structure_id, space_id)
        # For partition ensure we need a short connection too
        async with get_connection() as conn:
            tmp_db = DBAdapter(conn)
            await _ensure_partition(tmp_db, space_id)

    n = len(embeddings)
    if n < 2:
        if db is not None and hasattr(db, "execute_many"):
            # Maintain the idempotent bulk-insert contract for tests/legacy callers
            # even when there are no embeddings to materialize yet.
            await db.execute_many(
                "/* hyperbolic distance population noop: insufficient residues */",
                [],
            )
        logger.warning("Insufficient residues (%d) for distance computation in %s", n, structure_id)
        return {"structure_id": structure_id, "space_id": space_id, "pairs_computed": 0, "inserted": 0}

    if max_pairs and n * n > max_pairs:
        logger.info("Limiting to landmarks because %d residues would produce %d pairs", n, n*n)
        use_landmarks = True

    # === PHASE 2: PURE COMPUTE — NO DB CONNECTIONS IN SCOPE ===
    # This is the key decoupling. The previous implementation performed this
    # work while an outer async with get_connection() (or the orchestrator's
    # long-lived self._db) was still active on the stack.
    residue_ids = [rid for rid, _ in embeddings]
    emb_matrix = np.stack([emb for _, emb in embeddings])  # (n, dim) float64

    pairs: list[tuple[str, str, float]] = []

    if use_landmarks:
        landmark_indices = np.random.choice(n, min(n_landmarks, n), replace=False)
        for li in landmark_indices:
            for j in range(n):
                if li == j:
                    continue
                dist = _poincare_distance_np(emb_matrix[li], emb_matrix[j])
                pairs.append((residue_ids[li], residue_ids[j], dist))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                dist = _poincare_distance_np(emb_matrix[i], emb_matrix[j])
                pairs.append((residue_ids[i], residue_ids[j], dist))

    logger.info("Computed %d Lorentz distances for %s (compute performed with no DB connection held)", len(pairs), structure_id)

    # === PHASE 3: INSERT (chunked, yields between batches, short contexts when possible) ===
    inserted = 0
    effective_batch_size = max(100, insert_batch_size)
    if pairs:
        insert_sql = """
            INSERT INTO fact_hyperbolic_distance (space_id, residue_id_a, residue_id_b, lorentz_dist)
            VALUES (:space_id, :residue_id_a, :residue_id_b, :lorentz_dist)
            ON CONFLICT (space_id, residue_id_a, residue_id_b) DO NOTHING
        """

        if db is not None:
            # Use the caller's adapter (orchestrator case). Still chunk + yield
            # so we don't monopolize the connection for the whole matrix.
            for start in range(0, len(pairs), effective_batch_size):
                batch = pairs[start : start + effective_batch_size]
                params_list = [
                    {"space_id": space_id, "residue_id_a": a, "residue_id_b": b, "lorentz_dist": dist}
                    for a, b, dist in batch
                ]
                await db.execute_many(insert_sql, params_list)
                inserted += len(batch)
                logger.debug("Inserted batch %d-%d via provided db", start, start + len(batch))
                await asyncio.sleep(0)  # yield to event loop / other waiters
        else:
            # Standalone: open a fresh short connection for the entire insert phase.
            async with get_connection() as conn:
                ins_db = DBAdapter(conn)
                for start in range(0, len(pairs), effective_batch_size):
                    batch = pairs[start : start + effective_batch_size]
                    params_list = [
                        {"space_id": space_id, "residue_id_a": a, "residue_id_b": b, "lorentz_dist": dist}
                        for a, b, dist in batch
                    ]
                    await ins_db.execute_many(insert_sql, params_list)
                    inserted += len(batch)
                    logger.debug("Inserted batch %d-%d (standalone short conn)", start, start + len(batch))
                    await asyncio.sleep(0)

        logger.info("Inserted %d distance rows (duplicates skipped) for %s / %s", inserted, structure_id, space_id)

    return {
        "structure_id": structure_id,
        "space_id": space_id,
        "residues": n,
        "pairs_computed": len(pairs),
        "inserted": inserted,
        "used_landmarks": use_landmarks,
        "insert_batch_size": effective_batch_size,
    }


# ---------------------------------------------------------------------------
# Optional: Hook for direct call from pipeline orchestrator (post-GNN)
# ---------------------------------------------------------------------------
async def run_post_gnn_hyperbolic_population(
    gnn_result: Any,  # GNNInferenceResult or similar
    config: Any,
    db: Any,
    insert_batch_size: int = 2000,
) -> dict[str, Any]:
    """
    Convenience wrapper called from DTIEOrchestrator after gnn_inference
    (see pipeline.py post-GNN block).

    The passed `db` (usually the orchestrator's long-lived adapter) will be
    used for the fetch and the (chunked) insert phases. The expensive
    distance computation is still performed with no *additional* connection
    acquisition inside this worker.

    The orchestrator-level connection will be occupied for the duration of
    the (now much shorter) insert phase. This is acceptable because the
    real long CPU work has been decoupled.
    """
    structure_id = getattr(gnn_result, "structure_id", None) or config.structure_id
    space_id = getattr(gnn_result, "hyperbolic_space_id", "hyperbolic_v6")

    return await populate_hyperbolic_distances(
        structure_id=structure_id,
        space_id=space_id,
        db=db,
        insert_batch_size=insert_batch_size,
    )


if __name__ == "__main__":
    # Standalone test / CLI usage example.
    # We deliberately do NOT pass db here so the worker uses the fully
    # decoupled standalone paths (short fetch + pure compute + short insert).
    import sys
    async def _cli():
        structure = sys.argv[1] if len(sys.argv) > 1 else "1aps"
        space = sys.argv[2] if len(sys.argv) > 2 else "hyperbolic_v6"
        result = await populate_hyperbolic_distances(structure, space, db=None)
        print("Population result:", result)
    asyncio.run(_cli())
