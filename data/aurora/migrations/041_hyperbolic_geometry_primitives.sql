-- ============================================================================
-- Migration 041: Hyperbolic Geometry Primitives (Phase 1)
-- Date: 2026-06-08
-- Purpose: Registers PL/Python extensions, dual-precision columns for float64
--          hyperbolic embeddings, Lorentz manifold UDFs (inner product + distance
--          with numerical stability), and the precomputed distance matrix table
--          (partitioned by space_id for hot-path lookups).
--
-- This is the foundational DB upgrade to support native non-Euclidean vector
-- analysis (Lorentz model) alongside existing pgvector Euclidean paths.
-- See architectural plan for full context (UDFs, hybrid queries, precomputes, float64).
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS plpython3u;

-- 1. Lorentz Inner Product UDF (Minkowski inner product for hyperboloid model)
--    For u, v in R^{n+1}: -u0*v0 + sum_{i=1 to n} ui*vi
CREATE OR REPLACE FUNCTION lorentz_inner_product(
    u DOUBLE PRECISION[], 
    v DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION 
LANGUAGE plpython3u AS $$
    import numpy as np
    
    if len(u) != len(v):
        raise ValueError("Vector dimensions must match for Lorentz inner product.")
        
    u_arr = np.array(u, dtype=np.float64)
    v_arr = np.array(v, dtype=np.float64)
    
    # Minkowski inner product: -u0*v0 + dot(u[1:], v[1:])
    return float(-u_arr[0] * v_arr[0] + np.dot(u_arr[1:], v_arr[1:]))
$$;

COMMENT ON FUNCTION lorentz_inner_product(DOUBLE PRECISION[], DOUBLE PRECISION[]) IS 
'Computes the Lorentz (Minkowski) inner product for hyperbolic embeddings in the hyperboloid model. Used as building block for manifold distance. Expects float64 arrays from embedding_double column.';

-- 2. Lorentz Manifold Distance UDF
--    d_L(u, v) = arcosh( - <u, v>_L )
--    Includes boundary guard: floor val at 1.0 to prevent NaN from float rounding (val < 1.0)
CREATE OR REPLACE FUNCTION lorentz_distance(
    u DOUBLE PRECISION[], 
    v DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION 
LANGUAGE plpython3u AS $$
    import numpy as np
    
    if len(u) != len(v):
        raise ValueError("Vector dimensions must match for Lorentz distance.")
        
    u_arr = np.array(u, dtype=np.float64)
    v_arr = np.array(v, dtype=np.float64)
    
    # Calculate Lorentz inner product
    minkowski_dot = -u_arr[0] * v_arr[0] + np.dot(u_arr[1:], v_arr[1:])
    
    # Target value for arcosh is -minkowski_dot (should be >= 1)
    val = -minkowski_dot
    
    # Numerical stability floor: prevent arcosh(<1) -> NaN due to float64 rounding near boundary
    if val < 1.0:
        val = 1.0
        
    return float(np.arccosh(val))
$$;

COMMENT ON FUNCTION lorentz_distance(DOUBLE PRECISION[], DOUBLE PRECISION[]) IS 
'Computes geodesic distance on the Lorentz hyperboloid manifold using arcosh(-<u,v>_L). Includes guard against floating-point underflow producing values <1.0. Primary distance function for hyperbolic residue similarity / Phase 7 / Poincaré queries.';

-- 3. Schema: Dual-store high-precision embeddings (float64) for hyperbolic spaces
--    Leaves existing `embedding` (VECTOR, float32) untouched for Euclidean/pgvector compatibility.
ALTER TABLE fact_gnn_node_embedding 
    ADD COLUMN IF NOT EXISTS embedding_double DOUBLE PRECISION[];

COMMENT ON COLUMN fact_gnn_node_embedding.embedding_double 
IS 'High-precision (float64) array for non-Euclidean / hyperbolic embeddings. Populated for space_type=hyperbolic. Enables exact Lorentz math without precision loss near manifold boundary. Dual-stored alongside pgvector embedding for hybrid use.';

-- 4. Precomputed distance matrix table (partitioned by space_id)
--    Enables O(1) hot-path lookups for hyperbolic distances without recomputing arcosh on every query.
--    Populated by post-GNN background worker. For full N x N use per-structure or landmark approximations for scale.
CREATE TABLE IF NOT EXISTS fact_hyperbolic_distance (
    space_id         TEXT NOT NULL,
    residue_id_a     TEXT NOT NULL REFERENCES dim_residue(residue_id),
    residue_id_b     TEXT NOT NULL REFERENCES dim_residue(residue_id),
    lorentz_dist     DOUBLE PRECISION NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (space_id, residue_id_a, residue_id_b)
) PARTITION BY LIST (space_id);

COMMENT ON TABLE fact_hyperbolic_distance IS 
'Precomputed Lorentz manifold distances for hyperbolic embedding spaces. Enables fast hybrid queries and Poincaré / collapse simulations without repeated UDF calls. Partitioned by space_id (e.g., hyperbolic_v6). Worker populates selectively (e.g., per-structure all-pairs for small N, or top-k / landmarks for larger).';

-- Example partition (created on-demand by worker or manually; add more as new spaces appear)
-- CREATE TABLE IF NOT EXISTS fact_hyperbolic_distance_hyperbolic_v6 PARTITION OF fact_hyperbolic_distance
--     FOR VALUES IN ('hyperbolic_v6');

-- Indexes for hybrid queries (pre-filter on structure/residue + distance lookup)
CREATE INDEX IF NOT EXISTS idx_hyp_dist_space_a ON fact_hyperbolic_distance(space_id, residue_id_a);
CREATE INDEX IF NOT EXISTS idx_hyp_dist_space_b ON fact_hyperbolic_distance(space_id, residue_id_b);
CREATE INDEX IF NOT EXISTS idx_hyp_dist_dist ON fact_hyperbolic_distance(lorentz_dist);

-- 5. Helper view for common hyperbolic residue lookups (optional convenience)
CREATE OR REPLACE VIEW v_hyperbolic_residues AS
SELECT 
    e.structure_id,
    e.residue_id,
    e.embedding_double,
    e.cone_depth,
    e.epistemic_uncertainty,
    e.space_id
FROM fact_gnn_node_embedding e
JOIN embedding_space es ON e.space_id = es.space_id
WHERE es.space_type = 'hyperbolic'
  AND e.embedding_double IS NOT NULL;

COMMENT ON VIEW v_hyperbolic_residues IS 
'Convenience view over hyperbolic embeddings with double-precision vectors. Use for testing lorentz_distance UDF and hybrid pre-filter patterns.';

-- ============================================================================
-- USAGE NOTES / VERIFICATION
-- ============================================================================
-- 1. After migration, test UDFs:
--    SELECT lorentz_distance(
--      ARRAY[-1.0, 0.1, 0.2]::double precision[], 
--      ARRAY[-1.0, 0.11, 0.19]::double precision[]
--    );
--
-- 2. Hybrid query example (pre-filter scalars, then custom distance on filtered set):
--    (See user architectural plan for the 1APS-G53 pattern query.)
--
-- 3. Populate embedding_double and fact_hyperbolic_distance via post-GNN worker (next phase).
--
-- 4. Risks addressed:
--    - Sequential scan: Keep filters (structure_id, cone_depth, leak_score) highly selective.
--    - PL/Python deps: Ensure numpy in the DB's plpython3u environment (docker: add to image; prod: extension config).
--    - Precision: DOUBLE PRECISION[] + guard in UDF.
--
-- Next: Background worker logic to compute & insert distances after GNNv6 run.
-- ============================================================================
