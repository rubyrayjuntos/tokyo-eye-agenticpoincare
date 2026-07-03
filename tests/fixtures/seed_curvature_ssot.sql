-- Minimal SSOT seed for P_CURV_01 (CI + local integration).
-- Requires full migrations applied first (052 repopulates curvature_hash).
-- curvature_hash = SHA256(big-endian IEEE754 float64) — curvature_loader.curvature_hash_for().

INSERT INTO embedding_space (
    space_id,
    name,
    space_type,
    dimensionality,
    curvature,
    curvature_hash,
    model_name,
    is_active
) VALUES (
    'space_gospconemapper_v6_hyp128',
    'gospconemapper_v6_hyp128',
    'hyperbolic',
    128,
    0.7026273608207703,
    '90383c360f60edddafebcb6c119b4b62366543eb733d7a9536bcb54d92c2e0e8',
    'GOSPConeMapper-v6',
    TRUE
)
ON CONFLICT (space_id) DO UPDATE SET
    curvature = EXCLUDED.curvature,
    curvature_hash = EXCLUDED.curvature_hash,
    space_type = EXCLUDED.space_type,
    dimensionality = EXCLUDED.dimensionality;

INSERT INTO dim_structure (structure_id, pdb_id, method, source)
VALUES ('11qe', '11QE', 'X-RAY', 'rcsb')
ON CONFLICT (structure_id) DO NOTHING;

INSERT INTO dim_chain (chain_id, structure_id, chain_label, entity_type)
VALUES ('11qe:A', '11qe', 'A', 'protein')
ON CONFLICT (chain_id) DO NOTHING;

INSERT INTO dim_residue (residue_id, chain_id, residue_index, residue_name, residue_name_3)
VALUES ('11qe:A:1', '11qe:A', 1, 'ALA', 'ALA')
ON CONFLICT (residue_id) DO NOTHING;

INSERT INTO provenance_run (
    run_id,
    structure_id,
    model_version,
    pipeline_name,
    run_type,
    source_type,
    parameters,
    completed_at
) VALUES (
    'p_curv01_ci_run',
    '11qe',
    'GOSPConeMapper-v6',
    'p_curv01_ci',
    'inference',
    'deterministic',
    '{}'::jsonb,
    NOW()
)
ON CONFLICT (run_id) DO NOTHING;

INSERT INTO fact_gnn_node_embedding (
    embedding_id,
    run_id,
    structure_id,
    residue_id,
    space_id,
    embedding,
    source_type,
    model_version
) VALUES (
    'p_curv01_emb_1',
    'p_curv01_ci_run',
    '11qe',
    '11qe:A:1',
    'space_gospconemapper_v6_hyp128',
    (SELECT ('[' || repeat('0,', 127) || '0]')::vector),
    'deterministic',
    'GOSPConeMapper-v6'
)
ON CONFLICT (embedding_id) DO NOTHING;
