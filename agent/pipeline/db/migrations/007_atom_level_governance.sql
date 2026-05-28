-- ============================================================================
-- Migration 007: Atom-Level Governance
--
-- Three changes:
--   1. Add fact_atom_feature — first fact table at atom grain, making
--      dim_atom a live analytical anchor rather than a static spatial ref.
--   2. Fix dim_clash — add proper FK columns alongside the legacy TEXT
--      atom labels (table has no write path yet so no data migration needed).
--   3. Fix fact_metabolism_site — add atom_id FK alongside the legacy
--      atom_idx integer; backfill existing rows by coordinate matching.
-- ============================================================================


-- ============================================================================
-- 1. fact_atom_feature
--    Generic per-atom fact table. Keyed to dim_atom so every row is
--    traceable to a specific atom in a specific structure.
--    run_type / feature_type pair identifies what computed the value:
--      ('metabolism',   'som')           — site of metabolism
--      ('sasa',         'sasa_percent')  — per-atom solvent exposure
--      ('dehydron',     'apolar_contact')— apolar neighbour contributing to rho
--      ('pharmacophore','h_donor')       — pharmacophore constraint atom
--      ('clash',        'clash_partner') — atom in a steric clash
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_atom_feature (
    feature_id    TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    atom_id       TEXT        NOT NULL REFERENCES dim_atom(atom_id),
    run_id        TEXT        NOT NULL,
    run_type      TEXT        NOT NULL,
    feature_type  TEXT        NOT NULL,
    value         DOUBLE PRECISION,
    bool_value    BOOLEAN,
    metadata      JSONB,
    computed_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_atom_feature_atom
    ON fact_atom_feature(atom_id);

CREATE INDEX IF NOT EXISTS idx_atom_feature_run
    ON fact_atom_feature(run_id);

CREATE INDEX IF NOT EXISTS idx_atom_feature_type
    ON fact_atom_feature(run_type, feature_type);

COMMENT ON TABLE fact_atom_feature IS
    'Per-atom computational outputs. Finest grain in the fact layer. '
    'run_type + feature_type identify the pipeline phase and measurement.';


-- ============================================================================
-- 2. dim_clash — add FK columns
--    atom_1 / atom_2 TEXT columns are retained for display and for clash
--    records originating from external validation tools that may not map
--    cleanly to a dim_atom row (e.g. hydrogens not modelled in dim_atom).
--    atom_1_id / atom_2_id are the authoritative FK columns; populate them
--    when the write path for dim_clash is built.
-- ============================================================================

ALTER TABLE dim_clash
    ADD COLUMN IF NOT EXISTS atom_1_id TEXT REFERENCES dim_atom(atom_id),
    ADD COLUMN IF NOT EXISTS atom_2_id TEXT REFERENCES dim_atom(atom_id);

CREATE INDEX IF NOT EXISTS idx_clash_atom_1
    ON dim_clash(atom_1_id) WHERE atom_1_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_clash_atom_2
    ON dim_clash(atom_2_id) WHERE atom_2_id IS NOT NULL;

COMMENT ON COLUMN dim_clash.atom_1 IS
    'Deprecated display label. Use atom_1_id FK where available.';
COMMENT ON COLUMN dim_clash.atom_2 IS
    'Deprecated display label. Use atom_2_id FK where available.';


-- ============================================================================
-- 3. fact_metabolism_site — add atom_id FK
--    atom_idx INTEGER is retained as the backfill anchor and for
--    downstream callers that have not yet been updated to provide atom_id.
-- ============================================================================

ALTER TABLE fact_metabolism_site
    ADD COLUMN IF NOT EXISTS atom_id TEXT REFERENCES dim_atom(atom_id);

CREATE INDEX IF NOT EXISTS idx_metabolism_site_atom
    ON fact_metabolism_site(atom_id) WHERE atom_id IS NOT NULL;

COMMENT ON COLUMN fact_metabolism_site.atom_idx IS
    'Deprecated biotite array index. Use atom_id FK. '
    'Retained as backfill anchor until all rows carry a resolved atom_id.';

-- Backfill: resolve atom_id by matching stored coordinates to dim_atom.
-- Coordinates are stored to 3 decimal places in the PDB standard; rounding
-- to 3dp avoids floating-point equality mismatches.
-- Rows where coord_x IS NULL (the legacy simplified write path) are skipped
-- and will be resolved when re-processed by the updated write path.
UPDATE fact_metabolism_site fms
SET atom_id = (
    SELECT da.atom_id
    FROM fact_metabolism_run   fmr
    JOIN dim_chain              dc  ON dc.structure_id  = fmr.structure_id
    JOIN dim_residue            dr  ON dr.chain_id      = dc.chain_id
    JOIN dim_atom               da  ON da.residue_id    = dr.residue_id
                                   AND ROUND(da.x::numeric, 3) = ROUND(fms.coord_x::numeric, 3)
                                   AND ROUND(da.y::numeric, 3) = ROUND(fms.coord_y::numeric, 3)
                                   AND ROUND(da.z::numeric, 3) = ROUND(fms.coord_z::numeric, 3)
    WHERE fmr.metabolism_run_id = fms.metabolism_run_id
    LIMIT 1
)
WHERE fms.atom_id   IS NULL
  AND fms.coord_x   IS NOT NULL;
