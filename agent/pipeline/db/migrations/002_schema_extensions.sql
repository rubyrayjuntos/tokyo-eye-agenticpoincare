-- 002_schema_extensions.sql
-- Extensions to base governance schema per data-layer-design spec §7

-- ============================================================
-- SECTION 7a: Fix FKs + add source_type to existing fact tables
-- ============================================================

-- fact_dehydron: add direct structure link, make run_id nullable
ALTER TABLE fact_dehydron
    ADD COLUMN structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    ADD COLUMN source_type  TEXT NOT NULL DEFAULT 'deterministic',
    ALTER COLUMN run_id DROP NOT NULL;

CREATE INDEX idx_dehydron_structure ON fact_dehydron(structure_id);

-- fact_void: same treatment
ALTER TABLE fact_void
    ADD COLUMN structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    ADD COLUMN source_type  TEXT NOT NULL DEFAULT 'deterministic',
    ALTER COLUMN run_id DROP NOT NULL;

CREATE INDEX idx_void_structure ON fact_void(structure_id);

-- fact_synthesis_feasibility: add structure_id FK
ALTER TABLE fact_synthesis_feasibility
    ADD COLUMN structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    ADD COLUMN source_type  TEXT NOT NULL DEFAULT 'external';

CREATE INDEX idx_synthesis_structure ON fact_synthesis_feasibility(structure_id);

-- fact_autoprotocol: add structure_id FK
ALTER TABLE fact_autoprotocol
    ADD COLUMN structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    ADD COLUMN source_type  TEXT NOT NULL DEFAULT 'deterministic';

CREATE INDEX idx_autoprotocol_structure ON fact_autoprotocol(structure_id);

-- fact_audit_event: add nullable structure_id (may exist before structure resolved)
ALTER TABLE fact_audit_event
    ADD COLUMN structure_id TEXT REFERENCES dim_structure(structure_id);

CREATE INDEX idx_audit_structure ON fact_audit_event(structure_id);

-- dim_residue: add per-residue SASA (populated by Tier 1 FreeSASA step)
ALTER TABLE dim_residue ADD COLUMN sasa DOUBLE PRECISION;

-- Remaining source_type additions
ALTER TABLE fact_energy_calculation    ADD COLUMN source_type TEXT NOT NULL DEFAULT 'deterministic';
ALTER TABLE fact_hdx_correlation       ADD COLUMN source_type TEXT NOT NULL DEFAULT 'empirical';
ALTER TABLE fact_immunogenicity_run    ADD COLUMN source_type TEXT NOT NULL DEFAULT 'external';
ALTER TABLE fact_metabolism_run        ADD COLUMN source_type TEXT NOT NULL DEFAULT 'deterministic';

-- Fix v_validation_mining_summary: join via structure_id (run_id now nullable on dehydron/void)
CREATE OR REPLACE VIEW v_validation_mining_summary AS
SELECT
    vmr.run_id,
    ds.pdb_id,
    vmr.status,
    COUNT(DISTINCT fd.dehydron_id)   AS dehydron_count,
    COUNT(DISTINCT fv.void_id)       AS void_count,
    COUNT(DISTINCT fgs.glue_site_id) AS glue_site_count,
    vmr.duration_sec,
    vmr.started_at
FROM fact_validation_mining_run vmr
JOIN  dim_structure ds        ON vmr.structure_id  = ds.structure_id
LEFT JOIN fact_dehydron fd    ON vmr.structure_id  = fd.structure_id
LEFT JOIN fact_void fv        ON vmr.structure_id  = fv.structure_id
LEFT JOIN fact_glue_site fgs  ON vmr.run_id        = fgs.run_id
GROUP BY vmr.run_id, ds.pdb_id, vmr.status, vmr.duration_sec, vmr.started_at;

-- ============================================================
-- SECTION 7b: Job status table
-- ============================================================

CREATE TABLE fact_job_status (
    job_status_id   TEXT PRIMARY KEY,
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id),
    job_type        TEXT NOT NULL,
    -- job_type values: fast_path | void | cdd | immunogenicity | metabolism |
    --                  lerp | validation_mining | synthesis | autoprotocol |
    --                  hdx | gnn_inference
    status          TEXT NOT NULL,
    -- status values: queued | running | complete | failed
    error_message   TEXT,
    queued_at       TIMESTAMPTZ,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    cloud_task_name TEXT
);

CREATE INDEX idx_job_status_structure ON fact_job_status(structure_id);
CREATE INDEX idx_job_status_type      ON fact_job_status(structure_id, job_type);

-- ============================================================
-- SECTION 7c: GNN output tables
-- ============================================================

CREATE TABLE fact_gnn_inference (
    gnn_run_id                 TEXT PRIMARY KEY,
    structure_id               TEXT NOT NULL REFERENCES dim_structure(structure_id),
    model_version              TEXT NOT NULL,
    projection_dim             INTEGER NOT NULL,
    num_nodes                  INTEGER NOT NULL,
    num_edges                  INTEGER NOT NULL,
    curvature_value            DOUBLE PRECISION,
    depth_conditioning_enabled BOOLEAN,
    balance_loss               DOUBLE PRECISION,
    audit_trail                JSONB,
    sasa_null_warnings         JSONB,
    source_type                TEXT NOT NULL DEFAULT 'probabilistic',
    computed_at                TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_gnn_inference_structure ON fact_gnn_inference(structure_id);

CREATE TABLE fact_gnn_node_output (
    node_id               TEXT PRIMARY KEY,
    gnn_run_id            TEXT NOT NULL REFERENCES fact_gnn_inference(gnn_run_id),
    residue_id            TEXT NOT NULL REFERENCES dim_residue(residue_id),
    -- Input features (provenance: what was sent to the GNN)
    input_rho             DOUBLE PRECISION NOT NULL,
    input_tau_flag        DOUBLE PRECISION NOT NULL,
    input_ss_type         DOUBLE PRECISION NOT NULL,
    input_sasa            DOUBLE PRECISION NOT NULL,
    -- Poincare disc outputs
    projections           JSONB NOT NULL,
    cone_depth            DOUBLE PRECISION NOT NULL,
    cone_width            DOUBLE PRECISION NOT NULL,
    -- Expert routing
    expert_weights        JSONB NOT NULL,
    -- Evidential uncertainty
    epistemic_uncertainty DOUBLE PRECISION,
    aleatoric_uncertainty DOUBLE PRECISION,
    total_uncertainty     DOUBLE PRECISION,
    -- NIG evidence parameters
    evidence_mu           DOUBLE PRECISION,
    evidence_nu           DOUBLE PRECISION,
    evidence_alpha        DOUBLE PRECISION,
    evidence_beta         DOUBLE PRECISION,
    source_type           TEXT NOT NULL DEFAULT 'probabilistic'
);

CREATE INDEX idx_gnn_node_run     ON fact_gnn_node_output(gnn_run_id);
CREATE INDEX idx_gnn_node_residue ON fact_gnn_node_output(residue_id);

-- ============================================================
-- SECTION 7d: CDD annotation table
-- ============================================================

CREATE TABLE fact_cdd_annotation (
    annotation_id  TEXT PRIMARY KEY,
    structure_id   TEXT NOT NULL REFERENCES dim_structure(structure_id),
    chain_label    TEXT NOT NULL,
    domain_id      TEXT NOT NULL,
    domain_name    TEXT NOT NULL,
    start_residue  INTEGER NOT NULL,
    end_residue    INTEGER NOT NULL,
    e_value        DOUBLE PRECISION,
    bit_score      DOUBLE PRECISION,
    is_synthetic   BOOLEAN NOT NULL DEFAULT FALSE,
    source_type    TEXT NOT NULL DEFAULT 'external',
    computed_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cdd_annotation_structure ON fact_cdd_annotation(structure_id);

-- ============================================================
-- SECTION 7e: LERP folding path
-- ============================================================

CREATE TABLE fact_folding_path (
    path_id      TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    method       TEXT NOT NULL DEFAULT 'lerp',
    num_frames   INTEGER NOT NULL,
    num_atoms    INTEGER NOT NULL,
    gcs_uri      TEXT NOT NULL,
    source_type  TEXT NOT NULL DEFAULT 'deterministic',
    computed_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_folding_path_structure ON fact_folding_path(structure_id);
