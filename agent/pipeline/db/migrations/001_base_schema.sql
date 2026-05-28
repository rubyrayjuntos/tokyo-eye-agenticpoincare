-- GOSP Data Warehouse Schema (PostgreSQL DDL)
-- Generated from data governance audit
-- Date: 2026-02-15

-- ============================================================================
-- DIMENSION TABLES (Core Entities)
-- ============================================================================

CREATE TABLE dim_structure (
    structure_id TEXT PRIMARY KEY,
    pdb_id TEXT NOT NULL UNIQUE,
    resolution DOUBLE PRECISION,
    num_models INTEGER,
    total_residues INTEGER,
    total_atoms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_structure_pdb_id ON dim_structure(pdb_id);

CREATE TABLE dim_chain (
    chain_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    chain_label TEXT NOT NULL,
    sequence TEXT NOT NULL
);

CREATE INDEX idx_chain_structure_id ON dim_chain(structure_id);

CREATE TABLE dim_residue (
    residue_id TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL REFERENCES dim_chain(chain_id),
    residue_index INTEGER NOT NULL,
    residue_name TEXT NOT NULL,
    sse_code TEXT
);

CREATE INDEX idx_residue_chain_id ON dim_residue(chain_id);

CREATE TABLE dim_atom (
    atom_id TEXT PRIMARY KEY,
    residue_id TEXT NOT NULL REFERENCES dim_residue(residue_id),
    atom_name TEXT NOT NULL,
    element TEXT NOT NULL,
    occupancy DOUBLE PRECISION,
    b_factor DOUBLE PRECISION,
    x DOUBLE PRECISION NOT NULL,
    y DOUBLE PRECISION NOT NULL,
    z DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_atom_residue_id ON dim_atom(residue_id);

CREATE TABLE dim_validation_outlier (
    outlier_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    chain_label TEXT NOT NULL,
    residue_index INTEGER NOT NULL,
    residue_type TEXT,
    atom_name_1 TEXT,
    atom_name_2 TEXT,
    outlier_type TEXT NOT NULL,  -- length | angle
    z_score DOUBLE PRECISION,
    observed DOUBLE PRECISION,
    ideal DOUBLE PRECISION,
    deviation DOUBLE PRECISION
);

CREATE INDEX idx_validation_outlier_structure ON dim_validation_outlier(structure_id);

CREATE TABLE dim_clash (
    clash_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    atom_1 TEXT NOT NULL,
    atom_2 TEXT NOT NULL,
    distance DOUBLE PRECISION,
    clash_magnitude DOUBLE PRECISION
);

CREATE INDEX idx_clash_structure ON dim_clash(structure_id);

CREATE TABLE dim_sequence (
    sequence_id TEXT PRIMARY KEY,
    sequence TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL  -- manual | structure
);

-- ============================================================================
-- FACT TABLES (Pipeline Results & Measurements)
-- ============================================================================

CREATE TABLE fact_validation_mining_run (
    run_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    chain_label TEXT,
    focus_residues TEXT,
    status TEXT NOT NULL,  -- success | partial | error
    rho_threshold INTEGER,
    correlation_radius DOUBLE PRECISION,
    top_n_correlations INTEGER,
    min_dehydrons_per_site INTEGER,
    warnings JSONB,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    duration_sec DOUBLE PRECISION
);

CREATE INDEX idx_validation_mining_run_structure ON fact_validation_mining_run(structure_id);
CREATE INDEX idx_validation_mining_run_status ON fact_validation_mining_run(status);

CREATE TABLE fact_dehydron (
    dehydron_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES fact_validation_mining_run(run_id),
    donor_chain TEXT NOT NULL,
    donor_residue_index INTEGER NOT NULL,
    acceptor_chain TEXT NOT NULL,
    acceptor_residue_index INTEGER NOT NULL,
    midpoint_x DOUBLE PRECISION NOT NULL,
    midpoint_y DOUBLE PRECISION NOT NULL,
    midpoint_z DOUBLE PRECISION NOT NULL,
    distance DOUBLE PRECISION NOT NULL,
    wrapping_count INTEGER NOT NULL,
    is_dehydron BOOLEAN NOT NULL,
    is_interchain BOOLEAN NOT NULL
);

CREATE INDEX idx_dehydron_run_id ON fact_dehydron(run_id);
CREATE INDEX idx_dehydron_wrapping ON fact_dehydron(wrapping_count);

CREATE TABLE fact_void (
    void_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES fact_validation_mining_run(run_id),
    center_x DOUBLE PRECISION NOT NULL,
    center_y DOUBLE PRECISION NOT NULL,
    center_z DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    point_count INTEGER NOT NULL
);

CREATE INDEX idx_void_run_id ON fact_void(run_id);
CREATE INDEX idx_void_volume ON fact_void(volume);

CREATE TABLE bridge_void_dehydron (
    void_id TEXT NOT NULL REFERENCES fact_void(void_id),
    dehydron_id TEXT NOT NULL REFERENCES fact_dehydron(dehydron_id),
    PRIMARY KEY (void_id, dehydron_id)
);

CREATE INDEX idx_bridge_void_dehydron_dehydron ON bridge_void_dehydron(dehydron_id);

CREATE TABLE fact_outlier_correlation (
    correlation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES fact_validation_mining_run(run_id),
    outlier_id TEXT NOT NULL REFERENCES dim_validation_outlier(outlier_id),
    dehydron_id TEXT NOT NULL REFERENCES fact_dehydron(dehydron_id),
    distance DOUBLE PRECISION NOT NULL,
    correlation_score DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_outlier_correlation_run ON fact_outlier_correlation(run_id);
CREATE INDEX idx_outlier_correlation_score ON fact_outlier_correlation(correlation_score DESC);

CREATE TABLE fact_glue_site (
    glue_site_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES fact_validation_mining_run(run_id),
    avg_rho DOUBLE PRECISION NOT NULL,
    void_volume_est DOUBLE PRECISION,
    score DOUBLE PRECISION NOT NULL,
    representative_label TEXT
);

CREATE INDEX idx_glue_site_run ON fact_glue_site(run_id);
CREATE INDEX idx_glue_site_score ON fact_glue_site(score DESC);

CREATE TABLE bridge_glue_site_dehydron (
    glue_site_id TEXT NOT NULL REFERENCES fact_glue_site(glue_site_id),
    dehydron_id TEXT NOT NULL REFERENCES fact_dehydron(dehydron_id),
    PRIMARY KEY (glue_site_id, dehydron_id)
);

CREATE INDEX idx_bridge_glue_dehydron ON bridge_glue_site_dehydron(dehydron_id);

CREATE TABLE fact_wrapper_suggestion (
    wrapper_suggestion_id TEXT PRIMARY KEY,
    glue_site_id TEXT NOT NULL REFERENCES fact_glue_site(glue_site_id),
    wrapper_type TEXT NOT NULL,
    wrapping_gain INTEGER,
    predicted_ddg DOUBLE PRECISION
);

CREATE INDEX idx_wrapper_glue_site ON fact_wrapper_suggestion(glue_site_id);

-- ============================================================================
-- PHYSICS & ENERGY CALCULATIONS
-- ============================================================================

CREATE TABLE fact_energy_calculation (
    energy_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    requested_force_field TEXT NOT NULL,
    actual_method TEXT NOT NULL,
    delta_g DOUBLE PRECISION NOT NULL,
    potential_energy DOUBLE PRECISION NOT NULL,
    solvation_term DOUBLE PRECISION NOT NULL,
    sasa DOUBLE PRECISION NOT NULL,
    converged BOOLEAN NOT NULL,
    confidence TEXT NOT NULL,
    duration_sec DOUBLE PRECISION,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_energy_structure ON fact_energy_calculation(structure_id);
CREATE INDEX idx_energy_computed_at ON fact_energy_calculation(computed_at);

CREATE TABLE fact_energy_provenance_step (
    step_id TEXT PRIMARY KEY,
    energy_id TEXT NOT NULL REFERENCES fact_energy_calculation(energy_id),
    method TEXT NOT NULL,
    status TEXT NOT NULL,  -- success | failed | skipped
    duration_sec DOUBLE PRECISION,
    inputs JSONB,
    outputs JSONB,
    reason TEXT,
    approximations JSONB
);

CREATE INDEX idx_provenance_step_energy ON fact_energy_provenance_step(energy_id);

-- ============================================================================
-- EMPIRICAL VALIDATION (HDX-MS)
-- ============================================================================

CREATE TABLE fact_hdx_correlation (
    hdx_run_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    r_squared DOUBLE PRECISION NOT NULL,
    spearman_rho DOUBLE PRECISION NOT NULL,
    p_value DOUBLE PRECISION NOT NULL,
    passed BOOLEAN NOT NULL,
    num_residues INTEGER NOT NULL,
    duration_sec DOUBLE PRECISION,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hdx_structure ON fact_hdx_correlation(structure_id);
CREATE INDEX idx_hdx_passed ON fact_hdx_correlation(passed);

CREATE TABLE fact_hdx_residue (
    hdx_residue_id TEXT PRIMARY KEY,
    hdx_run_id TEXT NOT NULL REFERENCES fact_hdx_correlation(hdx_run_id),
    residue_index INTEGER NOT NULL,
    fractional_uptake DOUBLE PRECISION NOT NULL,
    wrapping_count INTEGER,
    num_peptides INTEGER
);

CREATE INDEX idx_hdx_residue_run ON fact_hdx_residue(hdx_run_id);

-- ============================================================================
-- RED ZONE SCREENING (Immunogenicity & Metabolism)
-- ============================================================================

CREATE TABLE fact_immunogenicity_run (
    immuno_run_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    alleles JSONB,
    ic50_threshold DOUBLE PRECISION,
    sasa_threshold DOUBLE PRECISION,
    peptide_length INTEGER,
    total_peptides INTEGER,
    flagged_count INTEGER,
    warning TEXT,
    duration_sec DOUBLE PRECISION,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_immunogenicity_structure ON fact_immunogenicity_run(structure_id);

CREATE TABLE fact_immunogenic_epitope (
    epitope_id TEXT PRIMARY KEY,
    immuno_run_id TEXT NOT NULL REFERENCES fact_immunogenicity_run(immuno_run_id),
    sequence TEXT NOT NULL,
    start_res INTEGER NOT NULL,
    end_res INTEGER NOT NULL,
    allele TEXT NOT NULL,
    ic50 DOUBLE PRECISION NOT NULL,
    sasa_percent DOUBLE PRECISION,
    flagged BOOLEAN NOT NULL
);

CREATE INDEX idx_epitope_immuno_run ON fact_immunogenic_epitope(immuno_run_id);
CREATE INDEX idx_epitope_flagged ON fact_immunogenic_epitope(flagged);

CREATE TABLE fact_metabolism_run (
    metabolism_run_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES dim_structure(structure_id),
    isoform TEXT NOT NULL,
    sasa_threshold DOUBLE PRECISION,
    flag_threshold INTEGER,
    som_count INTEGER,
    flagged BOOLEAN NOT NULL,
    warning TEXT,
    duration_sec DOUBLE PRECISION,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metabolism_structure ON fact_metabolism_run(structure_id);

CREATE TABLE fact_metabolism_site (
    som_id TEXT PRIMARY KEY,
    metabolism_run_id TEXT NOT NULL REFERENCES fact_metabolism_run(metabolism_run_id),
    atom_idx INTEGER NOT NULL,
    atom_type TEXT,
    pattern TEXT,
    coord_x DOUBLE PRECISION,
    coord_y DOUBLE PRECISION,
    coord_z DOUBLE PRECISION,
    sasa DOUBLE PRECISION,
    is_exposed BOOLEAN
);

CREATE INDEX idx_metabolism_site_run ON fact_metabolism_site(metabolism_run_id);

-- ============================================================================
-- SYNTHESIS & AUTOPROTOCOL
-- ============================================================================

CREATE TABLE fact_synthesis_feasibility (
    synthesis_id TEXT PRIMARY KEY,
    sequence_id TEXT NOT NULL REFERENCES dim_sequence(sequence_id),
    manufacturable BOOLEAN NOT NULL,
    complexity_score DOUBLE PRECISION,
    issues JSONB,
    gc_content DOUBLE PRECISION,
    aa_length INTEGER,
    dna_length INTEGER,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_synthesis_sequence ON fact_synthesis_feasibility(sequence_id);
CREATE INDEX idx_synthesis_manufacturable ON fact_synthesis_feasibility(manufacturable);

CREATE TABLE fact_autoprotocol (
    protocol_id TEXT PRIMARY KEY,
    sequence_id TEXT NOT NULL REFERENCES dim_sequence(sequence_id),
    instruction_count INTEGER,
    schema_valid BOOLEAN,
    protocol_json JSONB,
    generated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_autoprotocol_sequence ON fact_autoprotocol(sequence_id);

-- ============================================================================
-- AUDIT TRAIL
-- ============================================================================

CREATE TABLE fact_audit_event (
    event_id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    state_from TEXT NOT NULL,
    state_to TEXT NOT NULL,
    event_name TEXT NOT NULL,
    calc_values JSONB,
    user_id TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    prev_hash TEXT
);

CREATE INDEX idx_audit_timestamp ON fact_audit_event(timestamp DESC);
CREATE INDEX idx_audit_user_id ON fact_audit_event(user_id);
CREATE INDEX idx_audit_state ON fact_audit_event(state_from, state_to);

-- ============================================================================
-- CONSTRAINTS & CHECKS
-- ============================================================================

ALTER TABLE dim_structure
    ADD CONSTRAINT check_resolution_positive CHECK (resolution IS NULL OR resolution > 0);

ALTER TABLE dim_chain
    ADD CONSTRAINT check_chain_label_not_empty CHECK (chain_label != '');

ALTER TABLE fact_dehydron
    ADD CONSTRAINT check_wrapping_count_non_negative CHECK (wrapping_count >= 0);

ALTER TABLE fact_void
    ADD CONSTRAINT check_void_volume_positive CHECK (volume > 0);

ALTER TABLE fact_energy_calculation
    ADD CONSTRAINT check_sasa_non_negative CHECK (sasa >= 0);

ALTER TABLE fact_hdx_correlation
    ADD CONSTRAINT check_r_squared_range CHECK (r_squared >= 0 AND r_squared <= 1);

ALTER TABLE fact_hdx_correlation
    ADD CONSTRAINT check_p_value_range CHECK (p_value >= 0 AND p_value <= 1);

ALTER TABLE fact_immunogenic_epitope
    ADD CONSTRAINT check_ic50_positive CHECK (ic50 > 0);

ALTER TABLE fact_metabolism_site
    ADD CONSTRAINT check_sasa_range CHECK (sasa IS NULL OR (sasa >= 0 AND sasa <= 100));

-- ============================================================================
-- VIEWS (Useful Aggregations & Joins)
-- ============================================================================

CREATE VIEW v_validation_mining_summary AS
SELECT
    vmr.run_id,
    ds.pdb_id,
    vmr.status,
    COUNT(DISTINCT fd.dehydron_id) as dehydron_count,
    COUNT(DISTINCT fv.void_id) as void_count,
    COUNT(DISTINCT fgs.glue_site_id) as glue_site_count,
    vmr.duration_sec,
    vmr.started_at
FROM fact_validation_mining_run vmr
JOIN dim_structure ds ON vmr.structure_id = ds.structure_id
LEFT JOIN fact_dehydron fd ON vmr.run_id = fd.run_id
LEFT JOIN fact_void fv ON vmr.run_id = fv.run_id
LEFT JOIN fact_glue_site fgs ON vmr.run_id = fgs.run_id
GROUP BY vmr.run_id, ds.pdb_id, vmr.status, vmr.duration_sec, vmr.started_at;

CREATE VIEW v_glue_site_dehydrons AS
SELECT
    fgs.glue_site_id,
    fgs.run_id,
    fgs.score,
    fgs.avg_rho,
    COUNT(bgsd.dehydron_id) as num_dehydrons,
    STRING_AGG(
        CONCAT(fd.donor_chain, fd.donor_residue_index, '-', fd.acceptor_chain, fd.acceptor_residue_index),
        ','
    ) as dehydron_labels
FROM fact_glue_site fgs
LEFT JOIN bridge_glue_site_dehydron bgsd ON fgs.glue_site_id = bgsd.glue_site_id
LEFT JOIN fact_dehydron fd ON bgsd.dehydron_id = fd.dehydron_id
GROUP BY fgs.glue_site_id, fgs.run_id, fgs.score, fgs.avg_rho;

CREATE VIEW v_structure_energy_summary AS
SELECT
    ds.pdb_id,
    fec.requested_force_field,
    fec.actual_method,
    fec.delta_g,
    fec.potential_energy,
    fec.sasa,
    fec.confidence,
    fec.computed_at
FROM fact_energy_calculation fec
JOIN dim_structure ds ON fec.structure_id = ds.structure_id;

CREATE VIEW v_red_zone_flags AS
SELECT
    ds.pdb_id,
    'immunogenicity' as flag_type,
    COUNT(fie.epitope_id) as flag_count,
    MIN(fie.ic50) as min_ic50
FROM fact_immunogenic_epitope fie
JOIN fact_immunogenicity_run fir ON fie.immuno_run_id = fir.immuno_run_id
JOIN dim_structure ds ON fir.structure_id = ds.structure_id
WHERE fie.flagged = TRUE
GROUP BY ds.pdb_id
UNION ALL
SELECT
    ds.pdb_id,
    'metabolism' as flag_type,
    COUNT(fms.som_id) as flag_count,
    NULL as min_ic50
FROM fact_metabolism_site fms
JOIN fact_metabolism_run fmr ON fms.metabolism_run_id = fmr.metabolism_run_id
JOIN dim_structure ds ON fmr.structure_id = ds.structure_id
WHERE fms.is_exposed = TRUE
GROUP BY ds.pdb_id;
