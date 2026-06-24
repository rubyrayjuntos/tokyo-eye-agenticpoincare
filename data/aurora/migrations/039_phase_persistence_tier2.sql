-- ============================================================================
-- Migration 039: Phase Persistence Tier 2 Fact Tables
-- Date: 2026-05-30
-- Purpose: Dedicated fact tables for phases 2, 3.5, 4, 5, and 6 outputs.
--          Closes the persistence gap so all computed scientific data flows
--          to governed fact tables with provenance linkage.
-- Requirements: 1.2, 2.2, 3.2, 4.2, 5.2
-- ============================================================================

-- ============================================================================
-- FACT TABLE: Phase 2 Vulnerability Doorways (per-residue)
-- Stores residues identified as structural vulnerabilities at the boundary
-- of the hyperbolic cone with epistemic uncertainty metrics.
-- Natural key: (run_id, residue_id) — one record per residue per run.
-- Requirements: 1.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_phase2_vulnerability (
    vulnerability_id        TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    residue_id              TEXT NOT NULL REFERENCES dim_residue(residue_id),
    cone_depth              DOUBLE PRECISION NOT NULL,
    epistemic_uncertainty   DOUBLE PRECISION NOT NULL,
    aleatoric_uncertainty   DOUBLE PRECISION DEFAULT 0.0,
    depth_threshold         DOUBLE PRECISION,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_phase2_natural
    ON fact_phase2_vulnerability(run_id, residue_id);
CREATE INDEX IF NOT EXISTS idx_phase2_structure
    ON fact_phase2_vulnerability(structure_id);
CREATE INDEX IF NOT EXISTS idx_phase2_run
    ON fact_phase2_vulnerability(run_id);
CREATE INDEX IF NOT EXISTS idx_phase2_residue
    ON fact_phase2_vulnerability(residue_id);

-- ============================================================================
-- FACT TABLE: Phase 3.5 Topological Lift
-- Stores lifted allosteric site coordinates from 2D disc to 3D ball.
-- Natural key: (run_id, site_index) — one record per site per run.
-- Requirements: 2.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_topological_lift (
    lift_id                 TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    site_index              INTEGER NOT NULL,
    lifted_x                DOUBLE PRECISION,
    lifted_y                DOUBLE PRECISION,
    lifted_z                DOUBLE PRECISION,
    vertex_count            INTEGER,
    source_method           TEXT,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_lift_natural
    ON fact_topological_lift(run_id, site_index);
CREATE INDEX IF NOT EXISTS idx_lift_structure
    ON fact_topological_lift(structure_id);
CREATE INDEX IF NOT EXISTS idx_lift_run
    ON fact_topological_lift(run_id);

-- ============================================================================
-- FACT TABLE: Phase 4 Resistance Pathways (per-edge)
-- Stores resistance pathways through the contact graph.
-- Natural key: (run_id, source_node, target_node) — one record per edge per run.
-- Requirements: 3.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_resistance_pathway (
    pathway_id              TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    source_node             INTEGER NOT NULL,
    target_node             INTEGER NOT NULL,
    source_residue          INTEGER NOT NULL,
    target_residue          INTEGER NOT NULL,
    effective_resistance    DOUBLE PRECISION NOT NULL,
    coupling_strength       DOUBLE PRECISION NOT NULL,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pathway_natural
    ON fact_resistance_pathway(run_id, source_node, target_node);
CREATE INDEX IF NOT EXISTS idx_pathway_structure
    ON fact_resistance_pathway(structure_id);
CREATE INDEX IF NOT EXISTS idx_pathway_run
    ON fact_resistance_pathway(run_id);

-- ============================================================================
-- FACT TABLE: Phase 4 Resistance Spectral Summary (one row per run)
-- Stores spectral analysis summary: algebraic connectivity, hinge residues.
-- Natural key: (run_id) — one record per run.
-- Requirements: 3.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_resistance_spectral (
    spectral_id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    lambda_2                DOUBLE PRECISION,
    hinge_residues          JSONB,
    graph_nodes             INTEGER,
    graph_edges             INTEGER,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_spectral_natural
    ON fact_resistance_spectral(run_id);
CREATE INDEX IF NOT EXISTS idx_spectral_structure
    ON fact_resistance_spectral(structure_id);
CREATE INDEX IF NOT EXISTS idx_spectral_run
    ON fact_resistance_spectral(run_id);

-- ============================================================================
-- FACT TABLE: Phase 5 Pharmacophore
-- Stores druggable pharmacophore features from cone geometry.
-- Natural key: (run_id, pocket_index) — one record per pocket per run.
-- Requirements: 4.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_pharmacophore (
    pharmacophore_id        TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    pocket_index            INTEGER NOT NULL,
    center_x                DOUBLE PRECISION,
    center_y                DOUBLE PRECISION,
    center_z                DOUBLE PRECISION,
    druggability_score      DOUBLE PRECISION,
    residue_count           INTEGER,
    residue_indices         JSONB,
    allosteric_coupling     DOUBLE PRECISION DEFAULT 0.0,
    volume_estimate         DOUBLE PRECISION DEFAULT 0.0,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pharmacophore_natural
    ON fact_pharmacophore(run_id, pocket_index);
CREATE INDEX IF NOT EXISTS idx_pharmacophore_structure
    ON fact_pharmacophore(structure_id);
CREATE INDEX IF NOT EXISTS idx_pharmacophore_run
    ON fact_pharmacophore(run_id);

-- ============================================================================
-- FACT TABLE: Phase 6 Drug Candidates
-- Stores scored pockets with ADMET filtering and state-selectivity.
-- Natural key: (run_id, pocket_index) — one record per pocket per run.
-- Requirements: 5.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS fact_drug_candidate (
    candidate_id            TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    run_id                  TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id            TEXT NOT NULL REFERENCES dim_structure(structure_id),
    pocket_index            INTEGER NOT NULL,
    center_x                DOUBLE PRECISION,
    center_y                DOUBLE PRECISION,
    center_z                DOUBLE PRECISION,
    accessibility_score     DOUBLE PRECISION,
    binding_potential       DOUBLE PRECISION,
    admet_pass              BOOLEAN,
    selectivity_ratio       DOUBLE PRECISION,
    is_state_selective      BOOLEAN,
    combined_druggability   DOUBLE PRECISION,
    computed_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_natural
    ON fact_drug_candidate(run_id, pocket_index);
CREATE INDEX IF NOT EXISTS idx_candidate_structure
    ON fact_drug_candidate(structure_id);
CREATE INDEX IF NOT EXISTS idx_candidate_run
    ON fact_drug_candidate(run_id);

-- ============================================================================
-- NOTES
-- ============================================================================
-- - All tables use ON CONFLICT (natural key) DO UPDATE at the application
--   layer for idempotent upserts.
-- - All tables reference provenance_run(run_id) for full traceability.
-- - Phase 3.5 is Tier 2 (non-blocking); all others are Tier 1.
-- - fact_resistance_spectral stores one summary row per run; pathway details
--   are in fact_resistance_pathway (one row per edge).
