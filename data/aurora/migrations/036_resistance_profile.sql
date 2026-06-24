-- Migration 036: Resistance Profile fact table
-- Stores classification results from the Resistance Profiler module.
-- Each row represents one virtual mutation profiling result.

CREATE TABLE IF NOT EXISTS fact_resistance_profile (
    profile_id      TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES provenance_run(run_id),
    structure_id    TEXT NOT NULL,
    variant         TEXT NOT NULL,           -- e.g., "T315I"
    mechanism_class TEXT NOT NULL,           -- "Type_I_Steric", "Type_II_Allosteric", "Hybrid"
    confidence_score DOUBLE PRECISION NOT NULL,
    site_uncertainty_delta DOUBLE PRECISION,
    max_hub_delta   DOUBLE PRECISION,
    propagation_radius INTEGER,
    metrics         JSONB,                  -- Full metrics dict
    hub_details     JSONB,                  -- Per-hub comparison data
    affected_pathways JSONB,                -- List of affected pathway strings
    structural_impact TEXT,
    error           TEXT,                   -- NULL on success, error message on failure
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_resistance_profile_run_id
    ON fact_resistance_profile(run_id);

CREATE INDEX IF NOT EXISTS idx_fact_resistance_profile_structure_id
    ON fact_resistance_profile(structure_id);

CREATE INDEX IF NOT EXISTS idx_fact_resistance_profile_variant
    ON fact_resistance_profile(variant);

CREATE INDEX IF NOT EXISTS idx_fact_resistance_profile_mechanism
    ON fact_resistance_profile(mechanism_class);
