-- Migration 037: Ensemble Resistance Profile (SDRP)
-- Stores multi-structure ensemble resistance profiles computed by the SDRPEngine.
-- Each row represents one variant profiled across a set of conformational states.

CREATE TABLE IF NOT EXISTS fact_ensemble_resistance_profile (
    ensemble_id      TEXT PRIMARY KEY,
    variant          TEXT NOT NULL,
    sss_score        DOUBLE PRECISION NOT NULL,
    category         TEXT NOT NULL,           -- "Conformational_Switch", "Static_Disruptor", "Intermediate"
    state_profile    JSONB NOT NULL,          -- Full state_profile mapping
    mechanism_shifts JSONB,                   -- List of MechanismShift entries
    clinical_relevance TEXT,
    structure_ids    TEXT[] NOT NULL,          -- Array of structure_ids in the ensemble
    run_ids          TEXT[] NOT NULL,          -- Array of per-structure run_ids for provenance
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(variant, structure_ids)            -- Upsert key
);

CREATE INDEX IF NOT EXISTS idx_ensemble_profile_variant
    ON fact_ensemble_resistance_profile(variant);

CREATE INDEX IF NOT EXISTS idx_ensemble_profile_category
    ON fact_ensemble_resistance_profile(category);

CREATE INDEX IF NOT EXISTS idx_ensemble_profile_sss
    ON fact_ensemble_resistance_profile(sss_score);
