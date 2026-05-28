-- ============================================================================
-- Migration 032: Hypothesis Engine Tables (Governed)
-- Date: 2026-05-28
-- Purpose: Create governed tables for the hypothesis engine — structured
--          scientific reasoning about protein structures. Stores hypotheses,
--          testable predictions, and gathered evidence with full provenance.
-- Requirements: 1.3
-- ============================================================================

-- ============================================================================
-- TABLE: hypothesis
-- Stores structured scientific claims about protein structures.
-- Natural key: hypothesis_id (agent-generated UUID)
-- Requirements: 1.1, 1.3
-- ============================================================================

CREATE TABLE IF NOT EXISTS hypothesis (
    hypothesis_id   TEXT PRIMARY KEY,
    structure_id    TEXT NOT NULL REFERENCES dim_structure(structure_id),
    statement       TEXT NOT NULL,
    mechanism       TEXT,
    status          TEXT NOT NULL DEFAULT 'proposed',
    confidence      DOUBLE PRECISION DEFAULT 0.5,
    created_by      TEXT NOT NULL DEFAULT 'agent',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Lookup indexes
CREATE INDEX idx_hypothesis_structure
    ON hypothesis (structure_id);
CREATE INDEX idx_hypothesis_status
    ON hypothesis (status);

COMMENT ON TABLE hypothesis IS
    'Governed scientific hypotheses about protein structures. Writes via Normalizer only.';

-- ============================================================================
-- TABLE: hypothesis_prediction
-- Stores testable predictions derived from a hypothesis.
-- Natural key: prediction_id (agent-generated UUID)
-- Requirements: 1.4, 2.1, 2.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS hypothesis_prediction (
    prediction_id   TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL REFERENCES hypothesis(hypothesis_id),
    statement       TEXT NOT NULL,
    test_tool       TEXT,
    test_params     JSONB,
    threshold       TEXT,
    result          TEXT,
    passed          BOOLEAN,
    tested_at       TIMESTAMPTZ
);

-- Lookup index for joining predictions to hypothesis
CREATE INDEX idx_hypothesis_prediction_hypothesis
    ON hypothesis_prediction (hypothesis_id);

COMMENT ON TABLE hypothesis_prediction IS
    'Testable predictions derived from hypotheses. Each references a tool + threshold for evaluation.';

-- ============================================================================
-- TABLE: hypothesis_evidence
-- Stores evidence records (supporting or contradicting) gathered for a hypothesis.
-- Natural key: evidence_id (agent-generated UUID)
-- Requirements: 3.1, 3.2
-- ============================================================================

CREATE TABLE IF NOT EXISTS hypothesis_evidence (
    evidence_id     TEXT PRIMARY KEY,
    hypothesis_id   TEXT NOT NULL REFERENCES hypothesis(hypothesis_id),
    source_tool     TEXT NOT NULL,
    source_run_id   TEXT,
    supports        BOOLEAN NOT NULL,
    strength        DOUBLE PRECISION DEFAULT 0.5,
    description     TEXT NOT NULL,
    gathered_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Lookup index for joining evidence to hypothesis
CREATE INDEX idx_hypothesis_evidence_hypothesis
    ON hypothesis_evidence (hypothesis_id);

COMMENT ON TABLE hypothesis_evidence IS
    'Evidence records (supporting/contradicting) gathered for hypotheses. Writes via Normalizer only.';
