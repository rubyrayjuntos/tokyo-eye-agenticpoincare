-- ============================================================================
-- Migration 048: Agent workspace layout persistence
-- Date: 2026-06-25
-- Purpose: Persist Dockview layout snapshots and active phase group per session.
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_workspace_layout (
    session_id              TEXT PRIMARY KEY,
    user_id                 TEXT NOT NULL,
    workspace_id            TEXT NOT NULL DEFAULT 'default',
    layout_json             JSONB NOT NULL DEFAULT '{}'::jsonb,
    active_phase_group      TEXT NOT NULL DEFAULT 'exploration',
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_workspace_layout_user
    ON agent_workspace_layout (user_id, updated_at DESC);

COMMENT ON TABLE agent_workspace_layout IS
    'Per-session Dockview layout snapshots and workbench phase group for the agent UI.';
