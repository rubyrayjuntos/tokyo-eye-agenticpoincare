"""Shared test fixtures for Tokyo Eye tests."""

from __future__ import annotations

import os
from typing import Any

import pytest

from science.dtie.common.keys import make_chain_id, make_residue_id, make_structure_id


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_structure_id() -> str:
    """A canonical structure_id for testing."""
    return make_structure_id(pdb_id="4OBE", source="rcsb")


@pytest.fixture
def sample_chain_id(sample_structure_id: str) -> str:
    """A canonical chain_id for testing."""
    return make_chain_id(sample_structure_id, "A")


@pytest.fixture
def sample_residue_ids(sample_structure_id: str) -> list[str]:
    """A set of canonical residue_ids for testing (residues 10-14)."""
    return [
        make_residue_id(sample_structure_id, "A", i)
        for i in range(10, 15)
    ]


# ---------------------------------------------------------------------------
# Integration test fixtures (require DATABASE_URL or TEST_DATABASE_URL)
# ---------------------------------------------------------------------------


def _get_test_db_url() -> str | None:
    """Get the test database URL, or None if not configured."""
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        return None
    stripped = url.strip()
    # Reject doc-placeholder values copied literally from README snippets.
    if stripped in {"...", "…"} or "..." in stripped and "://" not in stripped:
        return None
    if not stripped.startswith(("postgresql://", "postgres://")):
        return None
    return stripped


def _integration_db_skip_reason() -> str | None:
    raw = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not raw:
        return "No TEST_DATABASE_URL or DATABASE_URL — skipping integration test"
    if raw.strip() in {"...", "…"} or ("..." in raw and "postgresql" not in raw):
        return (
            "TEST_DATABASE_URL looks like a placeholder (e.g. '...'). "
            "Use a real URL, e.g. postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/tokyoeye_dev"
        )
    if not _get_test_db_url():
        return f"Invalid TEST_DATABASE_URL/DATABASE_URL: {raw!r}"
    return None


@pytest.fixture
async def integration_db():
    """Provide a real DB connection for integration tests.

    Requires TEST_DATABASE_URL or DATABASE_URL to be set.
    Each test runs in a transaction that is rolled back after the test,
    so tests don't pollute each other.
    """
    url = _get_test_db_url()
    if not url:
        reason = _integration_db_skip_reason() or "No TEST_DATABASE_URL configured"
        pytest.skip(f"{reason} — skipping integration test")

    import psycopg

    async with await psycopg.AsyncConnection.connect(url) as conn:
        # Start a transaction that we'll roll back after the test
        await conn.execute("BEGIN")
        from data.db import DBAdapter
        adapter = DBAdapter(conn)
        yield adapter
        await conn.rollback()


@pytest.fixture
async def integration_db_with_schema(integration_db):
    """DB connection with schema already applied (migrations run).

    Uses the integration_db fixture and ensures core tables exist.
    Skips if migrations haven't been applied to the test DB.
    """
    # Verify core tables exist
    result = await integration_db.fetch_one(
        """
        SELECT COUNT(*) as cnt FROM information_schema.tables
        WHERE table_name IN (
            'provenance_run', 'fact_gnn_node_embedding',
            'embedding_space', 'governed_asset', 'normalization_audit'
        )
        """,
        {},
    )
    if not result or result["cnt"] < 5:
        pytest.skip("Test DB schema not applied — run migrations first")

    yield integration_db


# ---------------------------------------------------------------------------
# Query-validating mock DB
# ---------------------------------------------------------------------------


class QueryValidatingMockDB:
    """A mock DB that validates query structure and returns context-aware data.

    Unlike a simple mock that returns hardcoded data regardless of input,
    this mock:
    1. Validates that queries contain expected SQL keywords/table names
    2. Returns different data based on query content
    3. Tracks all executed queries for assertion
    4. Validates that params match placeholders in the query
    """

    def __init__(self):
        self.executed_queries: list[tuple[str, dict[str, Any] | None]] = []
        self._custom_responses: dict[str, list[dict[str, Any]]] = {}

    def register_response(self, table_or_view: str, rows: list[dict[str, Any]]) -> None:
        """Register a custom response for queries mentioning a specific table/view."""
        self._custom_responses[table_or_view] = rows

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        self._validate_query(query, params)
        self.executed_queries.append((query, params))

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        self._validate_query(query, params_list[0] if params_list else None)
        for params in params_list:
            self.executed_queries.append((query, params))

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self._validate_query(query, params)
        self.executed_queries.append((query, params))
        return self._get_response(query, single=True)

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._validate_query(query, params)
        self.executed_queries.append((query, params))
        return self._get_response(query, single=False) or []

    async def begin(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass

    def _validate_query(self, query: str, params: dict[str, Any] | None) -> None:
        """Validate that the query is well-formed and params match placeholders."""
        import re

        # Must contain at least one SQL keyword
        sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "BEGIN", "REFRESH"]
        query_upper = query.upper().strip()
        if not any(query_upper.startswith(kw) for kw in sql_keywords):
            raise ValueError(f"Query doesn't start with a SQL keyword: {query[:50]}...")

        # If params provided, validate that all :name placeholders have matching params
        if params:
            placeholders = set(re.findall(r":(\w+)", query))
            param_keys = set(params.keys())
            missing = placeholders - param_keys
            if missing:
                raise ValueError(
                    f"Query has placeholders {missing} not found in params. "
                    f"Query: {query[:80]}... Params: {list(param_keys)}"
                )

    def _get_response(self, query: str, single: bool) -> Any:
        """Return response based on which table/view the query references."""
        query_lower = query.lower()

        # Check custom responses first
        for table_or_view, rows in self._custom_responses.items():
            if table_or_view.lower() in query_lower:
                return rows[0] if single and rows else rows

        # Default responses based on query content
        if "count" in query_lower or "cnt" in query_lower:
            return {"cnt": 0} if single else [{"cnt": 0}]

        if "provenance_run" in query_lower and "select" in query_lower:
            return None if single else []

        if "embedding_space" in query_lower and "select" in query_lower:
            return None if single else []

        return None if single else []

    def assert_query_executed_containing(self, substring: str) -> None:
        """Assert that at least one executed query contains the given substring."""
        for query, _ in self.executed_queries:
            if substring.lower() in query.lower():
                return
        executed = [q[:60] for q, _ in self.executed_queries]
        raise AssertionError(
            f"No query containing '{substring}' was executed. "
            f"Executed queries: {executed}"
        )

    def assert_query_count(self, expected: int) -> None:
        """Assert the total number of queries executed."""
        actual = len(self.executed_queries)
        if actual != expected:
            raise AssertionError(f"Expected {expected} queries, got {actual}")


@pytest.fixture
def validating_mock_db() -> QueryValidatingMockDB:
    """A query-validating mock DB for unit tests."""
    return QueryValidatingMockDB()
