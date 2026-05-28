# Migrated from: new (Phase 1 dev infrastructure) on 2026-05-27
"""Database connection management for local development and production.

Local dev: PostgreSQL via docker-compose (pgvector/pgvector:pg16)
Production: Aurora PostgreSQL with pgvector extension

Usage:
    from data.db import get_connection, run_migrations

    # Get a pooled connection
    async with get_connection() as conn:
        await conn.execute("SELECT 1")

    # Run all migrations (uses a dedicated connection, not the pool)
    await run_migrations()

Pool lifecycle:
    The pool is initialized lazily on first use. For FastAPI apps,
    call open_pool() in the lifespan startup and close_pool() on shutdown.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import psycopg
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/tokyoeye_dev",
)

MIGRATIONS_DIR = Path(__file__).parent / "aurora" / "migrations"

# ---------------------------------------------------------------------------
# Connection Pool
# ---------------------------------------------------------------------------

_pool: AsyncConnectionPool | None = None


async def open_pool(
    min_size: int = 2,
    max_size: int = 10,
    conninfo: str | None = None,
) -> AsyncConnectionPool:
    """Initialize and open the connection pool.

    Call this during application startup (e.g., FastAPI lifespan).
    If the pool is already open, this is a no-op.
    """
    global _pool
    if _pool is not None:
        return _pool

    url = conninfo or DATABASE_URL
    _pool = AsyncConnectionPool(
        conninfo=url,
        min_size=min_size,
        max_size=max_size,
        open=False,
    )
    await _pool.open()
    logger.info("Connection pool opened (min=%d, max=%d)", min_size, max_size)
    return _pool


async def close_pool() -> None:
    """Close the connection pool.

    Call this during application shutdown.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Get an async database connection from the pool.

    The pool is initialized lazily if not already open.
    Connections are returned to the pool when the context exits.
    """
    global _pool
    if _pool is None:
        await open_pool()

    async with _pool.connection() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Named Parameter Conversion
# ---------------------------------------------------------------------------

# Regex that matches :name but NOT inside single-quoted string literals.
# Strategy: match either a quoted string (and skip it) or a :param (and capture it).
_NAMED_PARAM_RE = re.compile(
    r"'[^']*'"        # Match single-quoted strings (skip these)
    r"|"
    r":(\w+)"         # OR match :name and capture the name
)


def _convert_named_params(query: str) -> str:
    """Convert :name style params to %(name)s for psycopg.

    Correctly handles :name tokens inside single-quoted SQL strings
    by skipping them. Only bare :name tokens outside quotes are converted.
    """
    def _replacer(match: re.Match) -> str:
        # If group(1) is None, we matched a quoted string — return it unchanged
        if match.group(1) is None:
            return match.group(0)
        # Otherwise, convert :name to %(name)s
        return f"%({match.group(1)})s"

    return _NAMED_PARAM_RE.sub(_replacer, query)


# ---------------------------------------------------------------------------
# DB Adapter
# ---------------------------------------------------------------------------


class DBAdapter:
    """Adapter that wraps psycopg to match the Normalizer's DatabaseConnection protocol.

    This bridges the gap between psycopg's API and the protocol expected
    by the Normalizer, ProvenanceTracer, and other data layer components.

    Transaction semantics:
        psycopg connections default to autocommit=False, meaning a transaction
        is implicitly started on the first statement. begin() is provided for
        explicit transaction demarcation and will raise if the connection is
        in autocommit mode (where implicit transactions don't apply).
    """

    def __init__(self, conn: psycopg.AsyncConnection):
        self._conn = conn

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        if params:
            converted_query = _convert_named_params(query)
            await self._conn.execute(converted_query, params)
        else:
            await self._conn.execute(query)

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        converted_query = _convert_named_params(query)
        async with self._conn.cursor() as cur:
            for params in params_list:
                await cur.execute(converted_query, params)

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        converted_query = _convert_named_params(query)
        async with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(converted_query, params or {})
            return await cur.fetchone()

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        converted_query = _convert_named_params(query)
        async with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(converted_query, params or {})
            return await cur.fetchall()

    async def begin(self) -> None:
        """Begin an explicit transaction.

        In psycopg's default mode (autocommit=False), a transaction is
        implicitly active after the first statement. This method ensures
        we're in a clean transaction state. If the connection is in
        autocommit mode, this starts an explicit transaction block.
        """
        if self._conn.autocommit:
            # In autocommit mode, we need an explicit BEGIN
            await self._conn.execute("BEGIN")
        # In non-autocommit mode (the default), psycopg auto-begins
        # a transaction on the first statement. Nothing to do here,
        # but we verify the connection isn't in a failed transaction state.
        elif self._conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
            await self._conn.rollback()

    async def commit(self) -> None:
        await self._conn.commit()

    async def rollback(self) -> None:
        await self._conn.rollback()


# ---------------------------------------------------------------------------
# Migrations (uses dedicated connection, not the pool)
# ---------------------------------------------------------------------------


async def run_migrations(target_db_url: str | None = None) -> list[str]:
    """Run all SQL migrations in order against the database.

    Uses a dedicated connection (not the pool) since migrations may
    run before the pool is initialized or alter schema in ways that
    affect pooled connections.

    Args:
        target_db_url: Override database URL (uses DATABASE_URL if None).

    Returns:
        List of migration filenames that were executed.
    """
    url = target_db_url or DATABASE_URL
    executed: list[str] = []

    async with await psycopg.AsyncConnection.connect(url) as conn:
        # Create migrations tracking table if it doesn't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                filename TEXT PRIMARY KEY,
                executed_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.commit()

        # Get already-executed migrations
        async with conn.cursor() as cur:
            await cur.execute("SELECT filename FROM _migrations ORDER BY filename")
            already_done = {row[0] for row in await cur.fetchall()}

        # Find and run pending migrations in order
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for migration_file in migration_files:
            if migration_file.name in already_done:
                continue

            sql = migration_file.read_text()
            try:
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO _migrations (filename) VALUES (%s)",
                    (migration_file.name,),
                )
                await conn.commit()
                executed.append(migration_file.name)
            except Exception as e:
                await conn.rollback()
                raise RuntimeError(
                    f"Migration {migration_file.name} failed: {e}"
                ) from e

    return executed
