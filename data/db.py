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

Robustness against "pool-1" / exhaustion errors:
    - get_connection() has automatic exponential-backoff retry for transient
      pool acquisition failures (the classic "error connecting in 'pool-1'").
    - Configurable via DB_POOL_MIN_SIZE / MAX_SIZE / ACQUIRE_TIMEOUT /
      MAX_ACQUIRE_RETRIES etc. env vars.
    - reset_pool() + with_db_retry() helpers for manual or scripted recovery.
    - Every connection gets a "tokyo-eye-pidXXXX" application_name so you can
      easily inspect who is using connections in pg_stat_activity.
    - Batch scripts (e.g. batch_therapeutic_compiler) now acquire per-item
      and use the retry wrapper + semaphore.

When you see repeated pool-1 errors during heavy parallel work (many simultaneous
python -c, batch ingest + pipeline, agent tool calls, hypothesis formalization, etc.),
first try:
    from data.db import get_pool_stats, reset_pool
    print(get_pool_stats())
    await reset_pool()
Then re-run the operation. The next get_connection() will create a fresh pool
with the current (more generous) defaults.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Awaitable

import psycopg
from psycopg_pool import AsyncConnectionPool, PoolTimeout

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


class _GenerationToken:
    def __init__(self, value: int = 0):
        self.value = value
        self.previous_value = value

    def __iadd__(self, other: int):
        self.previous_value = self.value
        self.value += other
        return self

    def _coerce(self, other: object) -> int:
        if isinstance(other, _GenerationToken):
            if other is self:
                return self.previous_value
            return other.value
        return int(other)  # type: ignore[arg-type]

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        return self.value == self._coerce(other)

    def __ne__(self, other: object) -> bool:
        return self.value != self._coerce(other)

    def __gt__(self, other: object) -> bool:
        return self.value > self._coerce(other)

# Lifecycle protection against reset-while-await race (the "ghost in the async machine").
# _lifecycle_lock serializes open/close/reset so that no waiter sees a torn-down pool.
# _pool_generation is a monotonic token: waiters snapshot it before their await;
# if it changes on exception, they know a reset happened and should retry the new pool
# without treating it as a hard failure (avoids NoneType on await and spurious watchdog triggers).
_lifecycle_lock: asyncio.Lock = asyncio.Lock()
_pool_generation = _GenerationToken(0)

# Env-configurable pool tuning (override for heavy batch / agent / parallel diagnostic workloads)
DB_POOL_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", "8"))
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", "32"))
DB_POOL_ACQUIRE_TIMEOUT = float(os.environ.get("DB_POOL_ACQUIRE_TIMEOUT", "45.0"))
DB_POOL_MAX_LIFETIME = float(os.environ.get("DB_POOL_MAX_LIFETIME", "3600.0"))
DB_POOL_MAX_IDLE = float(os.environ.get("DB_POOL_MAX_IDLE", "300.0"))
DB_POOL_NUM_WORKERS = int(os.environ.get("DB_POOL_NUM_WORKERS", "3"))  # background maintenance workers

# How many times we retry a transient "pool-1" / acquire failure before giving up
DB_POOL_MAX_ACQUIRE_RETRIES = int(os.environ.get("DB_POOL_MAX_ACQUIRE_RETRIES", "5"))
DB_POOL_RETRY_BASE_DELAY = float(os.environ.get("DB_POOL_RETRY_BASE_DELAY", "0.4"))

# Global concurrency limiter for DB-heavy operations to prevent thundering herd
# (independent of pool size — protects the DB server itself).
#
# IMPORTANT for pool-1 stability:
# - Long-running operations that hold a connection while doing CPU work
#   (e.g. GNN inference + full hyperbolic all-pairs NumPy) will still starve
#   the pool even with this limit.
# - The hyperbolic_distance_populator and similar workers must decouple
#   fetch/compute/insert (see science/dtie/v5/workers/hyperbolic_distance_populator.py).
# - Rule of thumb: DB_CONCURRENCY_LIMIT should be <= pool max_size, and
#   heavy single-structure workers should be limited to 1-2 concurrent instances.
DB_CONCURRENCY_LIMIT = int(os.environ.get("DB_CONCURRENCY_LIMIT", "12"))
_db_concurrency_sem: asyncio.Semaphore | None = None


def _get_db_semaphore() -> asyncio.Semaphore:
    global _db_concurrency_sem
    if _db_concurrency_sem is None:
        _db_concurrency_sem = asyncio.Semaphore(DB_CONCURRENCY_LIMIT)
    return _db_concurrency_sem


async def _pool_configure(conn: psycopg.AsyncConnection) -> None:
    """Configure hook called by psycopg_pool for every new raw connection.

    This must be async so pool internals can await it safely.
    The configure function must leave the connection in a clean state
    (no open transaction, no INERROR status), otherwise the pool will
    discard the connection with "left in status INERROR by configure function".
    """
    try:
        import os as _os
        pid = _os.getpid()

        # Ensure the connection is not in INERROR before or after our work.
        if conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
            try:
                await conn.rollback()
            except Exception:
                pass

        # Use sql.Literal for SET to avoid "syntax error at $1" on utility statements
        # (Postgres extended query protocol rejects param binding for SET GUC in this form).
        # This was causing fresh conns from pool/standalone to be returned in INERROR state,
        # poisoning all subsequent fetch/execute with "current transaction is aborted".
        from psycopg import sql as _sql
        await conn.execute(
            _sql.SQL("SET application_name = {}").format(_sql.Literal(f"tokyo-eye-pid{pid}"))
        )
        # SET starts an implicit tx in non-autocommit mode; commit to leave conn
        # in clean IDLE state for the pool (otherwise "left in status INTRANS by configure").
        try:
            await conn.commit()
        except Exception:
            pass

        # Final safety: if SET somehow left us in error (unlikely), clean it.
        if conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
            try:
                await conn.rollback()
            except Exception:
                pass
    except Exception:
        # Completely non-fatal — app name is only for pg_stat_activity diagnostics.
        # Still try to leave the conn clean so the pool doesn't discard it.
        try:
            if conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
                await conn.rollback()
        except Exception:
            pass


async def _validate_connection(conn: psycopg.AsyncConnection) -> bool:
    """Lightweight health check for a connection from the pool.
    Used by psycopg_pool's 'check' mechanism and on-demand validation.
    Returns True if the connection is usable.
    """
    try:
        # Fast path: check transaction state first
        if conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
            await conn.rollback()
        # Cheap liveness probe
        await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.debug("Connection validation failed: %s", e)
        return False


async def _pool_check(conn: psycopg.AsyncConnection) -> bool:
    """Checker passed to AsyncConnectionPool for validating connections on checkout."""
    return await _validate_connection(conn)


async def open_pool(
    min_size: int | None = None,
    max_size: int | None = None,
    conninfo: str | None = None,
    force: bool = False,
) -> AsyncConnectionPool:
    """Initialize and open the connection pool.

    Call this during application startup (e.g., FastAPI lifespan) or at the
    beginning of long-running scripts.

    The pool is now significantly more robust against "pool-1" exhaustion:
    - Configurable via DB_POOL_* env vars (see defaults above).
    - Per-connection application_name for observability in pg_stat_activity.
    - get_connection() has built-in retry + backoff for transient pool pressure.
    - reset_pool() helper for recovery after a bad burst.

    If the pool already exists and is healthy, this is a no-op unless force=True.
    """
    global _pool, _pool_generation

    async with _lifecycle_lock:
        if _pool is not None and not force:
            # If the existing pool object reports closed, recreate (direct close under lock to avoid re-acquire deadlock).
            try:
                if getattr(_pool, "closed", False):
                    try:
                        await _pool.close()
                    except Exception:
                        pass
                    _pool = None
                    _pool_generation += 1
                else:
                    return _pool
            except Exception:
                try:
                    await _pool.close()
                except Exception:
                    pass
                _pool = None
                _pool_generation += 1

        msize = min_size or DB_POOL_MIN_SIZE
        Msize = max_size or DB_POOL_MAX_SIZE

        url = conninfo or DATABASE_URL

        _pool = AsyncConnectionPool(
            conninfo=url,
            min_size=msize,
            max_size=Msize,
            open=False,
            timeout=DB_POOL_ACQUIRE_TIMEOUT,
            max_lifetime=DB_POOL_MAX_LIFETIME,
            max_idle=DB_POOL_MAX_IDLE,
            num_workers=DB_POOL_NUM_WORKERS,  # background workers for maintenance / connection creation
            # 'check' validates connections before they are handed to callers — prevents
            # handing out dead connections that would cause immediate follow-on failures.
            check=_pool_check,
            # 'configure' is the supported hook (not 'setup') for every new connection.
            configure=_pool_configure,
        )
        await _pool.open()
        _pool_generation += 1
        logger.info(
            "Connection pool opened (min=%d, max=%d, acquire_timeout=%.1fs, workers=%d, check=enabled, generation=%d) — "
            "pool-1 robustness features active (retries + validation + app_name labels + global semaphore + lifecycle lock)",
            msize, Msize, DB_POOL_ACQUIRE_TIMEOUT, DB_POOL_NUM_WORKERS, _pool_generation,
        )
        return _pool


async def _close_pool_under_lock() -> None:
    """Internal close that assumes the caller already holds _lifecycle_lock."""
    global _pool, _pool_generation
    if _pool is not None:
        try:
            await _pool.close()
        except Exception as e:
            logger.warning("Error while closing pool (continuing): %s", e)
        _pool = None
        _pool_generation += 1
        logger.info("Connection pool closed (generation advanced to %d)", _pool_generation)


async def close_pool() -> None:
    """Close the connection pool.

    Call this during application shutdown or when you want to force a full reset.
    """
    async with _lifecycle_lock:
        await _close_pool_under_lock()


def get_pool_stats() -> dict[str, Any]:
    """Return current pool utilization stats + robustness config.

    Useful for health checks, /debug endpoints, and manual diagnostics.
    Call this when you see "pool-1" errors to see how much headroom remains.
    """
    if _pool is None:
        return {
            "status": "not_initialized",
            "min_size": None,
            "max_size": None,
        }
    try:
        stats = _pool.get_stats()
    except Exception:
        stats = {}
    return {
        "status": "open",
        "min_size": _pool.min_size,
        "max_size": _pool.max_size,
        "pool_size": stats.get("pool_size"),
        "pool_available": stats.get("pool_available"),
        "requests_waiting": stats.get("requests_waiting"),
        "acquire_timeout": DB_POOL_ACQUIRE_TIMEOUT,
        "max_acquire_retries": DB_POOL_MAX_ACQUIRE_RETRIES,
    }


async def reset_pool() -> None:
    """Force-close the current pool (if any) and clear the singleton.

    The next call to get_connection() or open_pool() will create a fresh pool.
    Use this as a recovery hammer after a bad burst of parallel work that
    left many connections in a bad state.

    Now uses the lifecycle lock + generation token so that any suspended
    waiters in get_connection() see the generation change and retry the
    new pool instead of crashing with "NoneType can't be awaited".
    """
    global _pool, _pool_generation
    async with _lifecycle_lock:
        if _pool is not None:
            logger.warning("reset_pool() called — forcing full pool recreation (recovery from pool-1 pressure)")
            old_pool = _pool
            try:
                stats = old_pool.get_stats()
            except Exception:
                stats = {}
            active_connections = max(
                0,
                int(stats.get("pool_size") or 0) - int(stats.get("pool_available") or 0),
            )
            _pool = None
            _pool_generation += 1
            logger.info(
                "Pool reset detached current pool; generation advanced to %d. Waiters will re-evaluate on next acquire.",
                _pool_generation,
            )
            if active_connections > 0:
                async def _delayed_close(pool_to_close: AsyncConnectionPool) -> None:
                    await asyncio.sleep(0.2)
                    try:
                        await pool_to_close.close()
                    except Exception as e:
                        logger.warning("Error while closing detached pool (continuing): %s", e)

                asyncio.create_task(_delayed_close(old_pool))
            else:
                try:
                    await old_pool.close()
                except Exception as e:
                    logger.warning("Error while closing reset pool (continuing): %s", e)
    # Next get_connection() will lazily re-open with current env-tuned settings.


# ---------------------------------------------------------------------------
# Resilient connection acquisition (the main defense against "pool-1" errors)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_connection(
    *,
    bypass_concurrency_limit: bool = False,
    validate: bool = True,
) -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Get an async database connection from the pool (with retry on transient exhaustion).

    The pool is initialized lazily if not already open.
    Connections are returned to the pool when the context exits.

    This is the recommended way to talk to the DB from tools, scripts, and routers.
    It now automatically retries on the "pool-1" / "error connecting" class of transient errors
    that appear when many parallel operations (batch, agent analysis, diagnostics, etc.)
    compete for connections.

    Additional hardening (generation token + lifecycle lock):
    - Captures _pool_generation before any await.
    - If pool becomes None (reset in progress), awaits the lifecycle lock (yields until new pool ready) then retries.
    - On transient failure, if generation changed, we know a reset occurred while we were suspended — we loop without counting a hard failure.
    - Prevents "object NoneType can't be used in 'await'" and stops the watchdog from firing on its own waiters.

    Additional hardening:
    - Respects a global DB_CONCURRENCY_LIMIT semaphore (backpressure).
    - Optional post-acquire validation (SELECT 1 + transaction cleanup).
    - Detailed pressure logging when the pool reports waiting requests.
    """
    global _pool, _pool_generation
    if _pool is None:
        await open_pool()

    sem = _get_db_semaphore() if not bypass_concurrency_limit else None

    last_exc: Exception | None = None
    for attempt in range(DB_POOL_MAX_ACQUIRE_RETRIES):
        acquired_sem = False
        yielded_to_caller = False
        # Snapshot generation *before* any await that could be interrupted by a reset.
        current_generation = _pool_generation
        try:
            if sem is not None:
                await sem.acquire()
                acquired_sem = True

            stats = get_pool_stats()
            if stats.get("requests_waiting", 0):
                logger.warning(
                    "DB pool under pressure: requests_waiting=%s | %s",
                    stats.get("requests_waiting"), stats
                )

            # Guard against the pool having been torn down by a concurrent reset.
            if _pool is None:
                # A reset is (or was) holding the lifecycle lock. Await it to park this
                # waiter until the replacement pool is ready, then loop and try the new one.
                async with _lifecycle_lock:
                    pass
                # Do not count this as a hard retry; the generation will have advanced.
                await asyncio.sleep(0.05)
                continue

            async with _pool.connection() as conn:
                if validate:
                    # Best-effort validation; if it fails we treat the acquire as bad and retry
                    if not await _validate_connection(conn):
                        # Force the pool to discard this one on exit
                        try:
                            await conn.rollback()
                        except Exception:
                            pass
                        raise PoolTimeout("validated connection failed liveness check")

                if not hasattr(conn, "fetchval"):
                    async def _fetchval(query: str, *params: Any) -> Any:
                        async with conn.cursor() as cur:
                            await cur.execute(query, params or None)
                            row = await cur.fetchone()
                            return row[0] if row else None

                    setattr(conn, "fetchval", _fetchval)

                yielded_to_caller = True
                yield conn
            return  # successful use + clean return to pool
        except Exception as e:
            if yielded_to_caller:
                raise
            last_exc = e
            msg = str(e).lower()
            is_transient = (
                isinstance(e, PoolTimeout)
                or "pool" in msg
                or "error connecting" in msg
                or "too many" in msg
                or "pool-1" in msg
                or isinstance(e, (psycopg.OperationalError, psycopg.InterfaceError))
            )
            # Generation check: if the token advanced while we were suspended in the
            # await, a reset happened under us — treat as "pool was replaced", not a
            # fatal pressure event. Loop back to use the fresh pool.
            if _pool_generation != current_generation:
                logger.debug(
                    "Pool generation advanced (was %d, now %d) while awaiting — retrying with new pool (not counting as failure).",
                    current_generation, _pool_generation,
                )
                await asyncio.sleep(0.05)
                continue

            if not is_transient or attempt == DB_POOL_MAX_ACQUIRE_RETRIES - 1:
                logger.error(
                    "DB pool acquire failed (attempt %d/%d): %s | stats=%s. "
                    "Actionable: await reset_pool() or reduce parallel DB work / DB_CONCURRENCY_LIMIT.",
                    attempt + 1, DB_POOL_MAX_ACQUIRE_RETRIES, e, get_pool_stats(),
                )
                # Auto-recovery: after final failure on a heavily waited pool, proactively reset
                # so the next caller gets a fresh pool instead of inheriting the damaged state.
                stats = get_pool_stats()
                if (stats.get("requests_waiting") or 0) >= 2:
                    try:
                        # Use create_task but the lock in reset will serialize it.
                        asyncio.create_task(reset_pool())
                        logger.warning("Auto-triggered reset_pool() due to sustained pool-1 pressure")
                    except Exception:
                        pass
                raise
            delay = DB_POOL_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.35)
            logger.warning(
                "Transient DB pool pressure (retry %d/%d in %.2fs): %s | stats=%s",
                attempt + 1, DB_POOL_MAX_ACQUIRE_RETRIES, delay, e, get_pool_stats(),
            )
            await asyncio.sleep(delay)
        finally:
            if acquired_sem and sem is not None:
                try:
                    sem.release()
                except Exception:
                    pass

    if last_exc:
        raise last_exc


async def with_db_retry(
    operation: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    operation_name: str = "db_operation",
) -> Any:
    """Run an async DB-using callable with the same retry/backoff policy as get_connection.

    Useful for batch scripts or complex multi-statement work where you want the
    whole logical operation to survive a temporary pool-1 spike.

    Example:
        result = await with_db_retry(
            lambda: process_one_structure(pdb, db),
            operation_name="process_structure"
        )
    """
    attempts = max_attempts or DB_POOL_MAX_ACQUIRE_RETRIES
    delay_base = base_delay or DB_POOL_RETRY_BASE_DELAY
    last_err = None
    for attempt in range(attempts):
        try:
            return await operation()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = (
                isinstance(e, (PoolTimeout, psycopg.OperationalError, psycopg.InterfaceError))
                or "pool" in msg or "error connecting" in msg or "pool-1" in msg
            )
            if not transient or attempt == attempts - 1:
                raise
            d = delay_base * (2 ** attempt) + random.uniform(0, 0.3)
            logger.warning("%s transient pool pressure (retry %d/%d in %.2fs): %s",
                           operation_name, attempt+1, attempts, d, e)
            await asyncio.sleep(d)
    if last_err:
        raise last_err
    return None


async def db_health_check() -> dict[str, Any]:
    """Run a resilient health check against the pool.

    Returns detailed status including pool stats, connectivity result,
    and pressure indicators. Safe to call from diagnostics, /health endpoints,
    or agent tools.
    """
    stats = get_pool_stats()
    result = {
        "pool": stats,
        "connectivity": False,
        "pressure": "unknown",
        "message": "",
    }

    try:
        async with get_connection(validate=True) as conn:
            await conn.execute("SELECT 1")
            result["connectivity"] = True

        waiting = stats.get("requests_waiting") or 0
        if waiting and waiting > 2:
            result["pressure"] = "high"
            result["message"] = f"Pool has {waiting} waiters — consider reset_pool() or reducing concurrency"
        elif waiting:
            result["pressure"] = "medium"
        else:
            result["pressure"] = "low"

    except Exception as e:
        result["message"] = str(e)
        result["pressure"] = "critical"

    return result


async def get_standalone_connection(
    url: str | None = None,
    *,
    timeout: float = 15.0,
    application_name: str | None = None,
) -> psycopg.AsyncConnection:
    """Direct (non-pooled) connection for special cases: migrations, one-off admin,
    or code paths that must run before/after the pool is available.

    Still gets basic hardening: connect timeout, application_name label, and
    a best-effort retry on transient connect errors.

    Prefer `get_connection()` (the pooled + retried + validated path) for normal work.
    """
    target = url or DATABASE_URL
    name = application_name or f"tokyo-eye-standalone-pid{os.getpid()}"
    last_err = None
    for attempt in range(3):
        try:
            conn = await psycopg.AsyncConnection.connect(
                target,
                connect_timeout=timeout,
            )
            try:
                # Use sql.Literal (not %s param) for SET GUC -- prevents syntax error / tx abort
                # that was returning dirty (INERROR) conns from get_standalone_connection.
                from psycopg import sql as _sql
                await conn.execute(
                    _sql.SQL("SET application_name = {}").format(_sql.Literal(name))
                )
            except Exception:
                pass
            return conn
        except Exception as e:
            last_err = e
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
    raise last_err or RuntimeError("Failed to obtain standalone connection")


async def log_pool_pressure() -> None:
    """Log current pool state at WARNING level if under any pressure.
    Call this from long-running loops or after bursts of work.
    """
    stats = get_pool_stats()
    waiting = stats.get("requests_waiting") or 0
    if waiting or stats.get("status") != "open":
        logger.warning("DB pool pressure report: %s", stats)
    else:
        logger.debug("DB pool healthy: %s", stats)


# Context manager for critical DB sections that need both semaphore + retry
@asynccontextmanager
async def db_critical_section(operation_name: str = "db_critical"):
    """Use this around multi-statement or high-value DB work for backpressure + resilience.

    Example:
        async with db_critical_section("formalize_d389_hypothesis"):
            async with get_connection() as conn:
                ...
    """
    sem = _get_db_semaphore()
    await sem.acquire()
    try:
        await log_pool_pressure()
        yield
    finally:
        try:
            sem.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Named Parameter Conversion
# ---------------------------------------------------------------------------

# Regex that matches :name but NOT inside single-quoted string literals
# and NOT the :Type portion of PostgreSQL ::cast syntax (e.g. :param::double precision[] or :p::text).
# Using negative lookbehind (?<!:) prevents matching the second colon in :: casts which would
# otherwise create spurious param names like "double", "text", "precision" leading to
# "query parameter missing: double, text" errors.
_NAMED_PARAM_RE = re.compile(
    r"'[^']*'"        # Match single-quoted strings (skip these)
    r"|"
    r"(?<!:):(\w+)"   # OR match :name (but not if immediately preceded by another :, i.e. part of ::cast)
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

    async def _rollback_failed_statement(self) -> None:
        """Clear aborted transaction state after a statement failure."""
        try:
            if self._conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
                await self._conn.rollback()
        except Exception:
            pass

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        try:
            if params:
                converted_query = _convert_named_params(query)
                await self._conn.execute(converted_query, params)
            else:
                await self._conn.execute(query)
        except Exception:
            await self._rollback_failed_statement()
            raise

    async def execute_many(self, query: str, params_list: list[dict[str, Any]] | list[tuple]) -> None:
        """Bulk execute with true driver batching (executemany) when possible.

        This is critical for large inserts (e.g. hyperbolic distance matrices with
        tens of thousands of rows). The previous per-row await loop in Python
        held the connection for a very long time.

        Accepts either list of dicts (named params after conversion) or list of
        tuples (positional $1 style). Uses a single cursor + executemany for
        efficiency.
        """
        if not params_list:
            return
        try:
            converted_query = _convert_named_params(query)
            async with self._conn.cursor() as cur:
                # psycopg async cursor supports executemany for batch efficiency
                await cur.executemany(converted_query, params_list)
        except Exception:
            await self._rollback_failed_statement()
            raise

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        try:
            converted_query = _convert_named_params(query)
            async with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(converted_query, params or {})
                return await cur.fetchone()
        except Exception:
            await self._rollback_failed_statement()
            raise

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            converted_query = _convert_named_params(query)
            async with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute(converted_query, params or {})
                return await cur.fetchall()
        except Exception:
            await self._rollback_failed_statement()
            raise

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
