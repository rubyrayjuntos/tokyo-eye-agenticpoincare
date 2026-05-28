"""Database connection factory for AlloyDB (Cloud Run + local dev).

Single async engine used by all API layers.

Connection modes:
  1. IAM auth (Cloud Run → AlloyDB): DB_USE_IAM=true, no password needed.
     Uses google.auth to get an OAuth2 token as the asyncpg password.
  2. Password auth (local dev): DB_PASSWORD set, standard asyncpg connection.
  3. Cloud SQL Auth Proxy: CLOUD_SQL_CONNECTION_NAME set, Unix socket.
"""
import logging
import os
from typing import AsyncGenerator, Optional
from urllib.parse import quote

from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)

logger = logging.getLogger("gosp.db")

_engine: Optional[AsyncEngine] = None
_secret_cache: dict[str, str] = {}


def _get_iam_token() -> str:
    """Get an OAuth2 access token for AlloyDB IAM auth."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/alloydb.login"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def _get_secret_value(secret_name: str) -> str:
    cached = _secret_cache.get(secret_name)
    if cached is not None:
        return cached

    from google.cloud import secretmanager

    project_id = (
        os.environ.get("GCP_PROJECT_ID")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("PROJECT_ID")
    )
    if not project_id and not secret_name.startswith("projects/"):
        raise RuntimeError("GCP project id is required to resolve DB_PASSWORD_SECRET")

    resource_name = (
        secret_name
        if secret_name.startswith("projects/")
        else f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    )

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(name=resource_name)
    value = response.payload.data.decode("utf-8")
    _secret_cache[secret_name] = value
    return value


def get_db_url() -> str:
    db_name = os.environ.get("DB_NAME", "postgres")
    db_user = os.environ.get("DB_USER", "")
    db_host = os.environ.get("DB_HOST", "")
    db_port = os.environ.get("DB_PORT", "5432")
    cloud_sql_connection = os.environ.get("CLOUD_SQL_CONNECTION_NAME")
    use_iam = os.environ.get("DB_USE_IAM", "").lower() in ("true", "1", "yes")
    ssl_mode = os.environ.get("DB_SSLMODE", "require" if use_iam else "prefer")
    ssl_query = f"ssl={ssl_mode}"

    if use_iam:
        # AlloyDB IAM auth: OAuth2 token as password
        token = _get_iam_token()
        token_encoded = quote(token, safe="")
        user_encoded = quote(db_user, safe="")
        if cloud_sql_connection and not db_host:
            socket_path = f"/cloudsql/{cloud_sql_connection}"
            return (
                f"postgresql+asyncpg://{user_encoded}:{token_encoded}"
                f"@/{db_name}?host={socket_path}&{ssl_query}"
            )
        return (
            f"postgresql+asyncpg://{user_encoded}:{token_encoded}"
            f"@{db_host}:{db_port}/{db_name}?{ssl_query}"
        )

    # Password auth (local dev)
    db_password = os.environ.get("DB_PASSWORD", "")
    db_password_secret = os.environ.get("DB_PASSWORD_SECRET", "").strip()
    if not db_password and db_password_secret:
        db_password = _get_secret_value(db_password_secret)
    db_password_encoded = quote(db_password, safe="")

    if cloud_sql_connection and not db_host:
        socket_path = f"/cloudsql/{cloud_sql_connection}"
        return (
            f"postgresql+asyncpg://{db_user}:{db_password_encoded}"
            f"@/{db_name}?host={socket_path}&{ssl_query}"
        )
    host = db_host or "localhost"
    return (
        f"postgresql+asyncpg://{db_user}:{db_password_encoded}"
        f"@{host}:{db_port}/{db_name}?{ssl_query}"
    )


def create_engine_from_env() -> AsyncEngine:
    """Create a new async engine from environment variables."""
    return create_async_engine(
        get_db_url(),
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


def get_engine() -> AsyncEngine:
    """Return the singleton async engine, creating it on first call.

    Thread-safe: worst case two engines are created on concurrent first
    access; the second is discarded. The global is set atomically.
    """
    global _engine
    if _engine is None:
        _engine = create_engine_from_env()
    return _engine


async def dispose_engine() -> None:
    """Dispose the singleton engine (call on shutdown)."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def get_connection() -> AsyncGenerator[AsyncConnection, None]:
    """FastAPI dependency: yields an AsyncConnection per request."""
    engine = get_engine()
    async with engine.connect() as conn:
        yield conn


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an AsyncSession per request with auto-commit."""
    engine = get_engine()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        async with session.begin():
            yield session
