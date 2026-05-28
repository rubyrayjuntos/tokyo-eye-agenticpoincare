"""Shared FastAPI dependencies for the coordinator."""

from __future__ import annotations

from data.db import DBAdapter, get_connection


async def get_db():
    """FastAPI dependency: yields a DBAdapter from the connection pool."""
    async with get_connection() as conn:
        yield DBAdapter(conn)
