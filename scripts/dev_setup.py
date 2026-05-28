#!/usr/bin/env python
"""Development environment setup script.

Usage:
    python scripts/dev_setup.py          # Full setup (start DB + run migrations)
    python scripts/dev_setup.py migrate  # Run migrations only
    python scripts/dev_setup.py reset    # Drop and recreate DB + run migrations

Prerequisites:
    - Docker running (for PostgreSQL)
    - Python dependencies installed (uv sync --extra dev)
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time


def start_database() -> None:
    """Start the local PostgreSQL container via docker-compose."""
    print("Starting local PostgreSQL (pgvector)...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "db"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Failed to start database: {result.stderr}")
        sys.exit(1)

    # Wait for healthy
    print("Waiting for database to be ready...")
    for i in range(30):
        check = subprocess.run(
            ["docker", "compose", "exec", "db", "pg_isready", "-U", "tokyoeye"],
            capture_output=True,
        )
        if check.returncode == 0:
            print("Database is ready.")
            return
        time.sleep(1)

    print("Database did not become ready in 30 seconds.")
    sys.exit(1)


async def run_migrations() -> None:
    """Run all pending migrations."""
    from data.db import run_migrations as _run_migrations

    print("Running migrations...")
    executed = await _run_migrations()
    if executed:
        print(f"Executed {len(executed)} migrations:")
        for f in executed:
            print(f"  ✓ {f}")
    else:
        print("No pending migrations.")


async def reset_database() -> None:
    """Drop and recreate the database, then run migrations."""
    import psycopg

    print("Resetting database...")
    # Connect to 'postgres' DB to drop/create tokyoeye_dev
    async with await psycopg.AsyncConnection.connect(
        "postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/postgres",
        autocommit=True,
    ) as conn:
        await conn.execute("DROP DATABASE IF EXISTS tokyoeye_dev")
        await conn.execute("CREATE DATABASE tokyoeye_dev OWNER tokyoeye")
        print("Database recreated.")

    await run_migrations()


async def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "setup"

    if command == "setup":
        start_database()
        await run_migrations()
        print("\nDev environment ready!")
        print("  Database: postgresql://tokyoeye:tokyoeye_dev_local@localhost:5432/tokyoeye_dev")
        print("  Run tests: PYTHONPATH=. pytest tests/")
        print("  Visualizer: cd visualizer/frontend && npm install && npm run dev")

    elif command == "migrate":
        await run_migrations()

    elif command == "reset":
        await reset_database()

    else:
        print(f"Unknown command: {command}")
        print("Usage: python scripts/dev_setup.py [setup|migrate|reset]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
