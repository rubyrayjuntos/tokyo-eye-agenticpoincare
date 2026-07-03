#!/usr/bin/env python3
"""Run audit event retention (aggregate + prune)."""

from __future__ import annotations

import argparse
import asyncio
import json

from data.audit.retention import run_audit_retention
from data.db import close_pool, get_connection, open_pool


async def _main(args: argparse.Namespace) -> int:
    await open_pool()
    try:
        async with get_connection() as conn:
            result = await run_audit_retention(
                conn,
                retention_days=args.days,
                aggregate=not args.no_aggregate,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                await conn.commit()
    finally:
        await close_pool()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune old pipeline audit events")
    parser.add_argument("--days", type=int, default=90, help="Retention window in days")
    parser.add_argument("--dry-run", action="store_true", help="Report counts only")
    parser.add_argument("--no-aggregate", action="store_true", help="Delete without rolling up")
    raise SystemExit(asyncio.run(_main(parser.parse_args())))
