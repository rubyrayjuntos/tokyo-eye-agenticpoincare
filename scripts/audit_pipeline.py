#!/usr/bin/env python3
"""CLI for querying pipeline audit events."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from data.audit.pipeline_events import (
    get_audit_events_for_structure,
    parse_since_duration,
    query_audit_events,
)
from data.db import get_connection, open_pool, close_pool


async def _run(args: argparse.Namespace) -> int:
    await open_pool()
    try:
        async with get_connection() as conn:
            since = parse_since_duration(args.since) if args.since else None
            if args.structure_id:
                events = await get_audit_events_for_structure(
                    conn,
                    args.structure_id,
                    since=since,
                    severity=args.severity,
                    limit=args.limit,
                )
            else:
                events = await query_audit_events(
                    conn,
                    structure_id=args.structure_id,
                    job_name=args.job_name,
                    event_type=args.event_type,
                    severity=args.severity,
                    since=since,
                    limit=args.limit,
                )
    finally:
        await close_pool()

    if args.summary:
        counts: dict[str, int] = {}
        for event in events:
            key = f"{event['severity']}:{event['event_type']}"
            counts[key] = counts.get(key, 0) + 1
        print(json.dumps({"total": len(events), "breakdown": counts}, indent=2))
    else:
        print(json.dumps({"count": len(events), "events": events}, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Query pipeline audit events")
    parser.add_argument("--structure-id", dest="structure_id", help="Filter by structure_id")
    parser.add_argument("--job-name", dest="job_name", help="Filter by job_name")
    parser.add_argument("--event-type", dest="event_type", help="Filter by event_type")
    parser.add_argument("--severity", choices=["info", "warning", "error"])
    parser.add_argument("--since", help="Duration ago: 7d, 24h, 30m")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--summary", action="store_true", help="Aggregate counts by severity:type")
    args = parser.parse_args()
    if not args.structure_id and not args.summary and not args.job_name and not args.event_type:
        parser.error(
            "Provide --structure-id, --job-name, --event-type, or --summary for audit query"
        )
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
