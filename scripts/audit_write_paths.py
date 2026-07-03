#!/usr/bin/env python3
"""Scan the repository for direct SQL writes and print an audit summary.

Usage:
    python scripts/audit_write_paths.py
    python scripts/audit_write_paths.py --markdown
"""

from __future__ import annotations

import argparse

from data.audit.write_paths import grouped_hits


def print_text(grouped: dict[str, list]) -> None:
    order = ("canonical", "bypass_fact", "bypass_governance", "other")
    labels = {
        "canonical": "Canonical normalizer writes",
        "bypass_fact": "Direct fact-table bypasses",
        "bypass_governance": "Direct governance/dimension bypasses",
        "other": "Other governed writes",
    }
    for key in order:
        rows = grouped.get(key, [])
        print(f"\n== {labels[key]} ({len(rows)}) ==")
        for hit in rows:
            op = getattr(hit, "operation", "insert")
            print(f"  {hit.path}:{hit.line_no} [{op}] -> {hit.table}")


def print_markdown(grouped: dict[str, list]) -> None:
    print("# Write Path Inventory (generated)")
    print()
    print("Regenerate with: `python scripts/audit_write_paths.py --markdown`")
    print()
    order = (
        ("bypass_fact", "P0 — Direct `fact_*` bypasses"),
        ("bypass_governance", "P1 — Governance/dimension bypasses"),
        ("canonical", "Canonical normalizer path"),
    )
    for key, title in order:
        rows = grouped.get(key, [])
        print(f"## {title} ({len(rows)})")
        print()
        print("| File | Line | Op | Table |")
        print("|------|------|----|-------|")
        for hit in rows:
            op = getattr(hit, "operation", "insert")
            print(f"| `{hit.path}` | {hit.line_no} | {op} | `{hit.table}` |")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="Emit markdown tables")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit 1 when CI write-path gate violations exist",
    )
    args = parser.parse_args()

    if args.gate:
        import sys

        from data.audit.write_paths import audit_write_path_gate

        errors = audit_write_path_gate()
        if not errors:
            print("write-path audit gate: OK")
            raise SystemExit(0)
        print(f"write-path audit gate: {len(errors)} violation(s)")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)

    grouped = grouped_hits()
    if args.markdown:
        print_markdown(grouped)
    else:
        print_text(grouped)


if __name__ == "__main__":
    main()
