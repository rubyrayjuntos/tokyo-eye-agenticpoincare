#!/usr/bin/env python3
"""CI gate: contract edits must bump version:."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audit.contract_codegen import audit_contract_version_bump


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify contract version bump on YAML changes")
    parser.add_argument(
        "--base-ref",
        default="HEAD",
        help="Git ref to diff against (CI: merge base, e.g. origin/main)",
    )
    args = parser.parse_args()

    errors = audit_contract_version_bump(base_ref=args.base_ref)
    if not errors:
        print(f"contract-version audit: OK (vs {args.base_ref})")
        return 0
    print(f"contract-version audit: {len(errors)} violation(s)")
    for err in errors:
        print(f"  {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
