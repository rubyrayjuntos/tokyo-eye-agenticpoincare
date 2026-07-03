#!/usr/bin/env python3
"""CI gate: fail on agent/compute import-boundary violations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audit.import_contract import audit_import_contracts


def main() -> int:
    errors = audit_import_contracts()
    if not errors:
        print("import-contract audit: OK")
        return 0
    print(f"import-contract audit: {len(errors)} violation(s)")
    for err in errors:
        print(f"  {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
