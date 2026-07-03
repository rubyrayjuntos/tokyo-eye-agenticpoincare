#!/usr/bin/env python3
"""CI gate: generated contract artifacts must match onboard_contract.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audit.contract_codegen import audit_contract_sync


def main() -> int:
    errors = audit_contract_sync()
    if not errors:
        print("contract-sync audit: OK")
        return 0
    print(f"contract-sync audit: {len(errors)} stale artifact(s)")
    for err in errors:
        print(f"  {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
