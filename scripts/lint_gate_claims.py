#!/usr/bin/env python3
"""CI gate: §0 enforcement-matrix claims (existence + liveness canaries)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audit.gate_canaries import audit_gate_liveness_canaries
from data.audit.gate_claims import audit_gate_claims


def main() -> int:
    errors = audit_gate_claims()
    errors.extend(audit_gate_liveness_canaries())
    if not errors:
        print("gate-claims audit: OK (existence + liveness)")
        return 0
    print(f"gate-claims audit: {len(errors)} issue(s)")
    for err in errors:
        print(f"  {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
