#!/usr/bin/env python3
"""CI gate: pipeline audit event_type must use shared.audit.events constants."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audit.audit_event_types import audit_event_type_literals


def main() -> int:
    errors = audit_event_type_literals()
    if not errors:
        print("audit-event-type audit: OK")
        return 0
    print(f"audit-event-type audit: {len(errors)} violation(s)")
    for err in errors:
        print(f"  {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
