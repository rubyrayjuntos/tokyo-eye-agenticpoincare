#!/usr/bin/env python3
"""CI gate: fail on ungrandfathered governed-table INSERT bypasses."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.audit.write_paths import audit_write_path_gate


def main() -> int:
    errors = audit_write_path_gate()
    if not errors:
        print("write-path audit: OK")
        return 0
    print(f"write-path audit: {len(errors)} violation(s)")
    for err in errors:
        print(f"  {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
