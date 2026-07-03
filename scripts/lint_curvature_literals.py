#!/usr/bin/env python3
"""Fail when production code hardcodes curvature fallbacks (TS-002 policy).

Usage:
    python scripts/lint_curvature_literals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science.dtie.common.curvature_literal_lint import scan_repository


def main() -> int:
    violations = scan_repository()
    if not violations:
        print("curvature literal lint: OK (no violations)")
        return 0

    print(f"curvature literal lint: {len(violations)} violation(s)")
    for item in violations:
        print(f"  {item.path}:{item.line_no} [{item.rule}] {item.line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
