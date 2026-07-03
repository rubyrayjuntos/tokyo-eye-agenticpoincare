#!/usr/bin/env python3
"""Print SHA256 pins for locked corpus artifacts (update corpus_governance.py in same commit)."""

from __future__ import annotations

import hashlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
REPORT = _REPO / "manifests" / "corpus_redundancy_report.json"
MANIFEST = _REPO / "manifests" / "v6_corpus_stage_a.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for label, path in (
        ("FROZEN_REPORT_SHA256", REPORT),
        ("LOCKED_MANIFEST_SHA256", MANIFEST),
    ):
        if not path.is_file():
            print(f"# missing: {path}")
            continue
        print(f'{label} = "{_sha256(path)}"  # {path.relative_to(_REPO)}')


if __name__ == "__main__":
    main()
