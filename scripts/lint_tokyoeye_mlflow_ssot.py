#!/usr/bin/env python3
"""Guard active TokyoEye surfaces against old production labels."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_PATHS = [
    ".env.example",
    "AGENTS.md",
    "MLproject",
    "Makefile",
    "docker-compose.yml",
    "data/gates/tokyoeye_mlflow_ssot.json",
    "docs/DEVELOPER_ONBOARDING.md",
    "docs/PROJECT_HUB.md",
    "docs/training/GNN_LIFECYCLE.md",
    "science/api/routers/compute.py",
    "science/api/routers/health.py",
    "science/compute/runners/gnn_inference.py",
    "science/contracts/README.md",
    "science/contracts/model_registry.py",
    "science/contracts/onboard_contract.yaml",
    "science/tokyo_eye/governance/entrypoints.py",
    "science/tokyo_eye/governance/import_weights.py",
    "science/tokyo_eye/governance/registry.py",
    "science/tokyo_eye/governance/resolve.py",
    "science/tokyo_eye/governance/taxonomy.py",
    "science/tokyo_eye/governance/vault.py",
    "docs/superpowers/specs/2026-08-23-tokyoeye-github-release-vault-design.md",
]

FORBIDDEN = [
    re.compile(r"\btokyo_eye_v8\b"),
    re.compile(r"\bTokyoEye-v8\b"),
    re.compile(r"\btokyo-eyes-v8\b"),
    re.compile(r"\bHEALTHY_V7_CKPT\b"),
    re.compile(r"checkpoints/v8/runs/[^\\s`'\"]+\\.pt"),
]


def main() -> int:
    offenders: list[str] = []
    for rel in ACTIVE_PATHS:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN:
                if pattern.search(line):
                    offenders.append(f"{rel}:{lineno}: {pattern.pattern}: {line.strip()}")
    if offenders:
        print("TokyoEye MLflow SSOT drift found:", file=sys.stderr)
        print("\n".join(offenders), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
