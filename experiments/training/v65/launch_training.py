"""V6.5 training launcher — isolated MLflow experiment + checkpoint namespace."""

from __future__ import annotations

import sys
from pathlib import Path


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in argv)


def main() -> None:
    argv = sys.argv[1:]
    injected: list[str] = []
    if not _has_flag(argv, "--gnn-lineage"):
        injected.extend(["--gnn-lineage", "v6.5"])
    if not _has_flag(argv, "--mlflow-experiment"):
        injected.extend(["--mlflow-experiment", "tokyo-eyes-v65"])
    if not _has_flag(argv, "--output-dir"):
        injected.extend(["--output-dir", str(Path("checkpoints/v65/runs/default"))])
    sys.argv = [sys.argv[0], *injected, *argv]
    from experiments.training.v6.launch_training import main as v6_main

    v6_main()


if __name__ == "__main__":
    main()
