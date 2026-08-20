"""Tokyo Eye v7 training launcher.

Forces ``gnn_lineage=v7`` / MLflow ``tokyo-eyes-v7`` / ``checkpoints/v7/runs``,
then delegates curriculum execution to the shared StageRunner stack.
"""

from __future__ import annotations

import sys

from experiments.training.v7 import V7_CHECKPOINT_ROOT, V7_MLFLOW_EXPERIMENT


def _inject_defaults(argv: list[str]) -> list[str]:
    out = list(argv)

    def has(flag: str) -> bool:
        return flag in out

    inject: list[str] = []
    if not has("--gnn-lineage"):
        inject += ["--gnn-lineage", "v7"]
    if not has("--mlflow-experiment"):
        inject += ["--mlflow-experiment", V7_MLFLOW_EXPERIMENT]
    if not has("--output-dir"):
        inject += ["--output-dir", str(V7_CHECKPOINT_ROOT / "default")]
    return inject + out


def main(argv: list[str] | None = None) -> None:
    args = _inject_defaults(list(argv) if argv is not None else sys.argv[1:])
    sys.argv = [sys.argv[0], *args]
    from experiments.training.v66.launch_training import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
