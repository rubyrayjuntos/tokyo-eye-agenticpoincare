"""Repair common local MLflow file-store issues that break `mlflow ui` with HTTP 500."""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

_RUN_ID_RE = re.compile(r"^run_id:\s*(.+)$", re.MULTILINE)


def _repair_missing_run_uuid(meta_path: Path) -> bool:
    text = meta_path.read_text()
    if "run_uuid:" in text:
        return False
    match = _RUN_ID_RE.search(text)
    if not match:
        return False
    run_id = match.group(1).strip().strip("'\"")
    meta_path.write_text(text.replace(match.group(0), f"{match.group(0)}\nrun_uuid: {run_id}", 1))
    return True


def _quarantine_orphan_experiment_dir(store_root: Path) -> bool:
    orphan = store_root / "tokyo-eyes-v6"
    if not orphan.is_dir() or (orphan / "meta.yaml").is_file():
        return False
    trash = store_root / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    shutil.move(str(orphan), str(trash / f"tokyo-eyes-v6-orphan-{stamp}"))
    return True


def repair_store(store_root: Path) -> list[str]:
    actions: list[str] = []
    if _quarantine_orphan_experiment_dir(store_root):
        actions.append("quarantined mlruns/tokyo-eyes-v6/ (artifact dir mistaken for experiment)")

    for meta_path in store_root.glob("*/*/meta.yaml"):
        if meta_path.parent.parent.name in {".trash", "models"}:
            continue
        if _repair_missing_run_uuid(meta_path):
            actions.append(f"added run_uuid to {meta_path.parent.name}")

    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair local MLflow file store")
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("mlruns"),
        help="MLflow tracking store root (default: mlruns)",
    )
    args = parser.parse_args()
    store_root = args.store.resolve()
    if not store_root.is_dir():
        raise SystemExit(f"Store not found: {store_root}")
    actions = repair_store(store_root)
    if not actions:
        print("MLflow store OK — no repairs needed.")
        return
    for line in actions:
        print(line)


if __name__ == "__main__":
    main()
