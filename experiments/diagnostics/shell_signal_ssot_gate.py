"""CLI: τ-rim outer-shell + uncertainty spread gate (post-ingest or offline).

Usage:
    python -m experiments.diagnostics.shell_signal_ssot_gate \\
        --structure-id 4obe \\
        --viewer-dir data/local_objects/gnn_viewer

    python -m experiments.diagnostics.shell_signal_ssot_gate \\
        --gate-json data/local_objects/gnn_viewer/4obe/4obe_shell_signal_gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_nodes_from_gate_json(path: Path) -> tuple[list, bool]:
    """If gate JSON exists, only re-print; nodes not required."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="τ-rim shell signal SSOT gate")
    parser.add_argument("--structure-id", type=str, default=None)
    parser.add_argument(
        "--viewer-dir",
        type=Path,
        default=Path("data/local_objects/gnn_viewer"),
    )
    parser.add_argument("--gate-json", type=Path, default=None, help="Existing gate report")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on gate failure")
    args = parser.parse_args(argv)

    if args.gate_json is not None:
        if not args.gate_json.is_file():
            logger.error("Gate JSON not found: %s", args.gate_json)
            return 2
        report = json.loads(args.gate_json.read_text(encoding="utf-8"))
        passed = bool(report.get("passed"))
        print(json.dumps(report, indent=2))
        if args.strict and not passed:
            return 1
        return 0

    sid = (args.structure_id or "").strip().lower()
    if not sid:
        logger.error("Provide --structure-id or --gate-json")
        return 2

    gate_path = args.viewer_dir / sid / f"{sid}_shell_signal_gate.json"
    if gate_path.is_file():
        report = json.loads(gate_path.read_text(encoding="utf-8"))
        print(json.dumps(report, indent=2))
        if args.strict and not report.get("passed"):
            return 1
        return 0

    # Evaluate from persisted nodes if gate not yet written (manual run after inference)
    from science.dtie.v6.visualization.shell_signal_gate import (
        evaluate_shell_signal_ssot,
        write_shell_gate_report,
    )

    nodes_path = args.viewer_dir / sid / f"{sid}_gnn_nodes.json"
    if nodes_path.is_file():
        raw = json.loads(nodes_path.read_text(encoding="utf-8"))
        from science.dtie.common.interfaces import GNNNodeOutput

        nodes = [GNNNodeOutput(**nd) for nd in raw]
    else:
        logger.error(
            "No gate at %s — run gnn_inference first or pass --gate-json",
            gate_path,
        )
        return 2

    verdict = evaluate_shell_signal_ssot(nodes, structure_id=sid, structural_disc_frozen=True)
    write_shell_gate_report(verdict, gate_path)
    print(json.dumps(verdict.to_dict(), indent=2))
    if args.strict and not verdict.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
