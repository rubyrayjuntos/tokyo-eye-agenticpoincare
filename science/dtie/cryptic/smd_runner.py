"""SMD runner entry point for the science container.

CLI interface for steered molecular dynamics validation of cryptic binding sites.
This module dispatches to the real OpenMM-based SMD runner.

Usage:
    python -m science.dtie.cryptic.smd_runner --spec-json <path> --protocol <name>

Output: Structured JSON on stdout with:
    - success: bool
    - status: "passed" | "failed"
    - protocol: str
    - work_kcal_mol: float
    - strain_delta: float
    - duration_ms: int
    - notes: str
    - metrics: dict

Supports protocols:
    SMD_three_phase, SMD_stent_stabilization, SMD_lid_restraint,
    SMD_clamp_stabilization, SMD_strain_relief

Requirements: 6.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from science.dtie.cryptic.stub_control import is_stub_mode

logger = logging.getLogger(__name__)

# Default timeout for GPU jobs (seconds)
DEFAULT_TIMEOUT_SECONDS = 3600

VALID_PROTOCOLS = {
    "SMD_three_phase",
    "SMD_stent_stabilization",
    "SMD_lid_restraint",
    "SMD_clamp_stabilization",
    "SMD_strain_relief",
}


def _run_stub(spec_json_path: str, protocol: str) -> dict[str, Any]:
    """Stub fallback — returns synthetic results for testing without GPU/OpenMM.

    Only used when stub mode is active (SMD_STUB_MODE=1 + user approval).
    See science.dtie.cryptic.stub_control for the gating logic.
    """
    stub_results: dict[str, dict[str, Any]] = {
        "SMD_three_phase": {
            "work_kcal_mol": 12.4,
            "strain_delta": -2.1,
        },
        "SMD_stent_stabilization": {
            "work_kcal_mol": 8.2,
            "strain_delta": -2.3,
        },
        "SMD_lid_restraint": {
            "work_kcal_mol": 15.1,
            "strain_delta": -1.8,
        },
        "SMD_clamp_stabilization": {
            "work_kcal_mol": 6.5,
            "strain_delta": -0.7,
        },
        "SMD_strain_relief": {
            "work_kcal_mol": 9.8,
            "strain_delta": -1.2,
        },
    }

    values = stub_results.get(protocol, {"work_kcal_mol": 10.0, "strain_delta": -1.5})

    return {
        "success": True,
        "status": "passed",
        "protocol": protocol,
        "work_kcal_mol": values["work_kcal_mol"],
        "strain_delta": values["strain_delta"],
        "duration_ms": 500,
        "notes": "STUB: Synthetic SMD result — no real physics performed.",
        "metrics": {},
    }


def _run_real(spec_json_path: str, protocol: str) -> dict[str, Any]:
    """Run the real OpenMM-based SMD simulation."""
    from science.dtie.cryptic.smd_runner_real import run_smd

    result = run_smd(spec_json_path, protocol)
    return json.loads(result.to_json())


def run(spec_json_path: str, protocol: str) -> dict[str, Any]:
    """Execute SMD for the given spec and protocol.

    Dispatches to either the real OpenMM runner or stub depending on
    the centralized stub_control module (SMD_STUB_MODE env var + explicit
    user approval).

    Args:
        spec_json_path: Path to site specification JSON.
        protocol: SMD protocol name.

    Returns:
        Dict with success, status, protocol, work_kcal_mol, strain_delta,
        duration_ms, notes, metrics.
    """
    if protocol not in VALID_PROTOCOLS:
        return {
            "success": False,
            "status": "failed",
            "protocol": protocol,
            "work_kcal_mol": 0.0,
            "strain_delta": 0.0,
            "duration_ms": 0,
            "notes": f"Invalid protocol '{protocol}'. Valid: {sorted(VALID_PROTOCOLS)}",
            "metrics": {},
        }

    if is_stub_mode():
        logger.warning("SMD STUB MODE ACTIVE — returning synthetic results.")
        return _run_stub(spec_json_path, protocol)

    return _run_real(spec_json_path, protocol)


def main() -> None:
    """CLI entry point.

    Usage:
        python -m science.dtie.cryptic.smd_runner --spec-json <path> --protocol <name>
    """
    parser = argparse.ArgumentParser(
        description="Steered MD runner for cryptic site validation"
    )
    parser.add_argument(
        "--spec-json",
        required=True,
        help="Path to site specification JSON file",
    )
    parser.add_argument(
        "--protocol",
        required=True,
        choices=sorted(VALID_PROTOCOLS),
        help="SMD protocol to execute",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    result = run(args.spec_json, args.protocol)
    print(json.dumps(result, indent=2))

    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
