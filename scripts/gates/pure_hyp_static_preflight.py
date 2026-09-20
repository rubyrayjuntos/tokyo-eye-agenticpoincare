#!/usr/bin/env python3
"""tokyo_eye_equ_correct_start pre-flight: static pure_hyp audit inside science container."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

GATE_ID = "tokyo_eye_equ_correct_start"
SCIENCE = Path("/app/science/tokyo_eye/v8")
TARGETS = [
    SCIENCE / "attention.py",
    SCIENCE / "affinity_head.py",
]

FORBIDDEN_TOKENS = (
    "_tangent_linear",
    "def _tangent_linear",
    "TangentLinear",
    "TangentPool",
    "use_tangent_shortcut=True",
    "use_tangent_shortcut = True",
)


def hits_for(src: str) -> list[str]:
    hits: list[str] = []
    for token in FORBIDDEN_TOKENS:
        if token in src:
            hits.append(token)
    # Classic substitute: Linear applied in tangent via log/exp
    if "def _tangent_linear" in src or "lin(log_map_zero" in src.replace(" ", ""):
        hits.append("log0_Linear_exp0_substitute")
    # Tangent residual mix as primary path
    if "u_self + u_att" in src or "u_self+u_att" in src.replace(" ", ""):
        hits.append("tangent_residual_mix")
    # Tangent Euclidean pool
    if "t_pool" in src or ("log_map_zero" in src and "torch.sum(w * u" in src):
        hits.append("tangent_weighted_pool")
    # Output map still W_o(log0)
    if "W_o(log_map_zero" in src.replace(" ", "") or "self.W_o(log_map_zero" in src:
        hits.append("W_o_log0_output")
    return sorted(set(hits))


def main() -> int:
    findings = {}
    all_hits: list[str] = []
    missing: list[str] = []
    for path in TARGETS:
        if not path.is_file():
            missing.append(str(path))
            continue
        h = hits_for(path.read_text(encoding="utf-8"))
        findings[str(path)] = h
        all_hits.extend(f"{path.name}:{x}" for x in h)

    pure_pass = (not missing) and (not all_hits)
    payload = {
        "gate_id": GATE_ID,
        "check": "pure_hyp_static_preflight",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pure_hyp_pass": pure_pass,
        "findings": findings,
        "missing": missing,
        "verdict": "PASS" if pure_pass else "FAIL",
        "note": (
            "Pass: no post-lift tangent geometry substitutes in attention/affinity."
            if pure_pass
            else "Fail: forbidden post-lift tangent patterns still present."
        ),
    }
    out = Path("/tmp/tokyo_eye_equ_correct_start_preflight.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"\nWROTE {out}", file=sys.stderr)
    return 0 if pure_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
