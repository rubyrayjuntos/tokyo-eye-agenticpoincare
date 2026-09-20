"""Run the real §2.4 pure_hyp_pass scanner against the production
TopologyAwareHardMoE (not a synthetic known-bad/known-good fixture).

Context: the freeze reconciliation addendum (2026-09-16) retired the old
grep-based pure_hyp_pass and requires the trunk-wide tracer in
science/tokyo_eye/v8/pure_hyp_pass.py to show green on the known-bad/known-good
property fixtures *before* it gates any train/promote stamp (tests/v8/
test_pure_hyp_pass.py, test_pure_hyp_static.py). Those fixtures were run
separately (23/24 passing after fixing a _route() signature drift in
test_soft_moe_routing_mix_fails — moe.py's _route now returns
(routing, override_mask) post-epsilon-greedy, the fixture still unpacked one
value).

This script complements the fixtures by scanning the actual module the spine
instantiates (hidden_dim=128 to match the EquiformerV3 frontend's observed
scalar width, gate_hidden=16 / num_experts=4 per moe.py defaults), in both
train mode (Gumbel hard STE + epsilon override) and eval mode (argmax), on a
few different explore_epsilon settings. A clean report here is necessary but
not sufficient: it does not replace running this scan inside a real spine
forward pass over real Stage-A-12 structures once the §3 cold rebuild reaches
step 4 (spine integration, "pure_hyp_pass re-run trunk-wide").

Writes data/gates/tokyo_eye_equ_pure_hyp_live_moe_scan.json (that directory is
the one writable path under /workspace for this account) and prints a summary.
Does not touch MLflow — this is a static/property check, not a training run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import torch

from science.tokyo_eye.v8.moe import TopologyAwareHardMoE
from science.tokyo_eye.v8.pure_hyp_pass import scan_forward

HIDDEN_DIM = 128
GATE_HIDDEN = 16
NUM_EXPERTS = 4
N_NODES = 64


def _synthetic_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N_NODES, HIDDEN_DIM, generator=g) * 0.03
    src = torch.randint(0, N_NODES, (4 * N_NODES,), generator=g)
    dst = torch.randint(0, N_NODES, (4 * N_NODES,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return z, edge_index


def run_one(*, mode: str, explore_epsilon: float, seed: int) -> dict:
    torch.manual_seed(seed)
    moe = TopologyAwareHardMoE(
        HIDDEN_DIM,
        num_experts=NUM_EXPERTS,
        gate_hidden=GATE_HIDDEN,
        temperature=0.5,
        explore_epsilon=explore_epsilon,
    )
    moe.train() if mode == "train" else moe.eval()
    z, edge_index = _synthetic_batch(seed)
    report = scan_forward(moe, lambda: moe(z, edge_index))
    return {
        "mode": mode,
        "explore_epsilon": explore_epsilon,
        "seed": seed,
        "pure_hyp_pass": report.passed,
        "violations": [
            {"kind": v.kind, "module": v.module_name, "detail": v.detail}
            for v in report.violations
        ],
    }


def main() -> int:
    cases = [
        dict(mode="eval", explore_epsilon=0.0, seed=0),
        dict(mode="train", explore_epsilon=0.0, seed=1),
        dict(mode="train", explore_epsilon=0.20, seed=2),
        dict(mode="train", explore_epsilon=1.0, seed=3),
    ]
    results = [run_one(**c) for c in cases]
    all_pass = all(r["pure_hyp_pass"] for r in results)

    stamp = {
        "gate_id": "tokyo_eye_equ_pure_hyp_live_moe_scan",
        "display_lineage": "Tokyo Eye EQU",
        "status": "PASS" if all_pass else "FAIL",
        "scope": "isolated TopologyAwareHardMoE forward (not full spine)",
        "hidden_dim": HIDDEN_DIM,
        "gate_hidden": GATE_HIDDEN,
        "num_experts": NUM_EXPERTS,
        "cases": results,
        "note": (
            "Necessary, not sufficient: real sign-off requires this scan "
            "wrapping the full TokyoEyesHyperbolicV8 forward on real "
            "Stage-A-12 batches (freeze addendum §3 step 4)."
        ),
    }
    out_path = Path("/workspace/data/gates/tokyo_eye_equ_pure_hyp_live_moe_scan.json")
    out_path.write_text(json.dumps(stamp, indent=2))

    print(f"pure_hyp_pass live MoE scan: {'PASS' if all_pass else 'FAIL'}")
    for r in results:
        status = "OK" if r["pure_hyp_pass"] else "VIOLATION"
        print(f"  [{status}] mode={r['mode']:5s} eps={r['explore_epsilon']:.2f} seed={r['seed']}")
        for v in r["violations"]:
            print(f"      - {v['kind']} in {v['module']}: {v['detail']}")
    print(f"stamp written: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
