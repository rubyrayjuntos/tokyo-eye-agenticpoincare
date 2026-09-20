"""§2.4 pure_hyp_pass, full-spine extension — real spine forward, Stage-A-12.

Extends ``scripts/verify_pure_hyp_pass_live_moe.py`` (that script's isolated
``TopologyAwareHardMoE``-only scan is unchanged and stays the canonical stamp
at ``data/gates/tokyo_eye_equ_pure_hyp_live_moe_scan.json``). Per CLAUDE.md
item 2: "wrap ``scan_forward`` around a real spine forward on Stage-A-12
(not MoE-only)."

Reuses the same real-trunk construction as the full-spine decay-floor-hold
script: ``experiments/training/v8/run_v8_experiment.build_system`` (real
``TokyoEyesHyperbolicV8`` + stub-but-live SE(3)-lite frontend, cold
random-init, addendum §2.3) and the real 12-structure Stage-A-12 manifest
(``manifests/v8_stage_a_small_v1.json``).

**Known scope gap (surfaced, not silently patched):** ``scan_forward``
(``science/tokyo_eye/v8/pure_hyp_pass.py:602``) calls ``model.eval()``
unconditionally before tracing the forward, regardless of what mode the
caller set beforehand. ``TopologyAwareHardMoE.forward`` only takes the
Gumbel-hard-STE + epsilon-override branch ``if self.training`` (moe.py:291);
eval mode always takes the argmax branch. So this scan — like the existing
live-MoE scan's mislabeled "train" cases — only ever traces the **eval-mode
argmax forward**. Train-mode Gumbel/epsilon-override ops are NOT exercised by
this check today. That is a tracer-semantics limitation of ``scan_forward``
itself (an addendum-track decision to fix, not something to work around here
by monkeypatching this script's own copy of ``model.training``).

Varies ``tau_ceiling`` (schedule start vs. end) per structure instead of
``explore_epsilon`` (which is moot under the forced-eval limitation above) —
``tau_start``=0.70 is the closest the schedule gets to the documented
near-origin false-negative edge case in ``pure_hyp_pass.py``'s own docstring
(ball detection needs radius > 1e-4).

Static/no-grad property check — does not touch MLflow, matching the
isolated live-MoE scan's convention.

Writes data/gates/tokyo_eye_equ_pure_hyp_full_spine_scan.json (sibling; does
NOT touch tokyo_eye_equ_pure_hyp_live_moe_scan.json).

Placement: /tmp/ because this account has no write access to
/workspace/scripts/ (same constraint as the decay-floor-hold full-spine
script). Host agent: copy into scripts/, ideally as this file's counterpart
next to verify_pure_hyp_pass_live_moe.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import torch

from experiments.training.v8.run_v8_experiment import build_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.freeze_reconciliation import PURE_HYP_PASS_VERSION
from science.tokyo_eye.v8.loader import TokyoEyeCuratedDataset
from science.tokyo_eye.v8.pure_hyp_pass import scan_forward

REPO_ROOT = Path("/workspace")
MANIFEST_PATH = REPO_ROOT / "manifests" / "v8_stage_a_small_v1.json"
PDB_DIR = REPO_ROOT / "pdb_cache"
GRAPH_CACHE_DIR = PDB_DIR / "v8_graph_cache"
GATE_ID = "tokyo_eye_equ_pure_hyp_full_spine_scan"
STAMP_PATH = REPO_ROOT / "data" / "gates" / f"{GATE_ID}.json"


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")

    cfg = load_weight_map(REPO_ROOT / DEFAULT_WEIGHT_MAP)
    system, load_info = build_system(
        cfg, equiformer_ckpt=None, device=device, freeze_backbone=False
    )
    system.eval()

    dataset = TokyoEyeCuratedDataset(
        manifest_path=MANIFEST_PATH,
        pdb_dir=PDB_DIR,
        use_graph_cache=True,
        graph_cache_dir=GRAPH_CACHE_DIR,
    )
    assert dataset.mode == "manifest"
    n_structures = len(dataset)
    structure_tags = [f"{e['pdb_id']}:{e['chain']}" for e in dataset.entries]

    tau_cases = [
        ("tau_end", float(cfg["tau_end"])),
        ("tau_start", float(cfg["tau_start"])),
    ]

    results: list[dict] = []
    for idx in range(n_structures):
        batch = dataset.get_on_device(idx, device)
        for tau_label, tau_val in tau_cases:
            report = scan_forward(
                system,
                lambda b=batch, t=tau_val: system(
                    b["x"], b["edge_index"], b["edge_type"],
                    tau_ceiling=t, chem=b.get("gate_chem"),
                ),
            )
            row = {
                "pdb_id": batch["pdb_id"],
                "chain": batch["chain"],
                "num_nodes": int(batch["num_nodes"]),
                "tau_case": tau_label,
                "tau_ceiling": tau_val,
                "pure_hyp_pass": bool(report.passed),
                "violations": [
                    {"kind": v.kind, "module": v.module_name, "detail": v.detail}
                    for v in report.violations
                ],
            }
            results.append(row)
            status = "OK" if row["pure_hyp_pass"] else "VIOLATION"
            print(
                f"[full_spine] [{status}] {row['pdb_id']}:{row['chain']} "
                f"N={row['num_nodes']} tau={tau_label}={tau_val:.3f}"
            )
            for v in row["violations"]:
                print(f"      - {v['kind']} in {v['module']}: {v['detail']}")

    all_pass = all(r["pure_hyp_pass"] for r in results)

    stamp = {
        "gate_id": GATE_ID,
        "display_lineage": "Tokyo Eye EQU",
        "status": "PASS" if all_pass else "FAIL",
        "scope": (
            "full spine (TokyoEyesHyperbolicV8 + stub-but-live SE(3)-lite "
            "frontend, cold random-init), real 12-structure Stage-A-12 "
            "manifest (manifests/v8_stage_a_small_v1.json)"
        ),
        "addendum_clause": "freeze_reconciliation_addendum_2.4_live_forward_tracer",
        "pure_hyp_pass_version": PURE_HYP_PASS_VERSION,
        "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "structures": structure_tags,
        "tau_cases": {label: val for label, val in tau_cases},
        "backbone_mode": load_info.get("backbone_mode"),
        "equiformer_mode": load_info.get("mode"),
        "cases": results,
        "root_cause_analysis": (
            "All 24 cases fail at the SAME site: science/tokyo_eye/v8/projector.py:53 "
            "(RadialAngularProjector.forward), 'tangent = alpha * a_hat + scalar_t'. "
            "Confirmed false positive, not a genuine ambient hyperbolic-purity defect: "
            "alpha*a_hat and scalar_t are both pre-exp0 TANGENT-SPACE Euclidean "
            "quantities (a_hat is a unit direction, alpha in (0, tau_ceiling), "
            "scalar_t = 0.01*tanh(...)) — the ball point 'z' is only produced by "
            "exp_map_zero(tangent, ...) on the NEXT line (projector.py:57). "
            "Two things conspire to mis-flag it: (1) the documented numeric-only "
            "'Euclidean-feature false positive' limitation in pure_hyp_pass.py's own "
            "docstring — alpha's range (0, tau_ceiling~0.7-0.995) coincidentally "
            "overlaps _is_open_ball_batch's detection band (1e-4, 0.999); (2) a real "
            "gap in _is_pre_lift_scope/_is_pre_lift_linear (pure_hyp_pass.py:144-161): "
            "the matcher requires the module name to START WITH 'projector' or contain "
            "'.projector.' as a MIDDLE segment, so it correctly exempts the projector's "
            "*children* (e.g. 'spine.projector.radial_mlp' matches '.projector.') but "
            "misses the projector module's OWN top-level forward body when nested as a "
            "trailing segment — 'spine.projector' (under TokyoEyeV8WithFrontend, i.e. "
            "exactly the full-spine wrapping this script uses) matches neither pattern, "
            "so _skip_prelift is never set while this line executes. Verified directly: "
            "_is_pre_lift_scope('spine.projector') == False, "
            "_is_pre_lift_scope('spine.projector.radial_mlp') == True. This is why the "
            "MoE-isolated scan (tokyo_eye_equ_pure_hyp_live_moe_scan.json) never saw it "
            "— it never runs the projector at all — and why unit tests on a bare "
            "TokyoEyesHyperbolicV8 (name=='projector', matches startswith) would not "
            "reproduce it either. Recommended fix (NOT applied here — tracer-semantics "
            "change, needs its own addendum sign-off, same principle as the "
            "forced-eval() caveat below): widen the suffix case, e.g. "
            "'n == \"projector\" or n.endswith(\".projector\") or ...'. status stays FAIL "
            "as the raw tracer verdict; this field records the confirmed root cause "
            "without unilaterally overriding it."
        ),
        "known_caveat_scan_forward_forces_eval": (
            "scan_forward (pure_hyp_pass.py) calls model.eval() unconditionally "
            "before tracing, so this scan — like the existing live-MoE scan's "
            "'train' cases — only ever exercises the eval-mode argmax forward. "
            "TopologyAwareHardMoE's Gumbel-hard-STE + epsilon-override branch "
            "(self.training-gated, moe.py:291) is NOT traced by this check. "
            "Fixing that is a tracer-semantics change on an addendum track, "
            "not something patched around inside this verify script."
        ),
        "note": (
            "Necessary, not sufficient, and eval-mode-only per the caveat above. "
            "Does not touch MLflow (static/property check, no training run). "
            "Does not overwrite tokyo_eye_equ_pure_hyp_live_moe_scan.json "
            "(MoE-isolated scope)."
        ),
    }
    STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAMP_PATH.write_text(json.dumps(stamp, indent=2))

    print(f"\nfull-spine pure_hyp_pass scan: {stamp['status']}")
    print(f"  structures: {', '.join(structure_tags)}")
    print(f"  cases: {len(results)} ({n_structures} structures x {len(tau_cases)} tau settings)")
    print(f"  stamp written: {STAMP_PATH}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
