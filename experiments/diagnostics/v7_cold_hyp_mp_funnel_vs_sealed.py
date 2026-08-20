#!/usr/bin/env python3
"""Sealed Θ vs cold Hyp MP funnel best — same Cα graph, report-only biology bars.

Fair compare (both arms use default Hyp MP / Cα contact edges):
  sealed = HEALTHY_V7_CKPT
  cold   = tokyo_eye_v7_cold_hyp_mp_funnel_v1 best

Pre-reg: docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-prereg.md
Bars are report-only after disc Pass (not a hard promote gate).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.diagnostics.v7_hyp_biology_mp_sealed_vs_biology import (
    BASIN_GAP_MIN,
    DISC_MIN,
    _hub_jaccard,
    _load_model,
    grade_basin,
    grade_migration,
    grade_telemetry,
)
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.tokyo_eye.TokyoEye import TokyoEye

DEFAULT_COLD = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_cold_hyp_mp_funnel_v1/v7_cold_hyp_mp_funnel_best.pt"
)
DEFAULT_OUT = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_cold_hyp_mp_funnel_v1/sealed_vs_cold_scorecard.json"
)


def _load_cold_model(ckpt: Path, device: str) -> torch.nn.Module:
    """Cold snaps may omit arch.version=v7; rebuild TokyoEye with train-time flags."""
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
    model = TokyoEye(
        node_dim=3,
        hidden=128,
        num_layers=6,
        num_experts=4,
        hyp_mp_primary=True,
        se3_aux=False,
        hyp_mp_layers=3,
        hyp_biology_mp=False,
        hyperbolic_gate=True,
        hyperbolic_expert_mix=False,
        topology_only_gate=True,
        gate_include_sasa=True,
        multi_rel_edge_mp=True,
        role_edge_mp=True,
        disc_projection_path="pre_routing",
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"cold load incomplete missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.to(device)
    model.eval()
    model.hyp_mp_primary = True
    model.hyp_biology_mp = False
    if hasattr(model, "se3_aux"):
        model.se3_aux = False
    return model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sealed", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--cold", type=Path, default=DEFAULT_COLD)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
    )
    p.add_argument("--skip-migration", action="store_true")
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    for path in (args.sealed, args.cold):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    arms: dict[str, Any] = {}
    for name, ckpt in (("sealed", args.sealed), ("cold", args.cold)):
        print(f"=== arm {name} ({ckpt}) ===", flush=True)
        if name == "cold":
            model = _load_cold_model(ckpt, args.device)
        else:
            model = _load_model(ckpt, args.device, biology=False)
        print("telemetry ...", flush=True)
        tel = grade_telemetry(model, args.pdb_dir, args.device, biology=False)
        print("basin ...", flush=True)
        basin = grade_basin(model, args.pdb_dir, args.device, biology=False)
        mig = None
        if not args.skip_migration:
            print("migration ...", flush=True)
            mig = grade_migration(
                model,
                args.pdb_dir,
                args.device,
                biology=False,
                corpus_cache=args.corpus_cache if args.corpus_cache.is_file() else None,
            )
        arms[name] = {
            "checkpoint": str(ckpt),
            "biology_mp": False,
            "simple_graph": "ca_contact",
            "telemetry": tel,
            "basin": basin,
            "migration": mig,
        }

    jaccards = {}
    sealed_tel = {r["pdb_id"]: r for r in arms["sealed"]["telemetry"]["targets"]}
    cold_tel = {r["pdb_id"]: r for r in arms["cold"]["telemetry"]["targets"]}
    for pdb_id in sealed_tel:
        jaccards[pdb_id] = _hub_jaccard(
            sealed_tel[pdb_id].get("hub_auth_resseqs") or [],
            cold_tel[pdb_id].get("hub_auth_resseqs") or [],
        )

    b_s, b_c = arms["sealed"]["basin"], arms["cold"]["basin"]
    basin_improve = bool(b_c["basin_sep_gap"] > b_s["basin_sep_gap"])
    m_s, m_c = arms["sealed"]["migration"], arms["cold"]["migration"]
    mig_improve = None
    if m_s and m_c:
        mig_improve = bool(m_c["delta_R"] > m_s["delta_R"])

    scorecard = {
        "mode": "report_only_after_disc_pass",
        "graph": "ca_contact_both_arms",
        "disc_cold": {
            "pass": bool(b_c["disc_pass"]),
            "disc_r_mean": b_c["disc_r_mean"],
            "bar": DISC_MIN,
        },
        "disc_sealed": {
            "pass": bool(b_s["disc_pass"]),
            "disc_r_mean": b_s["disc_r_mean"],
            "bar": DISC_MIN,
        },
        "basin_cold_vs_bar": {
            "pass": bool(b_c["basin_pass"]),
            "gap": b_c["basin_sep_gap"],
            "bar": BASIN_GAP_MIN,
        },
        "basin_sealed_vs_bar": {
            "pass": bool(b_s["basin_pass"]),
            "gap": b_s["basin_sep_gap"],
            "bar": BASIN_GAP_MIN,
        },
        "basin_improve_vs_sealed": {
            "pass": basin_improve,
            "gap_sealed": b_s["basin_sep_gap"],
            "gap_cold": b_c["basin_sep_gap"],
            "delta_gap": b_c["basin_sep_gap"] - b_s["basin_sep_gap"],
        },
        "migration_cold_vs_bar": {
            "pass": bool(m_c["pass_form"]) if m_c else None,
            "delta_R": None if m_c is None else m_c["delta_R"],
            "R_4OBE": None if m_c is None else m_c["R_4OBE"],
            "R_4DSO": None if m_c is None else m_c["R_4DSO"],
        },
        "migration_sealed_vs_bar": {
            "pass": bool(m_s["pass_form"]) if m_s else None,
            "delta_R": None if m_s is None else m_s["delta_R"],
            "R_4OBE": None if m_s is None else m_s["R_4OBE"],
            "R_4DSO": None if m_s is None else m_s["R_4DSO"],
        },
        "migration_improve_vs_sealed": {
            "pass": mig_improve,
            "delta_R_sealed": None if m_s is None else m_s["delta_R"],
            "delta_R_cold": None if m_c is None else m_c["delta_R"],
        },
        "telemetry_hub_jaccard_sealed_vs_cold": jaccards,
    }

    cold_signal = bool(basin_improve or (mig_improve is True))
    overall = {
        "report_only": True,
        "disc_cold_pass": bool(b_c["disc_pass"]),
        "cold_beats_sealed_basin": basin_improve,
        "cold_beats_sealed_migration": mig_improve,
        "cold_signal_vs_sealed": cold_signal,
        "cold_bar_pass": bool(b_c["basin_pass"])
        or (bool(m_c["pass_form"]) if m_c else False),
        "verdict": (
            "DISC_PASS_COLD_BEATS_SEALED"
            if bool(b_c["disc_pass"]) and cold_signal
            else "DISC_PASS_COLD_FLAT_OR_WORSE"
            if bool(b_c["disc_pass"])
            else "DISC_FAIL"
        ),
        "signal_strength": (
            "strong"
            if cold_signal
            and (
                b_c["basin_pass"]
                or (m_c and m_c["pass_form"] and (m_c["delta_R"] or 0) > 0.02)
            )
            else "weak"
            if cold_signal
            else "none"
        ),
    }

    payload = {
        "schema_version": 1,
        "probe": "tokyo_eye_v7_cold_hyp_mp_funnel_vs_sealed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "lineage_id": "tokyo_eye_v7_cold_hyp_mp_funnel_v1",
        "spec": "docs/specs/tokyo-eye-v7/cold-hyp-mp-funnel-prereg.md",
        "arms": arms,
        "scorecard": scorecard,
        "overall": overall,
    }
    for arm in payload["arms"].values():
        mig = arm.get("migration")
        if mig and "structures" in mig:
            for st in mig["structures"].values():
                st.pop("out_effect", None)
                st.pop("delta_rows", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"overall": overall, "out": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
