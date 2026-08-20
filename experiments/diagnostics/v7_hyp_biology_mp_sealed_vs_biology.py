#!/usr/bin/env python3
"""Sealed Θ vs hyp_biology_mp best — telemetry, basin gap, G12D hub migration.

Fair compare:
  sealed  = HEALTHY_V7_CKPT with default Hyp MP edges (Cα fallback allowed)
  biology = v7_hyp_biology_mp_best.pt with biology graph attach (Cα forbidden)

Bars (scorecard):
  - contract: biology arm ca_in_mp == false (hard)
  - disc:    disc_r_mean >= 0.25 (report + soft)
  - basin:   gap = mean_cross(OFF,ON) - mean_same; Pass if gap > 0.10;
             Improve if gap_biology > gap_sealed
  - migrate: Pass if R_4DSO > R_4OBE; Improve if delta_R_bio > delta_R_sealed
  - telemetry: report Jaccard hub overlap sealed↔bio (not a Pass bar)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.diagnostics.kras_hub_migration_ood import (
    EXPECTED,
    load_prot,
    resolve_hub_indices,
)
from experiments.diagnostics.kras_knockout_causal import _pdb_residue_names
from experiments.diagnostics.v7_kras_g12d_hub_migration import (
    knockout_scan_x_hyp,
    score_knockout,
)
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6._data import _download_pdb
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.dtie.common.kras_g12_graft import neighborhood_n12
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.tokyo_eye.basin_contrastive import (
    ball_distance_embeddings,
    r_star_indices,
    structure_embedding_logmap0,
)
from science.tokyo_eye.biology_graph import attach_biology_mp_graph
from science.tokyo_eye.hyp_mp_telemetry import attach_hyp_mp_telemetry
from science.tokyo_eye.thermo_edge_features import resolve_residue_records_for_prot
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_BIO = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/v7_hyp_biology_mp_best.pt"
)
DEFAULT_OUT = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/sealed_vs_biology_scorecard.json"
)
BASIN_ROSTER = {
    "OFF": [("4LPK", "A"), ("5US4", "A")],
    "ON": [("6GOD", "A"), ("6GOF", "A")],
}
TEL_TARGETS = [("4OBE", "A"), ("4LPK", "A"), ("6GOD", "A")]
BASIN_GAP_MIN = 0.10
DISC_MIN = 0.25


def _load_model(ckpt: Path, device: str, *, biology: bool) -> torch.nn.Module:
    model = load_model_from_checkpoint(ckpt, device)
    model.eval()
    model.hyp_mp_primary = True
    if biology:
        model.hyp_biology_mp = True
        if hasattr(model, "se3_aux"):
            model.se3_aux = False
    else:
        model.hyp_biology_mp = False
    return model


def _attach_if_biology(
    data: Any,
    prot: dict[str, Any],
    n: int,
    pdb_dir: Path,
    *,
    biology: bool,
) -> dict[str, Any]:
    if not biology:
        return {"ca_in_mp": None, "hyp_mp_edges": "default"}
    records = resolve_residue_records_for_prot(prot, pdb_dir=pdb_dir)
    coords = prot.get("ca_coords")
    if coords is None:
        raise RuntimeError(f"{prot.get('pdb_id')}: missing ca_coords")
    rho = prot["target_rho"].reshape(-1).detach().cpu().numpy()
    attach_biology_mp_graph(
        data,
        coords=coords[:n] if hasattr(coords, "__getitem__") else coords,
        rho=rho[:n],
        residue_ids=list(prot.get("residue_ids") or [])[:n],
        residue_records=records,
    )
    return getattr(data, "biology_mp_audit", {}) or {}


def _forward(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    pdb_dir: Path,
    *,
    biology: bool,
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    audit = _attach_if_biology(data, prot, n, pdb_dir, biology=biology)
    with torch.no_grad():
        out = model(data)
    return out, prot, n, audit


def _disc_r(out: dict[str, Any]) -> float:
    disc = out.get("hyp_projections_2d")
    if disc is None:
        disc = out.get("hyp_proj_2d")
    if disc is None or not torch.is_tensor(disc):
        return float("nan")
    return float(torch.linalg.vector_norm(disc.float(), dim=-1).mean().item())


def _curv(out: dict[str, Any], model: torch.nn.Module) -> float:
    c = (out.get("audit_trail") or {}).get("curvature_value")
    if c is None:
        return float(model.curvature.detach().cpu().reshape(-1)[0])
    return float(c.detach().cpu()) if torch.is_tensor(c) else float(c)


def grade_telemetry(
    model: torch.nn.Module,
    pdb_dir: Path,
    device: str,
    *,
    biology: bool,
) -> dict[str, Any]:
    rows = []
    for pdb_id, chain in TEL_TARGETS:
        prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
        if prot is None:
            raise FileNotFoundError(f"{pdb_id}:{chain}")
        print(f"  telemetry {pdb_id} biology={biology} ...", flush=True)
        prot = _align_prot_features(model, prot)
        data = prepare_training_batch(model, prot, device)
        n = residue_node_count(data, prot)
        bio_audit = _attach_if_biology(data, prot, n, pdb_dir, biology=biology)
        with torch.no_grad():
            out = model(data)
        attach_hyp_mp_telemetry(out, data, k_frac=0.10)
        tel = out["hyp_mp_telemetry"]
        ids = list(prot.get("residue_ids") or [])[:n]
        idx_map = residue_index_map(ids)
        rev = {i: rs for rs, i in idx_map.items()}
        hubs = [
            {
                "graph_index": int(i),
                "auth_resseq": rev.get(int(i)),
                "strength": float(s),
            }
            for i, s in zip(tel.get("hub_indices") or [], tel.get("hub_strengths") or [])
            if int(i) < n
        ]
        rows.append(
            {
                "pdb_id": pdb_id,
                "n_edges": tel.get("n_edges"),
                "n_hubs": tel.get("n_hubs"),
                "strength_cv": tel.get("strength_cv"),
                "disc_r_mean": _disc_r(out),
                "hub_auth_resseqs": [h["auth_resseq"] for h in hubs],
                "ca_in_mp": bio_audit.get("ca_in_mp"),
                "biology_edge_counts": (
                    bio_audit.get("edge_counts") if biology else None
                ),
            }
        )
    return {"targets": rows}


def grade_basin(
    model: torch.nn.Module,
    pdb_dir: Path,
    device: str,
    *,
    biology: bool,
) -> dict[str, Any]:
    prots: dict[str, dict[str, Any]] = {}
    for _basin, items in BASIN_ROSTER.items():
        for pdb_id, chain in items:
            prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
            if prot is None:
                raise FileNotFoundError(pdb_id)
            prots[pdb_id] = prot

    embeds: dict[str, torch.Tensor] = {}
    curvatures: dict[str, float] = {}
    disc_rs: list[float] = []
    audits: dict[str, Any] = {}

    for basin, items in BASIN_ROSTER.items():
        for pdb_id, chain in items:
            partner = None
            if basin == "OFF":
                partner = prots.get("5US4") if pdb_id == "4LPK" else prots.get("4LPK")
            else:
                partner = prots.get("6GOF") if pdb_id == "6GOD" else prots.get("6GOD")
            print(f"  basin embed {pdb_id} biology={biology} ...", flush=True)
            out, prot, n, bio_audit = _forward(
                model, prots[pdb_id], device, pdb_dir, biology=biology
            )
            audits[pdb_id] = bio_audit
            disc_rs.append(_disc_r(out))
            c = _curv(out, model)
            curvatures[pdb_id] = c
            idx_map = residue_index_map(list(prot.get("residue_ids") or [])[:n])
            present = set(idx_map.keys())
            try:
                n12 = (
                    neighborhood_n12(partner, prot)
                    if partner is not None
                    else neighborhood_n12(prot, prot)
                )
            except Exception:  # noqa: BLE001
                n12 = []
            idxs = [idx_map[r] for r in r_star_indices(present, n12) if r in idx_map]
            z = structure_embedding_logmap0(out["x_hyp"][:n], idxs, curvature=c)
            embeds[pdb_id] = z.cpu()

    c_ref = float(np.mean(list(curvatures.values())))
    cross = [
        ball_distance_embeddings(embeds[o], embeds[n], curvature=c_ref)
        for o, _ in BASIN_ROSTER["OFF"]
        for n, _ in BASIN_ROSTER["ON"]
    ]
    same = [
        ball_distance_embeddings(embeds["4LPK"], embeds["5US4"], curvature=c_ref),
        ball_distance_embeddings(embeds["6GOD"], embeds["6GOF"], curvature=c_ref),
    ]
    mean_cross = float(np.mean(cross))
    mean_same = float(np.mean(same))
    gap = mean_cross - mean_same
    disc_mean = float(np.nanmean(disc_rs))
    return {
        "basin_sep_gap": gap,
        "mean_cross_off_on": mean_cross,
        "mean_same_basin": mean_same,
        "disc_r_mean": disc_mean,
        "basin_pass": bool(gap > BASIN_GAP_MIN),
        "disc_pass": bool(np.isfinite(disc_mean) and disc_mean >= DISC_MIN),
        "ca_in_mp_any_true": any(a.get("ca_in_mp") is True for a in audits.values()),
    }


def _knockout_biology(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    pdb_dir: Path,
) -> dict[str, Any]:
    """Knockout scan with biology graph re-attached every forward."""
    from geoopt.manifolds.stereographic import math as pmath

    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    _attach_if_biology(data0, prot, n, pdb_dir, biology=True)
    with torch.no_grad():
        out0 = model(data0)
    x0 = out0["x_hyp"][:n]
    c = _curv(out0, model)
    k = torch.tensor(float(c), device=x0.device, dtype=x0.dtype)
    if float(k.item()) > 0:
        k = -k.abs()

    out_effect = np.zeros(n, dtype=np.float64)
    delta_rows: list[np.ndarray] = []
    for i in range(n):
        data = prepare_training_batch(model, prot, device)
        data.x = data.x.clone()
        data.x[i] = 0.0
        _attach_if_biology(data, prot, n, pdb_dir, biology=True)
        with torch.no_grad():
            x = model(data)["x_hyp"][:n]
            dist = pmath.dist(x0, x, k=k).detach().float().cpu().numpy().reshape(-1)
        delta = np.asarray(dist, dtype=np.float64)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out_effect[i] = float(np.mean(delta[mask]))
        delta_rows.append(delta)
    return {
        "n_residues": n,
        "curvature_c": float(c),
        "out_effect": out_effect,
        "delta_rows": delta_rows,
        "readout": "x_hyp_geodesic",
        "ca_in_mp": False,
    }


def grade_migration(
    model: torch.nn.Module,
    pdb_dir: Path,
    device: str,
    *,
    biology: bool,
    corpus_cache: Path | None,
) -> dict[str, Any]:
    structures: dict[str, Any] = {}
    for pdb_id, expected in EXPECTED.items():
        chain = "A"
        print(f"  migration knockout {pdb_id} biology={biology} ...", flush=True)
        prot = load_prot(
            pdb_id,
            chain=chain,
            pdb_dir=pdb_dir,
            corpus_cache=corpus_cache if pdb_id == "4OBE" else None,
        )
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.is_file():
            pdb_path = Path(_download_pdb(pdb_id, pdb_dir))
        indices = resolve_hub_indices(
            list(prot["residue_ids"]),
            _pdb_residue_names(pdb_path, chain),
            expected,
        )
        if biology:
            scan = _knockout_biology(model, prot, device, pdb_dir)
        else:
            scan = knockout_scan_x_hyp(model, prot, device)
        score = score_knockout(scan["out_effect"], scan["delta_rows"], indices)
        structures[pdb_id] = score

    r_obe = float(structures["4OBE"]["R_out_163"])
    r_dso = float(structures["4DSO"]["R_out_163"])
    delta = r_dso - r_obe
    return {
        "R_4OBE": r_obe,
        "R_4DSO": r_dso,
        "delta_R": delta,
        "pass_form": bool(r_dso > r_obe),
        "rank_163_4OBE": structures["4OBE"]["rank_out_163"],
        "rank_163_4DSO": structures["4DSO"]["rank_out_163"],
        "structures": {
            k: {kk: vv for kk, vv in v.items() if kk != "delta_rows"}
            for k, v in structures.items()
        },
    }


def _hub_jaccard(a: list[Any], b: list[Any]) -> float:
    sa, sb = set(x for x in a if x is not None), set(x for x in b if x is not None)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sealed", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--biology", type=Path, default=DEFAULT_BIO)
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
    for path in (args.sealed, args.biology):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    arms: dict[str, Any] = {}
    for name, ckpt, biology in (
        ("sealed", args.sealed, False),
        ("biology", args.biology, True),
    ):
        print(f"=== arm {name} ({ckpt}) ===", flush=True)
        model = _load_model(ckpt, args.device, biology=biology)
        print("telemetry ...", flush=True)
        tel = grade_telemetry(model, args.pdb_dir, args.device, biology=biology)
        print("basin ...", flush=True)
        basin = grade_basin(model, args.pdb_dir, args.device, biology=biology)
        mig = None
        if not args.skip_migration:
            print("migration ...", flush=True)
            mig = grade_migration(
                model,
                args.pdb_dir,
                args.device,
                biology=biology,
                corpus_cache=args.corpus_cache if args.corpus_cache.is_file() else None,
            )
        arms[name] = {
            "checkpoint": str(ckpt),
            "biology_mp": biology,
            "telemetry": tel,
            "basin": basin,
            "migration": mig,
        }

    # Telemetry hub Jaccard sealed vs biology (same PDB)
    jaccards = {}
    sealed_tel = {r["pdb_id"]: r for r in arms["sealed"]["telemetry"]["targets"]}
    bio_tel = {r["pdb_id"]: r for r in arms["biology"]["telemetry"]["targets"]}
    for pdb_id in sealed_tel:
        jaccards[pdb_id] = _hub_jaccard(
            sealed_tel[pdb_id].get("hub_auth_resseqs") or [],
            bio_tel[pdb_id].get("hub_auth_resseqs") or [],
        )

    b_s, b_b = arms["sealed"]["basin"], arms["biology"]["basin"]
    basin_improve = bool(b_b["basin_sep_gap"] > b_s["basin_sep_gap"])
    m_s, m_b = arms["sealed"]["migration"], arms["biology"]["migration"]
    mig_improve = None
    if m_s and m_b:
        mig_improve = bool(m_b["delta_R"] > m_s["delta_R"])

    contract_ok = not b_b.get("ca_in_mp_any_true", False)
    scorecard = {
        "contract_biology_no_ca": {
            "pass": contract_ok,
            "detail": "biology arm must keep ca_in_mp false",
        },
        "disc_biology": {
            "pass": bool(b_b["disc_pass"]),
            "disc_r_mean": b_b["disc_r_mean"],
            "bar": DISC_MIN,
        },
        "basin_biology_vs_bar": {
            "pass": bool(b_b["basin_pass"]),
            "gap": b_b["basin_sep_gap"],
            "bar": BASIN_GAP_MIN,
        },
        "basin_improve_vs_sealed": {
            "pass": basin_improve,
            "gap_sealed": b_s["basin_sep_gap"],
            "gap_biology": b_b["basin_sep_gap"],
            "delta_gap": b_b["basin_sep_gap"] - b_s["basin_sep_gap"],
        },
        "migration_biology_vs_bar": {
            "pass": bool(m_b["pass_form"]) if m_b else None,
            "delta_R": None if m_b is None else m_b["delta_R"],
            "R_4OBE": None if m_b is None else m_b["R_4OBE"],
            "R_4DSO": None if m_b is None else m_b["R_4DSO"],
        },
        "migration_improve_vs_sealed": {
            "pass": mig_improve,
            "delta_R_sealed": None if m_s is None else m_s["delta_R"],
            "delta_R_biology": None if m_b is None else m_b["delta_R"],
        },
        "telemetry_hub_jaccard_sealed_vs_biology": jaccards,
    }

    # Overall: wiring Pass required; biology signal = basin improve OR migration improve
    biology_signal = bool(basin_improve or (mig_improve is True))
    overall = {
        "wiring_pass": contract_ok and bool(b_b["disc_pass"]),
        "biology_signal_detected": biology_signal,
        "biology_bar_pass": bool(b_b["basin_pass"])
        or (bool(m_b["pass_form"]) if m_b else False),
        "verdict": (
            "WIRING_PASS_BIOLOGY_IMPROVE"
            if contract_ok and biology_signal
            else "WIRING_PASS_BIOLOGY_FLAT"
            if contract_ok
            else "WIRING_FAIL"
        ),
        "signal_strength": (
            "strong"
            if biology_signal
            and (b_b["basin_pass"] or (m_b and m_b["pass_form"] and (m_b["delta_R"] or 0) > 0.02))
            else "weak"
            if biology_signal
            else "none"
        ),
    }

    payload = {
        "schema_version": 1,
        "probe": "tokyo_eye_v7_hyp_biology_mp_sealed_vs_biology",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
        "scorecard": scorecard,
        "overall": overall,
    }
    # Strip heavy arrays from migration for JSON size
    for arm in payload["arms"].values():
        mig = arm.get("migration")
        if mig and "structures" in mig:
            for st in mig["structures"].values():
                st.pop("out_effect", None)
                st.pop("delta_rows", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"overall": overall, "out": str(args.output)}, indent=2))
    return 0 if overall["wiring_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
