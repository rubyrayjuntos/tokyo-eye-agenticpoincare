#!/usr/bin/env python3
"""Three-arm: classical vs sealed+biology vs trained Hyp-MP on fixed biology graph.

Pre-reg: docs/specs/tokyo-eye-v7/hyp-biology-mp-three-arm-prereg.md
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from experiments.diagnostics.kras_hub_migration_ood import (
    EXPECTED,
    HUB_TO,
    load_prot,
    resolve_hub_indices,
)
from experiments.diagnostics.kras_knockout_causal import _pdb_residue_names
from experiments.diagnostics.v7_hyp_biology_mp_sealed_vs_biology import (
    BASIN_GAP_MIN,
    BASIN_ROSTER,
    TEL_TARGETS,
    _load_model,
    grade_basin,
    grade_migration,
    grade_telemetry,
)
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6._data import _download_pdb
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.dtie.common.kras_g12_graft import neighborhood_n12
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.tokyo_eye.basin_contrastive import r_star_indices
from science.tokyo_eye.biology_graph import compute_biology_mp_graph
from science.tokyo_eye.thermo_edge_features import resolve_residue_records_for_prot

PREREG = Path("data/gates/tokyo_eye_v7_hyp_biology_mp_three_arm_prereg.json")
DEFAULT_TRAINED = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/v7_hyp_biology_mp_best.pt"
)
DEFAULT_OUT = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/three_arm_biology_graph_scorecard.json"
)


def _biology_undirected(
    prot: dict[str, Any],
    pdb_dir: Path,
) -> tuple[nx.Graph, np.ndarray, list[str], dict[str, Any]]:
    records = resolve_residue_records_for_prot(prot, pdb_dir=pdb_dir)
    coords = prot["ca_coords"]
    if torch.is_tensor(coords):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)
    rho = prot["target_rho"].reshape(-1).detach().cpu().numpy()
    ids = list(prot.get("residue_ids") or [])
    n = int(coords_np.shape[0])
    ei, _ea, _et, meta = compute_biology_mp_graph(
        coords_np,
        rho,
        residue_ids=ids[:n],
        residue_records=records,
    )
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if ei.size:
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            if int(s) == int(d):
                continue
            g.add_edge(int(s), int(d))
    return g, rho[:n], ids[:n], meta


def classical_hubs(g: nx.Graph, ids: list[str], *, k_frac: float = 0.10) -> dict[str, Any]:
    n = g.number_of_nodes()
    if n == 0:
        return {"hub_auth_resseqs": [], "betweenness": [], "n_hubs": 0}
    btw = nx.betweenness_centrality(g, normalized=True)
    arr = np.array([float(btw.get(i, 0.0)) for i in range(n)], dtype=np.float64)
    top_k = max(1, int(np.ceil(float(k_frac) * n)))
    order = np.argsort(-arr)
    hubs = order[:top_k]
    idx_map = residue_index_map(ids)
    rev = {i: rs for rs, i in idx_map.items()}
    return {
        "hub_auth_resseqs": [rev.get(int(i)) for i in hubs],
        "betweenness": arr.tolist(),
        "n_hubs": int(top_k),
        "n_edges_undirected": int(g.number_of_edges()),
        "strength_cv": float(arr.std() / (arr.mean() + 1e-12)),
    }


def classical_basin_vector(
    prot: dict[str, Any],
    rho: np.ndarray,
    ids: list[str],
    g: nx.Graph,
    partner: dict[str, Any] | None,
) -> np.ndarray:
    idx_map = residue_index_map(ids)
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
    if not idxs:
        idxs = list(range(min(len(rho), 10)))
    tau = prot.get("target_dehydron")
    if tau is None:
        tau_np = (rho < 13.0).astype(np.float64)
    else:
        tau_np = tau.reshape(-1).detach().cpu().numpy()[: len(rho)]
    deg = np.array([float(g.degree(i)) for i in range(len(rho))], dtype=np.float64)
    # Compact structure fingerprint on R*.
    sl = rho[idxs]
    tl = tau_np[idxs]
    dl = deg[idxs]
    return np.array(
        [
            float(sl.mean()),
            float(sl.std() if sl.size > 1 else 0.0),
            float(tl.mean()),
            float(dl.mean()),
            float(dl.std() if dl.size > 1 else 0.0),
            float(len(idxs)),
        ],
        dtype=np.float64,
    )


def grade_classical_basin(pdb_dir: Path) -> dict[str, Any]:
    prots: dict[str, dict[str, Any]] = {}
    graphs: dict[str, nx.Graph] = {}
    rhos: dict[str, np.ndarray] = {}
    id_map: dict[str, list[str]] = {}
    for _b, items in BASIN_ROSTER.items():
        for pdb_id, chain in items:
            prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
            if prot is None:
                raise FileNotFoundError(pdb_id)
            prots[pdb_id] = prot
            g, rho, ids, _meta = _biology_undirected(prot, pdb_dir)
            graphs[pdb_id] = g
            rhos[pdb_id] = rho
            id_map[pdb_id] = ids

    vecs: dict[str, np.ndarray] = {}
    for basin, items in BASIN_ROSTER.items():
        for pdb_id, _chain in items:
            partner = None
            if basin == "OFF":
                partner = prots.get("5US4") if pdb_id == "4LPK" else prots.get("4LPK")
            else:
                partner = prots.get("6GOF") if pdb_id == "6GOD" else prots.get("6GOD")
            vecs[pdb_id] = classical_basin_vector(
                prots[pdb_id], rhos[pdb_id], id_map[pdb_id], graphs[pdb_id], partner
            )

    def _d(a: str, b: str) -> float:
        return float(np.linalg.norm(vecs[a] - vecs[b]))

    cross = [_d(o, n) for o, _ in BASIN_ROSTER["OFF"] for n, _ in BASIN_ROSTER["ON"]]
    same = [_d("4LPK", "5US4"), _d("6GOD", "6GOF")]
    mean_cross = float(np.mean(cross))
    mean_same = float(np.mean(same))
    gap = mean_cross - mean_same
    return {
        "basin_sep_gap": gap,
        "mean_cross_off_on": mean_cross,
        "mean_same_basin": mean_same,
        "basin_pass": bool(gap > BASIN_GAP_MIN),
        "disc_r_mean": None,
        "disc_pass": None,
        "method": "Rstar_rho_tau_degree_L2",
    }


def grade_classical_migration(pdb_dir: Path, corpus_cache: Path | None) -> dict[str, Any]:
    structures: dict[str, Any] = {}
    for pdb_id, expected in EXPECTED.items():
        chain = "A"
        prot = load_prot(
            pdb_id,
            chain=chain,
            pdb_dir=pdb_dir,
            corpus_cache=corpus_cache if pdb_id == "4OBE" else None,
        )
        g, _rho, ids, meta = _biology_undirected(prot, pdb_dir)
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.is_file():
            pdb_path = Path(_download_pdb(pdb_id, pdb_dir))
        indices = resolve_hub_indices(
            list(prot["residue_ids"]),
            _pdb_residue_names(pdb_path, chain),
            expected,
        )
        btw = nx.betweenness_centrality(g, normalized=True)
        n = g.number_of_nodes()
        arr = np.array([float(btw.get(i, 0.0)) for i in range(n)], dtype=np.float64)
        i163 = indices[HUB_TO]
        med = float(np.median(arr[np.isfinite(arr)]))
        r = float(arr[i163] / med) if med > 0 else float("nan")
        order = np.argsort(-np.nan_to_num(arr, nan=-np.inf))
        rank = int(np.where(order == i163)[0][0]) + 1
        structures[pdb_id] = {
            "R_out_163": r,
            "rank_out_163": float(rank),
            "betweenness_163": float(arr[i163]),
            "median_betweenness": med,
            "n_biology_edges": meta.get("hbond", 0)
            + meta.get("dehydron", 0)
            + meta.get("pi_stack", 0)
            + meta.get("salt_bridge", 0),
        }
    r_obe = float(structures["4OBE"]["R_out_163"])
    r_dso = float(structures["4DSO"]["R_out_163"])
    return {
        "R_4OBE": r_obe,
        "R_4DSO": r_dso,
        "delta_R": r_dso - r_obe,
        "pass_form": bool(r_dso > r_obe),
        "rank_163_4OBE": structures["4OBE"]["rank_out_163"],
        "rank_163_4DSO": structures["4DSO"]["rank_out_163"],
        "structures": structures,
        "method": "biology_graph_betweenness",
    }


def grade_classical_telemetry(pdb_dir: Path) -> dict[str, Any]:
    rows = []
    for pdb_id, chain in TEL_TARGETS:
        prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
        if prot is None:
            raise FileNotFoundError(pdb_id)
        g, _rho, ids, meta = _biology_undirected(prot, pdb_dir)
        hubs = classical_hubs(g, ids)
        rows.append(
            {
                "pdb_id": pdb_id,
                "n_edges": hubs["n_edges_undirected"] * 2,
                "n_hubs": hubs["n_hubs"],
                "strength_cv": hubs["strength_cv"],
                "hub_auth_resseqs": hubs["hub_auth_resseqs"],
                "biology_meta": meta,
                "method": "betweenness",
            }
        )
    return {"targets": rows}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sealed", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--trained", type=Path, default=DEFAULT_TRAINED)
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
    if not PREREG.is_file():
        raise SystemExit(f"missing prereg {PREREG}")
    for path in (args.sealed, args.trained):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    arms: dict[str, Any] = {}

    print("=== arm classical ===", flush=True)
    arms["classical"] = {
        "checkpoint": None,
        "biology_mp": True,
        "telemetry": grade_classical_telemetry(args.pdb_dir),
        "basin": grade_classical_basin(args.pdb_dir),
        "migration": None
        if args.skip_migration
        else grade_classical_migration(
            args.pdb_dir,
            args.corpus_cache if args.corpus_cache.is_file() else None,
        ),
    }

    for name, ckpt in (
        ("sealed_biology", args.sealed),
        ("hyp_mp_trained", args.trained),
    ):
        print(f"=== arm {name} ({ckpt}) ===", flush=True)
        model = _load_model(ckpt, args.device, biology=True)
        tel = grade_telemetry(model, args.pdb_dir, args.device, biology=True)
        basin = grade_basin(model, args.pdb_dir, args.device, biology=True)
        mig = None
        if not args.skip_migration:
            mig = grade_migration(
                model,
                args.pdb_dir,
                args.device,
                biology=True,
                corpus_cache=args.corpus_cache if args.corpus_cache.is_file() else None,
            )
            # grade_migration uses knockout_scan for biology=False; force biology path
            # Our call already biology=True → _knockout_biology. Good.
        arms[name] = {
            "checkpoint": str(ckpt),
            "biology_mp": True,
            "telemetry": tel,
            "basin": basin,
            "migration": mig,
        }

    c_b = arms["classical"]["basin"]
    s_b = arms["sealed_biology"]["basin"]
    t_b = arms["hyp_mp_trained"]["basin"]
    c_m = arms["classical"]["migration"]
    s_m = arms["sealed_biology"]["migration"]
    t_m = arms["hyp_mp_trained"]["migration"]

    best_gnn_gap = max(float(s_b["basin_sep_gap"]), float(t_b["basin_sep_gap"]))
    best_gnn_dR = None
    if s_m and t_m:
        best_gnn_dR = max(float(s_m["delta_R"]), float(t_m["delta_R"]))

    beats_classical_basin = bool(best_gnn_gap > float(c_b["basin_sep_gap"]))
    beats_classical_mig = (
        None
        if c_m is None or best_gnn_dR is None
        else bool(best_gnn_dR > float(c_m["delta_R"]))
    )
    contract_ok = (not s_b.get("ca_in_mp_any_true")) and (
        not t_b.get("ca_in_mp_any_true")
    )

    scorecard = {
        "contract_no_ca": {"pass": contract_ok},
        "classical_basin": {
            "gap": c_b["basin_sep_gap"],
            "pass_bar": bool(c_b["basin_pass"]),
        },
        "sealed_biology_basin": {
            "gap": s_b["basin_sep_gap"],
            "pass_bar": bool(s_b["basin_pass"]),
            "disc_r_mean": s_b["disc_r_mean"],
        },
        "hyp_mp_trained_basin": {
            "gap": t_b["basin_sep_gap"],
            "pass_bar": bool(t_b["basin_pass"]),
            "disc_r_mean": t_b["disc_r_mean"],
        },
        "classical_migration": None
        if c_m is None
        else {
            "delta_R": c_m["delta_R"],
            "pass_form": c_m["pass_form"],
            "R_4OBE": c_m["R_4OBE"],
            "R_4DSO": c_m["R_4DSO"],
        },
        "sealed_biology_migration": None
        if s_m is None
        else {
            "delta_R": s_m["delta_R"],
            "pass_form": s_m["pass_form"],
            "R_4OBE": s_m["R_4OBE"],
            "R_4DSO": s_m["R_4DSO"],
            "rank_163_4OBE": s_m["rank_163_4OBE"],
            "rank_163_4DSO": s_m["rank_163_4DSO"],
        },
        "hyp_mp_trained_migration": None
        if t_m is None
        else {
            "delta_R": t_m["delta_R"],
            "pass_form": t_m["pass_form"],
            "R_4OBE": t_m["R_4OBE"],
            "R_4DSO": t_m["R_4DSO"],
            "rank_163_4OBE": t_m["rank_163_4OBE"],
            "rank_163_4DSO": t_m["rank_163_4DSO"],
        },
        "gnn_beats_classical_basin": {"pass": beats_classical_basin},
        "gnn_beats_classical_migration": {"pass": beats_classical_mig},
        "trained_vs_sealed_biology_basin_delta": float(t_b["basin_sep_gap"])
        - float(s_b["basin_sep_gap"]),
        "trained_vs_sealed_biology_delta_R": None
        if s_m is None or t_m is None
        else float(t_m["delta_R"]) - float(s_m["delta_R"]),
    }

    non_taut = bool(beats_classical_basin or beats_classical_mig is True)
    overall = {
        "wiring_pass": contract_ok,
        "non_tautology_pass": non_taut,
        "verdict": (
            "GNN_BEATS_CLASSICAL"
            if non_taut
            else "CLASSICAL_WINS_OR_TIE"
            if contract_ok
            else "WIRING_FAIL"
        ),
        "signal_strength": (
            "strong"
            if non_taut
            and (
                best_gnn_gap > BASIN_GAP_MIN
                or (best_gnn_dR is not None and best_gnn_dR > 0.02)
            )
            else "weak"
            if non_taut
            else "none"
        ),
    }

    payload = {
        "schema_version": 1,
        "probe": "tokyo_eye_v7_hyp_biology_mp_three_arm",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "prereg": str(PREREG),
        "arms": arms,
        "scorecard": scorecard,
        "overall": overall,
    }
    for arm in payload["arms"].values():
        mig = arm.get("migration")
        if isinstance(mig, dict) and "structures" in mig:
            for st in mig["structures"].values():
                if isinstance(st, dict):
                    st.pop("out_effect", None)
                    st.pop("delta_rows", None)
                    st.pop("score_knockout_compat", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"overall": overall, "scorecard": scorecard, "out": str(args.output)}, indent=2))
    return 0 if contract_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
