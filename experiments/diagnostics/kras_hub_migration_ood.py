#!/usr/bin/env python3
"""KRAS G12D conductance hub migration — trunk in-flow into residue 163.

Pre-registration: ``docs/specs/learned-flow-influence/ablation.md``
(section "KRAS G12D conductance hub migration").

Physics-native only: GLY151 (WT) → ILE163 (G12D). No epistemic doorways.
Logs both raw ``in(163)`` and normalized ``R = in(163)/median(in)``;
pass criterion remains ``R_4DSO > R_4OBE`` only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.PDB import PDBParser

from experiments.diagnostics.jacobian_flow_influence import (
    evaluate_structure,
    load_proteins_from_cache,
)
from experiments.training.v6._data import _download_pdb, load_protein_graph_from_pdb_legacy

# Deposited identities locked in the pre-registration.
EXPECTED: dict[str, dict[int, str]] = {
    "4OBE": {12: "GLY", 151: "GLY", 163: "ILE"},
    "4DSO": {12: "ASP", 151: "GLY", 163: "ILE"},
}
HUB_FROM = 151
HUB_TO = 163


def _pdb_residue_names(pdb_path: Path, chain: str) -> dict[int, str]:
    structure = PDBParser(QUIET=True).get_structure("x", str(pdb_path))
    model = next(structure.get_models())
    out: dict[int, str] = {}
    for residue in model[chain]:
        het, resseq, _ = residue.id
        if het.strip():
            continue
        out[int(resseq)] = residue.get_resname().upper()
    return out


def resolve_hub_indices(
    residue_ids: list[str],
    pdb_resnames: dict[int, str],
    expected: dict[int, str],
) -> dict[int, int]:
    """Map deposited resseq → graph row; refuse on identity mismatch."""
    graph_by_resnum: dict[int, int] = {}
    for idx, residue_id in enumerate(residue_ids):
        parts = str(residue_id).split(":")
        if len(parts) < 2:
            raise ValueError(f"Unparseable graph residue id: {residue_id!r}")
        resnum = int(parts[1])
        if resnum in graph_by_resnum:
            raise ValueError(f"Duplicate graph residue number {resnum}")
        graph_by_resnum[resnum] = idx

    resolved: dict[int, int] = {}
    for resnum, expected_name in expected.items():
        actual = str(pdb_resnames.get(resnum, "<missing>")).upper()
        if actual != expected_name:
            raise ValueError(
                f"Residue {resnum} expected {expected_name}, found {actual}; "
                "refusing hub-migration scoring (identity mismatch)"
            )
        if resnum not in graph_by_resnum:
            raise ValueError(
                f"Residue {resnum} ({expected_name}) absent from model graph"
            )
        resolved[resnum] = graph_by_resnum[resnum]
    return resolved


def score_in_centrality(
    in_centrality: np.ndarray,
    indices: dict[int, int],
) -> dict[str, float]:
    """Raw + normalized in-flow at hub sites (trunk)."""
    in_c = np.asarray(in_centrality, dtype=np.float64).reshape(-1)
    finite = in_c[np.isfinite(in_c)]
    if finite.size == 0:
        raise ValueError("in-centrality has no finite entries")
    median_in = float(np.median(finite))
    if not np.isfinite(median_in) or median_in <= 0.0:
        raise ValueError(f"median(in) not positive finite: {median_in}")

    i163 = indices[HUB_TO]
    i151 = indices[HUB_FROM]
    in_163 = float(in_c[i163])
    in_151 = float(in_c[i151])
    r_163 = in_163 / median_in
    h_ratio = in_163 / in_151 if np.isfinite(in_151) and abs(in_151) > 1e-30 else float("nan")

    # Rank: 1 = strongest in-hub (descending).
    order = np.argsort(-np.nan_to_num(in_c, nan=-np.inf))
    rank_163 = int(np.where(order == i163)[0][0]) + 1
    rank_151 = int(np.where(order == i151)[0][0]) + 1

    return {
        "in_163_raw": in_163,
        "in_151_raw": in_151,
        "median_in": median_in,
        "R_163": r_163,
        "H_163_over_151": h_ratio,
        "rank_163": float(rank_163),
        "rank_151": float(rank_151),
        "n_residues": float(in_c.size),
    }


def load_prot(
    pdb_id: str,
    *,
    chain: str,
    pdb_dir: Path,
    corpus_cache: Path | None,
) -> dict[str, Any]:
    pdb_id = pdb_id.upper()
    if pdb_id == "4OBE" and corpus_cache is not None and corpus_cache.is_file():
        try:
            return load_proteins_from_cache(corpus_cache, [pdb_id])[pdb_id]
        except RuntimeError:
            pass
    # 4DSO (enabled:false) and any cache miss: explicit PDB load — never silent skip.
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    if not pdb_path.is_file():
        pdb_path = Path(_download_pdb(pdb_id, pdb_dir))
    if not pdb_path.is_file():
        raise FileNotFoundError(
            f"Refuse: {pdb_id}.pdb missing under {pdb_dir} (no silent skip)"
        )
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Failed to build graph for {pdb_id}:{chain}")
    return prot


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        HEALTHY_FIX1_CKPT,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            FIX1_SPARSITY_CHAMPION_CKPT
            if FIX1_SPARSITY_CHAMPION_CKPT.is_file()
            else HEALTHY_FIX1_CKPT
        ),
        help=(
            "Default: sparsity phase champion (ep48) when present; "
            "else sealed Fix-1 trunk."
        ),
    )
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("pdb_cache"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/routing_sparsity/"
            "kras_hub_migration_4obe_4dso.json"
        ),
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    structures: dict[str, Any] = {}
    for pdb_id, expected in EXPECTED.items():
        chain = "A"
        prot = load_prot(
            pdb_id,
            chain=chain,
            pdb_dir=args.pdb_dir,
            corpus_cache=args.corpus_cache if pdb_id == "4OBE" else None,
        )
        pdb_path = args.pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.is_file():
            pdb_path = Path(_download_pdb(pdb_id, args.pdb_dir))
        pdb_resnames = _pdb_residue_names(pdb_path, chain)
        indices = resolve_hub_indices(
            list(prot["residue_ids"]), pdb_resnames, expected
        )
        print(f"evaluating {pdb_id} on {args.checkpoint} ...", flush=True)
        report = evaluate_structure(
            prot=prot, checkpoint=args.checkpoint, device=args.device
        )
        trunk_in = np.asarray(
            report["layers"]["encoder_h"]["centralities"]["in"], dtype=np.float64
        )
        disc_in = np.asarray(
            report["layers"]["hyp_projections_2d"]["centralities"]["in"],
            dtype=np.float64,
        )
        trunk_score = score_in_centrality(trunk_in, indices)
        disc_score = score_in_centrality(disc_in, indices)
        structures[pdb_id] = {
            "alignment": {
                str(r): {
                    "pdb_resname": expected[r],
                    "graph_index": indices[r],
                    "graph_residue_id": prot["residue_ids"][indices[r]],
                }
                for r in expected
            },
            "trunk": trunk_score,
            "disc": disc_score,
            "liveness_trunk": report["layers"]["encoder_h"]["liveness"],
            "probe_report_ref": {
                "n_residues": report["layers"]["encoder_h"]["n_residues"],
                "asymmetry_mean": report["layers"]["encoder_h"]["liveness"][
                    "asymmetry_mean"
                ],
            },
        }

    r_obe = structures["4OBE"]["trunk"]["R_163"]
    r_dso = structures["4DSO"]["trunk"]["R_163"]
    primary_pass = bool(r_dso > r_obe)
    verdict = {
        "primary_metric": "R_163 = in(163) / median(in) on encoder_h",
        "R_4OBE": r_obe,
        "R_4DSO": r_dso,
        "delta_R": r_dso - r_obe,
        "pass": primary_pass,
        "outcome": "Pass" if primary_pass else "Fail",
        "raw_logged_for_attribution": {
            "4OBE_in_163_raw": structures["4OBE"]["trunk"]["in_163_raw"],
            "4OBE_median_in": structures["4OBE"]["trunk"]["median_in"],
            "4DSO_in_163_raw": structures["4DSO"]["trunk"]["in_163_raw"],
            "4DSO_median_in": structures["4DSO"]["trunk"]["median_in"],
        },
        "secondary": {
            "H_4OBE": structures["4OBE"]["trunk"]["H_163_over_151"],
            "H_4DSO": structures["4DSO"]["trunk"]["H_163_over_151"],
            "H_pass": bool(
                structures["4DSO"]["trunk"]["H_163_over_151"]
                > structures["4OBE"]["trunk"]["H_163_over_151"]
            ),
            "rank_163_4OBE": structures["4OBE"]["trunk"]["rank_163"],
            "rank_163_4DSO": structures["4DSO"]["trunk"]["rank_163"],
        },
    }

    out = {
        "schema_version": 1,
        "probe": "kras_hub_migration_ood",
        "preregistration": (
            "docs/specs/learned-flow-influence/ablation.md"
            " §KRAS G12D conductance hub migration"
        ),
        "checkpoint": str(args.checkpoint),
        "excluded": [
            "epistemic doorways",
            "robust_experts.pt",
            "containment Path B checkpoint",
        ],
        "structures": structures,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
