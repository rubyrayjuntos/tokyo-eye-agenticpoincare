#!/usr/bin/env python3
"""KRAS G12D four-quadrant edge-ΔE rematch (inhibitor-free roster).

Roster: 4LPK (WT GDP) / 6GOD (WT GppNHp) / 5US4 (G12D GDP) / 6GOF (G12D GppNHp).
Primary Pass: Cα contact @ 8 Å — mimetic inactive + compressed mut span.
Complementarity dehydron-wrapper + coupled-lock Cα distances are report-only.

Spec: ``docs/specs/kras-topo-structural-inference/edge-delta-four-quadrant.md``
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from science.dtie.common.kras_topo_matrix import (
    CA_CONTACT_CUTOFF_A,
    COUPLED_LOCK_PAIRS,
    FOUR_Q_MUT_OFF,
    FOUR_Q_MUT_ON,
    FOUR_Q_WT_OFF,
    FOUR_Q_WT_ON,
    ca_contact_edges_by_resseq,
    ca_distance,
    dehydron_wrapper_edges_by_resseq,
    edge_symdiff_size,
    four_quadrant_edge_verdict,
    residue_index_map,
)

ROSTER: list[tuple[str, str, str]] = [
    (FOUR_Q_WT_OFF, "A", "wt_off"),
    (FOUR_Q_WT_ON, "A", "wt_on"),
    (FOUR_Q_MUT_OFF, "A", "mut_off"),
    (FOUR_Q_MUT_ON, "A", "mut_on"),
]


def _load_prot(pdb_id: str, chain: str, pdb_dir: Path) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain} from {pdb_dir}")
    return prot


def _structure_bundle(pdb_id: str, chain: str, role: str, pdb_dir: Path) -> dict[str, Any]:
    from experiments.training.v66._data import _download_pdb

    prot = _load_prot(pdb_id, chain, pdb_dir)
    ca = prot.get("ca_coords")
    if ca is None:
        raise ValueError(f"{pdb_id}: missing ca_coords")
    residue_ids = list(prot.get("residue_ids") or [])
    ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
    edges_ca = ca_contact_edges_by_resseq(
        ca_np, residue_ids, cutoff_angstrom=CA_CONTACT_CUTOFF_A
    )
    pdb_path = Path(_download_pdb(pdb_id, pdb_dir))
    wrap_graphs = dehydron_wrapper_edges_by_resseq(pdb_path, chain)
    idx_map = residue_index_map(residue_ids)
    def _finite_or_none(x: float) -> float | None:
        return float(x) if np.isfinite(x) else None

    coupled = {
        f"{a}-{b}": _finite_or_none(ca_distance(ca_np, residue_ids, a, b))
        for a, b in COUPLED_LOCK_PAIRS
    }
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "role": role,
        "n_residues": len(residue_ids),
        "n_contact_edges_ca": len(edges_ca),
        "n_complementarity_edges": len(wrap_graphs["complementarity"]),
        "coupled_lock_ca_distances": coupled,
        "_edges_ca": edges_ca,
        "_edges_complementarity": wrap_graphs["complementarity"],
        "_idx_map": idx_map,
    }


def _pair_symdiff(
    bundles: dict[str, dict[str, Any]],
    a: str,
    b: str,
    *,
    edge_key: str,
    shared: list[int],
) -> int:
    return edge_symdiff_size(
        bundles[a][edge_key],
        bundles[b][edge_key],
        shared_resseqs=shared,
    )


def grade(*, pdb_dir: Path) -> dict[str, Any]:
    bundles: dict[str, dict[str, Any]] = {}
    for pdb_id, chain, role in ROSTER:
        print(f"loading {role} {pdb_id}:{chain} ...", flush=True)
        bundles[pdb_id] = _structure_bundle(pdb_id, chain, role, pdb_dir)

    wt_off, wt_on = FOUR_Q_WT_OFF, FOUR_Q_WT_ON
    mut_off, mut_on = FOUR_Q_MUT_OFF, FOUR_Q_MUT_ON
    shared = sorted(
        set(bundles[wt_off]["_idx_map"])
        & set(bundles[wt_on]["_idx_map"])
        & set(bundles[mut_off]["_idx_map"])
        & set(bundles[mut_on]["_idx_map"])
    )

    d_mimetic_ca = _pair_symdiff(
        bundles, mut_off, wt_on, edge_key="_edges_ca", shared=shared
    )
    d_wt_to_on_ca = _pair_symdiff(
        bundles, wt_off, wt_on, edge_key="_edges_ca", shared=shared
    )
    d_mut_span_ca = _pair_symdiff(
        bundles, mut_off, mut_on, edge_key="_edges_ca", shared=shared
    )
    d_wt_span_ca = d_wt_to_on_ca  # same pair as wt_to_active for span bar

    ca_verdict = four_quadrant_edge_verdict(
        delta_e_mimetic=d_mimetic_ca,
        delta_e_wt_to_active=d_wt_to_on_ca,
        delta_e_mut_span=d_mut_span_ca,
        delta_e_wt_span=d_wt_span_ca,
    )
    ca_verdict["graph"] = "ca_contact_8A"
    ca_verdict["n_shared_residues"] = len(shared)

    comp_telemetry = {
        "delta_e_mimetic": _pair_symdiff(
            bundles, mut_off, wt_on, edge_key="_edges_complementarity", shared=shared
        ),
        "delta_e_wt_to_active": _pair_symdiff(
            bundles, wt_off, wt_on, edge_key="_edges_complementarity", shared=shared
        ),
        "delta_e_mut_span": _pair_symdiff(
            bundles, mut_off, mut_on, edge_key="_edges_complementarity", shared=shared
        ),
        "delta_e_wt_span": _pair_symdiff(
            bundles, wt_off, wt_on, edge_key="_edges_complementarity", shared=shared
        ),
        "note": "dehydron-wrapper complementarity — report-only telemetry",
    }
    comp_verdict = four_quadrant_edge_verdict(
        delta_e_mimetic=comp_telemetry["delta_e_mimetic"],
        delta_e_wt_to_active=comp_telemetry["delta_e_wt_to_active"],
        delta_e_mut_span=comp_telemetry["delta_e_mut_span"],
        delta_e_wt_span=comp_telemetry["delta_e_wt_span"],
    )
    comp_verdict["graph"] = "dehydron_wrapper_complementarity"
    comp_verdict["report_only"] = True

    coupled_report = {
        pdb_id: bundles[pdb_id]["coupled_lock_ca_distances"] for pdb_id in bundles
    }

    public = {
        pdb_id: {k: v for k, v in b.items() if not k.startswith("_")}
        for pdb_id, b in bundles.items()
    }

    return {
        "schema_version": 1,
        "probe": "kras_topo_edge_delta_four_quadrant",
        "spec": "docs/specs/kras-topo-structural-inference/edge-delta-four-quadrant.md",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "roster": {
            "wt_off": wt_off,
            "wt_on": wt_on,
            "mut_off": mut_off,
            "mut_on": mut_on,
            "rationale": (
                "inhibitor-free; isogenic ON pair 6GOD/6GOF; native nucleotide + Mg2+; "
                "avoid MRTX1133/BI-2865-class Switch I/II artifacts"
            ),
        },
        "structures": public,
        "gates": {
            "ca_four_quadrant": ca_verdict,
            "complementarity_telemetry": {**comp_verdict, **comp_telemetry},
            "coupled_lock_ca": {
                "report_only": True,
                "pairs": [f"{a}-{b}" for a, b in COUPLED_LOCK_PAIRS],
                "distances_by_structure": coupled_report,
                "note": (
                    "Expect G12D-OFF (5US4) closer to active-like 12–60 / Switch pattern "
                    "than WT-OFF (4LPK); not a blocking Pass"
                ),
            },
        },
        "verdict": {
            "pass": bool(ca_verdict["pass"]),
            "outcome": ca_verdict["outcome"],
            "blocking": "ca_four_quadrant",
            "gates": ca_verdict["gates"],
        },
        "historical_triad": {
            "status": "report_only_superseded",
            "artifact": (
                "checkpoints/v66/diagnostics/routing_sparsity/"
                "kras_topo_matrix_4obe_4dso_5vq2.json"
            ),
            "note": "4OBE/4DSO/5VQ2 complementarity Pass demoted (wrong allele active)",
        },
        "notes": [
            "Science claim: G12D rewires inactive networks toward WT-ON topology.",
            "Primary graph: Cα contacts @ 8 Å on shared deposited resseqs.",
            "Complementarity and coupled-lock are report-only companions.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/routing_sparsity/"
            "kras_topo_edge_delta_four_quadrant.json"
        ),
    )
    args = p.parse_args(argv)
    report = grade(pdb_dir=args.pdb_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps(report["gates"]["ca_four_quadrant"], indent=2))
    print(f"wrote {args.output}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
