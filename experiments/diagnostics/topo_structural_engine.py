#!/usr/bin/env python3
"""Generic topo-structural engine — flow↔centrality concordance (platform grade).

No KRAS residue-ID gates. Landmark audits (e.g. 81/114/156) are optional
side reports only and never enter ``verdict``.

Smoke roster (TRAINING_TARGETS, non-KRAS):
  - 3PP0  SRC kinase
  - 2SHP  SHP2 phosphatase
  - 2HHB  haemoglobin (blind fold; small — prefer over 2Z6H for smoke runtime)
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
from experiments.diagnostics.kras_knockout_causal import knockout_scan
from experiments.training.v66._data import TRAINING_TARGETS, load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.classical_network_metrics import classical_network_metrics
from science.dtie.common.topo_flow_concordance import (
    DEFAULT_PASS_RHO,
    DEFAULT_STABILITY_DRHO_MAX,
    DEFAULT_TOP_K_FRAC,
    concordance_report,
    panel_verdict,
    stability_ok,
)
from science.training.gnn_lineage import load_model_from_checkpoint

# Non-KRAS smoke panel from TRAINING_TARGETS (not 1STP/1PTP/1B0N — those are
# not in the manifest and 1STP is streptavidin, not a kinase).
DEFAULT_SMOKE_PANEL: list[tuple[str, str]] = [
    ("3PP0", "A"),  # SRC kinase
    ("2SHP", "A"),  # SHP2 phosphatase
    ("2HHB", "B"),  # haemoglobin — small blind fold (not 2Z6H: n≈533 too slow for smoke)
]

# Optional KRAS landmark audit only (never Pass/Fail).
KRAS_AUDIT_LANDMARKS = (81, 114, 156)


def _run_one(
    *,
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
    k_frac: float,
    pass_rho: float,
    stability_cutoffs: tuple[float, float] = (7.5, 8.5),
) -> dict[str, Any]:
    meta = TRAINING_TARGETS.get(pdb_id.upper(), {})
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")

    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)
    ca = prot.get("ca_coords")
    if ca is None:
        raise ValueError(f"{pdb_id}: missing ca_coords")
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    ca_np = ca.detach().cpu().numpy() if torch.is_tensor(ca) else np.asarray(ca)
    ca_np = ca_np[:n]

    classical = classical_network_metrics(ca_np)
    btw = np.asarray(classical["betweenness"], dtype=np.float64)
    scan = knockout_scan(model, prot, device)
    out_effect = np.asarray(scan["out_effect"], dtype=np.float64)[:n]

    report = concordance_report(
        out_effect, btw, k_frac=k_frac, pass_rho=pass_rho
    )

    # Stability: recompute CB at neighboring cutoffs; knockout fixed.
    from science.dtie.common.classical_network_metrics import (
        DEFAULT_CONTACT_CUTOFF_A,
        build_ca_contact_graph,
    )
    import networkx as nx

    def _btw_at(cutoff: float) -> np.ndarray:
        g = build_ca_contact_graph(ca_np, cutoff_angstrom=cutoff)
        raw = nx.betweenness_centrality(g, weight="weight", normalized=True)
        return np.asarray([raw.get(i, 0.0) for i in range(n)], dtype=np.float64)

    rho_lo = concordance_report(
        out_effect, _btw_at(stability_cutoffs[0]), k_frac=k_frac, pass_rho=pass_rho
    )["spearman_full"]
    rho_hi = concordance_report(
        out_effect, _btw_at(stability_cutoffs[1]), k_frac=k_frac, pass_rho=pass_rho
    )["spearman_full"]
    stab = bool(
        stability_ok(report["spearman_full"], rho_lo, max_abs_drho=DEFAULT_STABILITY_DRHO_MAX)
        and stability_ok(report["spearman_full"], rho_hi, max_abs_drho=DEFAULT_STABILITY_DRHO_MAX)
    )

    return {
        "pdb_id": pdb_id.upper(),
        "chain": chain,
        "gene": meta.get("gene"),
        "desc": meta.get("desc"),
        "in_training_targets": pdb_id.upper() in TRAINING_TARGETS,
        "n_residues": int(n),
        "contact_cutoff_a": DEFAULT_CONTACT_CUTOFF_A,
        **report,
        "stability": {
            "cutoffs_a": list(stability_cutoffs),
            "spearman_full_lo": rho_lo,
            "spearman_full_hi": rho_hi,
            "max_abs_drho": DEFAULT_STABILITY_DRHO_MAX,
            "pass": stab,
        },
        "stability_pass": stab,
        "pass": bool(report["pass"] and stab),
    }


def _optional_kras_audit(
    checkpoint: Path, pdb_dir: Path, device: str
) -> dict[str, Any] | None:
    """Side report only — written under audit_reports/, ignored by verdict."""
    from science.dtie.common.kras_topo_matrix import parse_resseq, residue_index_map

    pdb_id, chain = "4OBE", "A"
    try:
        row = _run_one(
            checkpoint=checkpoint,
            pdb_id=pdb_id,
            chain=chain,
            pdb_dir=pdb_dir,
            device=device,
            k_frac=DEFAULT_TOP_K_FRAC,
            pass_rho=DEFAULT_PASS_RHO,
        )
    except Exception as exc:  # noqa: BLE001 — audit must not fail the panel
        return {"error": str(exc), "landmarks": list(KRAS_AUDIT_LANDMARKS)}

    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        return {"error": "4OBE load failed", "landmarks": list(KRAS_AUDIT_LANDMARKS)}
    idx_map = residue_index_map(prot.get("residue_ids") or [])
    # Re-knock for landmark attribution values
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)
    scan = knockout_scan(model, prot, device)
    oe = np.asarray(scan["out_effect"], dtype=np.float64)
    landmarks = {}
    for rs in KRAS_AUDIT_LANDMARKS:
        if rs not in idx_map:
            landmarks[str(rs)] = None
            continue
        i = idx_map[rs]
        landmarks[str(rs)] = {
            "graph_index": i,
            "knockout_out_effect": float(oe[i]) if i < oe.size else None,
            "note": "audit-only; excluded from verdict",
        }
    return {
        "pdb_id": "4OBE",
        "concordance_audit": {
            "spearman_full": row.get("spearman_full"),
            "pass_would_be": row.get("pass"),
        },
        "landmarks": landmarks,
    }


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        HEALTHY_FIX1_CKPT,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            FIX1_SPARSITY_CHAMPION_CKPT
            if FIX1_SPARSITY_CHAMPION_CKPT.is_file()
            else HEALTHY_FIX1_CKPT
        ),
    )
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--k-frac", type=float, default=DEFAULT_TOP_K_FRAC)
    p.add_argument("--pass-rho", type=float, default=DEFAULT_PASS_RHO)
    p.add_argument(
        "--targets",
        nargs="*",
        default=[f"{a}:{b}" for a, b in DEFAULT_SMOKE_PANEL],
        help="PDB:CHAIN list (default non-KRAS TRAINING_TARGETS smoke panel)",
    )
    p.add_argument(
        "--with-kras-audit",
        action="store_true",
        help="Also write audit_reports/ landmark side file (ignored by verdict)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/routing_sparsity/"
            "general_hub_alignment_smoke.json"
        ),
    )
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    panel: list[tuple[str, str]] = []
    for tok in args.targets:
        if ":" in tok:
            pdb_id, chain = tok.split(":", 1)
        else:
            pdb_id, chain = tok, "A"
        panel.append((pdb_id.upper(), chain))

    rows: list[dict[str, Any]] = []
    for pdb_id, chain in panel:
        print(f"concordance {pdb_id}:{chain} ...", flush=True)
        row = _run_one(
            checkpoint=args.checkpoint,
            pdb_id=pdb_id,
            chain=chain,
            pdb_dir=args.pdb_dir,
            device=args.device,
            k_frac=float(args.k_frac),
            pass_rho=float(args.pass_rho),
        )
        print(
            f"  ρ_full={row['spearman_full']:.3f} pass={row['pass']} "
            f"stab={row['stability_pass']} gene={row.get('gene')}",
            flush=True,
        )
        rows.append(row)

    verdict = panel_verdict(rows)
    out = {
        "schema_version": 1,
        "probe": "topo_structural_engine",
        "grade": "general_hub_alignment",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "pass_rho": float(args.pass_rho),
        "k_frac": float(args.k_frac),
        "panel": rows,
        "verdict": verdict,
        "notes": [
            "Pass = full-structure Spearman(out_effect, CB) > pass_rho AND cutoff stability.",
            "Top-k% hubs are diagnostic enrichment only — not residue-ID gates.",
            "KRAS landmarks never enter verdict.",
            "Default panel is TRAINING_TARGETS non-KRAS (3PP0/2SHP/2HHB), not 1STP/1PTP/1B0N.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.output}")

    if args.with_kras_audit:
        audit_dir = args.output.parent / "audit_reports"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit = _optional_kras_audit(args.checkpoint, args.pdb_dir, args.device)
        audit_path = audit_dir / "kras_landmarks_4obe_audit_only.json"
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        print(f"audit-only wrote {audit_path}")

    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
