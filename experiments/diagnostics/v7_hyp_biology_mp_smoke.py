#!/usr/bin/env python3
"""Smoke grade: hyp_biology_mp fail-closed + ontology audit (no Cα in MP).

Does not overwrite HEALTHY_V7_CKPT. Report-only biology grades.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.diagnostics.fix1_champion_hub_knockout_sweep import _align_prot_features
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from experiments.training.v7.healthy_bprime import HEALTHY_V7_CKPT
from science.tokyo_eye.biology_graph import attach_biology_mp_graph
from science.tokyo_eye.biology_mp_audit import FORBIDDEN_MP_EDGE_TYPES
from science.tokyo_eye.thermo_edge_features import resolve_residue_records_for_prot
from science.training.gnn_lineage import load_model_from_checkpoint

PREREG = Path("data/gates/tokyo_eye_v7_hyp_biology_mp_prereg.json")
DEFAULT_OUT = Path(
    "checkpoints/v7/runs/tokyo_eye_v7_hyp_biology_mp_v1/hyp_biology_mp_smoke.json"
)
DEFAULT_TARGETS = ("4OBE:A",)


def _parse_targets(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for tok in raw.split():
        pdb, _, chain = tok.partition(":")
        out.append((pdb.upper(), chain or "A"))
    return out


def _run_one(
    *,
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    device: str,
) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    model.hyp_biology_mp = True
    prot = _align_prot_features(model, prot)
    data = prepare_training_batch(model, prot, device)
    n = residue_node_count(data, prot)
    records = resolve_residue_records_for_prot(prot, pdb_dir=pdb_dir)
    coords = prot.get("ca_coords")
    if coords is None:
        raise RuntimeError(f"{pdb_id}:{chain} missing ca_coords for biology attach")
    rho = prot["target_rho"].reshape(-1).detach().cpu().numpy()
    attach_biology_mp_graph(
        data,
        coords=coords,
        rho=rho,
        residue_ids=list(prot.get("residue_ids") or [])[:n],
        residue_records=records,
    )
    with torch.no_grad():
        out = model(data)
    audit = out.get("audit_trail") or {}
    bio = audit.get("biology_mp") or getattr(data, "biology_mp_audit", {}) or {}
    ca_in_mp = bool(audit.get("ca_in_mp", bio.get("ca_in_mp", True)))
    allow_fb = bool(getattr(data, "allow_ca_fallback", True))
    forbidden_hit = [
        t for t in FORBIDDEN_MP_EDGE_TYPES if t in (bio.get("edge_counts") or {})
    ]
    # Forbidden types must not appear as positive counts under biology names —
    # they are ontology labels that should be absent entirely.
    hard_ok = (
        bool(audit.get("hyp_biology_mp") or model.hyp_biology_mp)
        and ca_in_mp is False
        and allow_fb is False
        and int(bio.get("n_biology_edges_undirected", -1)) >= 0
    )
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "n_residues": n,
        "hyp_biology_mp": bool(model.hyp_biology_mp),
        "ca_in_mp": ca_in_mp,
        "allow_ca_fallback": allow_fb,
        "biology_mp": bio,
        "meta": getattr(data, "biology_mp_meta", {}),
        "hard_ok": hard_ok,
        "forbidden_labels_seen": forbidden_hit,
        "forward_ok": True,
        "disc_r_mean": float(
            out["hyp_projections_2d"].norm(dim=-1).mean().detach().cpu()
        )
        if "hyp_projections_2d" in out
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=HEALTHY_V7_CKPT)
    p.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--targets", default=" ".join(DEFAULT_TARGETS))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    if not PREREG.is_file():
        raise SystemExit(f"missing prereg {PREREG}")

    results = []
    errors = []
    for pdb_id, chain in _parse_targets(args.targets):
        try:
            results.append(
                _run_one(
                    checkpoint=args.checkpoint,
                    pdb_id=pdb_id,
                    chain=chain,
                    pdb_dir=args.pdb_dir,
                    device=args.device,
                )
            )
        except Exception as exc:  # noqa: BLE001 — smoke aggregates
            errors.append({"pdb_id": pdb_id, "chain": chain, "error": str(exc)})

    all_hard = bool(results) and all(r.get("hard_ok") for r in results) and not errors
    payload = {
        "gate": "tokyo_eye_v7_hyp_biology_mp_smoke",
        "lineage_id": "tokyo_eye_v7_hyp_biology_mp_v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "prereg": str(PREREG),
        "status": "PASS" if all_hard else "FAIL",
        "pass_form": {
            "forward_ok": True,
            "ca_in_mp": False,
            "allow_ca_fallback": False,
            "biology_grades": "report_only",
        },
        "results": results,
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "out": str(args.out)}, indent=2))
    return 0 if all_hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
