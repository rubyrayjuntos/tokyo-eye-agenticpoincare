#!/usr/bin/env python3
"""Phase 4b — extract full Ledger B hubs and draft literature map.

Recomputes knockout out_effect (grade JSON lacked hub lists), patches the
alignment artifact with hub inventories, and writes a literature map markdown.

Does NOT change pre-registered interface sets or Pass thresholds.
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
from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
from experiments.training.v6.train_loop import prepare_training_batch, residue_node_count
from science.dtie.common.kras_topo_matrix import residue_index_map
from science.dtie.common.ledger_b_interface import (
    interface_alignment_score,
    resolve_interface_set,
)
from science.training.gnn_lineage import load_model_from_checkpoint

DEFAULT_PREREG = Path("data/gates/ledger_b_interface_prereg_src_shp2.json")
DEFAULT_GRADE = Path(
    "checkpoints/v66/diagnostics/routing_sparsity/ledger_b_interface_alignment.json"
)
DEFAULT_OUT = Path(
    "checkpoints/v66/diagnostics/routing_sparsity/ledger_b_phase4b_hub_map.json"
)
DEFAULT_MD = Path(
    "docs/specs/kras-topo-structural-inference/ledger-b-phase4b-hub-literature-map.md"
)

# Curated literature notes keyed by literature/resseq numbering used in pre-reg
# (SHP2 = deposit auth; SRC = Roskoski/human KD numbers).
_SHP2_NOTES: dict[int, str] = {
    79: "N-SH2 latch (Q79)",
    80: "N-SH2 latch (Y80)",
    84: "N-SH2 latch (H84)",
    100: "N-SH2–PTP tunnel-2 / interface band",
    101: "N-SH2–PTP tunnel-2 / interface band",
    102: "N-SH2–PTP tunnel-2 / interface band",
    103: "N-SH2–PTP tunnel-2 / interface band",
    104: "N-SH2–PTP tunnel-2 / interface band",
    105: "N-SH2–PTP tunnel-2 / interface band",
    106: "N-SH2–PTP tunnel-2 / interface band",
    107: "N-SH2–PTP tunnel-2 / interface band",
    108: "N-SH2–PTP tunnel-2 / interface band",
    109: "N-SH2–PTP tunnel-2 / interface band",
    110: "N-SH2–PTP tunnel-2 / interface band",
    111: "C-SH2 tunnel contact (R111)",
    114: "C-SH2 tunnel contact (H114)",
    249: "PTP tunnel wall (E249)",
    250: "PTP tunnel wall (E250)",
    253: "PTP tunnel wall (T253)",
    254: "PTP tunnel wall (L254)",
    257: "PTP tunnel wall (Q257)",
    265: "PTP latch (R265)",
    269: "PTP latch (Q269)",
    281: "PTP latch (N281)",
    459: "Catalytic Cys459 (PTP active site; secondary)",
    491: "PTP tunnel / covalent-mapped (P491)",
    492: "PTP tunnel / covalent-mapped (K492)",
    495: "PTP tunnel / covalent-mapped (Q495)",
}

_SRC_LIT_NOTES: dict[int, str] = {
    273: "P-loop Gly-rich",
    274: "P-loop Gly-rich",
    275: "P-loop Gly-rich",
    276: "P-loop Gly-rich",
    277: "P-loop Gly-rich",
    278: "P-loop Gly-rich",
    279: "P-loop Gly-rich",
    280: "P-loop Gly-rich",
    281: "P-loop Gly-rich",
    295: "Catalytic Lys295",
    310: "αC Glu310 (Lys–Glu salt bridge)",
    339: "Hinge",
    340: "Hinge",
    341: "Hinge",
    342: "Hinge",
    343: "Hinge",
    344: "Hinge",
    345: "Hinge",
    384: "HRD catalytic loop",
    385: "HRD catalytic loop",
    386: "HRD catalytic loop",
    404: "DFG",
    405: "DFG",
    406: "DFG",
    416: "Autophosphorylation Tyr416 (A-loop)",
}


def _shp2_domain(auth: int) -> str:
    if auth < 1:
        return "unknown"
    if auth <= 103:
        return "N-SH2"
    if auth <= 111:
        return "N-SH2–C-SH2 linker"
    if auth <= 216:
        return "C-SH2"
    if auth <= 220:
        return "C-SH2–PTP linker"
    return "PTP"


def _src_domain(lit: int) -> str:
    if lit < 273:
        return "N-terminal / regulatory (if present)"
    if lit <= 350:
        return "N-lobe"
    if lit <= 403:
        return "C-lobe (pre-DFG)"
    if lit <= 432:
        return "activation segment / A-loop"
    return "C-lobe C-terminal"


def _resnames_from_pdb(pdb_id: str, chain: str, pdb_dir: Path) -> dict[int, str]:
    """Best-effort auth_seq → 3-letter resname from cached PDB."""
    from Bio.PDB import PDBParser

    path = pdb_dir / f"{pdb_id.lower()}.pdb"
    if not path.is_file():
        path = pdb_dir / f"{pdb_id.upper()}.pdb"
    if not path.is_file():
        return {}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(path))
    out: dict[int, str] = {}
    for model in structure:
        for ch in model:
            if ch.id.strip() != chain:
                continue
            for res in ch:
                het, seq, _ic = res.id
                if het.strip():
                    continue
                out[int(seq)] = res.get_resname().strip().upper()
        break
    return out


def _annotate_hubs(
    *,
    pdb_id: str,
    hubs_ranked: list[dict[str, Any]],
    interface: set[int],
    offset: int,
    resnames: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in hubs_ranked:
        auth = h.get("auth_resseq")
        if auth is None:
            continue
        auth = int(auth)
        lit = auth - int(offset)
        in_i = auth in interface
        if pdb_id == "2SHP":
            domain = _shp2_domain(auth)
            note = _SHP2_NOTES.get(auth, "no curated interface annotation — candidate discovery hub")
            lit_key = auth
        else:
            domain = _src_domain(lit)
            note = _SRC_LIT_NOTES.get(
                lit, "no curated KD-motif annotation — candidate discovery hub"
            )
            # A-loop window note
            if 407 <= lit <= 432 and lit not in _SRC_LIT_NOTES:
                note = "Activation-loop window (pre-reg S2)"
            lit_key = lit
        rows.append(
            {
                **h,
                "literature_resseq": lit_key if pdb_id == "3PP0" else auth,
                "resname": resnames.get(auth),
                "domain_bucket": domain,
                "in_prereg_interface": in_i,
                "partition": "H∩I" if in_i else "H\\I",
                "literature_note": note,
            }
        )
    return rows


def _grade_one(
    *,
    checkpoint: Path,
    pdb_id: str,
    chain: str,
    structure_spec: dict[str, Any],
    pass_form: dict[str, Any],
    pdb_dir: Path,
    device: str,
) -> dict[str, Any]:
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id}:{chain}")
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    prot = _align_prot_features(model, prot)
    data0 = prepare_training_batch(model, prot, device)
    n = residue_node_count(data0, prot)
    residue_ids = list(prot.get("residue_ids") or [])[:n]
    idx_map = residue_index_map(residue_ids)
    present = set(idx_map.keys())
    resolved = resolve_interface_set(structure_spec, present)
    scan = knockout_scan(model, prot, device)
    out_effect = np.asarray(scan["out_effect"], dtype=np.float64)[:n]
    scored = interface_alignment_score(
        out_effect,
        idx_map,
        resolved["interface_resseqs"],
        k_frac=0.10,
        recall_threshold=float(pass_form.get("primary_threshold", 0.25)),
        enrichment_threshold=float(pass_form.get("secondary_threshold", 1.25)),
    )
    offset = int(resolved.get("literature_to_auth_offset") or 0)
    resnames = _resnames_from_pdb(pdb_id, chain, pdb_dir)
    annotated = _annotate_hubs(
        pdb_id=pdb_id,
        hubs_ranked=scored.get("hubs_ranked") or [],
        interface=set(resolved["interface_resseqs"]),
        offset=offset,
        resnames=resnames,
    )
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "gene": structure_spec.get("gene"),
        "n_residues": int(n),
        "resolved": resolved,
        "primary": scored,
        "hubs_annotated": annotated,
        "n_hubs": len(annotated),
        "n_overlap": sum(1 for a in annotated if a["in_prereg_interface"]),
        "n_outside": sum(1 for a in annotated if not a["in_prereg_interface"]),
        "pass": bool(scored.get("pass")),
    }


def _markdown(panel: list[dict[str, Any]], *, checkpoint: str, recorded_at: str) -> str:
    lines = [
        "# Ledger B — Phase 4b Hub–Literature Map",
        "",
        f"**Recorded:** {recorded_at}  ",
        f"**Checkpoint:** `{checkpoint}`  ",
        "**Scope:** Interpretative only — does not change Ledger B Pass form or pre-reg *I*.",
        "",
        "## Method",
        "",
        "1. Forward-knockout `out_effect` on champion (same as Ledger B grade).",
        "2. *H* = top 10% by `out_effect`.",
        "3. Partition *H* into **H∩I** (overlap with locked pre-reg interface) and **H\\\\I** (discovery candidates).",
        "4. Annotate with domain buckets + curated motif notes (not smoke-test hub lists).",
        "",
    ]
    for row in panel:
        pdb = row["pdb_id"]
        offset = row["resolved"].get("literature_to_auth_offset") or 0
        lines += [
            f"## {pdb} ({row.get('gene')})",
            "",
            f"- *n* residues: {row['n_residues']}; |*H*|={row['n_hubs']}; "
            f"|H∩I|={row['n_overlap']}; |H\\\\I|={row['n_outside']}",
            f"- recall@top-10% (recomputed): {row['primary'].get('recall_at_top_k')}",
            f"- literature→auth offset: {offset}",
            "",
            "### H∩I (partial canonical hits)",
            "",
            "| rank | auth | lit | aa | domain | note |",
            "|-----:|-----:|----:|:--:|:-------|:-----|",
        ]
        for a in row["hubs_annotated"]:
            if not a["in_prereg_interface"]:
                continue
            lines.append(
                f"| {a['rank']} | {a['auth_resseq']} | {a['literature_resseq']} | "
                f"{a.get('resname') or '?'} | {a['domain_bucket']} | {a['literature_note']} |"
            )
        lines += [
            "",
            "### H\\\\I (discovery candidates — model hubs outside pre-reg *I*)",
            "",
            "| rank | auth | lit | aa | domain | note | out_effect |",
            "|-----:|-----:|----:|:--:|:-------|:-----|----------:|",
        ]
        for a in row["hubs_annotated"]:
            if a["in_prereg_interface"]:
                continue
            lines.append(
                f"| {a['rank']} | {a['auth_resseq']} | {a['literature_resseq']} | "
                f"{a.get('resname') or '?'} | {a['domain_bucket']} | {a['literature_note']} | "
                f"{a['out_effect']:.4f} |"
            )
        lines.append("")

    lines += [
        "## Reading guide",
        "",
        "- **H∩I** sites are the only residues that contributed to the failed recall; "
        "ask whether they are high-utility latch/tunnel/spine contacts or incidental bleed.",
        "- **H\\\\I** is the discovery claim: if these hubs are biologically coherent "
        "(domain clustering, known cryptic sites, or pathway contacts not in *I*), "
        "the dual-ledger story holds without lowering the 0.25 bar.",
        "- Do **not** promote H\\\\I residues into *I* without a new pre-registration stamp.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from experiments.training.v66.healthy_fix1 import (
        FIX1_SPARSITY_CHAMPION_CKPT,
        HEALTHY_FIX1_CKPT,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
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
    p.add_argument("--grade-artifact", type=Path, default=DEFAULT_GRADE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = p.parse_args(argv)
    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    prereg = json.loads(args.prereg.read_text())
    pass_form = prereg["pass_form"]
    panel: list[dict[str, Any]] = []
    for pdb_id, spec in prereg["structures"].items():
        chain = str(spec.get("chain") or "A")
        print(f"Phase 4b {pdb_id}:{chain} ...", flush=True)
        row = _grade_one(
            checkpoint=args.checkpoint,
            pdb_id=pdb_id,
            chain=chain,
            structure_spec=spec,
            pass_form=pass_form,
            pdb_dir=args.pdb_dir,
            device=args.device,
        )
        print(
            f"  |H|={row['n_hubs']} H∩I={row['n_overlap']} H\\I={row['n_outside']} "
            f"recall={row['primary'].get('recall_at_top_k')}",
            flush=True,
        )
        panel.append(row)

    recorded_at = datetime.now(timezone.utc).isoformat()
    out = {
        "schema_version": 1,
        "probe": "ledger_b_phase4b_hub_map",
        "recorded_at": recorded_at,
        "checkpoint": str(args.checkpoint),
        "prereg": str(args.prereg),
        "panel": panel,
        "notes": [
            "Interpretative validation only.",
            "Pre-reg I and Pass threshold 0.25 unchanged.",
            "Hub lists re-emitted because grade JSON previously lacked full H.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output}")

    # Patch grade artifact with hub lists (preserve prior verdict fields).
    if args.grade_artifact.is_file():
        grade = json.loads(args.grade_artifact.read_text())
        by_id = {r["pdb_id"]: r for r in panel}
        for prow in grade.get("panel") or []:
            src = by_id.get(prow.get("pdb_id"))
            if not src:
                continue
            prim = prow.setdefault("primary", {})
            for key in (
                "hub_resseqs",
                "hub_outside_interface",
                "hubs_ranked",
                "hub_overlap_resseqs",
                "n_hubs",
                "n_hub_overlap",
            ):
                if key in src["primary"]:
                    prim[key] = src["primary"][key]
            prow["hubs_annotated"] = src["hubs_annotated"]
            prow["phase4b_patched_at"] = recorded_at
        grade["phase4b"] = {
            "patched_at": recorded_at,
            "hub_map": str(args.output),
            "literature_map": str(args.markdown),
        }
        args.grade_artifact.write_text(json.dumps(grade, indent=2) + "\n")
        print(f"patched {args.grade_artifact}")

    md = _markdown(panel, checkpoint=str(args.checkpoint), recorded_at=recorded_at)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(md)
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
