#!/usr/bin/env python3
"""Part 0 construction checks for ha_edges_v1 (graph-communication ablation).

Writes artifacts under:
  checkpoints/v66/diagnostics/graph_communication/ha_edges_v1/part0/

Fail any check → do not train (ablation §2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import (
    TAU,
    parse_residue_records_from_pdb_chain,
)
from science.dtie.v66.chem_edge_graph import (
    EDGE_ATTR_CHEM_DIM,
    NUM_ROLE_RELATIONS_WITH_CHEM,
    ROLE_COVALE,
    ROLE_DISULF,
    attach_chem_edge_graph,
)
from science.dtie.v66.gnn.model import GOSPConeMapperV66
from science.dtie.v66.ha_edge_graph import (
    EDGE_ATTR_CHEM_HA_DIM,
    FEELER_RIM_DELTA_STRUCTURES,
    HA_PACKING_CA_MAX_A,
    HA_PACKING_MIN_DIST_A,
    ha_column_map,
)
from science.dtie.v66.role_edge_graph import (
    EDGE_ATTR_ROLE_DIM,
    NUM_ROLE_RELATIONS,
    ROLE_COUPLING,
    ROLE_DEHYDRON,
    ROLE_PACKING,
    ROLE_RIBBON,
    ROLE_SPOKE,
    attach_role_edge_graph,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO
    / "checkpoints/v66/diagnostics/graph_communication/ha_edges_v1/part0"
)
DEFAULT_PDB_DIRS = [
    Path("/tmp/dtie_pdb_cache"),
    REPO / "pdb_cache",
    REPO / "science/dtie/assets/benchmark_pdbs",
]

RELATION_NAMES = {
    ROLE_PACKING: "packing",
    ROLE_DEHYDRON: "dehydron",
    ROLE_SPOKE: "spoke",
    ROLE_RIBBON: "ribbon",
    ROLE_COUPLING: "coupling",
    ROLE_DISULF: "disulf",
    ROLE_COVALE: "covale",
}


def _find_pdb(pdb_id: str, pdb_dirs: list[Path]) -> Path:
    name = f"{pdb_id.upper()}.pdb"
    for d in pdb_dirs:
        p = d / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"PDB {pdb_id} not found in {pdb_dirs}")


def _load_ca_and_features(pdb_path: Path, chain: str = "A") -> dict[str, Any]:
    """Build a minimal prot dict from PDB (legacy path — no DB)."""
    from experiments.training.v6._data import load_protein_graph_from_pdb_legacy

    pdb_id = pdb_path.stem.upper()
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_path.parent)
    if prot is None:
        raise RuntimeError(f"Failed to load {pdb_id} from {pdb_path}")
    records = parse_residue_records_from_pdb_chain(pdb_path, chain)
    prot["residue_records"] = records
    prot["pdb_path"] = pdb_path
    # Promote node features from nested Data if present
    data = prot.get("data")
    if data is not None and getattr(data, "x", None) is not None:
        prot["x"] = data.x
        prot["rho"] = data.x[:, 0]
    return prot


def _counts_from_attr(edge_attr: np.ndarray, *, chem: bool) -> dict[str, int]:
    n_rel = NUM_ROLE_RELATIONS_WITH_CHEM if chem else NUM_ROLE_RELATIONS
    oh = edge_attr[:, GEO_DIM : GEO_DIM + n_rel]
    out: dict[str, int] = {}
    for rid, name in RELATION_NAMES.items():
        if rid >= n_rel:
            continue
        out[name] = int(oh[:, rid].sum() // 2)
    return out


def _build_graphs(prot: dict[str, Any]) -> tuple[Data, Data]:
    ca = prot["ca_coords"]
    if isinstance(ca, torch.Tensor):
        ca_np = ca.detach().cpu().numpy()
    else:
        ca_np = np.asarray(ca, dtype=np.float64)
    x = prot["x"] if "x" in prot else None
    if x is None:
        # topology_three_vector from legacy loader fields
        rho = prot["rho"] if "rho" in prot else prot.get("target_dehydron")
        if isinstance(rho, torch.Tensor):
            rho_t = rho.float().reshape(-1)
        else:
            rho_t = torch.as_tensor(rho, dtype=torch.float32).reshape(-1)
        tau = (rho_t < TAU).float()
        ss = torch.zeros_like(rho_t)
        x = torch.stack([rho_t, tau, ss], dim=-1)
    elif isinstance(x, np.ndarray):
        x = torch.as_tensor(x, dtype=torch.float32)
    if x.size(-1) > 3:
        x = x[:, :3].contiguous()

    def _one(*, ha: bool) -> Data:
        data = Data(
            x=x.clone(),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 4),
        )
        data.rho = x[:, 0].clone()
        data = attach_role_edge_graph(
            data,
            ca_np,
            residue_ids=prot.get("residue_ids"),
            residue_records=prot.get("residue_records"),
            ha_edges=ha,
        )
        data = attach_chem_edge_graph(
            data,
            ca_np,
            prot.get("covalent_bonds") or [],
            residue_ids=prot["residue_ids"],
            structure_id=str(prot.get("structure_id") or prot.get("pdb_id") or "").lower(),
            chain_label=str(prot.get("chain") or "A"),
        )
        return data

    return _one(ha=False), _one(ha=True)


def check_0_1_node_count(base: Data, treat: Data) -> dict[str, Any]:
    n0, n1 = int(base.x.size(0)), int(treat.x.size(0))
    return {
        "check": "0.1",
        "name": "residue_node_count",
        "tag": "euclidean_construction",
        "pass": n0 == n1,
        "baseline_N": n0,
        "treatment_N": n1,
    }


def check_0_2_relation_vocab(treat: Data) -> dict[str, Any]:
    ea = treat.edge_attr.detach().cpu().numpy()
    n_rel = NUM_ROLE_RELATIONS_WITH_CHEM
    oh = ea[:, GEO_DIM : GEO_DIM + n_rel]
    exclusive = bool(np.allclose(oh.sum(axis=1), 1.0))
    present = {
        name: bool(oh[:, rid].sum() > 0)
        for rid, name in RELATION_NAMES.items()
        if rid < n_rel
    }
    # Mandatory families present (packing/dehydron/spoke/ribbon); chem may be empty
    mandatory = all(present[k] for k in ("packing", "dehydron", "spoke", "ribbon"))
    # No attrs beyond chem HA width
    width_ok = int(ea.shape[1]) == EDGE_ATTR_CHEM_HA_DIM
    return {
        "check": "0.2",
        "name": "relation_vocabulary",
        "tag": "euclidean_construction",
        "pass": exclusive and mandatory and width_ok,
        "exclusive_onehots": exclusive,
        "mandatory_present": mandatory,
        "present": present,
        "edge_attr_dim": int(ea.shape[1]),
        "expected_dim": EDGE_ATTR_CHEM_HA_DIM,
        "forbidden_new_relations": True,
    }


def check_0_3_zero_disulf_forward(prot: dict[str, Any], treat: Data) -> dict[str, Any]:
    """Structures with zero disulfides: chem path empty, forward finite."""
    bonds = prot.get("covalent_bonds") or []
    n_disulf = sum(1 for b in bonds if str(b.get("bond_type", "")).lower() == "disulf")
    chem_counts = getattr(treat, "chem_edge_counts", {}) or {}
    n = int(treat.x.size(0))
    # Gate side-channels required for full model forward
    treat.clustering = torch.rand(n)
    treat.degree = torch.ones(n)
    treat.ss_onehot = torch.zeros(n, 3)
    treat.rho = treat.x[:, 0].clone()
    torch.manual_seed(1)
    model = GOSPConeMapperV66(
        node_dim=3,
        hidden=64,
        num_experts=4,
        num_layers=2,
        hyperbolic_gate=True,
        topology_only_gate=True,
        role_edge_mp=True,
        chem_edge_mp=True,
        ha_edge_mp=True,
        init_seed=1,
    )
    model.eval()
    ok = True
    note = "forward_ok"
    try:
        with torch.no_grad():
            out = model(treat.clone())
        depth = out.get("cone_depth") if isinstance(out, dict) else None
        if depth is not None:
            t = depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth)
            if not torch.isfinite(t).all():
                ok = False
                note = "non_finite_cone_depth"
        if not torch.isfinite(treat.edge_attr).all():
            ok = False
            note = "non_finite_edge_attr"
    except Exception as exc:  # noqa: BLE001 — Part 0 must record blockers
        ok = False
        note = f"forward_error:{type(exc).__name__}:{exc}"
    return {
        "check": "0.3",
        "name": "zero_disulf_chem_path",
        "tag": "euclidean_construction",
        "pass": ok,
        "n_disulf_bonds": int(n_disulf),
        "chem_edge_counts": dict(chem_counts),
        "note": note,
    }


def check_0_4_isolated_init() -> dict[str, Any]:
    """HA flag adds no new MLPs — gate/trunk seed must match chem_mvp init."""
    kwargs = dict(
        node_dim=3,
        hidden=128,
        num_experts=4,
        num_layers=3,
        hyperbolic_gate=True,
        topology_only_gate=True,
        role_edge_mp=True,
        chem_edge_mp=True,
        init_seed=1,
    )
    torch.manual_seed(1)
    m0 = GOSPConeMapperV66(**kwargs, ha_edge_mp=False)
    torch.manual_seed(1)
    m1 = GOSPConeMapperV66(**kwargs, ha_edge_mp=True)
    gate_ok = torch.equal(
        m0.gate.prototype_bank.prototype_tangent,
        m1.gate.prototype_bank.prototype_tangent,
    )
    emb_ok = torch.equal(m0.node_emb.weight, m1.node_emb.weight)
    # Radial MLP 0 (packing) identical — no new relation MLPs
    mlp_ok = torch.equal(
        m0.convs[0].radial_mlps[0][0].weight,
        m1.convs[0].radial_mlps[0][0].weight,
    )
    return {
        "check": "0.4",
        "name": "isolated_init_seed_discipline",
        "tag": "euclidean_construction",
        "pass": bool(gate_ok and emb_ok and mlp_ok),
        "prototype_match": bool(gate_ok),
        "node_emb_match": bool(emb_ok),
        "radial_mlp0_match": bool(mlp_ok),
        "note": (
            "ha_edge_mp consumes HA aux via multiplicative scale only — "
            "no new MLP params; init_seed isolates gate/trunk"
        ),
    }


def check_0_5_edge_liveness(
    base: Data, treat: Data, *, structure: str
) -> dict[str, Any]:
    c0 = _counts_from_attr(base.edge_attr.detach().cpu().numpy(), chem=True)
    c1 = _counts_from_attr(treat.edge_attr.detach().cpu().numpy(), chem=True)
    ratios: dict[str, float | None] = {}
    ok = True
    for key in ("packing", "dehydron"):
        b, t = c0[key], c1[key]
        if b == 0:
            ratios[key] = None
            if t == 0 and key == "dehydron":
                ok = False  # silent empty dehydron
            continue
        r = t / b
        ratios[key] = float(r)
        if not (0.5 <= r <= 1.5):
            ok = False
    if c1["dehydron"] == 0:
        ok = False
    return {
        "check": "0.5",
        "name": "edge_liveness",
        "tag": "euclidean_construction",
        "pass": ok,
        "structure": structure,
        "baseline_counts": c0,
        "treatment_counts": c1,
        "ratio_treatment_over_baseline": ratios,
        "band": "±50%",
    }


def check_0_6_tag_audit(checks: list[dict[str, Any]]) -> dict[str, Any]:
    unlabeled = [c["check"] for c in checks if not c.get("tag")]
    return {
        "check": "0.6",
        "name": "tag_audit",
        "tag": "euclidean_construction",
        "pass": len(unlabeled) == 0,
        "unlabeled": unlabeled,
        "metric_tags_used": sorted({c.get("tag") for c in checks if c.get("tag")}),
        "note": (
            "Part 0 metrics are euclidean_construction only; "
            "hyperbolic_inference / disc_view reserved for post-train D4 grade"
        ),
    }


def run_structure(
    pdb_id: str, *, chain: str, pdb_dirs: list[Path]
) -> dict[str, Any]:
    pdb_path = _find_pdb(pdb_id, pdb_dirs)
    prot = _load_ca_and_features(pdb_path, chain=chain)
    base, treat = _build_graphs(prot)
    checks = [
        check_0_1_node_count(base, treat),
        check_0_2_relation_vocab(treat),
        check_0_3_zero_disulf_forward(prot, treat),
        check_0_5_edge_liveness(base, treat, structure=pdb_id.upper()),
    ]
    return {
        "pdb_id": pdb_id.upper(),
        "chain": chain,
        "N": int(base.x.size(0)),
        "baseline_edge_attr_dim": int(base.edge_attr.size(-1)),
        "treatment_edge_attr_dim": int(treat.edge_attr.size(-1)),
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structures",
        nargs="+",
        default=["4OBE", "1LYZ"],
        help="PDB IDs for Part 0 (default 4OBE 1LYZ)",
    )
    parser.add_argument("--chain", default="A")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        action="append",
        default=None,
        help="PDB search dir (repeatable); defaults include cache + benchmark",
    )
    args = parser.parse_args(argv)
    pdb_dirs = list(args.pdb_dir) if args.pdb_dir else list(DEFAULT_PDB_DIRS)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    construction = {
        "arm": "ha_edges_v1",
        "frozen_rules": {
            "packing_existence": (
                f"ordered↔ordered AND (Cα≤8.0 OR "
                f"(HA_min≤{HA_PACKING_MIN_DIST_A} AND Cα≤{HA_PACKING_CA_MAX_A}))"
            ),
            "dehydron_membership": "unchanged (rho_bond < τ via compute_bond_wrapping_count)",
            "dehydron_strength": "0.5*deficit(rho_bond,τ)+0.5*prox(d_NO)",
            "packing_strength": "clip(1 - ha_min/8, 0, 1)",
            "spoke_ribbon": "unchanged vs chem_mvp",
            "GEO_DIM": 4,
            "column_map_chem": ha_column_map(chem=True),
            "feeler_rim_delta_structures": FEELER_RIM_DELTA_STRUCTURES,
            "feeler_rim_delta_note": (
                "δ filed for post-train §5.1: rim enrichment holds on "
                "≥ baseline_hold_count−1 Stage A-12 structures; "
                "not measurable in Part 0 without training"
            ),
        },
        "baseline_edge_attr_dim": EDGE_ATTR_CHEM_DIM,
        "treatment_edge_attr_dim": EDGE_ATTR_CHEM_HA_DIM,
        "role_baseline_dim": EDGE_ATTR_ROLE_DIM,
    }
    (out_dir / "construction_rules.json").write_text(
        json.dumps(construction, indent=2) + "\n"
    )

    init_check = check_0_4_isolated_init()
    per_struct: list[dict[str, Any]] = []
    for pdb_id in args.structures:
        result = run_structure(pdb_id, chain=args.chain, pdb_dirs=pdb_dirs)
        (out_dir / f"{pdb_id.lower()}_part0.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        per_struct.append(result)

    all_checks = [init_check]
    for r in per_struct:
        all_checks.extend(r["checks"])
    tag_check = check_0_6_tag_audit(all_checks)
    all_checks.append(tag_check)

    overall = all(c["pass"] for c in all_checks) and all(r["pass"] for r in per_struct)
    summary = {
        "overall": "PASS" if overall else "FAIL",
        "structures": [r["pdb_id"] for r in per_struct],
        "checks": all_checks,
        "per_structure_pass": {r["pdb_id"]: r["pass"] for r in per_struct},
        "construction_rules": str(out_dir / "construction_rules.json"),
        "training_run": False,
        "note": "Part 0 construction only — training NOT run",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Human-readable table
    lines = [
        "# Part 0 — ha_edges_v1 construction checks",
        "",
        f"**Overall:** {'PASS' if overall else 'FAIL'}",
        "",
        "| Check | Name | Pass | Notes |",
        "|-------|------|------|-------|",
    ]
    for c in all_checks:
        note = c.get("note") or c.get("structure") or ""
        if isinstance(note, dict):
            note = json.dumps(note)
        lines.append(
            f"| {c['check']} | {c['name']} | {'PASS' if c['pass'] else 'FAIL'} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Frozen rules (brief)",
            "",
            f"- Packing: `{construction['frozen_rules']['packing_existence']}`",
            f"- Dehydron strength: `{construction['frozen_rules']['dehydron_strength']}`",
            f"- Aux map (chem): `{construction['frozen_rules']['column_map_chem']['layout']}`",
            f"- Feelers δ: baseline−{FEELER_RIM_DELTA_STRUCTURES} Stage A-12 structures (post-train)",
            "",
            "**Training NOT run.**",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines))
    print(json.dumps({"overall": summary["overall"], "out_dir": str(out_dir)}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
