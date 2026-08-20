"""Shared manifold / graph construction SSOT for v66 euc-reach.

Train, grade, and Part0 diagnostics MUST use this module for suite proteins
and the same ``load_protein_graph_from_pdb_legacy`` path as Stage A corpus load.

Path ``part0_legacy`` (constant ρ multichain) is retained for reconcile/autopsy
only — never for train or grade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from experiments.training.v66._data import load_protein_graph_from_pdb_legacy

REPO = Path(__file__).resolve().parents[3]

# Locked manifold for train / grade / Part0 (default).
MANIFOLD_LOADER = "grade_ssot"
MANIFOLD_RHO_SOURCE = "load_protein_graph_from_pdb_legacy"

PDB_DIRS = [
    REPO / "pdb_cache",
    Path("/tmp/dtie_pdb_cache"),
    REPO / "science/dtie/assets/benchmark_pdbs",
]

EUCLIDEAN_REACH_SUITE = (
    {"pdb_id": "1BE9", "tier": "small_rigid", "chains": ("A",)},
    {"pdb_id": "1GPW", "tier": "medium_void", "chains": ("A", "B")},
    {"pdb_id": "1F88", "tier": "large_7tm", "chains": ("A",)},
)

# Matched cold retrain after manifold reconcile (do not overwrite prior runs).
DEFAULT_BASELINE_RUN_ID = "chem_mvp_stage_a12_cold_v1"
DEFAULT_TREATMENT_RUN_ID = "chem_mvp_euc_reach_manifold_v1"
INVALIDATED_PRIOR_RUN_ID = "chem_mvp_euc_reach_v1"


def find_pdb(pdb_id: str, *, pdb_dirs: list[Path] | None = None) -> Path:
    name = f"{pdb_id.upper()}.pdb"
    for d in pdb_dirs or PDB_DIRS:
        p = d / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"{pdb_id}: PDB not found in {pdb_dirs or PDB_DIRS}")


def _topology_slice(prot: dict[str, Any]) -> dict[str, Any]:
    """Match Stage A topology node_dim=3 (ρ, τ, ss); keep SASA on data.sasa."""
    data = prot["data"]
    if data.x.size(-1) > 3:
        if getattr(data, "sasa", None) is None:
            data.sasa = data.x[:, 3].detach().clone()
        data.x = data.x[:, :3].contiguous()
    return prot


def _concat_prots(parts: list[dict[str, Any]], *, pdb_id: str) -> dict[str, Any]:
    """Concatenate single-chain legacy graphs into one multichain prot."""
    xs, sasas, cas, rids = [], [], [], []
    for p in parts:
        d = p["data"]
        xs.append(d.x.detach().cpu())
        s = getattr(d, "sasa", None)
        if s is None:
            s = torch.zeros(d.x.size(0), dtype=torch.float32)
        sasas.append(s.detach().cpu().reshape(-1))
        ca = p["ca_coords"]
        cas.append(ca.detach().cpu() if torch.is_tensor(ca) else torch.as_tensor(ca))
        rids.extend(list(p["residue_ids"]))
    x = torch.cat(xs, dim=0)
    sasa = torch.cat(sasas, dim=0)
    ca = torch.cat(cas, dim=0)
    n = int(x.size(0))
    data = Data(
        x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 4),
    )
    data.sasa = sasa
    data.clustering = torch.zeros(n, dtype=torch.float32)
    return {
        "pdb_id": pdb_id,
        "structure_id": pdb_id.lower(),
        "chain": "+".join(str(p["chain"]) for p in parts),
        "data": data,
        "ca_coords": ca,
        "residue_ids": rids,
        "n_residues": n,
        "covalent_bonds": [],
        "pdb_path": parts[0].get("pdb_path"),
        "pdb_dir": parts[0].get("pdb_dir"),
        "source": "pdb_legacy_concat",
        "loader": MANIFOLD_LOADER,
        "rho_source": MANIFOLD_RHO_SOURCE,
    }


def load_suite_prot(
    pdb_id: str,
    chains: tuple[str, ...],
    *,
    pdb_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Suite protein load — identical path for grade, Part0 default, and preflight."""
    pdb_dir = find_pdb(pdb_id, pdb_dirs=pdb_dirs).parent
    parts: list[dict[str, Any]] = []
    for ch in chains:
        prot = load_protein_graph_from_pdb_legacy(pdb_id, ch, pdb_dir)
        if prot is None:
            raise RuntimeError(f"Failed to load {pdb_id} chain {ch} from {pdb_dir}")
        prot["structure_id"] = pdb_id.lower()
        prot["loader"] = MANIFOLD_LOADER
        prot["rho_source"] = MANIFOLD_RHO_SOURCE
        parts.append(_topology_slice(prot))
    if len(parts) == 1:
        out = parts[0]
        out["loader"] = MANIFOLD_LOADER
        out["rho_source"] = MANIFOLD_RHO_SOURCE
        return out
    return _concat_prots(parts, pdb_id=pdb_id)


def enrich_prot_residue_records(prot: dict[str, Any], chains: tuple[str, ...]) -> dict[str, Any]:
    """Attach per-chain residue_records when multichain concat omitted them."""
    if prot.get("residue_records"):
        return prot
    pdb_path = prot.get("pdb_path")
    if not pdb_path or not Path(pdb_path).is_file():
        pdb_id = str(prot.get("pdb_id") or prot.get("structure_id", "")).upper()
        try:
            pdb_path = find_pdb(pdb_id)
        except FileNotFoundError:
            return prot
    from science.dtie.common.residue_features import parse_residue_records_from_pdb_chain

    records: list[Any] = []
    for ch in chains:
        records.extend(parse_residue_records_from_pdb_chain(Path(pdb_path), ch))
    if records:
        prot["residue_records"] = records
        prot["pdb_path"] = str(pdb_path)
    return prot
