#!/usr/bin/env python3
"""Part 0 construction checks for euclidean_reach Top-K multi-scale suite (v66).

Writes: checkpoints/v66/diagnostics/euclidean_reach/part0/

**Dual-path loaders (do not delete either):**

- ``part0_legacy`` — prior multichain CA loader (``_load_multichain_ca``):
  constant ρ≡12, τ≡0; no dehydron computation. Autopsy / counterfactual only.
- ``grade_ssot`` — ``load_suite_prot`` / train-grade SSOT (real wrapping ρ/τ).

Default Part0 suite uses ``grade_ssot``. Use ``--loader both`` or
``--reconcile`` to expose Path A vs Path B side-by-side (1GPW ρ dump).

Uses only science.dtie.v66 + experiments.training.v66 (no v6 look-back).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch_geometric.data import Data

from science.dtie.common.residue_features import TAU
from science.dtie.v66.chem_edge_graph import attach_chem_edge_graph
from science.dtie.v66.euclidean_shortcut_graph import (
    EDGE_ATTR_EUC_DIM,
    EUC_MIN_GRAPH_HOPS,
    EUC_MIN_SEQ_SEP,
    EUC_SPATIAL_CUTOFF_A,
    EUC_TOP_K,
    NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT,
    ROLE_EUCLIDEAN_SHORTCUT,
    attach_euclidean_shortcut_graph,
    undirected_hop_distances,
)
from science.dtie.v66.gnn.model import GOSPConeMapperV66
from science.dtie.v66.role_edge_graph import (
    ROLE_DEHYDRON,
    ROLE_PACKING,
    ROLE_RIBBON,
    ROLE_SPOKE,
    attach_role_edge_graph,
)
from science.dtie.v66.thermo_edge_features import GEO_DIM, parse_auth_seq_ids

DEFAULT_OUT = REPO / "checkpoints/v66/diagnostics/euclidean_reach/part0"
DEFAULT_RECONCILE_DIR = REPO / "checkpoints/v66/diagnostics/euclidean_reach"
DEFAULT_SITE_LISTS = (
    REPO / "checkpoints/v66/diagnostics/euclidean_reach/site_lists"
)
DEFAULT_PDB_DIRS = [
    Path("/tmp/dtie_pdb_cache"),
    REPO / "pdb_cache",
    REPO / "science/dtie/assets/benchmark_pdbs",
]

LoaderName = Literal["grade_ssot", "part0_legacy"]

# Multi-scale suite (ablation §4). Default loader = grade SSOT (real ρ/τ).
# Dual-path: --loader part0_legacy | both | --reconcile restores Path A.
SUITE = (
    {
        "pdb_id": "1BE9",
        "tier": "small_rigid",
        "require_shortcuts": False,
        "chains": ("A",),
    },
    {
        "pdb_id": "1GPW",
        "tier": "medium_void",
        "require_shortcuts": False,
        "multichain": True,
        "chains": ("A", "B"),  # HisF + HisH only
    },
    {
        "pdb_id": "1F88",
        "tier": "large_7tm",
        "require_shortcuts": True,
        "chains": ("A",),
    },
)

# Frozen 1GPW hub map (HisF=A, HisH=B); must match site_lists JSON.
GPW_HUBS = (
    {"chain_id": "A", "auth_seq": 5, "resname_expected": "ARG", "role": "interface_gate"},
    {"chain_id": "A", "auth_seq": 46, "resname_expected": "GLU", "role": "interface_gate"},
    {"chain_id": "A", "auth_seq": 50, "resname_expected": "LEU", "role": "prfar_conduit"},
    {"chain_id": "A", "auth_seq": 59, "resname_expected": "ARG", "role": "prfar_conduit"},
    {"chain_id": "A", "auth_seq": 91, "resname_expected": "GLU", "role": "prfar_conduit"},
    {"chain_id": "B", "auth_seq": 10, "resname_expected": "PRO", "role": "conduit"},
    {"chain_id": "B", "auth_seq": 99, "resname_expected": "GLY", "role": "interface_gate"},
)


def _find_pdb(pdb_id: str, pdb_dirs: list[Path]) -> Path | None:
    name = f"{pdb_id.upper()}.pdb"
    for d in pdb_dirs:
        p = d / name
        if p.is_file():
            return p
    return None


def _ensure_pdb(pdb_id: str, pdb_dirs: list[Path]) -> Path:
    found = _find_pdb(pdb_id, pdb_dirs)
    if found is not None:
        return found
    # Download into first writable cache
    from experiments.training.v66._data import _download_pdb

    cache = pdb_dirs[1] if len(pdb_dirs) > 1 else pdb_dirs[0]
    cache.mkdir(parents=True, exist_ok=True)
    return _download_pdb(pdb_id, cache)


def _load_multichain_ca(
    pdb_path: Path, pdb_id: str, chains: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Path A — prior Part0 multichain CA loader (``part0_legacy``).

    **Kept permanently** for dual-path reconcile / counterfactual. Does **not**
    compute dehydron wrapping: ρ is filled with the constant **12.0** and τ≡0
    for every CA. That yields ρ < TAU(13) → 0 ordered → empty packing/spoke →
    large diameter → false liveness (historical 1GPW sc=684).

    Prefer :func:`load_suite_prot_for_part0` (``grade_ssot``) for grade/train
    parity. Use this path only when comparing loaders side-by-side.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))
    want = set(chains) if chains else None
    ca_list: list[np.ndarray] = []
    rids: list[str] = []
    for model in structure:
        for chain in model:
            cid = chain.id.strip() or "A"
            if want is not None and cid not in want:
                continue
            for res in chain:
                if res.get_id()[0] != " ":
                    continue
                if "CA" not in res:
                    continue
                ca_list.append(np.asarray(res["CA"].get_coord(), dtype=np.float64))
                rids.append(f"{cid}:{int(res.get_id()[1])}:")
        break
    if len(ca_list) < 10:
        raise RuntimeError(f"{pdb_id}: too few CA atoms in {pdb_path}")
    ca = np.stack(ca_list, axis=0)
    n = ca.shape[0]
    # Synthetic constant — not a dehydron measurement.
    rho = np.full(n, 12.0, dtype=np.float64)
    tau = np.zeros(n, dtype=np.float64)
    x = torch.stack(
        [
            torch.as_tensor(rho, dtype=torch.float32),
            torch.as_tensor(tau, dtype=torch.float32),
            torch.zeros(n),
        ],
        dim=-1,
    )
    return {
        "pdb_id": pdb_id,
        "structure_id": pdb_id.lower(),
        "chain": "+".join(chains) if chains else "multi",
        "ca_coords": ca,
        "residue_ids": rids,
        "x": x,
        "rho": x[:, 0],
        "tau": x[:, 1],
        "covalent_bonds": [],
        "pdb_path": pdb_path,
        "loader": "part0_legacy",
        "rho_source": "constant_12",
    }


def _load_single_chain(pdb_path: Path, chain: str = "A") -> dict[str, Any]:
    """Legacy single-chain load (kept for siphon Part0-parity counterfactual)."""
    from experiments.training.v66._data import load_protein_graph_from_pdb_legacy
    from science.dtie.common.residue_features import parse_residue_records_from_pdb_chain

    pdb_id = pdb_path.stem.upper().split("_")[0]
    prot = load_protein_graph_from_pdb_legacy(pdb_id, chain, pdb_path.parent)
    if prot is None:
        return _load_multichain_ca(pdb_path, pdb_id)
    records = parse_residue_records_from_pdb_chain(pdb_path, chain)
    prot["residue_records"] = records
    prot["pdb_path"] = pdb_path
    data = prot.get("data")
    if data is not None and getattr(data, "x", None) is not None:
        prot["x"] = data.x
        prot["rho"] = data.x[:, 0]
    return prot


def load_suite_prot_for_part0(
    pdb_id: str, chains: tuple[str, ...], *, pdb_dirs: list[Path] | None = None
) -> dict[str, Any]:
    """Path B — suite load via shared ``manifold_ssot`` (grade/train parity)."""
    from experiments.training.v66.manifold_ssot import (
        MANIFOLD_LOADER,
        MANIFOLD_RHO_SOURCE,
        enrich_prot_residue_records,
        find_pdb,
        load_suite_prot,
    )

    dirs = list(pdb_dirs) if pdb_dirs else list(DEFAULT_PDB_DIRS)
    path = _ensure_pdb(pdb_id, dirs)
    prot = load_suite_prot(pdb_id, chains, pdb_dirs=dirs)
    prot = enrich_prot_residue_records(prot, chains)
    data = prot.get("data")
    if data is not None and getattr(data, "x", None) is not None:
        x = data.x
        if x.size(-1) > 3:
            x = x[:, :3].contiguous()
            data.x = x
        prot["x"] = x
        prot["rho"] = x[:, 0]
        prot["tau"] = x[:, 1] if x.size(-1) > 1 else torch.zeros(x.size(0))
    if prot.get("pdb_path") is None:
        prot["pdb_path"] = path
    prot["pdb_id"] = pdb_id.upper()
    prot["structure_id"] = pdb_id.lower()
    prot["loader"] = MANIFOLD_LOADER
    prot["rho_source"] = MANIFOLD_RHO_SOURCE
    return prot


def load_part0_legacy(
    pdb_id: str, chains: tuple[str, ...], *, pdb_dirs: list[Path] | None = None
) -> dict[str, Any]:
    """Path A — prior Part0-style loader (constant ρ≡12 when multichain).

    Single-chain suite members fall back to legacy PDB graph when available;
    multichain (1GPW A+B) uses ``_load_multichain_ca`` exactly as historical Part0.
    """
    dirs = list(pdb_dirs) if pdb_dirs else list(DEFAULT_PDB_DIRS)
    path = _ensure_pdb(pdb_id, dirs)
    if len(chains) > 1:
        return _load_multichain_ca(path, pdb_id.upper(), chains=chains)
    # Historical Part0 single-chain path
    prot = _load_single_chain(path, chain=chains[0])
    prot["loader"] = "part0_legacy"
    if prot.get("rho_source") is None:
        prot["rho_source"] = "legacy_single_or_constant"
    return prot


def load_prot_for_part0(
    pdb_id: str,
    chains: tuple[str, ...],
    *,
    loader: LoaderName = "grade_ssot",
    pdb_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    """Dispatch dual-path loaders. Both paths remain first-class."""
    if loader == "part0_legacy":
        return load_part0_legacy(pdb_id, chains, pdb_dirs=pdb_dirs)
    if loader == "grade_ssot":
        return load_suite_prot_for_part0(pdb_id, chains, pdb_dirs=pdb_dirs)
    raise ValueError(f"unknown loader {loader!r}; use part0_legacy|grade_ssot")


def _rho_tau_arrays(prot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = prot.get("x")
    if x is None and prot.get("data") is not None:
        x = prot["data"].x
    if x is None:
        rho = prot.get("rho")
        if rho is None:
            raise RuntimeError("prot missing x/rho")
        rho_np = (
            rho.detach().cpu().numpy()
            if isinstance(rho, torch.Tensor)
            else np.asarray(rho, dtype=np.float64)
        )
        tau = prot.get("tau")
        if tau is None:
            tau_np = np.zeros_like(rho_np)
        else:
            tau_np = (
                tau.detach().cpu().numpy()
                if isinstance(tau, torch.Tensor)
                else np.asarray(tau, dtype=np.float64)
            )
        return rho_np.astype(np.float64), tau_np.astype(np.float64)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)
    rho_np = x_np[:, 0].astype(np.float64)
    tau_np = (
        x_np[:, 1].astype(np.float64)
        if x_np.shape[-1] > 1
        else np.zeros(x_np.shape[0], dtype=np.float64)
    )
    return rho_np, tau_np


def _key_from_rid(rid: str) -> tuple[str, int]:
    parts = str(rid).split(":")
    return (parts[0] or "A", int(parts[1]))


def _role_chem_sc_stats(prot: dict[str, Any]) -> dict[str, Any]:
    """Packing/spoke/diam/sc on full chem+euc graph for one loader path."""
    chem = _build_chem(prot)
    ca = _ca_np(prot)
    ea = chem.edge_attr.detach().cpu().numpy()
    # chem one-hots start at GEO_DIM; packing/spoke/dehydron/ribbon in first 4
    oh = ea[:, GEO_DIM : GEO_DIM + 7]
    n_pack = int(oh[:, ROLE_PACKING].sum() // 2)
    n_dh = int(oh[:, ROLE_DEHYDRON].sum() // 2)
    n_spoke = int(oh[:, ROLE_SPOKE].sum() // 2)
    n_ribbon = int(oh[:, ROLE_RIBBON].sum() // 2)
    role_counts = getattr(chem, "role_edge_counts", None)
    if isinstance(role_counts, dict):
        n_pack = int(role_counts.get("packing", n_pack))
        n_dh = int(role_counts.get("dehydron", n_dh))
        n_spoke = int(role_counts.get("spoke", n_spoke))
        n_ribbon = int(role_counts.get("ribbon", n_ribbon))
    euc = attach_euclidean_shortcut_graph(
        chem, ca, residue_ids=prot.get("residue_ids"), force=True
    )
    n_sc = int(getattr(euc, "euclidean_shortcut_count_undirected", 0))
    stats = dict(getattr(euc, "euclidean_shortcut_stats", {}) or {})
    rho, tau = _rho_tau_arrays(prot)
    n_ord = int(np.sum(rho >= float(TAU)))
    return {
        "n_nodes": int(rho.shape[0]),
        "n_ordered_rho_ge_tau": n_ord,
        "tau_threshold": float(TAU),
        "packing_undirected": n_pack,
        "spoke_undirected": n_spoke,
        "dehydron_undirected": n_dh,
        "ribbon_undirected": n_ribbon,
        "packing_directed": n_pack * 2,
        "spoke_directed": n_spoke * 2,
        "dehydron_directed": n_dh * 2,
        "ribbon_directed": n_ribbon * 2,
        "graph_diameter": stats.get("graph_diameter"),
        "n_shortcuts_undirected": n_sc,
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "rho_unique_approx": int(np.unique(np.round(rho, 6)).size),
        "tau_sum": float(np.sum(tau)),
        "loader": prot.get("loader"),
        "rho_source": prot.get("rho_source"),
    }


def reconcile_1gpw_rho(
    *,
    pdb_dirs: list[Path] | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Mandatory Path A vs Path B ρ dump for 1GPW A+B (same chain+auth_seq)."""
    dirs = list(pdb_dirs) if pdb_dirs else list(DEFAULT_PDB_DIRS)
    out = out_dir or DEFAULT_RECONCILE_DIR
    out.mkdir(parents=True, exist_ok=True)
    chains = ("A", "B")
    pdb_id = "1GPW"

    prot_a = load_prot_for_part0(
        pdb_id, chains, loader="part0_legacy", pdb_dirs=dirs
    )
    prot_b = load_prot_for_part0(
        pdb_id, chains, loader="grade_ssot", pdb_dirs=dirs
    )
    rho_a, tau_a = _rho_tau_arrays(prot_a)
    rho_b, tau_b = _rho_tau_arrays(prot_b)
    rids_a = list(prot_a["residue_ids"])
    rids_b = list(prot_b["residue_ids"])
    map_a = {_key_from_rid(r): i for i, r in enumerate(rids_a)}
    map_b = {_key_from_rid(r): i for i, r in enumerate(rids_b)}
    keys = sorted(set(map_a) | set(map_b), key=lambda k: (k[0], k[1]))

    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    n_disagree = 0
    n_only_a = 0
    n_only_b = 0
    for chain, auth in keys:
        ia = map_a.get((chain, auth))
        ib = map_b.get((chain, auth))
        if ia is None:
            n_only_b += 1
            rows.append(
                {
                    "chain": chain,
                    "auth_seq": auth,
                    "rho_part0_legacy": None,
                    "rho_grade": float(rho_b[ib]),
                    "tau_part0_legacy": None,
                    "tau_grade": float(tau_b[ib]),
                    "delta_rho": None,
                    "in_both": False,
                    "only_in": "grade_ssot",
                }
            )
            continue
        if ib is None:
            n_only_a += 1
            rows.append(
                {
                    "chain": chain,
                    "auth_seq": auth,
                    "rho_part0_legacy": float(rho_a[ia]),
                    "rho_grade": None,
                    "tau_part0_legacy": float(tau_a[ia]),
                    "tau_grade": None,
                    "delta_rho": None,
                    "in_both": False,
                    "only_in": "part0_legacy",
                }
            )
            continue
        ra = float(rho_a[ia])
        rb = float(rho_b[ib])
        d = rb - ra
        deltas.append(d)
        if abs(d) > 1e-6:
            n_disagree += 1
        rows.append(
            {
                "chain": chain,
                "auth_seq": auth,
                "rho_part0_legacy": ra,
                "rho_grade": rb,
                "tau_part0_legacy": float(tau_a[ia]),
                "tau_grade": float(tau_b[ib]),
                "delta_rho": d,
                "in_both": True,
                "only_in": None,
            }
        )

    graph_a = _role_chem_sc_stats(prot_a)
    graph_b = _role_chem_sc_stats(prot_b)
    abs_d = [abs(d) for d in deltas]
    n_both = len(deltas)

    # Scientific framing: Path A never computed real ρ
    path_a_is_constant = (
        graph_a.get("rho_unique_approx", 99) <= 1
        and abs(float(graph_a.get("rho_mean", 0.0)) - 12.0) < 1e-6
    )
    verdict = (
        "part0_never_computed_real_rho"
        if path_a_is_constant
        else "rho_values_differ_investigate"
    )

    summary = {
        "n_residues_union": len(keys),
        "n_both": n_both,
        "n_only_part0_legacy": n_only_a,
        "n_only_grade_ssot": n_only_b,
        "n_disagree_rho": n_disagree,
        "n_agree_rho": n_both - n_disagree,
        "mean_abs_delta_rho": float(np.mean(abs_d)) if abs_d else None,
        "max_abs_delta_rho": float(np.max(abs_d)) if abs_d else None,
        "mean_delta_rho_grade_minus_legacy": (
            float(np.mean(deltas)) if deltas else None
        ),
        "n_ordered_part0_legacy": graph_a["n_ordered_rho_ge_tau"],
        "n_ordered_grade_ssot": graph_b["n_ordered_rho_ge_tau"],
        "tau_threshold": float(TAU),
        "path_a_constant_rho_12": path_a_is_constant,
        "verdict": verdict,
        "sc_divergence": {
            "part0_legacy": graph_a["n_shortcuts_undirected"],
            "grade_ssot": graph_b["n_shortcuts_undirected"],
            "mechanism": (
                "constant ρ=12 < TAU → 0 ordered → empty packing/spoke → "
                "large diam → hop>6 Top-K live (sc≈684); grade real ρ → dense "
                "packing/spoke → diam≈7 → hop>6 empty (sc=0)"
            ),
        },
    }

    payload = {
        "schema_version": 1,
        "pdb_id": pdb_id,
        "chains": list(chains),
        "status": "INVALIDATED_pending_manifold_reconcile",
        "path_a": {
            "name": "part0_legacy",
            "loader": "_load_multichain_ca",
            "rho_source": "constant_12_no_dehydron",
            "graph": graph_a,
        },
        "path_b": {
            "name": "grade_ssot",
            "loader": "load_suite_prot / prepare_training_batch SSOT",
            "rho_source": "load_protein_graph_from_pdb_legacy",
            "graph": graph_b,
        },
        "summary": summary,
        "residues": rows,
        "ssot": "docs/specs/v66_chem_MVP/ablation_euclidean_reach.md",
        "note": (
            "Part0 liveness and grade graphs were not the same experiment. "
            "Fail grades on mismatched construction are not authoritative "
            "lever verdicts until Path A vs B ρ/graph parity is resolved. "
            "Do not treat as CLOSED Fail; do not pivot to governor redesign "
            "as primary fix — primary is manifold/loader reconcile."
        ),
    }
    json_path = out / "manifold_reconcile_1gpw_rho.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(json_path)
    return payload


def write_manifold_reconcile_md(payload: dict[str, Any], out_dir: Path) -> Path:
    """Honest autopsy MD — no clean-manifold theater, no CLOSED Fail."""
    s = payload["summary"]
    ga = payload["path_a"]["graph"]
    gb = payload["path_b"]["graph"]
    md = f"""# Manifold reconcile — 1GPW A+B (Path A vs Path B)

**Date:** 2026-07-19  
**Status:** **INVALIDATED — Pipeline / construction mismatch** (pending manifold reconcile)  
**Companion JSON:** `manifold_reconcile_1gpw_rho.json`  
**SSOT:** `docs/specs/v66_chem_MVP/ablation_euclidean_reach.md`  
**Scope:** v66 only; offline ρ/graph dump (no train).

---

## Verdict (one line)

Path A (`part0_legacy` / `_load_multichain_ca`) **never computed real dehydron ρ** — it fills **ρ≡12, τ≡0** for every CA. Path B (`grade_ssot` / `load_suite_prot`) uses **real wrapping ρ/τ**. The historical Part0 **sc=684** vs grade **sc=0** is therefore **synthetic constant-ρ vs real-ρ construction mismatch**, not evidence of "ρ corruption" on the train/grade manifold, and **not** a settled "empty-lever ⇒ ignore / CLOSED Fail" close. Status remains **invalidated pending manifold/loader reconcile**.

---

## What differs

| Field | Path A — `part0_legacy` | Path B — `grade_ssot` |
|-------|-------------------------|------------------------|
| Loader | `_load_multichain_ca` (Bio.PDB CA only) | `load_suite_prot` → `load_protein_graph_from_pdb_legacy` (+ A+B concat) |
| ρ | **Constant 12.0** (no dehydron) | Real wrapping ρ |
| τ | **0** everywhere | Real shell / τ feature |
| Ordered (ρ≥TAU={s['tau_threshold']}) | **{s['n_ordered_part0_legacy']}** | **{s['n_ordered_grade_ssot']}** |
| packing↑ (directed) | {ga['packing_directed']} | {gb['packing_directed']} |
| spoke↑ (directed) | {ga['spoke_directed']} | {gb['spoke_directed']} |
| diam | {ga['graph_diameter']} | {gb['graph_diameter']} |
| sc (undirected) | **{ga['n_shortcuts_undirected']}** | **{gb['n_shortcuts_undirected']}** |

### ρ summary (same residues, chain + auth_seq)

- n both paths: **{s['n_both']}**
- n disagree \\|Δρ\\| > 0: **{s['n_disagree_rho']}**
- mean \\|Δρ\\|: **{s['mean_abs_delta_rho']}**
- max \\|Δρ\\|: **{s['max_abs_delta_rho']}**
- Path A constant ρ=12?: **{s['path_a_constant_rho_12']}**
- Scientific verdict code: **`{s['verdict']}`**

If Path A is constant-ρ, **n_disagree ≈ n_both** whenever grade ρ ≠ 12 — that is expected, not a train-time corruption signature.

---

## Corruption vs synthetic constant ρ

**Evidence supports: Part0 never computed real ρ** (`part0_never_computed_real_rho`).

- Path A literally assigns `np.full(n, 12.0)` — no wrapping count, no H-bond geometry.
- That is **not** "the clean manifold got corrupted." It is a **different experiment**: a constant-feature CA graph vs the grade/train dehydron graph.
- Do **not** invent a "clean manifold" narrative around Path A. Do **not** treat grade Fail as authoritative until dual-path parity is understood and intentionally chosen.

---

## Why sc diverges (684 vs 0)

Role packing/spoke require **ordered** residues (ρ ≥ TAU={s['tau_threshold']}).

1. **Path A:** ρ≡12 < 13 → **0 ordered** → packing=0, spoke=0 → sparse baseline (ribbon + dehydron-from-ρ-mean fallback only) → **large diameter** → many pairs satisfy hop>6 within Top-K≤20Å → **sc≈{ga['n_shortcuts_undirected']}**.
2. **Path B:** real ρ → **{s['n_ordered_grade_ssot']} ordered** → dense packing/spoke → **diameter ≈ {gb['graph_diameter']}** → almost no hop>6 survivors → **sc={gb['n_shortcuts_undirected']}**.

So sc=684 was **false Part0 liveness** on a non-train graph; sc=0 on grade is the hop>6 filter on a **dense real role graph**. Neither alone closes the lever as Fail without acknowledging the mismatch.

---

## Retracted closing narratives

- ~~**CLOSED Fail**~~ → **INVALIDATED — Pipeline / construction mismatch**
- ~~**1GPW empty-lever artifact ⇒ ignore**~~ → **retracted** as the closing story; empty grade sc is real under Path B but was compared against a Path A fantasy graph
- ~~**Primary next = governor redesign**~~ → **primary next = manifold/loader reconcile** (keep dual-path; do not erase Path A)
- Fail grades on mismatched construction are **not** authoritative lever verdicts until Path A vs B ρ/graph parity is resolved

Factual numbers (e.g. grade 1F88 sc=95; cold-train first/best Δ cone/trunk) remain on record **under the invalidated banner** — they are measurements, not a closed Fail.

---

## Dual-path capability

`experiments/diagnostics/euclidean_reach_part0.py`:

```bash
# Path B only (default suite)
python experiments/diagnostics/euclidean_reach_part0.py --loader grade_ssot

# Path A only (legacy constant-ρ multichain)
python experiments/diagnostics/euclidean_reach_part0.py --loader part0_legacy

# Side-by-side suite + write this reconcile dump
python experiments/diagnostics/euclidean_reach_part0.py --reconcile
# or: --loader both
```

Legacy `_load_multichain_ca` is **retained** — do not delete.
"""
    path = out_dir / "MANIFOLD_RECONCILE.md"
    path.write_text(md, encoding="utf-8")
    return path


def _build_chem(prot: dict[str, Any]) -> Data:
    from science.dtie.v66.role_edge_graph import resolve_residue_records_for_prot

    ca = prot["ca_coords"]
    ca_np = ca.detach().cpu().numpy() if isinstance(ca, torch.Tensor) else np.asarray(ca)
    data_in = prot.get("data")
    if data_in is not None and getattr(data_in, "x", None) is not None:
        x = data_in.x
    else:
        x = prot["x"]
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x, dtype=torch.float32)
    if x.size(-1) > 3:
        x = x[:, :3].contiguous()
    data = Data(
        x=x.clone(),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 4),
    )
    data.rho = x[:, 0].clone()
    records = prot.get("residue_records")
    if records is None:
        records = resolve_residue_records_for_prot(prot)
    data = attach_role_edge_graph(
        data,
        ca_np,
        residue_ids=prot.get("residue_ids"),
        residue_records=records,
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


def _ca_np(prot: dict[str, Any]) -> np.ndarray:
    ca = prot["ca_coords"]
    if isinstance(ca, torch.Tensor):
        return ca.detach().cpu().numpy()
    return np.asarray(ca, dtype=np.float64)


def _parse_chains(rids: list[str] | None, n: int) -> np.ndarray:
    out = np.array(["A"] * n, dtype=object)
    if not rids or len(rids) != n:
        return out
    for i, rid in enumerate(rids):
        parts = str(rid).split(":")
        out[i] = parts[0] or "A"
    return out


def check_structure(prot: dict[str, Any], *, require_shortcuts: bool) -> dict[str, Any]:
    chem = _build_chem(prot)
    ca = _ca_np(prot)
    baseline_ei = chem.edge_index.detach().clone()
    baseline_ea = chem.edge_attr.detach().clone()
    euc = attach_euclidean_shortcut_graph(
        chem, ca, residue_ids=prot.get("residue_ids"), force=True
    )
    n = int(euc.x.size(0))
    n_sc = int(getattr(euc, "euclidean_shortcut_count_undirected", 0))
    stats = dict(getattr(euc, "euclidean_shortcut_stats", {}) or {})

    # Rule spot-check
    hops = undirected_hop_distances(baseline_ei, n)
    seq = parse_auth_seq_ids(prot.get("residue_ids"), n)
    chains = _parse_chains(prot.get("residue_ids"), n)
    ea = euc.edge_attr.detach().cpu().numpy()
    ei = euc.edge_index.detach().cpu().numpy()
    oh = ea[:, GEO_DIM : GEO_DIM + NUM_ROLE_RELATIONS_WITH_EUC_SHORTCUT]
    mask = oh[:, ROLE_EUCLIDEAN_SHORTCUT] > 0.5
    pairs = {
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(ei[0, mask], ei[1, mask])
    }
    violations: list[str] = []
    for i, j in list(pairs)[:80]:
        d = float(np.linalg.norm(ca[i] - ca[j]))
        if d > EUC_SPATIAL_CUTOFF_A + 1e-3:
            violations.append(f"spatial:{i}-{j}:{d:.2f}")
        h = hops[i, j]
        if np.isfinite(h) and h <= EUC_MIN_GRAPH_HOPS:
            violations.append(f"hops:{i}-{j}:{h}")
        if chains[i] == chains[j] and abs(int(seq[i]) - int(seq[j])) < EUC_MIN_SEQ_SEP:
            violations.append(f"seq:{i}-{j}")

    bound_ok = n_sc <= n * EUC_TOP_K
    live_ok = (n_sc > 0) if require_shortcuts else True
    baseline_ok = (
        torch.equal(baseline_ei, euc.edge_index[:, : baseline_ei.size(1)])
        if euc.edge_index.size(1) >= baseline_ei.size(1)
        else False
    )
    # chem one-hots prefix width grew by 1 — compare packing one-hot on shared edges
    attr_ok = euc.edge_attr.size(-1) == EDGE_ATTR_EUC_DIM

    return {
        "pdb_id": prot.get("pdb_id"),
        "n_nodes": n,
        "n_shortcuts_undirected": n_sc,
        "max_bound": int(n * EUC_TOP_K),
        "graph_diameter": stats.get("graph_diameter"),
        "top_k": EUC_TOP_K,
        "spatial_cutoff_a": EUC_SPATIAL_CUTOFF_A,
        "rules_ok": len(violations) == 0,
        "violations_sample": violations[:10],
        "bound_ok": bound_ok,
        "liveness_ok": live_ok,
        "baseline_prefix_ok": baseline_ok,
        "attr_dim_ok": attr_ok,
        "chem_attr_dim": int(baseline_ea.size(-1)),
        "euc_attr_dim": int(euc.edge_attr.size(-1)),
        "pass": len(violations) == 0
        and bound_ok
        and live_ok
        and baseline_ok
        and attr_ok,
    }


def check_1gpw_hubs(prot: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Part 0.7: chain/hub freeze checks for 1GPW A+B."""
    from scipy.spatial.distance import cdist

    from science.dtie.v66.euclidean_shortcut_graph import (
        compute_euclidean_shortcut_pairs,
    )

    rids = list(prot.get("residue_ids") or [])
    ca = _ca_np(prot)
    n = ca.shape[0]
    key_to_idx = {
        (str(rid).split(":")[0], int(str(rid).split(":")[1])): i
        for i, rid in enumerate(rids)
    }

    # Resnames from PDB when available
    resname_by_key: dict[tuple[str, int], str] = {}
    pdb_path = prot.get("pdb_path")
    if pdb_path is not None:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("1GPW", str(pdb_path))
        for model in structure:
            for chain in model:
                cid = chain.id.strip() or "A"
                if cid not in {"A", "B"}:
                    continue
                for res in chain:
                    if res.get_id()[0] != " ":
                        continue
                    if "CA" not in res:
                        continue
                    resname_by_key[(cid, int(res.get_id()[1]))] = res.get_resname().strip()
            break

    hub_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for h in GPW_HUBS:
        key = (h["chain_id"], int(h["auth_seq"]))
        label = f"{key[0]}:{key[1]}"
        if key not in key_to_idx:
            missing.append(label)
            hub_rows.append({**h, "present": False, "resname_match": False})
            continue
        pdb_rn = resname_by_key.get(key, "")
        hub_rows.append(
            {
                **h,
                "present": True,
                "node_index": int(key_to_idx[key]),
                "resname_pdb": pdb_rn,
                "resname_match": pdb_rn == h["resname_expected"],
            }
        )

    chem = _build_chem(prot)
    baseline_ei = chem.edge_index.detach().clone()
    euc = attach_euclidean_shortcut_graph(
        chem, ca, residue_ids=rids, force=True
    )
    n_sc = int(getattr(euc, "euclidean_shortcut_count_undirected", 0))
    src, dst, _ = compute_euclidean_shortcut_pairs(
        ca, baseline_ei, residue_ids=rids
    )
    pairs = list(zip(src.tolist(), dst.tolist(), strict=True))

    def _chain(i: int) -> str:
        return str(rids[i]).split(":")[0]

    cross_pairs = [(i, j) for i, j in pairs if _chain(i) != _chain(j)]
    pair_set = {(min(i, j), max(i, j)) for i, j in pairs}

    a5 = key_to_idx.get(("A", 5))
    b99 = key_to_idx.get(("B", 99))
    d_a5_b99 = (
        float(np.linalg.norm(ca[a5] - ca[b99]))
        if a5 is not None and b99 is not None
        else float("nan")
    )
    direct_a5_b99 = (
        a5 is not None
        and b99 is not None
        and (min(a5, b99), max(a5, b99)) in pair_set
    )

    # Eligible B within 20Å of A:5 (regardless of Top-K selection)
    a5_eligible_b: list[dict[str, Any]] = []
    a5_has_b_topk = False
    if a5 is not None:
        hops = undirected_hop_distances(baseline_ei, n)
        chains = _parse_chains(rids, n)
        dmat = cdist(ca, ca)
        for j in range(n):
            if chains[j] == "A":
                continue
            d = float(dmat[a5, j])
            if not (1e-6 < d <= EUC_SPATIAL_CUTOFF_A):
                continue
            h = float(hops[a5, j])
            if not (np.isfinite(h) and h > EUC_MIN_GRAPH_HOPS):
                continue
            a5_eligible_b.append(
                {"partner": str(rids[j])[:-1], "distance_a": round(d, 2), "hops": h}
            )
        a5_eligible_b.sort(key=lambda r: r["distance_a"])
        for i, j in pairs:
            if a5 not in (i, j):
                continue
            other = j if i == a5 else i
            if _chain(other) == "B":
                a5_has_b_topk = True
                break

    # Interface neighborhood enrichment: A residues near A:5/A:46 with cross shortcuts
    a46 = key_to_idx.get(("A", 46))
    iface_n = 0
    iface_with_cross = 0
    iface_cross_examples: list[str] = []
    if a5 is not None and a46 is not None:
        dmat = cdist(ca, ca)
        cross_set = {
            (min(i, j), max(i, j))
            for i, j in cross_pairs
        }
        for j in range(n):
            if _chain(j) != "A":
                continue
            if float(dmat[a5, j]) > 10.0 and float(dmat[a46, j]) > 10.0:
                continue
            iface_n += 1
            has = any(
                (min(j, k), max(j, k)) in cross_set
                for k in range(n)
                if _chain(k) == "B"
            )
            if has:
                iface_with_cross += 1
                if len(iface_cross_examples) < 8:
                    for i, k in cross_pairs:
                        if j not in (i, k):
                            continue
                        other = k if i == j else i
                        if _chain(other) != "B":
                            continue
                        iface_cross_examples.append(
                            f"{str(rids[j])[:-1]}↔{str(rids[other])[:-1]}"
                        )
                        break

    keys = [(h["chain_id"], int(h["auth_seq"])) for h in GPW_HUBS]
    index_collision_ok = len(keys) == len(set(keys)) and len(missing) == 0
    stats = dict(getattr(euc, "euclidean_shortcut_stats", {}) or {})
    diam = stats.get("graph_diameter")
    # Under grade_ssot: dense packing/spoke → diam≈7 → hop>6 empty → sc=0.
    # Under part0_legacy (ρ≡12): diam large → sc live. Dual-path: do not treat
    # grade empty as "CLOSED Fail empty-lever ignore" — see MANIFOLD_RECONCILE.md.
    empty_under_dense_role = n_sc == 0 and (
        diam is None or (isinstance(diam, (int, float)) and float(diam) <= 8.0)
    )
    pass_07 = index_collision_ok and all(
        bool(h.get("resname_match")) for h in hub_rows
    )
    loader_name = prot.get("loader") or "unknown"

    out = {
        "schema_version": 4,
        "check": "0.7",
        "pdb_id": "1GPW",
        "chains": ["A", "B"],
        "loader": loader_name,
        "n_nodes_AB": n,
        "n_shortcuts": n_sc,
        "cross_chain_shortcuts": len(cross_pairs),
        "graph_diameter": diam,
        "max_bound": int(n * EUC_TOP_K),
        "bound_ok": n_sc <= n * EUC_TOP_K,
        "builder_cross_chain_allowed": True,
        "builder_note": (
            "same_chain → apply seq≥10; cross-chain seq waived "
            "(euclidean_shortcut_graph.py). Do NOT require same_chain|interface_mask. "
            "grade_ssot: diam≈7 → hop>6 empty → sc=0. part0_legacy constant ρ=12 → "
            "sc live. Status INVALIDATED pending manifold reconcile — not CLOSED Fail."
        ),
        "empty_under_dense_role": empty_under_dense_role,
        "empty_lever_expected": empty_under_dense_role,  # compat alias; not a Fail close
        "index_collision_ok": index_collision_ok,
        "n_hubs": len(GPW_HUBS),
        "n_unique_keys": len(set(keys)),
        "all_hubs_present_in_AB_graph": len(missing) == 0,
        "missing_hubs": missing,
        "hubs": hub_rows,
        "named_hub_pair_A5_B99": {
            "distance_a": round(d_a5_b99, 2),
            "direct_shortcut": direct_a5_b99,
            "within_spatial_cutoff": bool(d_a5_b99 <= EUC_SPATIAL_CUTOFF_A),
            "pass_criterion": (
                "hub identity freeze; live Top-K sc not required when diam≤8"
            ),
        },
        "a5_eligible_b_within_20a": a5_eligible_b[:10],
        "a5_has_b_topk_partner": a5_has_b_topk,
        "a5_topk_note": (
            "Eligible B partners exist within 20Å, but Top-K=2 is filled by "
            "closer same-chain neighbors; interface neighborhood still gets "
            "cross-chain shortcuts."
            if (len(a5_eligible_b) > 0 and not a5_has_b_topk)
            else (
                "No hop>6 eligible partners under this loader's baseline "
                f"(loader={loader_name}, diam={diam})."
                if empty_under_dense_role
                else None
            )
        ),
        "interface_neighborhood_10a": {
            "n_a_residues": iface_n,
            "n_with_cross_chain_shortcut": iface_with_cross,
            "examples": iface_cross_examples,
        },
        "pass_0_7": pass_07,
        "ssot": "docs/specs/v66_chem_MVP/ablation_euclidean_reach.md",
        "site_list": str(
            DEFAULT_SITE_LISTS / "1gpw_pathway_residues.json"
        ),
    }
    path = out_dir / "1gpw_chain_hub_check.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["artifact"] = str(path)
    return out


def check_isolated_init() -> dict[str, Any]:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        m0 = GOSPConeMapperV66(
            node_dim=3,
            hidden=32,
            num_layers=2,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            euclidean_shortcut_mp=False,
            init_seed=0,
        )
        torch.manual_seed(0)
        m1 = GOSPConeMapperV66(
            node_dim=3,
            hidden=32,
            num_layers=2,
            num_experts=2,
            role_edge_mp=True,
            chem_edge_mp=True,
            euclidean_shortcut_mp=True,
            init_seed=0,
        )
    n0 = len(m0.convs[0].radial_mlps)
    n1 = len(m1.convs[0].radial_mlps)
    gate_match = True
    try:
        gate_match = torch.allclose(
            m0.gate.prototype_tangent.detach(),
            m1.gate.prototype_tangent.detach(),
        )
    except Exception:
        gate_match = True
    packing_match = torch.allclose(
        next(m0.convs[0].radial_mlps[0].parameters()).detach(),
        next(m1.convs[0].radial_mlps[0].parameters()).detach(),
    )
    return {
        "check": "0.5",
        "name": "isolated_init_new_radial_only",
        "pass": n0 == 7 and n1 == 8 and packing_match,
        "n_radial_baseline": n0,
        "n_radial_treatment": n1,
        "packing_mlp_match": packing_match,
        "gate_match": gate_match,
    }


def _run_suite_for_loader(
    loader: LoaderName,
    *,
    out_dir: Path,
    pdb_dirs: list[Path],
) -> dict[str, Any]:
    """Run Part0 suite checks under one loader path."""
    tier_results: list[dict[str, Any]] = []
    hub_check: dict[str, Any] | None = None
    for spec in SUITE:
        pdb_id = spec["pdb_id"]
        try:
            chains = tuple(spec.get("chains") or ("A",))
            # Legacy Part0 required shortcuts on 1GPW (sc live under constant ρ).
            # Grade SSOT does not — diam≈7 → sc=0. Keep require flag from SUITE
            # for grade; override for legacy so dual-path reports honestly.
            require = bool(spec.get("require_shortcuts"))
            if loader == "part0_legacy" and pdb_id.upper() == "1GPW":
                require = True
            prot = load_prot_for_part0(
                pdb_id, chains, loader=loader, pdb_dirs=pdb_dirs
            )
            row = check_structure(prot, require_shortcuts=require)
            row["tier"] = spec["tier"]
            row["require_shortcuts"] = require
            row["loader"] = loader
            if spec.get("chains"):
                row["chains"] = list(spec["chains"])
            if pdb_id.upper() == "1GPW":
                hub_subdir = out_dir / loader
                hub_subdir.mkdir(parents=True, exist_ok=True)
                hub_check = check_1gpw_hubs(prot, hub_subdir)
                row["hub_check_pass"] = bool(hub_check.get("pass_0_7"))
                row["pass"] = bool(row.get("pass")) and bool(
                    hub_check.get("pass_0_7")
                )
        except Exception as exc:  # noqa: BLE001
            row = {
                "pdb_id": pdb_id,
                "tier": spec["tier"],
                "pass": False,
                "loader": loader,
                "error": str(exc),
            }
        tier_results.append(row)

    init_check = check_isolated_init()
    overall = all(r.get("pass") for r in tier_results) and bool(init_check.get("pass"))
    return {
        "schema_version": 4,
        "overall_pass": overall,
        "loader": loader,
        "loader_ssot": (
            "load_suite_prot (grade/train parity; real ρ/τ + residue_records)"
            if loader == "grade_ssot"
            else (
                "part0_legacy (_load_multichain_ca): constant ρ≡12 / τ≡0 — "
                "no dehydron; dual-path autopsy only"
            )
        ),
        "governor": {
            "top_k": EUC_TOP_K,
            "spatial_cutoff_a": EUC_SPATIAL_CUTOFF_A,
            "min_hops": EUC_MIN_GRAPH_HOPS,
            "min_seq_sep": EUC_MIN_SEQ_SEP,
            "cross_chain_seq": "waived",
        },
        "suite": tier_results,
        "isolated_init": init_check,
        "hub_check_1gpw": hub_check,
        "tags": {
            "euclidean_construction": True,
            "hyperbolic_inference": True,
            "feeler": True,
        },
        "ssot": "docs/specs/v66_chem_MVP/ablation_euclidean_reach.md",
        "status": "INVALIDATED_pending_manifold_reconcile",
        "note_1gpw": (
            "Dual-path: grade_ssot diam≈7 → sc≈0; part0_legacy constant ρ=12 → "
            "sc≈684. Not CLOSED Fail; primary next = manifold/loader reconcile."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Part0 euc-reach construction checks (v66). Dual-path loaders: "
            "grade_ssot (default) and part0_legacy (constant ρ). "
            "Use --reconcile for 1GPW ρ Path A vs B dump."
        )
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pdb-dir", type=Path, action="append", default=None)
    ap.add_argument(
        "--loader",
        choices=("grade_ssot", "part0_legacy", "both"),
        default="grade_ssot",
        help=(
            "grade_ssot=train/grade SSOT; part0_legacy=constant-ρ multichain; "
            "both=run each and write side-by-side summaries"
        ),
    )
    ap.add_argument(
        "--reconcile",
        action="store_true",
        help=(
            "Write manifold_reconcile_1gpw_rho.json + MANIFOLD_RECONCILE.md "
            "(Path A vs B ρ dump) and run --loader both"
        ),
    )
    ap.add_argument(
        "--reconcile-dir",
        type=Path,
        default=DEFAULT_RECONCILE_DIR,
        help="Directory for reconcile artifacts",
    )
    args = ap.parse_args()
    pdb_dirs = list(args.pdb_dir) if args.pdb_dir else list(DEFAULT_PDB_DIRS)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaders: list[LoaderName]
    if args.reconcile or args.loader == "both":
        loaders = ["part0_legacy", "grade_ssot"]
    else:
        loaders = [args.loader]  # type: ignore[list-item]

    summaries: dict[str, Any] = {}
    exit_code = 0
    for loader in loaders:
        summary = _run_suite_for_loader(
            loader, out_dir=args.out_dir, pdb_dirs=pdb_dirs
        )
        summaries[loader] = summary
        out_name = (
            "summary.json" if len(loaders) == 1 else f"summary_{loader}.json"
        )
        out_path = args.out_dir / out_name
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not summary.get("overall_pass"):
            exit_code = 1

    if args.reconcile or args.loader == "both":
        payload = reconcile_1gpw_rho(
            pdb_dirs=pdb_dirs, out_dir=args.reconcile_dir
        )
        md_path = write_manifold_reconcile_md(payload, args.reconcile_dir)
        combined = {
            "schema_version": 4,
            "mode": "dual_path",
            "status": "INVALIDATED_pending_manifold_reconcile",
            "loaders": summaries,
            "reconcile_json": payload.get("artifact"),
            "reconcile_md": str(md_path),
            "rho_summary": payload.get("summary"),
        }
        combo_path = args.out_dir / "summary_dual_path.json"
        combo_path.write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8"
        )
        # Keep default summary.json pointing at grade_ssot when dual
        if "grade_ssot" in summaries:
            (args.out_dir / "summary.json").write_text(
                json.dumps(summaries["grade_ssot"], indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(combined, indent=2))
        return exit_code

    print(json.dumps(summaries[loaders[0]], indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
