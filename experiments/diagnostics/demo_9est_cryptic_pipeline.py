"""
demo_9est_cryptic_pipeline.py
==============================

Separate **exploratory demonstration** — NOT part of the frozen inference-only
pre-registration (PREREG_9EST_1FLE_cryptic_interface.md, REFUTE).

Tests the ingest cryptic-pocket pipeline on static 9EST:
  GNN channels → seed DBSCAN → fpocket merge → overlap with 1FLE-derived I_gold

Compares v5-legacy channel thresholds vs v6 lever_a profile to diagnose whether
the binding-site scan needs architectural updates for the new model.

Optional: --smd attempts md_validate on rank-1 site (requires science GPU path).

Usage:
    python -m experiments.diagnostics.demo_9est_cryptic_pipeline
    python -m experiments.diagnostics.demo_9est_cryptic_pipeline --write-json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from uuid import UUID

import numpy as np


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value

DEFAULT_PDB_DIR = Path("/tmp/dtie_pdb_cache")
DEFAULT_LOCK_PATH = Path("data/benchmarks/9est_1fle_interface_lock.json")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"
)
DEFAULT_OUTPUT = Path("data/benchmarks/9est_cryptic_pipeline_demo.json")
STRUCTURE_ID = "9est"
CHAIN = "A"


@dataclass
class SiteOverlap:
    site_rank: int
    site_type: str
    discovery_method: str
    n_residues: int
    n_gold_hit: int
    recall_on_gold: float
    precision_in_site: float
    gold_resnums_hit: list[int]


@dataclass
class ProfileReport:
    profile: str
    thresholds: dict
    n_qualifying: int
    n_sites: int
    best_site: SiteOverlap | None
    top3_gold_recall: float
    warnings: list[str]


def _download_pdb(pdb_id: str, pdb_dir: Path) -> Path:
    pdb_dir.mkdir(parents=True, exist_ok=True)
    path = pdb_dir / f"{pdb_id.upper()}.pdb"
    if path.exists():
        return path
    import urllib.request

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, path)
    return path


def _residue_index(residue_id: str) -> int:
    # canonical: 9est:A:35 or legacy A:35:
    parts = residue_id.split(":")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    raise ValueError(f"cannot parse residue index from {residue_id!r}")


def _gold_resnums_from_lock(lock_path: Path) -> list[int]:
    with open(lock_path, encoding="utf-8") as f:
        lock = json.load(f)
    return list(lock["interface_label"]["gold_9est_resnums"])


def extract_v6_gnn_nodes(
    checkpoint: Path,
    pdb_dir: Path,
    *,
    device: str = "cpu",
) -> tuple[list, dict[str, tuple[float, float, float]]]:
    from agent.tools.cryptic.seed_generator import GNNNodeOutput
    from experiments.diagnostics.embedding_occupancy_audit import (
        _forward_audit,
        load_audit_model,
    )
    from experiments.training.v6._data import load_protein_graph
    from science.dtie.common.keys import make_residue_id

    prot = load_protein_graph("9EST", CHAIN, pdb_dir)
    if prot is None:
        raise RuntimeError("failed to load 9EST graph")

    model, version = load_audit_model(checkpoint, device)
    out = _forward_audit(model, version, prot, device)

    epi = out["uncertainty"]["epistemic"].detach().cpu().numpy().reshape(-1)
    depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)
    xy = out["hyp_projections_2d"].detach().cpu().numpy()
    disc_r = np.linalg.norm(xy, axis=1)
    ca = prot["ca_coords"].detach().cpu().numpy()

    nodes: list[GNNNodeOutput] = []
    ca_coords: dict[str, tuple[float, float, float]] = {}
    for i, rid in enumerate(prot["residue_ids"]):
        resnum = int(str(rid).split(":")[1])
        canonical = make_residue_id(STRUCTURE_ID, CHAIN, resnum)
        nodes.append(
            GNNNodeOutput(
                residue_id=canonical,
                epistemic_uncertainty=float(epi[i]),
                cone_depth=float(depth[i]),
                disc_r=float(disc_r[i]),
            )
        )
        ca_coords[canonical] = (float(ca[i, 0]), float(ca[i, 1]), float(ca[i, 2]))

    return nodes, ca_coords


def run_fpocket_on_pdb(pdb_path: Path, structure_id: str) -> list:
    from agent.tools.cryptic.pocket_detector import GeometryPocket, _run_fpocket_subprocess

    raw = _run_fpocket_subprocess(pdb_path, structure_id)
    if raw["status"] != "success":
        return []

    pockets: list[GeometryPocket] = []
    for p in raw["pockets"]:
        centroid = p.get("centroid_xyz")
        if centroid is None:
            continue
        pockets.append(
            GeometryPocket(
                pocket_index=int(p["id"]),
                centroid_xyz=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                volume_angstrom3=float(p.get("volume", 0.0)),
                druggability_score=float(p.get("druggability", 0.0)),
                residue_ids=p.get("residue_ids", []),
            )
        )
    return pockets


def site_overlap_report(
    candidates: list,
    gold_resnums: set[int],
    *,
    top_k: int = 5,
) -> list[SiteOverlap]:
    reports: list[SiteOverlap] = []
    for cand in candidates[:top_k]:
        site_resnums = {_residue_index(r) for r in cand.residue_ids}
        hits = sorted(site_resnums & gold_resnums)
        n_site = len(site_resnums)
        n_hit = len(hits)
        reports.append(
            SiteOverlap(
                site_rank=cand.site_rank,
                site_type=cand.site_type,
                discovery_method=cand.discovery_method,
                n_residues=n_site,
                n_gold_hit=n_hit,
                recall_on_gold=n_hit / max(1, len(gold_resnums)),
                precision_in_site=n_hit / max(1, n_site),
                gold_resnums_hit=hits,
            )
        )
    return reports


def evaluate_profile(
    *,
    structure_id: str,
    gnn_nodes: list,
    ca_coords: dict,
    geometry_pockets: list,
    gold_resnums: set[int],
    profile,
) -> ProfileReport:
    from agent.tools.cryptic.gnn_channel_profile import (
        filter_qualifying_residues_profile,
        resolve_profile_thresholds,
    )
    from agent.tools.cryptic.offline_scan import run_offline_binding_scan

    epi_thr, shell_thr = resolve_profile_thresholds(gnn_nodes, profile)
    qualifying = filter_qualifying_residues_profile(gnn_nodes, profile)

    scan = run_offline_binding_scan(
        structure_id=structure_id,
        gnn_nodes=gnn_nodes,
        ca_coords=ca_coords,
        geometry_pockets=geometry_pockets,
        channel_profile=profile,
    )

    overlaps = site_overlap_report(scan.candidates, gold_resnums)
    best = overlaps[0] if overlaps else None
    top3_union: set[int] = set()
    for cand in scan.candidates[:3]:
        top3_union |= {_residue_index(r) for r in cand.residue_ids}
    top3_recall = len(top3_union & gold_resnums) / max(1, len(gold_resnums))

    return ProfileReport(
        profile=profile.name,
        thresholds={
            "epistemic": epi_thr,
            "shell": shell_thr,
            "shell_field": profile.shell_field,
        },
        n_qualifying=len(qualifying),
        n_sites=len(scan.candidates),
        best_site=best,
        top3_gold_recall=top3_recall,
        warnings=list(scan.warnings),
    )


def run_demo(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    pdb_dir: Path = DEFAULT_PDB_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    device: str = "cpu",
) -> dict:
    from agent.tools.cryptic.gnn_channel_profile import V5_LEGACY, V6_LEVER_A

    gold_list = _gold_resnums_from_lock(lock_path)
    gold_set = set(gold_list)

    gnn_nodes, ca_coords = extract_v6_gnn_nodes(checkpoint, pdb_dir, device=device)

    pdb_path = _download_pdb("9EST", pdb_dir)
    with tempfile.TemporaryDirectory(prefix="fpocket_demo_") as tmp:
        # fpocket expects a clean PDB path; copy chain file if needed
        from experiments.training.v6._data import _extract_chain, _chain_cache_dir

        chain_path = _extract_chain(pdb_path, CHAIN, _chain_cache_dir(pdb_dir))
        geometry_pockets = run_fpocket_on_pdb(chain_path, STRUCTURE_ID)

    v5_report = evaluate_profile(
        structure_id=STRUCTURE_ID,
        gnn_nodes=gnn_nodes,
        ca_coords=ca_coords,
        geometry_pockets=geometry_pockets,
        gold_resnums=gold_set,
        profile=V5_LEGACY,
    )
    v6_report = evaluate_profile(
        structure_id=STRUCTURE_ID,
        gnn_nodes=gnn_nodes,
        ca_coords=ca_coords,
        geometry_pockets=geometry_pockets,
        gold_resnums=gold_set,
        profile=V6_LEVER_A,
    )

    geometry_only_hits: list[int] = []
    if geometry_pockets:
        best_geo = max(geometry_pockets, key=lambda p: p.druggability_score)
        geo_nums = {_residue_index(r) for r in best_geo.residue_ids}
        geometry_only_hits = sorted(geo_nums & gold_set)

    return {
        "demo": "9est_cryptic_pipeline",
        "generated_utc": datetime.now(UTC).isoformat(),
        "note": (
            "Exploratory pipeline demo — separate from inference-only prereg (REFUTE). "
            "Does not amend frozen S1/S2 score definition."
        ),
        "structure_id": STRUCTURE_ID,
        "checkpoint": str(checkpoint),
        "n_gold": len(gold_set),
        "n_fpocket_pockets": len(geometry_pockets),
        "geometry_best_druggability_gold_hits": geometry_only_hits,
        "profiles": {
            "v5_legacy": _profile_to_dict(v5_report),
            "v6_lever_a": _profile_to_dict(v6_report),
        },
        "diagnosis": _diagnosis(v5_report, v6_report),
    }


def _profile_to_dict(report: ProfileReport) -> dict:
    d = asdict(report)
    return d


def _diagnosis(v5: ProfileReport, v6: ProfileReport) -> dict:
    return {
        "v5_legacy_usable_on_v6_checkpoint": v5.n_qualifying > 0,
        "v6_profile_produces_sites": v6.n_sites > 0,
        "recommended_ingest_profile": "v6_lever_a" if v6.n_sites > 0 else "needs_calibration",
        "summary": (
            f"v5_legacy: {v5.n_qualifying} qualifying residues, {v5.n_sites} sites; "
            f"v6_lever_a: {v6.n_qualifying} qualifying, {v6.n_sites} sites; "
            f"top-site gold recall v6={v6.best_site.recall_on_gold if v6.best_site else 0:.2f}"
        ),
    }


def _print_report(payload: dict) -> None:
    diag = payload["diagnosis"]
    print("# 9EST cryptic pipeline demo (exploratory — not inference prereg)")
    print()
    print(f"|I_gold| = {payload['n_gold']}  |  fpocket pockets = {payload['n_fpocket_pockets']}")
    print()
    print("## Channel profile comparison (lever_a checkpoint on 9EST)")
    print()
    print("| Profile | Qualifying | Sites | Top-site gold recall | Top3 union recall |")
    print("|---|---|---|---|---|")
    for key in ("v5_legacy", "v6_lever_a"):
        p = payload["profiles"][key]
        best = p.get("best_site") or {}
        print(
            f"| {key} | {p['n_qualifying']} | {p['n_sites']} | "
            f"{best.get('recall_on_gold', 0):.3f} | {p['top3_gold_recall']:.3f} |"
        )
    print()
    print("## Diagnosis")
    print(diag["summary"])
    print()
    print(f"Recommended ingest profile: **{diag['recommended_ingest_profile']}**")
    if payload["geometry_best_druggability_gold_hits"]:
        print(
            f"Best fpocket pocket gold hits: {payload['geometry_best_druggability_gold_hits']}"
        )


def asyncio_run(coro):
    return asyncio.run(coro)


async def evaluate_persisted_sites(
    structure_id: str,
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> dict:
    """Query fact_cryptic_site after ingest pipeline and compare to I_gold."""
    from agent.tools.cryptic.tool import query_binding_sites
    from data.db import DBAdapter, get_standalone_connection

    gold_list = _gold_resnums_from_lock(lock_path)
    gold_set = set(gold_list)
    sid = structure_id.strip().lower()

    # One-shot CLI: bypass shared pool to avoid pool-1 contention with agent/science.
    conn = await get_standalone_connection(
        application_name="demo-9est-cryptic-pipeline",
    )
    try:
        db = DBAdapter(conn)
        scan_row = await db.fetch_one(
            """
            SELECT scan_id, status, sites_found, model_version, scan_parameters, created_at
            FROM fact_binding_site_scan
            WHERE structure_id = :sid
            ORDER BY created_at DESC
            LIMIT 1
            """,
            {"sid": sid},
        )
        tool_result = await query_binding_sites(sid, db=db)
    finally:
        await conn.close()

    sites = tool_result.data.get("sites", []) if tool_result.data else []
    site_reports: list[dict] = []
    top3_union: set[int] = set()

    for row in sites[:5]:
        res_ids = row.get("residue_ids") or []
        if isinstance(res_ids, str):
            res_ids = json.loads(res_ids)
        site_nums = {_residue_index(r) for r in res_ids}
        hits = sorted(site_nums & gold_set)
        if row.get("site_rank", 99) <= 3:
            top3_union |= site_nums
        site_reports.append(
            {
                "site_rank": row.get("site_rank"),
                "site_type": row.get("site_type"),
                "discovery_method": row.get("discovery_method"),
                "druggability_score": row.get("druggability_score"),
                "n_residues": len(site_nums),
                "n_gold_hit": len(hits),
                "recall_on_gold": len(hits) / max(1, len(gold_set)),
                "gold_resnums_hit": hits,
                "fpocket_druggability": row.get("fpocket_druggability"),
            }
        )

    return {
        "demo": "9est_cryptic_pipeline_persisted",
        "generated_utc": datetime.now(UTC).isoformat(),
        "structure_id": structure_id.lower(),
        "n_gold": len(gold_set),
        "scan_metadata": _json_safe(dict(scan_row)) if scan_row else None,
        "n_sites": len(sites),
        "top3_gold_recall": len(top3_union & gold_set) / max(1, len(gold_set)),
        "sites": site_reports,
        "fpocket_used": any(
            (s.get("discovery_method") in ("geometry", "hybrid"))
            or s.get("fpocket_druggability")
            for s in site_reports
        ),
        "tool_message": tool_result.message,
    }


def _print_db_report(payload: dict) -> None:
    print("# 9EST ingest pipeline — persisted cryptic sites")
    print()
    scan = payload.get("scan_metadata") or {}
    print(
        f"structure={payload['structure_id']}  sites={payload['n_sites']}  "
        f"scan_status={scan.get('status')}  model={scan.get('model_version')}"
    )
    params = scan.get("scan_parameters")
    if isinstance(params, str):
        params = json.loads(params)
    if isinstance(params, dict):
        print(f"channel_profile={params.get('channel_profile')}  fpocket_in_merge={payload['fpocket_used']}")
    print(f"|I_gold|={payload['n_gold']}  top3_union_recall={payload['top3_gold_recall']:.3f}")
    print()
    if not payload["sites"]:
        print(payload.get("tool_message", "No sites in DB."))
        return
    print("| rank | type | method | druggability | gold hit | recall |")
    print("|---|---|---|---|---|---|")
    for s in payload["sites"]:
        print(
            f"| {s['site_rank']} | {s['site_type']} | {s['discovery_method']} | "
            f"{s.get('druggability_score', 0):.3f} | {s['n_gold_hit']} | {s['recall_on_gold']:.3f} |"
        )
    if payload["sites"] and payload["sites"][0].get("gold_resnums_hit"):
        print(f"\nRank-1 gold resnums: {payload['sites'][0]['gold_resnums_hit']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    ap.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write-json", action="store_true")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--from-db",
        type=str,
        default=None,
        metavar="STRUCTURE_ID",
        help="Evaluate persisted fact_cryptic_site after ingest (e.g. rcsb:9est)",
    )
    args = ap.parse_args()

    if args.from_db:
        payload = asyncio_run(evaluate_persisted_sites(args.from_db, lock_path=args.lock_path))
        _print_db_report(payload)
        if args.write_json:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\nWrote → {args.output}")
        return 0

    payload = run_demo(
        checkpoint=args.checkpoint,
        pdb_dir=args.pdb_dir,
        lock_path=args.lock_path,
        device=args.device,
    )
    _print_report(payload)
    if args.write_json:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
