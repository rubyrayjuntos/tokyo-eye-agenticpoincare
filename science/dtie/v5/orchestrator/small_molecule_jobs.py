"""Specialized DTIE-native small-molecule jobs for agent-driven execution.

These jobs do not pretend to be external engines like fpocket, AutoDock, or
LigandScout. Instead, they expose the real DTIE Phase 5/6 surrogate machinery
as first-class, parameterized science-container jobs that the agent can call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any

import numpy as np
import psycopg

from data.db import DBAdapter
from science.dtie.common.interfaces import PhaseResult
from science.dtie.v5.orchestrator.pipeline import DTIEOrchestrator, PipelineConfig
from science.dtie.v5.phases.phase35_topological_lift import run_phase35_topological_lift
from science.dtie.v5.phases.phase4_resistance import run_phase4_resistance
from science.dtie.v5.phases.phase5_pharmacophore import run_phase5_pharmacophore
from science.dtie.v5.phases.phase6_drug_discovery import run_phase6_drug_discovery

AA_FEATURES = {
    "hydrophobic": {"A", "ALA", "V", "VAL", "L", "LEU", "I", "ILE", "M", "MET", "P", "PRO", "F", "PHE", "W", "TRP", "Y", "TYR"},
    "aromatic": {"F", "PHE", "W", "TRP", "Y", "TYR", "H", "HIS"},
    "donor": {"R", "ARG", "K", "LYS", "H", "HIS", "S", "SER", "T", "THR", "Y", "TYR", "N", "ASN", "Q", "GLN", "W", "TRP"},
    "acceptor": {"D", "ASP", "E", "GLU", "N", "ASN", "Q", "GLN", "S", "SER", "T", "THR", "Y", "TYR", "H", "HIS"},
    "positive": {"R", "ARG", "K", "LYS", "H", "HIS"},
    "negative": {"D", "ASP", "E", "GLU"},
    "polar": {"S", "SER", "T", "THR", "N", "ASN", "Q", "GLN", "C", "CYS", "H", "HIS", "Y", "TYR"},
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _classify_residue(residue_name: str | None) -> set[str]:
    name = (residue_name or "").upper()
    return {feature for feature, members in AA_FEATURES.items() if name in members}


async def _compute_dependencies(
    db: Any,
    gnn_result: Any,
    structure_id: str,
    spatial_cutoff: float,
) -> tuple[PhaseResult, PhaseResult]:
    phase35_result = await run_phase35_topological_lift(
        db=db,
        gnn_result=gnn_result,
        structure_id=structure_id,
    )
    phase4_result = await run_phase4_resistance(
        db=db,
        gnn_result=gnn_result,
        structure_id=structure_id,
        phase35_result=phase35_result,
        spatial_cutoff=spatial_cutoff,
    )
    return phase35_result, phase4_result


async def _fetch_residue_records(db: Any, structure_id: str) -> list[dict[str, Any]]:
    return await db.fetch_all(
        """
        SELECT
            r.residue_id,
            r.residue_index,
            r.residue_name,
            c.chain_label,
            a.x,
            a.y,
            a.z
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        JOIN dim_structure s ON s.structure_id = c.structure_id
        LEFT JOIN dim_atom a ON a.residue_id = r.residue_id AND a.atom_name = 'CA'
        WHERE s.structure_id = :structure_id
        ORDER BY c.chain_label, r.residue_index
        """,
        {"structure_id": structure_id},
    )


def _summarize_pockets(phase5_result: PhaseResult) -> list[dict[str, Any]]:
    pockets = []
    for pocket in phase5_result.outputs.get("pharmacophores", []):
        pockets.append(
            {
                "pocket_index": pocket["pocket_index"],
                "druggability_score": round(float(pocket["druggability_score"]), 4),
                "residue_count": int(pocket["residue_count"]),
                "allosteric_coupling": round(float(pocket.get("allosteric_coupling", 0.0)), 4),
                "volume_estimate_A3": round(float(pocket.get("volume_estimate_A3", 0.0)), 2),
                "center_xyz": _json_safe(pocket.get("center_xyz", [0.0, 0.0, 0.0])),
                "residue_indices": pocket.get("residue_indices", []),
            }
        )
    return pockets


def _feature_vector(
    pocket: dict[str, Any],
    feature_map: dict[str, Any],
    candidate: dict[str, Any] | None = None,
) -> list[float]:
    counts = feature_map["feature_counts"]
    return [
        float(pocket.get("druggability_score", 0.0)),
        float(pocket.get("allosteric_coupling", 0.0)),
        float(pocket.get("volume_estimate_A3", 0.0)) / 100.0,
        float(pocket.get("residue_count", 0.0)) / 10.0,
        float(counts["hydrophobic"]),
        float(counts["aromatic"]),
        float(counts["donor"]),
        float(counts["acceptor"]),
        float(counts["positive"]),
        float(counts["negative"]),
        float(candidate.get("combined_druggability", 0.0) if candidate else 0.0),
    ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=float)
    bv = np.array(b, dtype=float)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)


async def _map_features(
    db: Any,
    structure_id: str,
    pockets: list[dict[str, Any]],
    radius_angstrom: float,
    selected_pocket_index: int | None = None,
) -> list[dict[str, Any]]:
    rows = await _fetch_residue_records(db, structure_id)
    radius_angstrom = max(float(radius_angstrom), 2.0)
    results: list[dict[str, Any]] = []
    for pocket in pockets:
        pocket_index = int(pocket["pocket_index"])
        if selected_pocket_index is not None and pocket_index != int(selected_pocket_index):
            continue
        center = np.array(pocket.get("center_xyz", [0.0, 0.0, 0.0]), dtype=float)
        nearby_residues: list[str] = []
        feature_counts = {name: 0 for name in AA_FEATURES}
        for row in rows:
            x, y, z = row.get("x"), row.get("y"), row.get("z")
            if x is None or y is None or z is None:
                continue
            coord = np.array([x, y, z], dtype=float)
            if np.linalg.norm(coord - center) > radius_angstrom:
                continue
            nearby_residues.append(str(row["residue_id"]))
            for feature in _classify_residue(row.get("residue_name")):
                feature_counts[feature] += 1
        dominant_features = [
            name for name, count in sorted(feature_counts.items(), key=lambda item: (-item[1], item[0]))
            if count > 0
        ][:3]
        results.append(
            {
                "pocket_index": pocket_index,
                "radius_angstrom": radius_angstrom,
                "nearby_residue_ids": nearby_residues,
                "feature_counts": feature_counts,
                "dominant_features": dominant_features,
            }
        )
    return results


def _build_docking_surrogates(
    pockets: list[dict[str, Any]],
    feature_maps: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    docking_box_padding: float,
    pose_count: int,
    selected_pocket_index: int | None = None,
) -> list[dict[str, Any]]:
    feature_by_index = {entry["pocket_index"]: entry for entry in feature_maps}
    candidate_by_index = {entry["pocket_index"]: entry for entry in candidates}
    surrogates = []
    for pocket in pockets:
        pocket_index = int(pocket["pocket_index"])
        if selected_pocket_index is not None and pocket_index != int(selected_pocket_index):
            continue
        feature_map = feature_by_index.get(pocket_index)
        candidate = candidate_by_index.get(pocket_index, {})
        if feature_map is None:
            continue
        score = (
            0.35 * float(candidate.get("combined_druggability", pocket.get("druggability_score", 0.0)))
            + 0.25 * float(candidate.get("binding_potential", 0.0))
            + 0.15 * float(candidate.get("accessibility_score", 0.0))
            + 0.15 * (1.0 if candidate.get("is_state_selective") else 0.0)
            + 0.10 * min(len(feature_map["dominant_features"]) / 3.0, 1.0)
        )
        residue_count = max(int(pocket.get("residue_count", 1)), 1)
        box_span = max(10.0, (residue_count * 1.75) + float(docking_box_padding))
        surrogates.append(
            {
                "pocket_index": pocket_index,
                "surrogate_score": round(score, 4),
                "box_center_xyz": _json_safe(pocket.get("center_xyz", [0.0, 0.0, 0.0])),
                "box_size_xyz": [round(box_span, 2)] * 3,
                "interaction_priors": feature_map["dominant_features"],
                "anchor_residue_ids": feature_map["nearby_residue_ids"][: max(pose_count, 1)],
                "state_selective": bool(candidate.get("is_state_selective", False)),
            }
        )
    surrogates.sort(key=lambda entry: -entry["surrogate_score"])
    return surrogates


async def _run_extract_pockets(
    db: Any,
    orchestrator: DTIEOrchestrator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_id = f"smol_{uuid.uuid4().hex[:12]}"
    config = PipelineConfig(structure_id=args.structure, spatial_cutoff=args.spatial_cutoff)
    gnn_result = await orchestrator._run_gnn(config, run_id)
    phase35_result, phase4_result = await _compute_dependencies(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        spatial_cutoff=args.spatial_cutoff,
    )
    phase5_result = await run_phase5_pharmacophore(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        phase4_result=phase4_result,
        phase35_result=phase35_result,
        candidate_percentile=args.candidate_percentile,
        cluster_distance_angstrom=args.cluster_distance_angstrom,
        min_cluster_size=args.min_cluster_size,
        max_pockets=args.max_pockets,
    )
    if args.persist:
        await orchestrator._persist_phase_result(phase5_result, run_id, run_id, config)
    return {
        "success": phase5_result.success,
        "run_id": run_id,
        "structure_id": config.structure_id,
        "phase": phase5_result.phase_name,
        "pocket_count": phase5_result.outputs.get("pharmacophore_count", 0),
        "parameters": phase5_result.outputs.get("parameters", {}),
        "pockets": _summarize_pockets(phase5_result),
    }


async def _run_map_pharmacophore(
    db: Any,
    orchestrator: DTIEOrchestrator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    extract_result = await _run_extract_pockets(db, orchestrator, args)
    feature_maps = await _map_features(
        db=db,
        structure_id=extract_result["structure_id"],
        pockets=extract_result["pockets"],
        radius_angstrom=args.radius_angstrom,
        selected_pocket_index=args.pocket_index,
    )
    return {
        **extract_result,
        "feature_maps": _json_safe(feature_maps),
    }


async def _run_screen_fragments(
    db: Any,
    orchestrator: DTIEOrchestrator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_id = f"smol_{uuid.uuid4().hex[:12]}"
    config = PipelineConfig(structure_id=args.structure, spatial_cutoff=args.spatial_cutoff)
    gnn_result = await orchestrator._run_gnn(config, run_id)
    phase35_result, phase4_result = await _compute_dependencies(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        spatial_cutoff=args.spatial_cutoff,
    )
    phase5_result = await run_phase5_pharmacophore(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        phase4_result=phase4_result,
        phase35_result=phase35_result,
        candidate_percentile=args.candidate_percentile,
        cluster_distance_angstrom=args.cluster_distance_angstrom,
        min_cluster_size=args.min_cluster_size,
        max_pockets=args.max_pockets,
    )
    phase6_result = await run_phase6_drug_discovery(
        db=db,
        gnn_result=gnn_result,
        structure_id=config.structure_id,
        phase5_result=phase5_result,
        fallback_top_pockets=args.fallback_top_pockets,
        max_mean_depth=args.max_mean_depth,
        max_mean_epistemic=args.max_mean_epistemic,
        max_mean_total_uncertainty=args.max_mean_total_uncertainty,
        selectivity_ratio_threshold=args.selectivity_ratio_threshold,
        druggability_weight=args.druggability_weight,
        binding_weight=args.binding_weight,
        accessibility_weight=args.accessibility_weight,
        selectivity_weight=args.selectivity_weight,
        focus_pocket_index=args.pocket_index,
        top_k=args.top_k,
    )
    if args.persist:
        await orchestrator._persist_phase_result(phase5_result, run_id, run_id, config)
        await orchestrator._persist_phase_result(phase6_result, run_id, run_id, config)
    return {
        "success": phase6_result.success,
        "run_id": run_id,
        "structure_id": config.structure_id,
        "phase": phase6_result.phase_name,
        "parameters": phase6_result.outputs.get("parameters", {}),
        "scored_pocket_count": phase6_result.outputs.get("scored_pocket_count", 0),
        "top_candidate": _json_safe(phase6_result.outputs.get("top_candidate")),
        "scored_pockets": _json_safe(phase6_result.outputs.get("scored_pockets", [])),
        "phase5_pockets": _summarize_pockets(phase5_result),
    }


async def _run_docking_surrogate(
    db: Any,
    orchestrator: DTIEOrchestrator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    screen_result = await _run_screen_fragments(db, orchestrator, args)
    feature_maps = await _map_features(
        db=db,
        structure_id=screen_result["structure_id"],
        pockets=screen_result["phase5_pockets"],
        radius_angstrom=args.radius_angstrom,
        selected_pocket_index=args.pocket_index,
    )
    docking_surrogates = _build_docking_surrogates(
        pockets=screen_result["phase5_pockets"],
        feature_maps=feature_maps,
        candidates=screen_result["scored_pockets"],
        docking_box_padding=args.docking_box_padding,
        pose_count=args.pose_count,
        selected_pocket_index=args.pocket_index,
    )
    return {
        **screen_result,
        "feature_maps": _json_safe(feature_maps),
        "docking_surrogates": _json_safe(docking_surrogates),
    }


async def _run_search_pocket_vectors(
    db: Any,
    orchestrator: DTIEOrchestrator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_result = await _run_map_pharmacophore(db, orchestrator, args)
    query_pocket_index = args.pocket_index
    if query_pocket_index is None:
        if not query_result["pockets"]:
            return {
                "success": False,
                "error": "No query pockets were generated",
            }
        query_pocket_index = query_result["pockets"][0]["pocket_index"]
    query_feature = next(
        (entry for entry in query_result["feature_maps"] if entry["pocket_index"] == int(query_pocket_index)),
        None,
    )
    query_pocket = next(
        (entry for entry in query_result["pockets"] if entry["pocket_index"] == int(query_pocket_index)),
        None,
    )
    if query_feature is None or query_pocket is None:
        return {
            "success": False,
            "error": f"Pocket {query_pocket_index} was not found for vector search",
        }

    compare_structures = [s.strip().lower() for s in (args.compare_structures or args.structure).split(",") if s.strip()]
    candidates: list[dict[str, Any]] = []
    for structure_id in compare_structures:
        nested_args = argparse.Namespace(**vars(args))
        nested_args.structure = structure_id
        nested_args.pocket_index = None
        nested_result = await _run_map_pharmacophore(db, orchestrator, nested_args)
        if not nested_result.get("success"):
            candidates.append(
                {
                    "structure_id": structure_id,
                    "error": nested_result.get("error", "comparison_structure_failed"),
                }
            )
            continue
        candidate_by_index = {entry["pocket_index"]: entry for entry in nested_result["feature_maps"]}
        for pocket in nested_result["pockets"]:
            feature_map = candidate_by_index.get(pocket["pocket_index"])
            if feature_map is None:
                continue
            vector = _feature_vector(pocket, feature_map, None)
            similarity = _cosine_similarity(_feature_vector(query_pocket, query_feature, None), vector)
            same_query = structure_id == args.structure.lower() and pocket["pocket_index"] == int(query_pocket_index)
            if same_query:
                continue
            candidates.append(
                {
                    "structure_id": structure_id,
                    "pocket_index": pocket["pocket_index"],
                    "similarity": round(similarity, 4),
                    "druggability_score": pocket["druggability_score"],
                    "dominant_features": feature_map["dominant_features"],
                }
            )
    candidates.sort(key=lambda entry: -entry["similarity"])
    top_k = max(int(args.top_k), 1)
    return {
        "success": True,
        "structure_id": args.structure.lower(),
        "query_pocket_index": int(query_pocket_index),
        "matches": candidates[:top_k],
        "compare_structures": compare_structures,
    }


async def _run_small_molecule_pipeline(
    db: Any,
    orchestrator: DTIEOrchestrator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    docking_result = await _run_docking_surrogate(db, orchestrator, args)
    return {
        "success": docking_result.get("success", False),
        "structure_id": docking_result.get("structure_id"),
        "run_id": docking_result.get("run_id"),
        "pocket_count": len(docking_result.get("phase5_pockets", [])),
        "feature_map_count": len(docking_result.get("feature_maps", [])),
        "scored_pocket_count": docking_result.get("scored_pocket_count", 0),
        "top_candidate": docking_result.get("top_candidate"),
        "top_docking_surrogate": docking_result.get("docking_surrogates", [None])[0],
        "phase5_pockets": docking_result.get("phase5_pockets", []),
        "feature_maps": docking_result.get("feature_maps", []),
        "scored_pockets": docking_result.get("scored_pockets", []),
        "docking_surrogates": docking_result.get("docking_surrogates", []),
    }


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--structure", type=str, required=True, help="Canonical structure ID")
    parser.add_argument("--candidate-percentile", type=float, default=75.0)
    parser.add_argument("--cluster-distance-angstrom", type=float, default=8.0)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--max-pockets", type=int, default=10)
    parser.add_argument("--spatial-cutoff", type=float, default=8.0)
    parser.add_argument("--persist", action="store_true")


async def _cli_main() -> None:
    sys.path.insert(0, "/app")
    os.environ.setdefault(
        "DATABASE_URL",
        "postgresql://tokyoeye:tokyoeye_dev_local@db:5432/tokyoeye_dev",
    )
    parser = argparse.ArgumentParser(description="DTIE small-molecule specialized jobs")
    subparsers = parser.add_subparsers(dest="job", required=True)

    extract = subparsers.add_parser("extract_pockets")
    _add_common_args(extract)

    pharmacophore = subparsers.add_parser("map_pharmacophore_features")
    _add_common_args(pharmacophore)
    pharmacophore.add_argument("--radius-angstrom", type=float, default=6.0)
    pharmacophore.add_argument("--pocket-index", type=int)

    screen = subparsers.add_parser("screen_fragments")
    _add_common_args(screen)
    screen.add_argument("--fallback-top-pockets", type=int, default=5)
    screen.add_argument("--max-mean-depth", type=float, default=3.0)
    screen.add_argument("--max-mean-epistemic", type=float, default=0.8)
    screen.add_argument("--max-mean-total-uncertainty", type=float, default=1.5)
    screen.add_argument("--selectivity-ratio-threshold", type=float, default=1.3)
    screen.add_argument("--druggability-weight", type=float, default=0.3)
    screen.add_argument("--binding-weight", type=float, default=0.3)
    screen.add_argument("--accessibility-weight", type=float, default=0.2)
    screen.add_argument("--selectivity-weight", type=float, default=0.2)
    screen.add_argument("--pocket-index", type=int)
    screen.add_argument("--top-k", type=int, default=5)

    docking = subparsers.add_parser("run_docking_surrogate")
    _add_common_args(docking)
    docking.add_argument("--fallback-top-pockets", type=int, default=5)
    docking.add_argument("--max-mean-depth", type=float, default=3.0)
    docking.add_argument("--max-mean-epistemic", type=float, default=0.8)
    docking.add_argument("--max-mean-total-uncertainty", type=float, default=1.5)
    docking.add_argument("--selectivity-ratio-threshold", type=float, default=1.3)
    docking.add_argument("--druggability-weight", type=float, default=0.3)
    docking.add_argument("--binding-weight", type=float, default=0.3)
    docking.add_argument("--accessibility-weight", type=float, default=0.2)
    docking.add_argument("--selectivity-weight", type=float, default=0.2)
    docking.add_argument("--pocket-index", type=int)
    docking.add_argument("--top-k", type=int, default=5)
    docking.add_argument("--radius-angstrom", type=float, default=6.0)
    docking.add_argument("--docking-box-padding", type=float, default=6.0)
    docking.add_argument("--pose-count", type=int, default=5)

    vector_search = subparsers.add_parser("search_pocket_vectors")
    _add_common_args(vector_search)
    vector_search.add_argument("--radius-angstrom", type=float, default=6.0)
    vector_search.add_argument("--pocket-index", type=int)
    vector_search.add_argument("--compare-structures", type=str)
    vector_search.add_argument("--top-k", type=int, default=10)

    small_pipeline = subparsers.add_parser("run_small_molecule_pipeline")
    _add_common_args(small_pipeline)
    small_pipeline.add_argument("--fallback-top-pockets", type=int, default=5)
    small_pipeline.add_argument("--max-mean-depth", type=float, default=3.0)
    small_pipeline.add_argument("--max-mean-epistemic", type=float, default=0.8)
    small_pipeline.add_argument("--max-mean-total-uncertainty", type=float, default=1.5)
    small_pipeline.add_argument("--selectivity-ratio-threshold", type=float, default=1.3)
    small_pipeline.add_argument("--druggability-weight", type=float, default=0.3)
    small_pipeline.add_argument("--binding-weight", type=float, default=0.3)
    small_pipeline.add_argument("--accessibility-weight", type=float, default=0.2)
    small_pipeline.add_argument("--selectivity-weight", type=float, default=0.2)
    small_pipeline.add_argument("--pocket-index", type=int)
    small_pipeline.add_argument("--top-k", type=int, default=5)
    small_pipeline.add_argument("--radius-angstrom", type=float, default=6.0)
    small_pipeline.add_argument("--docking-box-padding", type=float, default=6.0)
    small_pipeline.add_argument("--pose-count", type=int, default=5)

    args = parser.parse_args()

    async with await psycopg.AsyncConnection.connect(os.environ["DATABASE_URL"]) as conn:
        db = DBAdapter(conn)
        orchestrator = DTIEOrchestrator(db=db)
        job_map = {
            "extract_pockets": _run_extract_pockets,
            "map_pharmacophore_features": _run_map_pharmacophore,
            "screen_fragments": _run_screen_fragments,
            "run_docking_surrogate": _run_docking_surrogate,
            "search_pocket_vectors": _run_search_pocket_vectors,
            "run_small_molecule_pipeline": _run_small_molecule_pipeline,
        }
        payload = await job_map[args.job](db, orchestrator, args)
        await conn.commit()
        print(json.dumps(_json_safe(payload)))


if __name__ == "__main__":
    asyncio.run(_cli_main())
