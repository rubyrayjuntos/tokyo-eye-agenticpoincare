"""CLI for Poincaré embedding property suite.

Usage:
  python -m experiments.diagnostics.poincare_property_suite --structure-id 4uj1 --from-db
  python -m experiments.diagnostics.poincare_property_suite --structure-id 4uj1 --from-db --export-npz /tmp/p.npz
  python -m experiments.diagnostics.poincare_property_suite --npz /tmp/p.npz
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from science.dtie.common.curvature_registry import (
    CANONICAL_TS002_CURVATURE,
    curvature_match,
    format_curvature_report_redacted,
    probe_curvature_sources,
)
from science.dtie.common.poincare_properties import PropertySuiteResult, run_property_suite


DEFAULT_INFERENCE_CHECKPOINT = "checkpoints/v6/tokyo_eyes_v6.pt"
# Dashboard screenshot pan where left-boundary stripe appears (4uj1).
DEFAULT_DASHBOARD_MOBIUS = (-0.464, 0.018)


async def load_from_inference(
    structure_id: str,
    checkpoint_path: str = DEFAULT_INFERENCE_CHECKPOINT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Live V6 inference — full arrays for invariant 3 (x_routed_hyp + ball_3d)."""
    import torch

    from data.db import DBAdapter, get_connection
    from science.dtie.common.graph_builder import GraphBuilder
    from science.dtie.v6.gnn.runner import V6GNNRunner

    structure_id = structure_id.strip().lower()
    async with get_connection() as conn:
        db = DBAdapter(conn)
        builder = GraphBuilder(db=db)
        protein_graph = await builder.build_graph(structure_id)
        pyg_data = builder.to_pyg(protein_graph)

    runner = V6GNNRunner(checkpoint_path=checkpoint_path, device=device)
    await runner._ensure_model_loaded()
    graph_data = runner._ensure_v6_features(pyg_data)
    if not hasattr(graph_data, "clustering") or graph_data.clustering is None:
        from science.dtie.v5.gnn.model import precompute_clustering

        graph_data = precompute_clustering(graph_data)
    graph_data = graph_data.to(device)

    with torch.no_grad():
        raw = runner._model(graph_data)

    def _flat(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().reshape(-1)

    curvature = float(runner._model.curvature.detach().cpu().item())

    raw_depth = raw["radial_features"].squeeze(-1).detach()
    depth_max = raw_depth.max() + 1e-8
    cone_depth = ((raw_depth / depth_max) * 8.0).cpu().numpy()

    return {
        "structure_id": structure_id,
        "curvature": curvature,
        "source": "inference",
        "disc_2d": raw["hyp_projections_2d"].detach().cpu().numpy(),
        "ball_3d": raw["hyp_projections_3d"].detach().cpu().numpy(),
        "x_routed_hyp": raw["x_routed_hyp"].detach().cpu().numpy(),
        "x_hyp_highd": None,
        "cone_depth": cone_depth,
        "rho": graph_data.x[:, 0].detach().cpu().numpy(),
        "sasa": graph_data.x[:, 3].detach().cpu().numpy(),
        "epistemic": _flat(raw["uncertainty"]["epistemic"]),
    }


def print_convention_probe(result: PropertySuiteResult) -> None:
    p = result.convention
    print("=== Convention probe (run this first) ===")
    print(f"  disc max raw norm     : {p.disc_max_raw_norm:.6f}")
    print(f"  r_ball (suite)        : {p.r_ball:.6f}")
    print(f"  disc max / r_ball     : {p.disc_max_over_r_ball:.6f}")
    print(f"  model clamp ceiling   : {p.model_clamp_ceiling:.6f}")
    print(f"  legacy unit-clamp?    : {p.legacy_unit_clamp_detected}")
    print(f"  → {p.note}\n")


async def load_from_db(structure_id: str, run_id: str | None = None) -> dict[str, Any]:
    from data.db import DBAdapter, get_connection

    structure_id = structure_id.strip().lower()
    params: dict[str, Any] = {"structure_id": structure_id}
    run_filter = ""
    if run_id:
        run_filter = " AND e.run_id = :run_id"
        params["run_id"] = run_id

    async with get_connection() as conn:
        db = DBAdapter(conn)
        if not run_id:
            latest = await db.fetch_one(
                """
                SELECT f.run_id
                FROM fact_gnn_node_embedding f
                JOIN provenance_run p ON p.run_id = f.run_id
                JOIN embedding_space es ON es.space_id = f.space_id
                WHERE f.structure_id = :structure_id
                  AND es.space_type = 'hyperbolic'
                  AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
                  AND COALESCE(p.parameters->>'superseded', 'false') != 'true'
                ORDER BY COALESCE(p.completed_at, p.started_at) DESC, f.computed_at DESC
                LIMIT 1
                """,
                {"structure_id": structure_id},
            )
            if latest and latest.get("run_id"):
                run_id = str(latest["run_id"])
                run_filter = " AND e.run_id = :run_id"
                params["run_id"] = run_id

        rows = await db.fetch_all(
            f"""
            SELECT r.residue_id, r.residue_index,
                   e.hyp_projection_2d, e.hyp_projections, e.embedding_double,
                   e.cone_depth, e.epistemic_uncertainty, e.input_rho, e.input_sasa,
                   es.curvature
            FROM fact_gnn_node_embedding e
            JOIN dim_residue r ON r.residue_id = e.residue_id
            JOIN embedding_space es ON es.space_id = e.space_id
            WHERE e.structure_id = :structure_id
              AND es.space_type = 'hyperbolic'
              {run_filter}
            ORDER BY r.residue_index
            """,
            params,
        )

    if not rows:
        raise RuntimeError(f"No hyperbolic embeddings for {structure_id}")

    disc_pts: list[list[float]] = []
    ball_high: list[list[float]] = []
    cone_depth: list[float] = []
    rho: list[float] = []
    sasa: list[float] = []
    epistemic: list[float] = []

    for row in rows:
        x, y = 0.0, 0.0
        if row.get("hyp_projection_2d"):
            coords = row["hyp_projection_2d"]
            if isinstance(coords, str):
                coords = [float(v) for v in coords.strip("[]").split(",")]
            x, y = float(coords[0]), float(coords[1])
        elif row.get("hyp_projections"):
            proj = row["hyp_projections"]
            if isinstance(proj, list) and len(proj) >= 2:
                x, y = float(proj[0]), float(proj[1])

        emb = row.get("embedding_double")
        if isinstance(emb, str):
            emb = json.loads(emb)
        if emb:
            ball_high.append([float(v) for v in emb])

        disc_pts.append([x, y])
        cone_depth.append(float(row.get("cone_depth") or 0.0))
        rho.append(float(row.get("input_rho") or 0.0))
        sasa.append(float(row.get("input_sasa") or 0.0))
        epistemic.append(float(row.get("epistemic_uncertainty") or 0.0))

    return {
        "structure_id": structure_id,
        "curvature": float(rows[0].get("curvature") or 1.0),
        "source": f"db run_id={run_id}" if run_id else "db",
        "disc_2d": np.asarray(disc_pts, dtype=np.float64),
        "x_hyp_highd": np.asarray(ball_high, dtype=np.float64) if ball_high else None,
        "ball_3d": None,
        "x_routed_hyp": None,
        "cone_depth": np.asarray(cone_depth, dtype=np.float64),
        "rho": np.asarray(rho, dtype=np.float64),
        "sasa": np.asarray(sasa, dtype=np.float64),
        "epistemic": np.asarray(epistemic, dtype=np.float64),
    }


def load_from_npz(path: Path) -> dict[str, Any]:
    z = np.load(path)
    data: dict[str, Any] = {
        "structure_id": path.stem,
        "curvature": float(z["curvature"][0]),
        "source": f"npz:{path}",
        "disc_2d": z["disc_2d"],
        "cone_depth": z["cone_depth"],
        "ball_3d": z["ball_3d"] if "ball_3d" in z else None,
        "x_routed_hyp": z["x_routed_hyp"] if "x_routed_hyp" in z else None,
        "x_hyp_highd": z["x_hyp_highd"] if "x_hyp_highd" in z else None,
        "rho": z["rho"] if "rho" in z else None,
        "sasa": z["sasa"] if "sasa" in z else None,
        "epistemic": z["epistemic"] if "epistemic" in z else None,
    }
    return data


def print_report(result: PropertySuiteResult) -> None:
    print_convention_probe(result)
    print(f"=== Poincaré property suite: {result.structure_id} ===")
    print(f"source={result.source}  r_ball={result.convention.r_ball:.6f}\n")

    for title, rep in [
        ("1) Norm distribution (disc)", result.norm_disc),
        ("1b) Norm distribution (ball 3D)", result.norm_ball3d),
        ("1c) Norm distribution (high-D ball)", result.norm_routed_highd),
    ]:
        if rep is None:
            continue
        print(f"--- {title} [{rep.label}] n={rep.n} ---")
        print(f"  max_raw_norm={rep.max_raw_norm:.6f}  clamp_ceiling={rep.clamp_ceiling:.6f}")
        print(
            f"  ||x||/ceiling: mean={rep.ratio_vs_ceiling_mean:.4f} p90={rep.ratio_vs_ceiling_p90:.4f} "
            f"p99={rep.ratio_vs_ceiling_p99:.4f}"
        )
        print(
            f"  frac>0.9(ceiling)={rep.fraction_above_0_9_of_ceiling:.3f}  "
            f"frac>0.99(ceiling)={rep.fraction_above_0_99_of_ceiling:.3f}"
        )
        print(
            f"  ||x||/r_ball: p50={rep.ratio_vs_r_ball_p50:.4f} p90={rep.ratio_vs_r_ball_p90:.4f} "
            f"p99={rep.ratio_vs_r_ball_p99:.4f}"
        )
        print(f"  frac>0.8(r_ball)={rep.fraction_above_0_8_r_ball:.3f}")
        if rep.clamp_applied_fraction is not None:
            print(f"  clamp_applied_fraction={rep.clamp_applied_fraction:.4f}")

    b = result.boundary_disc
    print(f"\n--- 2) Boundary margin ---")
    print(
        f"  max/ceiling={b.max_ratio_vs_clamp_ceiling:.6f} max/r_ball={b.max_ratio_vs_r_ball:.6f} "
        f"geoopt_violations={b.n_violations_geoopt} near_clamp={b.n_within_epsilon_of_clamp_ceiling} "
        f"PASS={b.pass_strict_interior}"
    )

    if result.ball_disc:
        bd = result.ball_disc
        print(f"\n--- 3) Ball-disc consistency ---")
        if bd.skipped:
            print(f"  SKIPPED: {bd.skip_reason}")
            print(f"  {bd.note}")
        else:
            print(f"  pearson={bd.pearson_distances:.6f} mean|d|={bd.mean_abs_dist_diff:.6e}")

    m = result.mobius
    print(f"\n--- 4) Möbius isometry center={m.mobius_center} ---")
    print(
        f"  geodesic(c): max_drift={m.max_distance_drift_geodesic:.6e} PASS={m.pass_isometry_geodesic}"
    )
    print(
        f"  viewer(c=1): max_drift={m.max_distance_drift_viewer:.6e} PASS={m.pass_isometry_viewer}"
    )
    print(f"  {m.note}")

    v = result.viewer_clip
    print(f"\n--- 6) Viewer clip / left-stripe (pan={m.mobius_center}) ---")
    print(
        f"  clip_fraction={v.viewer_clip_fraction:.4f} max_render_norm={v.max_render_norm:.4f} "
        f"rscale={v.rscale:.4f} diagnosis={v.diagnosis}"
    )
    print(f"  {v.note}")

    p = result.physics
    print(f"\n--- 5) Physics fidelity ---")
    print(f"  cone_depth std={p.cone_depth_std:.6f}")
    if p.skipped_depth_correlation:
        print(f"  {p.note}")
    else:
        print(f"  radial vs depth r={p.radial_vs_cone_depth_r:.4f}  vs rho r={p.radial_vs_rho_r:.4f}")
        if p.radial_vs_sasa_r is not None:
            print(f"  radial vs sasa r={p.radial_vs_sasa_r:.4f}")
    print()


def export_npz(data: dict[str, Any], path: Path) -> None:
    arrays: dict[str, Any] = {
        "disc_2d": data["disc_2d"],
        "cone_depth": data["cone_depth"],
        "curvature": np.array([data["curvature"]]),
    }
    for key in ("ball_3d", "x_routed_hyp", "x_hyp_highd", "rho", "sasa", "epistemic"):
        if data.get(key) is not None:
            arrays[key] = data[key]
    np.savez_compressed(path, **arrays)
    print(f"Exported → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--from-db", action="store_true")
    parser.add_argument(
        "--from-inference",
        action="store_true",
        help="Live V6 inference (x_routed_hyp + ball_3d for invariant 3)",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_INFERENCE_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--npz", default=None, help="Load arrays from export (probe without DB)")
    parser.add_argument("--export-npz", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--mobius-x", type=float, default=DEFAULT_DASHBOARD_MOBIUS[0])
    parser.add_argument("--mobius-y", type=float, default=DEFAULT_DASHBOARD_MOBIUS[1])
    parser.add_argument("--curvature-probe-only", action="store_true")
    args = parser.parse_args()
    mobius_center = (args.mobius_x, args.mobius_y)

    if args.curvature_probe_only:
        if not args.structure_id:
            parser.error("--structure-id required")
        report = asyncio.run(
            probe_curvature_sources(
                args.structure_id,
                run_id=args.run_id,
                checkpoint_path=args.checkpoint,
            )
        )
        print(format_curvature_report_redacted(report))
        return

    if args.npz:
        data = load_from_npz(Path(args.npz))
    elif args.from_inference:
        if not args.structure_id:
            parser.error("--structure-id required with --from-inference")
        data = asyncio.run(
            load_from_inference(args.structure_id, args.checkpoint, args.device)
        )
    elif args.from_db:
        if not args.structure_id:
            parser.error("--structure-id required with --from-db")
        data = asyncio.run(load_from_db(args.structure_id, args.run_id))
    else:
        parser.error("Use --from-db, --from-inference, or --npz")

    c_report = asyncio.run(
        probe_curvature_sources(
            data["structure_id"],
            run_id=args.run_id,
            checkpoint_path=args.checkpoint,
            suite_c=data["curvature"],
        )
    )
    print(format_curvature_report_redacted(c_report))
    print()
    effective_c = data["curvature"]
    if (
        c_report.model_checkpoint_c is not None
        and c_report.db_embedding_space_c is not None
        and not c_report.model_vs_db_match
    ):
        effective_c = c_report.model_checkpoint_c
        data["source"] = data["source"] + " [suite c→model checkpoint; DB stale]"

    result = run_property_suite(
        structure_id=data["structure_id"],
        curvature=effective_c,
        source=data["source"],
        disc_2d=data["disc_2d"],
        cone_depth=data["cone_depth"],
        ball_3d=data.get("ball_3d"),
        x_routed_hyp=data.get("x_routed_hyp"),
        x_hyp_highd=data.get("x_hyp_highd"),
        rho=data.get("rho"),
        sasa=data.get("sasa"),
        epistemic=data.get("epistemic"),
        epsilon=args.epsilon,
        mobius_center=mobius_center,
    )
    print_report(result)
    if args.export_npz:
        export_npz(data, Path(args.export_npz))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
