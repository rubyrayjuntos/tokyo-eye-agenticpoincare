"""Single place for Poincaré curvature constants and cross-source equality checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from science.dtie.common.curvature_loader import (
    CANONICAL_V6_CURVATURE,
    V6_HYP_SPACE_NAME,
    get_curvature,
)

# Deprecated TS-002 training reference — do not use for v6 runtime distance math.
CANONICAL_TS002_CURVATURE = 0.6054342985153198

CURVATURE_MATCH_RTOL = 1e-4
CURVATURE_MATCH_ATOL = 1e-5


def curvature_match(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=CURVATURE_MATCH_RTOL, abs_tol=CURVATURE_MATCH_ATOL)


def curvature_bucket(c: float) -> str:
    """Redacted label for logs (no raw trade-secret curvature)."""
    if curvature_match(c, CANONICAL_V6_CURVATURE):
        return "v6_lever_a_canonical"
    if curvature_match(c, CANONICAL_TS002_CURVATURE):
        return "ts002_deprecated"
    if c < 0.62:
        return "low_c_band"
    if c < 0.72:
        return "mid_c_band"
    return "high_c_band"


@dataclass
class CurvatureSourceReport:
    structure_id: str
    run_id: str | None
    model_checkpoint_c: float | None
    db_embedding_space_c: float | None
    vector_queries_default_c: float
    suite_c_used: float | None
    geoopt_k_neg_c: bool | None
    k_c_crossing_suspect: bool
    all_four_match: bool
    model_vs_db_match: bool
    model_vs_vector_queries_match: bool
    ts002_canonical_match: bool
    note: str


async def probe_curvature_sources(
    structure_id: str,
    *,
    run_id: str | None = None,
    checkpoint_path: str | None = None,
    suite_c: float | None = None,
) -> CurvatureSourceReport:
    """Assert curvature equality across model, DB, vector_queries default, suite."""
    import torch

    from data.db import DBAdapter, get_connection
    from science.dtie.v6.gnn.runner import V6GNNRunner

    from science.contracts.model_registry import get_production_checkpoint_path

    if checkpoint_path is None:
        checkpoint_path = get_production_checkpoint_path()

    structure_id = structure_id.strip().lower()
    model_c: float | None = None
    geoopt_k_ok: bool | None = None
    k_cross: bool = False

    try:
        runner = V6GNNRunner(checkpoint_path=checkpoint_path, device="cpu")
        await runner._ensure_model_loaded()
        m = runner._model
        model_c = float(m.curvature.detach().cpu().item())
        k = float((-m.curvature).detach().cpu().item())
        geoopt_k_ok = curvature_match(k, -model_c)
        # c↔k footgun: using |k| as c would invert ball radius
        k_cross = curvature_match(model_c, abs(k)) and not geoopt_k_ok
    except Exception:
        model_c = None

    db_c: float | None = None
    vq_c: float | None = None
    resolved_run: str | None = run_id
    async with get_connection() as conn:
        db = DBAdapter(conn)
        if not resolved_run:
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
            if latest:
                resolved_run = str(latest["run_id"])

        if resolved_run:
            row = await db.fetch_one(
                """
                SELECT es.curvature
                FROM fact_gnn_node_embedding e
                JOIN embedding_space es ON es.space_id = e.space_id
                WHERE e.structure_id = :structure_id AND e.run_id = :run_id
                LIMIT 1
                """,
                {"structure_id": structure_id, "run_id": resolved_run},
            )
            if row and row.get("curvature") is not None:
                db_c = float(row["curvature"])

        try:
            vq_c = await get_curvature(V6_HYP_SPACE_NAME, db=db)
        except Exception:
            vq_c = None

    used = suite_c if suite_c is not None else (db_c if db_c is not None else model_c)

    sources = [x for x in (model_c, db_c, vq_c, used) if x is not None]
    all_match = len(sources) >= 2 and all(
        curvature_match(sources[0], s) for s in sources[1:]
    )

    model_db = model_c is not None and db_c is not None and curvature_match(model_c, db_c)
    model_vq = model_c is not None and curvature_match(model_c, vq_c)
    ts002 = model_c is not None and curvature_match(model_c, CANONICAL_TS002_CURVATURE)
    v6_pin = model_c is not None and curvature_match(model_c, CANONICAL_V6_CURVATURE)

    if not all_match:
        if db_c is not None and model_c is not None and not model_db:
            note = (
                "SPLIT-BRAIN: embedding_space.curvature stale (ON CONFLICT DO NOTHING). "
                "Suite/dashboard use DB c; model/checkpoint use learned c."
            )
        elif model_c is not None and not v6_pin:
            note = "Learned c diverged from v6 lever_a pin — verify checkpoint and registry."
        else:
            note = "Curvature sources disagree — check run provenance and vector_queries default."
    else:
        note = "All four curvature sources match within tolerance."

    return CurvatureSourceReport(
        structure_id=structure_id,
        run_id=resolved_run,
        model_checkpoint_c=model_c,
        db_embedding_space_c=db_c,
        vector_queries_default_c=vq_c,
        suite_c_used=used,
        geoopt_k_neg_c=geoopt_k_ok,
        k_c_crossing_suspect=k_cross,
        all_four_match=all_match,
        model_vs_db_match=model_db,
        model_vs_vector_queries_match=model_vq,
        ts002_canonical_match=ts002,
        note=note,
    )


def format_curvature_report_redacted(report: CurvatureSourceReport) -> str:
    """Stdout-safe curvature probe (buckets only, no raw c)."""
    lines = [
        "=== Curvature four-source probe (redacted) ===",
        f"  structure     : {report.structure_id}",
        f"  run_id        : {report.run_id or 'n/a'}",
        f"  model_ckpt    : {curvature_bucket(report.model_checkpoint_c) if report.model_checkpoint_c else 'n/a'}",
        f"  db_embed_space: {curvature_bucket(report.db_embedding_space_c) if report.db_embedding_space_c else 'n/a'}",
        f"  vector_queries: {curvature_bucket(report.vector_queries_default_c)}",
        f"  suite_used    : {curvature_bucket(report.suite_c_used) if report.suite_c_used else 'n/a'}",
        f"  geoopt k=-c   : {report.geoopt_k_neg_c}",
        f"  k/c crossing  : {report.k_c_crossing_suspect}",
        f"  all_four_match: {report.all_four_match}",
        f"  model_vs_db   : {report.model_vs_db_match}",
        f"  model_vs_vq   : {report.model_vs_vector_queries_match}",
        f"  ts002_match   : {report.ts002_canonical_match}",
        f"  → {report.note}",
    ]
    return "\n".join(lines)
