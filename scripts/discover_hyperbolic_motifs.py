#!/usr/bin/env python3
"""Hyperbolic motif discovery over hydrated residue embeddings.

Discovers atlas-wide residue motifs by clustering hydrated Poincare-ball
embeddings with K-medoids over an exact geodesic distance matrix, then ranking
clusters by vulnerability enrichment, uncertainty, diversity, and compactness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import numpy as np
import pandas as pd
import psycopg

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.db import DBAdapter, get_connection as pooled_get_connection, open_pool
from science.dtie.common.curvature_loader import get_curvature
from science.dtie.common.hyperbolic_utils import hyperbolic_pairwise_distance

LOGGER = logging.getLogger(__name__)

DEFAULT_SPACE = "space_gospconemapper_v6_hyp128"
DEFAULT_K_RANGE = [8, 10, 12, 16, 20]
DEFAULT_RANDOM_STATE = 42
DEFAULT_BLOCKWISE_THRESHOLD = 7000
DEFAULT_BLOCK_SIZE = 512
EXPECTED_EMBEDDING_DIM = 128


@dataclass(frozen=True)
class KMedoidsSweepResult:
    k: int
    labels: np.ndarray
    medoid_indices: np.ndarray
    cluster_membership: pd.DataFrame
    cluster_summary: pd.DataFrame


def parse_csv_arg(value: str | None) -> list[str]:
    """Parse a comma-separated argument into a normalized string list."""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_k_range(value: str) -> list[int]:
    """Parse a comma-separated list of positive cluster counts."""
    values: list[int] = []
    for item in parse_csv_arg(value):
        try:
            parsed = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid k value {item!r}; expected positive integer") from exc
        if parsed <= 1:
            raise ValueError(f"Invalid k value {item!r}; expected integer > 1")
        values.append(parsed)
    if not values:
        raise ValueError("k-range must include at least one integer > 1")
    return sorted(set(values))


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for motif discovery."""
    parser = argparse.ArgumentParser(
        description=(
            "Discover hyperbolic residue motifs via exact Poincare distances and "
            "K-medoids clustering over governed Tokyo Eye embeddings."
        ),
    )
    parser.add_argument("--space", type=str, default=DEFAULT_SPACE, help="Embedding space id or name")
    parser.add_argument("--dsn", type=str, default=None, help="Optional Postgres DSN override")
    parser.add_argument(
        "--cone-depth-min",
        type=float,
        default=3.0,
        help="Minimum cone depth filter before clustering",
    )
    parser.add_argument(
        "--leak-score-min",
        type=float,
        default=20.0,
        help="Minimum leak score filter before clustering",
    )
    parser.add_argument(
        "--epistemic-min",
        type=float,
        default=None,
        help="Optional minimum epistemic uncertainty filter",
    )
    parser.add_argument(
        "--include-structures",
        type=str,
        default=None,
        help="Optional comma-separated structure ids to include",
    )
    parser.add_argument(
        "--exclude-structures",
        type=str,
        default=None,
        help="Optional comma-separated structure ids to exclude",
    )
    parser.add_argument(
        "--protein-family-filter",
        type=str,
        default=None,
        help="Optional comma-separated protein family values from dim_structure.metadata",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for debugging or memory-bounded runs",
    )
    parser.add_argument(
        "--k-range",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_RANGE),
        help=f"Candidate K sweep (default: {','.join(str(k) for k in DEFAULT_K_RANGE)})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for K-medoids reproducibility",
    )
    parser.add_argument(
        "--top-k-clusters",
        type=int,
        default=5,
        help="Number of top motif clusters to export as medoid seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hyperbolic_motifs",
        help="Artifact output directory",
    )
    parser.add_argument(
        "--write-distance-matrix",
        action="store_true",
        help="Write distance_matrix.npy",
    )
    parser.add_argument(
        "--write-membership-matrix",
        action="store_true",
        help="Write labels.npy and membership_matrix.npy",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Row block size for blockwise distance construction when N is large",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and normalize list-like values."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.k_range = parse_k_range(args.k_range)
    args.include_structures = [value.lower() for value in parse_csv_arg(args.include_structures)]
    args.exclude_structures = [value.lower() for value in parse_csv_arg(args.exclude_structures)]
    args.protein_family_filter = [value.lower() for value in parse_csv_arg(args.protein_family_filter)]
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be a positive integer")
    if args.top_k_clusters <= 0:
        parser.error("--top-k-clusters must be a positive integer")
    if args.block_size <= 0:
        parser.error("--block-size must be a positive integer")
    return args


@asynccontextmanager
async def get_db_adapter(dsn: str | None) -> AsyncIterator[DBAdapter]:
    """Yield a DBAdapter using either the shared pool or a DSN override."""
    if dsn:
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            yield DBAdapter(conn)
        finally:
            await conn.close()
        return

    await open_pool()
    async with pooled_get_connection() as conn:
        yield DBAdapter(conn)


def _build_filter_clause(field_sql: str, values: Sequence[str], param_name: str) -> tuple[str, dict[str, Any]]:
    if not values:
        return "", {}
    return f" AND {field_sql} = ANY(:{param_name})", {param_name: list(values)}


def build_load_query(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Build the governed SQL query used to hydrate clustering rows."""
    params: dict[str, Any] = {
        "space": args.space,
        "cone_depth_min": args.cone_depth_min,
        "leak_score_min": args.leak_score_min,
    }
    filters: list[str] = []

    include_clause, include_params = _build_filter_clause(
        "le.structure_id", args.include_structures, "include_structures"
    )
    if include_clause:
        filters.append(include_clause)
        params.update(include_params)

    exclude_clause, exclude_params = _build_filter_clause(
        "le.structure_id", args.exclude_structures, "exclude_structures"
    )
    if exclude_clause:
        filters.append(f" AND NOT ({exclude_clause[5:]})")
        params.update(exclude_params)

    family_clause, family_params = _build_filter_clause(
        "LOWER(COALESCE(NULLIF(ds.metadata->>'protein_family', ''), NULLIF(ds.metadata->>'family', ''), le.structure_id))",
        args.protein_family_filter,
        "protein_family_filter",
    )
    if family_clause:
        filters.append(family_clause)
        params.update(family_params)

    if args.epistemic_min is not None:
        filters.append(
            " AND COALESCE(ls.epistemic_uncertainty, le.epistemic_uncertainty, 0.0) >= :epistemic_min"
        )
        params["epistemic_min"] = args.epistemic_min

    limit_clause = ""
    if args.max_rows is not None:
        limit_clause = "LIMIT :max_rows"
        params["max_rows"] = args.max_rows

    extra_filters = "".join(filters)
    query = f"""
        WITH requested_space AS (
            SELECT es.space_id, es.name AS space_name, es.curvature
            FROM embedding_space es
            WHERE es.space_id = :space OR es.name = :space
        ),
        latest_embeddings AS (
            SELECT
                e.residue_id,
                e.structure_id,
                e.space_id,
                e.embedding_double,
                e.cone_depth,
                e.epistemic_uncertainty,
                e.computed_at,
                e.run_id,
                ROW_NUMBER() OVER (
                    PARTITION BY e.space_id, e.residue_id
                    ORDER BY e.computed_at DESC, e.run_id DESC
                ) AS rn
            FROM fact_gnn_node_embedding e
            JOIN requested_space rs ON rs.space_id = e.space_id
            WHERE e.embedding_double IS NOT NULL
        ),
        latest_source_leak AS (
            SELECT
                sl.residue_id,
                sl.leak_score,
                sl.cone_depth,
                sl.epistemic_uncertainty,
                sl.computed_at,
                sl.run_id,
                ROW_NUMBER() OVER (
                    PARTITION BY sl.residue_id
                    ORDER BY sl.computed_at DESC, sl.run_id DESC
                ) AS rn
            FROM fact_source_leak sl
        ),
        latest_graph_metrics AS (
            SELECT
                gm.residue_id,
                gm.betweenness,
                gm.degree,
                gm.computed_at,
                gm.run_id,
                ROW_NUMBER() OVER (
                    PARTITION BY gm.residue_id
                    ORDER BY gm.computed_at DESC, gm.run_id DESC
                ) AS rn
            FROM fact_graph_node_metrics gm
        )
        SELECT
            le.residue_id,
            le.structure_id,
            dc.chain_id,
            dc.chain_label,
            dr.residue_index AS residue_number,
            COALESCE(dr.residue_name_3, dr.residue_name) AS residue_type,
            le.embedding_double,
            COALESCE(ls.leak_score, 0.0) AS leak_score,
            COALESCE(ls.cone_depth, le.cone_depth, 0.0) AS cone_depth,
            COALESCE(ls.epistemic_uncertainty, le.epistemic_uncertainty, 0.0) AS epistemic_uncertainty,
            COALESCE(
                NULLIF(ds.metadata->>'gene_symbol', ''),
                NULLIF(ds.metadata->>'protein_id', ''),
                NULLIF(ds.pdb_id, ''),
                le.structure_id
            ) AS gene_symbol,
            COALESCE(
                NULLIF(ds.metadata->>'protein_family', ''),
                NULLIF(ds.metadata->>'family', ''),
                COALESCE(NULLIF(ds.metadata->>'gene_symbol', ''), NULLIF(ds.pdb_id, ''), le.structure_id)
            ) AS protein_family,
            gm.betweenness,
            gm.degree,
            rs.space_name,
            rs.curvature AS stored_curvature
        FROM latest_embeddings le
        JOIN requested_space rs ON rs.space_id = le.space_id
        JOIN dim_residue dr ON dr.residue_id = le.residue_id
        JOIN dim_chain dc ON dc.chain_id = dr.chain_id
        JOIN dim_structure ds ON ds.structure_id = le.structure_id
        LEFT JOIN latest_source_leak ls ON ls.residue_id = le.residue_id AND ls.rn = 1
        LEFT JOIN latest_graph_metrics gm ON gm.residue_id = le.residue_id AND gm.rn = 1
        WHERE le.rn = 1
          AND COALESCE(ls.cone_depth, le.cone_depth, 0.0) >= :cone_depth_min
          AND COALESCE(ls.leak_score, 0.0) >= :leak_score_min
          {extra_filters}
        ORDER BY COALESCE(ls.leak_score, 0.0) DESC, le.residue_id
        {limit_clause}
    """
    return query.strip(), params


async def load_filtered_rows(db: DBAdapter, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and filter residue rows directly from governed Postgres tables."""
    query, params = build_load_query(args)
    LOGGER.info("discover_hyperbolic_motifs SQL:\n%s", query)
    LOGGER.info("discover_hyperbolic_motifs params: %s", params)
    rows = await db.fetch_all(query, params)
    df = pd.DataFrame([dict(row) for row in rows]) if rows else pd.DataFrame()
    metadata = {
        "sql_query": query,
        "sql_params": params,
        "row_count": int(len(df)),
        "structure_count": int(df["structure_id"].nunique()) if not df.empty else 0,
        "space_name": df["space_name"].iloc[0] if not df.empty and "space_name" in df else None,
        "stored_curvature": (
            float(df["stored_curvature"].iloc[0])
            if not df.empty and "stored_curvature" in df and pd.notna(df["stored_curvature"].iloc[0])
            else None
        ),
    }
    return df, metadata


def validate_embeddings(
    df: pd.DataFrame,
    *,
    dim: int = EXPECTED_EMBEDDING_DIM,
    curvature: float,
    stored_curvature: float | None = None,
) -> None:
    """Validate embedding dimensionality, finiteness, and ball bounds."""
    if df.empty:
        raise ValueError("No filtered rows available for motif discovery")
    if df["residue_id"].duplicated().any():
        duplicates = df.loc[df["residue_id"].duplicated(), "residue_id"].tolist()
        raise ValueError(f"Duplicate residue_id rows remain after canonicalization: {duplicates[:5]}")
    if stored_curvature is not None and not np.isclose(stored_curvature, curvature, atol=1e-9):
        raise ValueError(
            f"Requested curvature {curvature} does not match stored embedding-space curvature {stored_curvature}"
        )

    max_sq_norm = (1.0 / curvature) - 1e-10
    for index, value in enumerate(df["embedding_double"].tolist()):
        vector = np.asarray(value, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != dim:
            raise ValueError(
                f"Embedding at row {index} has shape {vector.shape}; expected ({dim},)"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Embedding at row {index} contains non-finite values")
        sq_norm = float(np.dot(vector, vector))
        if sq_norm >= max_sq_norm:
            residue_id = df.iloc[index]["residue_id"]
            raise ValueError(
                f"Embedding for {residue_id} violates Poincare ball bounds for curvature {curvature}"
            )


def embeddings_to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert embedding_double arrays into an (N, D) float64 matrix."""
    vectors = [np.asarray(value, dtype=np.float64) for value in df["embedding_double"].tolist()]
    return np.vstack(vectors)


def _blockwise_poincare_distance_matrix(
    x: np.ndarray,
    c: float,
    block_size: int,
) -> np.ndarray:
    sq_norm = np.sum(x * x, axis=1)
    dist = np.empty((x.shape[0], x.shape[0]), dtype=np.float64)

    for start in range(0, x.shape[0], block_size):
        stop = min(start + block_size, x.shape[0])
        block = x[start:stop]
        sq_diff = np.sum((block[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        denom = (1.0 - c * np.sum(block * block, axis=1))[:, None] * (1.0 - c * sq_norm)[None, :]
        arg = 1.0 + (2.0 * c * sq_diff) / np.clip(denom, 1e-12, None)
        arg = np.clip(arg, 1.0 + 1e-12, None)
        dist[start:stop, :] = np.arccosh(arg) / np.sqrt(c)

    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    return dist


def poincare_distance_matrix(
    x: np.ndarray,
    c: float,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    blockwise_threshold: int = DEFAULT_BLOCKWISE_THRESHOLD,
) -> np.ndarray:
    """Compute an exact pairwise Poincare-ball distance matrix."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {x.shape}")
    if c <= 0:
        raise ValueError("Curvature must be positive")
    if x.shape[0] == 0:
        raise ValueError("Cannot build a distance matrix for zero rows")

    if x.shape[0] <= blockwise_threshold:
        return hyperbolic_pairwise_distance(x, c=c)
    return _blockwise_poincare_distance_matrix(x, c=c, block_size=block_size)


def validate_distance_matrix(distance_matrix: np.ndarray) -> None:
    """Validate symmetry, finiteness, and zero diagonal."""
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    if not np.isfinite(distance_matrix).all():
        raise ValueError("Distance matrix contains non-finite values")
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-8):
        raise ValueError("Distance matrix is not symmetric")
    if not np.allclose(np.diag(distance_matrix), 0.0, atol=1e-10):
        raise ValueError("Distance matrix diagonal must be zero")


def run_kmedoids(distance_matrix: np.ndarray, k: int, random_state: int) -> dict[str, Any]:
    """Run K-medoids over a precomputed exact distance matrix."""
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn-extra is required for discover_hyperbolic_motifs.py. "
            "Install the science dependencies or regenerate requirements-science.txt."
        ) from exc

    model = KMedoids(
        n_clusters=k,
        metric="precomputed",
        method="pam",
        init="k-medoids++",
        random_state=random_state,
    )
    labels = model.fit_predict(distance_matrix)
    medoid_indices = np.asarray(model.medoid_indices_, dtype=np.int64)
    return {
        "labels": labels,
        "medoid_indices": medoid_indices,
        "inertia": float(model.inertia_),
    }


def _zscore(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    sigma = float(numeric.std(ddof=0))
    if sigma == 0.0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=values.index)
    mu = float(numeric.mean())
    return (numeric - mu) / sigma


def score_clusters(
    df: pd.DataFrame,
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    medoid_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-residue membership and per-cluster summary tables."""
    membership = df.copy()
    membership["cluster_id"] = labels.astype(int)
    membership["medoid_residue_id"] = [membership.iloc[idx]["residue_id"] for idx in medoid_indices[labels]]
    membership["distance_to_medoid"] = distance_matrix[np.arange(len(membership)), medoid_indices[labels]]
    membership["is_medoid"] = False
    membership.loc[membership.index[medoid_indices], "is_medoid"] = True
    membership["protein_id"] = membership["gene_symbol"].fillna(membership["structure_id"])
    membership["protein_family"] = membership["protein_family"].fillna(membership["protein_id"])

    summaries: list[dict[str, Any]] = []
    for cluster_id in sorted(set(labels.tolist())):
        cluster_rows = membership[membership["cluster_id"] == cluster_id].copy()
        if cluster_rows.empty:
            continue
        medoid_row = membership.iloc[medoid_indices[cluster_id]]
        top_examples = ";".join(
            cluster_rows.nsmallest(min(5, len(cluster_rows)), "distance_to_medoid")["residue_id"].tolist()
        )
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "medoid_residue_id": medoid_row["residue_id"],
                "structure_id": medoid_row["structure_id"],
                "gene_symbol": medoid_row["gene_symbol"],
                "protein_family": medoid_row["protein_family"],
                "n_members": int(len(cluster_rows)),
                "mean_leak": float(cluster_rows["leak_score"].mean()),
                "mean_epistemic": float(cluster_rows["epistemic_uncertainty"].mean()),
                "mean_cone_depth": float(cluster_rows["cone_depth"].mean()),
                "unique_structures": int(cluster_rows["structure_id"].nunique()),
                "unique_proteins": int(cluster_rows["protein_id"].nunique()),
                "unique_families": int(cluster_rows["protein_family"].nunique()),
                "mean_distance_to_medoid": float(cluster_rows["distance_to_medoid"].mean()),
                "std_distance_to_medoid": float(cluster_rows["distance_to_medoid"].std(ddof=0)),
                "max_distance_to_medoid": float(cluster_rows["distance_to_medoid"].max()),
                "top_member_examples": top_examples,
            }
        )

    summary = pd.DataFrame(summaries)
    if summary.empty:
        raise ValueError("K-medoids produced no non-empty clusters")

    summary["std_distance_to_medoid"] = summary["std_distance_to_medoid"].fillna(0.0)
    summary["cluster_score_family"] = (
        _zscore(summary["mean_leak"])
        + _zscore(summary["mean_epistemic"])
        + _zscore(summary["unique_families"])
        - _zscore(summary["std_distance_to_medoid"])
    )
    summary["cluster_score_protein"] = (
        _zscore(summary["mean_leak"])
        + _zscore(summary["mean_epistemic"])
        + _zscore(summary["unique_proteins"])
        - _zscore(summary["std_distance_to_medoid"])
    )
    summary = summary.sort_values(
        ["cluster_score_family", "cluster_score_protein", "n_members"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return membership, summary


def summarize_k_sweep(cluster_summary_by_k: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Summarize K sweep candidates and derive a selection score."""
    rows: list[dict[str, Any]] = []
    for k, summary in sorted(cluster_summary_by_k.items()):
        if summary.empty:
            continue
        top_n = min(3, len(summary))
        rows.append(
            {
                "k": k,
                "top_cluster_score_family": float(summary["cluster_score_family"].max()),
                "top_cluster_score_protein": float(summary["cluster_score_protein"].max()),
                "mean_top_cluster_score_family": float(summary["cluster_score_family"].nlargest(top_n).mean()),
                "mean_family_diversity": float(summary["unique_families"].mean()),
                "median_mean_distance": float(summary["mean_distance_to_medoid"].median()),
                "median_spread": float(summary["std_distance_to_medoid"].median()),
                "largest_cluster_size": int(summary["n_members"].max()),
            }
        )

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        raise ValueError("No valid K sweep results were produced")

    sweep["selection_score"] = (
        _zscore(sweep["top_cluster_score_family"])
        + _zscore(sweep["mean_top_cluster_score_family"])
        + _zscore(sweep["mean_family_diversity"])
        - _zscore(sweep["median_mean_distance"])
        - _zscore(sweep["median_spread"])
    )
    return sweep.sort_values(
        ["selection_score", "top_cluster_score_family", "k"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def choose_best_k(cluster_summary_by_k: dict[int, pd.DataFrame]) -> int:
    """Choose the best K based on cluster signal, diversity, and compactness."""
    sweep = summarize_k_sweep(cluster_summary_by_k)
    return int(sweep.iloc[0]["k"])


def _resolve_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_ROOT,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_artifacts(
    *,
    output_dir: Path,
    cluster_membership: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    k_sweep_summary: pd.DataFrame,
    selected_k: int,
    args: argparse.Namespace,
    curvature: float,
    load_metadata: dict[str, Any],
    runtime_seconds: float,
    distance_matrix: np.ndarray | None,
) -> dict[str, str]:
    """Write motif discovery artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters_full = cluster_membership[
        [
            "cluster_id",
            "residue_id",
            "structure_id",
            "gene_symbol",
            "protein_family",
            "chain_id",
            "chain_label",
            "residue_number",
            "residue_type",
            "leak_score",
            "cone_depth",
            "epistemic_uncertainty",
            "betweenness",
            "degree",
            "distance_to_medoid",
            "medoid_residue_id",
            "is_medoid",
        ]
    ].sort_values(["cluster_id", "distance_to_medoid", "residue_id"])
    clusters_full_path = output_dir / "clusters_full.csv"
    clusters_full.to_csv(clusters_full_path, index=False)

    cluster_summary_path = output_dir / "cluster_summaries.csv"
    cluster_summary.to_csv(cluster_summary_path, index=False)

    k_sweep_path = output_dir / "k_sweep_summary.csv"
    k_sweep_summary.to_csv(k_sweep_path, index=False)

    top_cluster_rows = cluster_summary.head(min(args.top_k_clusters, len(cluster_summary)))
    medoid_seeds = []
    for _, row in top_cluster_rows.iterrows():
        medoid_seeds.append(
            {
                "cluster_id": int(row["cluster_id"]),
                "medoid_residue_id": row["medoid_residue_id"],
                "structure_id": row["structure_id"],
                "gene_symbol": row["gene_symbol"],
                "protein_family": row["protein_family"],
                "cluster_score_family": float(row["cluster_score_family"]),
                "cluster_score_protein": float(row["cluster_score_protein"]),
                "mean_leak": float(row["mean_leak"]),
                "mean_epistemic": float(row["mean_epistemic"]),
                "query_template": {
                    "target_residue_id": row["medoid_residue_id"],
                    "cone_depth_min": float(args.cone_depth_min),
                    "leak_score_min": float(args.leak_score_min),
                    "distance_max": float(row["max_distance_to_medoid"]),
                },
            }
        )

    medoid_seeds_path = output_dir / "medoid_seeds.json"
    with medoid_seeds_path.open("w", encoding="utf-8") as fh:
        json.dump(medoid_seeds, fh, indent=2, default=_json_default)

    run_metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _resolve_git_commit(),
        "space": args.space,
        "requested_curvature": curvature,
        "stored_curvature": load_metadata.get("stored_curvature"),
        "filters": {
            "cone_depth_min": args.cone_depth_min,
            "leak_score_min": args.leak_score_min,
            "epistemic_min": args.epistemic_min,
            "include_structures": args.include_structures,
            "exclude_structures": args.exclude_structures,
            "protein_family_filter": args.protein_family_filter,
            "max_rows": args.max_rows,
        },
        "effective_row_count": load_metadata["row_count"],
        "structure_count": load_metadata["structure_count"],
        "candidate_k_values": args.k_range,
        "chosen_k": selected_k,
        "random_seed": args.random_state,
        "runtime_seconds": runtime_seconds,
        "distance_matrix_shape": list(distance_matrix.shape) if distance_matrix is not None else None,
        "distance_matrix_bytes": int(distance_matrix.nbytes) if distance_matrix is not None else None,
        "sql_query": load_metadata["sql_query"],
        "sql_params": load_metadata["sql_params"],
    }
    run_metadata_path = output_dir / "run_metadata.json"
    with run_metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2, default=_json_default)

    artifact_paths = {
        "clusters_full": str(clusters_full_path),
        "cluster_summaries": str(cluster_summary_path),
        "medoid_seeds": str(medoid_seeds_path),
        "run_metadata": str(run_metadata_path),
        "k_sweep_summary": str(k_sweep_path),
    }

    if args.write_distance_matrix and distance_matrix is not None:
        distance_path = output_dir / "distance_matrix.npy"
        np.save(distance_path, distance_matrix)
        artifact_paths["distance_matrix"] = str(distance_path)

    if args.write_membership_matrix:
        labels = cluster_membership["cluster_id"].to_numpy(dtype=np.int64)
        labels_path = output_dir / "labels.npy"
        np.save(labels_path, labels)
        artifact_paths["labels"] = str(labels_path)

        membership_matrix = np.eye(selected_k, dtype=np.int8)[labels]
        membership_path = output_dir / "membership_matrix.npy"
        np.save(membership_path, membership_matrix)
        artifact_paths["membership_matrix"] = str(membership_path)

    return artifact_paths


async def async_main(argv: list[str] | None = None) -> int:
    """Async CLI entrypoint."""
    args = parse_args(argv)
    start_time = time.perf_counter()

    async with get_db_adapter(args.dsn) as db:
        curvature = await get_curvature(args.space, db=db)
        df, load_metadata = await load_filtered_rows(db, args)

    validate_embeddings(
        df,
        dim=EXPECTED_EMBEDDING_DIM,
        curvature=curvature,
        stored_curvature=load_metadata.get("stored_curvature"),
    )
    embedding_matrix = embeddings_to_matrix(df)
    distance_matrix = poincare_distance_matrix(
        embedding_matrix,
        curvature,
        block_size=args.block_size,
    )
    validate_distance_matrix(distance_matrix)

    valid_k_values = [k for k in args.k_range if k < len(df)]
    if not valid_k_values:
        raise ValueError(
            f"No valid K values remain after filtering {len(df)} rows; "
            f"all requested K values were >= row count"
        )

    sweep_results: dict[int, KMedoidsSweepResult] = {}
    cluster_summary_by_k: dict[int, pd.DataFrame] = {}
    for k in valid_k_values:
        run_result = run_kmedoids(distance_matrix, k=k, random_state=args.random_state)
        cluster_membership, cluster_summary = score_clusters(
            df,
            distance_matrix,
            labels=run_result["labels"],
            medoid_indices=run_result["medoid_indices"],
        )
        sweep_results[k] = KMedoidsSweepResult(
            k=k,
            labels=run_result["labels"],
            medoid_indices=run_result["medoid_indices"],
            cluster_membership=cluster_membership,
            cluster_summary=cluster_summary,
        )
        cluster_summary_by_k[k] = cluster_summary

    k_sweep_summary = summarize_k_sweep(cluster_summary_by_k)
    selected_k = choose_best_k(cluster_summary_by_k)
    best_result = sweep_results[selected_k]

    artifacts = write_artifacts(
        output_dir=Path(args.output_dir),
        cluster_membership=best_result.cluster_membership,
        cluster_summary=best_result.cluster_summary,
        k_sweep_summary=k_sweep_summary,
        selected_k=selected_k,
        args=args,
        curvature=curvature,
        load_metadata=load_metadata,
        runtime_seconds=time.perf_counter() - start_time,
        distance_matrix=distance_matrix,
    )

    summary = {
        "status": "ok",
        "selected_k": selected_k,
        "row_count": load_metadata["row_count"],
        "structure_count": load_metadata["structure_count"],
        "top_medoid_residue": (
            best_result.cluster_summary.iloc[0]["medoid_residue_id"]
            if not best_result.cluster_summary.empty
            else None
        ),
        "artifacts": artifacts,
    }
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


def main() -> None:
    """Synchronous script entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        raise SystemExit(asyncio.run(async_main()))
    except Exception as exc:
        LOGGER.exception("discover_hyperbolic_motifs failed")
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
