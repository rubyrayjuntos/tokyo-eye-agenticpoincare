from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.discover_hyperbolic_motifs import (
    choose_best_k,
    load_filtered_rows,
    parse_args,
    poincare_distance_matrix,
    score_clusters,
    validate_embeddings,
    write_artifacts,
)
from science.dtie.common.hyperbolic_utils import hyperbolic_pairwise_distance


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "residue_id": "4obe:A:10",
                "structure_id": "4obe",
                "chain_id": "4obe:A",
                "chain_label": "A",
                "residue_number": 10,
                "residue_type": "GLY",
                "embedding_double": [0.10, 0.05, 0.01] + [0.0] * 125,
                "leak_score": 70.0,
                "cone_depth": 4.1,
                "epistemic_uncertainty": 2.2,
                "gene_symbol": "KRAS",
                "protein_family": "ras",
                "betweenness": 0.2,
                "degree": 5,
            },
            {
                "residue_id": "4obe:A:11",
                "structure_id": "4obe",
                "chain_id": "4obe:A",
                "chain_label": "A",
                "residue_number": 11,
                "residue_type": "ASP",
                "embedding_double": [0.11, 0.04, 0.00] + [0.0] * 125,
                "leak_score": 67.0,
                "cone_depth": 4.0,
                "epistemic_uncertainty": 2.0,
                "gene_symbol": "KRAS",
                "protein_family": "ras",
                "betweenness": 0.18,
                "degree": 4,
            },
            {
                "residue_id": "3oxz:A:420",
                "structure_id": "3oxz",
                "chain_id": "3oxz:A",
                "chain_label": "A",
                "residue_number": 420,
                "residue_type": "TYR",
                "embedding_double": [0.35, 0.02, 0.03] + [0.0] * 125,
                "leak_score": 88.0,
                "cone_depth": 5.7,
                "epistemic_uncertainty": 3.1,
                "gene_symbol": "ABL1",
                "protein_family": "kinase",
                "betweenness": 0.35,
                "degree": 7,
            },
            {
                "residue_id": "3oxz:A:421",
                "structure_id": "3oxz",
                "chain_id": "3oxz:A",
                "chain_label": "A",
                "residue_number": 421,
                "residue_type": "LEU",
                "embedding_double": [0.36, 0.03, 0.02] + [0.0] * 125,
                "leak_score": 85.0,
                "cone_depth": 5.5,
                "epistemic_uncertainty": 2.9,
                "gene_symbol": "ABL1",
                "protein_family": "kinase",
                "betweenness": 0.30,
                "degree": 6,
            },
        ]
    )


def test_parse_args_normalizes_csv_inputs() -> None:
    args = parse_args(
        [
            "--space",
            "space_gospconemapper_v6_hyp128",
            "--k-range",
            "16,8,12",
            "--include-structures",
            "4OBE,3OXZ",
            "--exclude-structures",
            "6GQO",
            "--protein-family-filter",
            "kinase,RAS",
        ]
    )

    assert args.k_range == [8, 12, 16]
    assert args.include_structures == ["4obe", "3oxz"]
    assert args.exclude_structures == ["6gqo"]
    assert args.protein_family_filter == ["kinase", "ras"]


def test_validate_embeddings_and_distance_matrix_match_shared_geometry() -> None:
    df = _sample_df()
    validate_embeddings(df, dim=128, curvature=0.6054342985153198, stored_curvature=0.6054342985153198)

    x = np.vstack(df["embedding_double"].apply(lambda value: np.asarray(value, dtype=np.float64)))
    expected = hyperbolic_pairwise_distance(x, c=0.6054342985153198)
    actual = poincare_distance_matrix(x, c=0.6054342985153198)
    blockwise = poincare_distance_matrix(
        x,
        c=0.6054342985153198,
        block_size=2,
        blockwise_threshold=2,
    )

    assert np.allclose(actual, expected)
    assert np.allclose(blockwise, expected)


def test_validate_embeddings_rejects_out_of_ball() -> None:
    df = _sample_df().iloc[:1].copy()
    df.at[df.index[0], "embedding_double"] = [1.6] + [0.0] * 127

    with pytest.raises(ValueError, match="violates Poincare ball bounds"):
        validate_embeddings(df, dim=128, curvature=0.6054342985153198)


def test_score_clusters_and_choose_best_k() -> None:
    df = _sample_df()
    x = np.vstack(df["embedding_double"].apply(lambda value: np.asarray(value, dtype=np.float64)))
    distance_matrix = poincare_distance_matrix(x, c=0.6054342985153198)

    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    medoid_indices = np.array([0, 2], dtype=np.int64)
    membership, summary = score_clusters(df, distance_matrix, labels, medoid_indices)

    assert {"cluster_id", "medoid_residue_id", "distance_to_medoid", "is_medoid"} <= set(membership.columns)
    assert summary.iloc[0]["medoid_residue_id"] in {"4obe:A:10", "3oxz:A:420"}

    k8 = summary.copy()
    k12 = summary.copy()
    k8["cluster_score_family"] = [2.5, 1.8]
    k8["cluster_score_protein"] = [2.3, 1.6]
    k8["unique_families"] = [2, 2]
    k8["mean_distance_to_medoid"] = [0.4, 0.5]
    k8["std_distance_to_medoid"] = [0.05, 0.06]

    k12["cluster_score_family"] = [1.2, 0.9]
    k12["cluster_score_protein"] = [1.0, 0.8]
    k12["unique_families"] = [1, 1]
    k12["mean_distance_to_medoid"] = [0.9, 1.1]
    k12["std_distance_to_medoid"] = [0.20, 0.25]

    assert choose_best_k({8: k8, 12: k12}) == 8


@pytest.mark.asyncio
async def test_load_filtered_rows_builds_governed_query(validating_mock_db) -> None:
    validating_mock_db.register_response(
        "fact_gnn_node_embedding",
        [
            {
                "residue_id": "4obe:A:10",
                "structure_id": "4obe",
                "chain_id": "4obe:A",
                "chain_label": "A",
                "residue_number": 10,
                "residue_type": "GLY",
                "embedding_double": [0.1] * 128,
                "leak_score": 50.0,
                "cone_depth": 4.0,
                "epistemic_uncertainty": 2.1,
                "gene_symbol": "KRAS",
                "protein_family": "ras",
                "betweenness": 0.2,
                "degree": 5,
                "space_name": "space_gospconemapper_v6_hyp128",
                "stored_curvature": 0.6054342985153198,
            }
        ],
    )
    args = parse_args(
        [
            "--include-structures",
            "4obe",
            "--protein-family-filter",
            "ras",
            "--epistemic-min",
            "2.0",
        ]
    )

    df, metadata = await load_filtered_rows(validating_mock_db, args)

    assert len(df) == 1
    assert metadata["row_count"] == 1
    query, params = validating_mock_db.executed_queries[0]
    assert "latest_embeddings" in query
    assert params["include_structures"] == ["4obe"]
    assert params["protein_family_filter"] == ["ras"]
    assert params["epistemic_min"] == 2.0


def test_write_artifacts_emits_expected_files(tmp_path) -> None:
    df = _sample_df()
    x = np.vstack(df["embedding_double"].apply(lambda value: np.asarray(value, dtype=np.float64)))
    distance_matrix = poincare_distance_matrix(x, c=0.6054342985153198)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    medoid_indices = np.array([0, 2], dtype=np.int64)
    membership, summary = score_clusters(df, distance_matrix, labels, medoid_indices)
    k_sweep = pd.DataFrame(
        [
            {
                "k": 2,
                "top_cluster_score_family": 2.1,
                "top_cluster_score_protein": 2.0,
                "mean_top_cluster_score_family": 1.9,
                "mean_family_diversity": 2.0,
                "median_mean_distance": 0.5,
                "median_spread": 0.1,
                "largest_cluster_size": 2,
                "selection_score": 1.4,
            }
        ]
    )
    args = parse_args(["--top-k-clusters", "2", "--write-membership-matrix"])
    artifacts = write_artifacts(
        output_dir=tmp_path,
        cluster_membership=membership,
        cluster_summary=summary,
        k_sweep_summary=k_sweep,
        selected_k=2,
        args=args,
        load_metadata={
            "row_count": 4,
            "structure_count": 2,
            "stored_curvature": 0.6054342985153198,
            "sql_query": "SELECT ...",
            "sql_params": {"space": "space_gospconemapper_v6_hyp128"},
        },
        runtime_seconds=1.2,
        distance_matrix=distance_matrix,
    )

    assert (tmp_path / "clusters_full.csv").exists()
    assert (tmp_path / "cluster_summaries.csv").exists()
    assert (tmp_path / "medoid_seeds.json").exists()
    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "membership_matrix.npy").exists()
    with open(artifacts["medoid_seeds"], encoding="utf-8") as fh:
        seeds = json.load(fh)
    assert len(seeds) == 2
    assert "target_residue_id" in seeds[0]["query_template"]
