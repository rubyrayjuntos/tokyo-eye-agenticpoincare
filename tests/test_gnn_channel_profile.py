"""Tests for v6 GNN channel profiles in binding-site seed generation."""

from __future__ import annotations

from agent.tools.cryptic.gnn_channel_profile import (
    V5_LEGACY,
    V6_LEVER_A,
    filter_qualifying_residues_profile,
    profile_for_model_version,
    resolve_profile_thresholds,
)
from agent.tools.cryptic.seed_generator import GNNNodeOutput, generate_seeds_from_gnn


def _v6_scale_nodes(n: int = 10) -> list[GNNNodeOutput]:
    """Synthetic nodes matching lever_a scale on 9EST."""
    return [
        GNNNodeOutput(
            residue_id=f"9est:A:{i}",
            epistemic_uncertainty=1.57 + 0.01 * (i % 5),
            cone_depth=0.4 + 0.05 * (i % 3),
            disc_r=0.2 + 0.08 * (i % 7),
        )
        for i in range(16, 16 + n)
    ]


def test_v5_legacy_filters_zero_on_v6_scale():
    nodes = _v6_scale_nodes()
    assert filter_qualifying_residues_profile(nodes, V5_LEGACY) == []


def test_v6_profile_qualifies_subset():
    nodes = _v6_scale_nodes(20)
    qualified = filter_qualifying_residues_profile(nodes, V6_LEVER_A)
    assert len(qualified) > 0
    assert len(qualified) < len(nodes)


def test_profile_for_model_version_detects_v6():
    assert profile_for_model_version("lever_a_clean_slate_v1").name == "v6_lever_a"
    assert profile_for_model_version("GOSPConeMapper-v5").name == "v5_legacy"


def test_generate_seeds_with_v6_profile_produces_clusters():
    nodes = _v6_scale_nodes(30)
    # Place CA coords in two tight groups
    ca: dict[str, tuple[float, float, float]] = {}
    for i, node in enumerate(nodes):
        if i < 15:
            ca[node.residue_id] = (10.0 + i * 0.5, 10.0, 10.0)
        else:
            ca[node.residue_id] = (50.0 + (i - 15) * 0.5, 50.0, 50.0)

    clusters = generate_seeds_from_gnn(
        nodes,
        ca,
        channel_profile=V6_LEVER_A,
        min_cluster_size=3,
        eps_angstrom=8.0,
    )
    assert len(clusters) >= 1

    epi_thr, shell_thr = resolve_profile_thresholds(nodes, V6_LEVER_A)
    assert epi_thr > 1.5
    assert shell_thr < 1.0
