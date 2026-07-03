"""Tests for Phase 4 normalized spectral analysis."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from science.dtie.v5.phases.phase4_resistance import _spectral_analysis


def test_spectral_lambda2_is_bounded_with_exponential_weights():
    """Normalized Laplacian keeps λ₂ in [0, 2] even when conductance weights explode."""
    G = nx.Graph()
    for i in range(6):
        G.add_node(i)
    heavy_weight = math.exp(16.0)
    for i in range(5):
        G.add_edge(i, i + 1, conductance=heavy_weight)

    spectral = _spectral_analysis(G)

    assert 0.0 <= spectral["lambda_2"] <= 2.0
    assert spectral["lambda_2"] < 10.0


def test_spectral_lambda2_zero_for_disconnected_graph():
    G = nx.Graph()
    G.add_edge(0, 1, conductance=1.0)
    G.add_edge(2, 3, conductance=1.0)

    spectral = _spectral_analysis(G)

    assert spectral["lambda_2"] == 0.0
    assert spectral["hinge_residues"] == []


def test_spectral_fiedler_vector_length_matches_nodes():
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["conductance"] = np.exp(4.0 + 5.0)

    spectral = _spectral_analysis(G)

    assert len(spectral["fiedler_vector"]) == G.number_of_nodes()
