"""Unit tests for cheap Hyp-MP telemetry helpers."""

from __future__ import annotations

import torch

from science.tokyo_eye.hyp_mp_telemetry import hyp_mp_edge_node_telemetry


def test_hyp_mp_telemetry_hub_count() -> None:
    # Two nodes near origin in ball; one edge both ways.
    x = torch.tensor([[0.1, 0.0], [0.2, 0.0]], dtype=torch.float64)
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    tel = hyp_mp_edge_node_telemetry(x, ei, curvature=1.0, k_frac=0.50)
    assert tel["n_nodes"] == 2
    assert tel["n_hubs"] == 1
    assert tel["n_edges"] == 2
    assert len(tel["hub_indices"]) == 1
