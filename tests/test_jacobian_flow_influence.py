from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from experiments.diagnostics.jacobian_flow_influence import (
    ASYMMETRY_FLOOR,
    FLOW_CV_FLOOR,
    asymmetry_index,
    bootstrap_spearman,
    coefficient_of_variation,
    flow_centralities,
    safe_norm_sq,
)
from science.dtie.common.classical_network_metrics import (
    anm_msf,
    build_ca_contact_graph,
    classical_network_metrics,
)


def test_safe_norm_sq_floors_near_zero() -> None:
    x = torch.zeros(3, dtype=torch.float64)
    assert float(safe_norm_sq(x)) >= 1e-6


def test_asymmetry_index_zero_on_symmetric() -> None:
    m = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    asym = asymmetry_index(m)
    assert asym["mean"] == pytest.approx(0.0)


def test_asymmetry_index_nonzero_on_directed() -> None:
    m = np.array([[0.0, 10.0], [1.0, 0.0]])
    asym = asymmetry_index(m)
    assert asym["mean"] == pytest.approx(9.0 / 11.0)
    assert asym["mean"] > ASYMMETRY_FLOOR


def test_flow_centralities_and_cv_floor() -> None:
    m = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 4.0, 0.0]])
    cents = flow_centralities(m)
    assert cents["out"].tolist() == pytest.approx([1.0, 5.0, 4.0])
    assert cents["in"].tolist() == pytest.approx([2.0, 5.0, 3.0])
    cv = coefficient_of_variation(cents["total"])
    assert cv > FLOW_CV_FLOOR


def test_classical_metrics_on_toy_chain() -> None:
    # Linear chain of 12 Cα spaced 3.8Å — connected at 8Å cutoff.
    n = 12
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n) * 3.8
    g = build_ca_contact_graph(coords, cutoff_angstrom=8.0)
    assert g.number_of_edges() > 0
    metrics = classical_network_metrics(coords, cutoff_angstrom=8.0)
    assert metrics["n_residues"] == n
    assert np.all(np.isfinite(metrics["betweenness"]))
    assert np.all(np.isfinite(metrics["current_flow_betweenness"]))
    # Ends should have lower betweenness than middle on a path.
    assert metrics["betweenness"][0] < metrics["betweenness"][n // 2]
    msf = anm_msf(coords)
    assert msf.shape == (n,)
    assert np.any(np.isfinite(msf))


def test_classical_metrics_refuse_empty_graph() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="zero edges"):
        classical_network_metrics(coords, cutoff_angstrom=8.0)


def test_bootstrap_spearman_recovers_perfect_rank() -> None:
    x = np.arange(20, dtype=np.float64)
    y = x * 2.0 + 1.0
    out = bootstrap_spearman(x, y, n_boot=50, seed=1)
    assert out["spearman"] == pytest.approx(1.0)
    assert out["ci_lo"] > 0.9


def test_structure_holds_and_stage_a12_verdict_floors() -> None:
    from experiments.diagnostics.jacobian_flow_influence import (
        aggregate_stage_a12_verdict,
        structure_holds_trunk,
    )

    def layer(rho: float, ci_lo: float, *, alive: bool = True, asym: bool = True) -> dict:
        return {
            "liveness": {"alive": alive, "asymmetry_non_zero": asym},
            "classical_correlation": {
                "betweenness": {"spearman": rho, "ci_lo": ci_lo, "ci_hi": 0.9},
                "anm_msf": {"spearman": -0.5, "ci_lo": -0.7, "ci_hi": -0.3},
            },
        }

    assert structure_holds_trunk(layer(0.5, 0.2))["holds"] is True
    assert structure_holds_trunk(layer(0.25, 0.1))["holds"] is False  # below 0.30
    assert structure_holds_trunk(layer(0.5, -0.01))["holds"] is False  # CI includes 0

    structs = [
        {"pdb_id": f"P{i:02d}", "layers": {"encoder_h": layer(0.5, 0.2)}}
        for i in range(10)
    ] + [
        {"pdb_id": "X1", "layers": {"encoder_h": layer(0.1, -0.1)}},
        {"pdb_id": "X2", "layers": {"encoder_h": layer(0.1, -0.1)}},
    ]
    v = aggregate_stage_a12_verdict(structs)
    assert v["n_holds"] == 10
    assert v["outcome"] == "win_graph_scaffolded_flow"


def test_jacobian_probe_node_count_honors_n_residue_nodes() -> None:
    from experiments.diagnostics.jacobian_flow_influence import jacobian_probe_node_count

    n_res = 4
    data = Data(x=torch.randn(n_res + 1, 3))
    data.n_residue_nodes = n_res
    data.n_parent_nodes = 1
    assert jacobian_probe_node_count(data) == n_res
    assert jacobian_probe_node_count(data, {"n_residues": 99}) == n_res


class _MockFlowGNN(torch.nn.Module):
    """Minimal GNN stub: encoder_h = data.x[:, :hidden] for Jacobian hooks."""

    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.radial_head = torch.nn.Linear(hidden, 1, bias=False)
        torch.nn.init.ones_(self.radial_head.weight)

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        enc = data.x[:, : self.hidden]
        if enc.shape[1] < self.hidden:
            enc = torch.cat(
                [enc, enc.new_zeros(enc.shape[0], self.hidden - enc.shape[1])],
                dim=1,
            )
        _ = self.radial_head(enc)
        return {"hyp_projections_2d": enc[:, :2]}


def test_compute_influence_matrix_excludes_parent_nodes() -> None:
    from experiments.diagnostics.jacobian_flow_influence import compute_influence_matrix

    n_res = 3
    hidden = 4
    data = Data(
        x=torch.randn(n_res + 1, hidden, dtype=torch.float64) + 1.0,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    data.n_residue_nodes = n_res
    data.n_parent_nodes = 1
    model = _MockFlowGNN(hidden=hidden).double()
    report = compute_influence_matrix(
        model,
        data,
        layer="encoder_h",
        device="cpu",
        prot={"n_residues": n_res},
    )
    assert report["n_residues"] == n_res
    assert report["n_total_nodes"] == n_res + 1
    assert report["influence"].shape == (n_res, n_res)
    assert len(report["centralities"]["out"]) == n_res


def test_pc1_sq_score_mode_runs_and_tags_report() -> None:
    from experiments.diagnostics.jacobian_flow_influence import (
        compute_influence_matrix,
        pc1_unit_direction,
        score_scalar,
    )

    n_res = 5
    hidden = 4
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(n_res, hidden, dtype=torch.float64, generator=rng) + 0.5
    data = Data(x=x.clone(), edge_index=torch.zeros(2, 0, dtype=torch.long))
    data.n_residue_nodes = n_res
    model = _MockFlowGNN(hidden=hidden).double()
    report = compute_influence_matrix(
        model,
        data,
        layer="encoder_h",
        device="cpu",
        prot={"n_residues": n_res},
        score_mode="pc1_sq",
    )
    assert report["score_mode"] == "pc1_sq"
    assert report["grad_site"] == "raw_x"
    assert report["influence"].shape == (n_res, n_res)
    assert np.isfinite(report["centralities"]["total"]).all()
    û = pc1_unit_direction(x)
    s = score_scalar(x[0], score_mode="pc1_sq", pc1_dir=û)
    assert float(s) >= 1e-6


class _MockEmbFlowGNN(torch.nn.Module):
    """Stub with node_emb so post_zscore / post_node_emb grad sites work."""

    def __init__(self, in_dim: int = 3, hidden: int = 4) -> None:
        super().__init__()
        self.node_emb = torch.nn.Linear(in_dim, hidden, bias=False)
        self.radial_head = torch.nn.Linear(hidden, 1, bias=False)
        torch.nn.init.eye_(self.node_emb.weight[:, : min(in_dim, hidden)])
        torch.nn.init.ones_(self.radial_head.weight)

    def forward(self, data: Data) -> dict[str, torch.Tensor]:
        enc = self.node_emb(data.x)
        _ = self.radial_head(enc)
        return {"hyp_projections_2d": enc[:, :2]}


def test_grad_site_post_node_emb_runs() -> None:
    from experiments.diagnostics.jacobian_flow_influence import compute_influence_matrix

    n_res = 4
    rng = torch.Generator().manual_seed(1)
    data = Data(
        x=torch.randn(n_res, 3, dtype=torch.float64, generator=rng) + 0.5,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    data.n_residue_nodes = n_res
    model = _MockEmbFlowGNN(in_dim=3, hidden=4).double()
    report = compute_influence_matrix(
        model,
        data,
        layer="encoder_h",
        device="cpu",
        prot={"n_residues": n_res},
        score_mode="pc1_sq",
        grad_site="post_node_emb",
    )
    assert report["grad_site"] == "post_node_emb"
    assert report["influence"].shape == (n_res, n_res)
    assert np.isfinite(report["centralities"]["total"]).all()


def test_grad_site_post_zscore_matches_raw_when_no_transform() -> None:
    """Without z-score, node_emb input is data.x → sites must agree."""
    from experiments.diagnostics.jacobian_flow_influence import compute_influence_matrix

    n_res = 4
    rng = torch.Generator().manual_seed(2)
    x = torch.randn(n_res, 3, dtype=torch.float64, generator=rng) + 0.5
    model = _MockEmbFlowGNN(in_dim=3, hidden=4).double()

    def _run(site: str) -> np.ndarray:
        data = Data(x=x.clone(), edge_index=torch.zeros(2, 0, dtype=torch.long))
        data.n_residue_nodes = n_res
        return compute_influence_matrix(
            model,
            data,
            layer="encoder_h",
            device="cpu",
            prot={"n_residues": n_res},
            score_mode="pc1_sq",
            grad_site=site,  # type: ignore[arg-type]
        )["influence"]

    raw = _run("raw_x")
    z = _run("post_zscore")
    assert np.allclose(raw, z, rtol=1e-8, atol=1e-10)
