"""Edge telemetry MVP property tests (spec §4)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from science.dtie.v5.gnn.model import precompute_clustering
from science.dtie.v6.gnn.model import GOSPConeMapperV6
from science.training.edge_telemetry import (
    assess_telemetry_alive,
    collect_edge_telemetry,
    same_expert_stats,
)

_REPO = Path(__file__).resolve().parents[1]
COLD_CKPT = _REPO / "checkpoints/v6/runs/slim_moe_structural_ssot_cold_v1/v6_best.pt"
ROUTE_CKPT = _REPO / "checkpoints/v6/runs/slim_moe_route_v1/v6_best.pt"
STAGE_A_MANIFEST = _REPO / "manifests/v6_corpus_stage_a_small_v1.json"


def _synthetic_prot(n: int = 24) -> dict:
    rng = np.random.default_rng(0)
    x = torch.tensor(
        np.column_stack(
            [
                rng.uniform(8.0, 18.0, n),
                rng.integers(0, 2, n).astype(np.float64),
                rng.uniform(0.0, 1.0, n),
                rng.uniform(0.05, 0.9, n),
            ]
        ),
        dtype=torch.float32,
    )
    m = n * 2
    src = torch.randint(0, n, (m,))
    dst = torch.randint(0, n, (m,))
    edge_index = torch.stack([src, dst], dim=0)
    rel = torch.randn(m, 3) * 0.5
    dist = rel.norm(dim=-1, keepdim=True).clamp(min=0.1)
    edge_attr = torch.cat([rel, dist], dim=-1)
    return {
        "pdb_id": "SYN",
        "chain": "A",
        "data": Data(x=x, edge_index=edge_index, edge_attr=edge_attr),
        "ca_coords": torch.tensor(rng.normal(size=(n, 3)), dtype=torch.float32),
        "residue_ids": [f"A:{i}:" for i in range(1, n + 1)],
    }


def _forward_telemetry(n: int = 24) -> tuple:
    from experiments.training.v6.train_loop import prepare_training_batch

    model = GOSPConeMapperV6(
        hidden=32,
        num_layers=2,
        num_experts=4,
        hyperbolic_gate=False,
        topology_only_gate=True,
    )
    model.eval()
    prot = _synthetic_prot(n)
    data = prepare_training_batch(model, prot, "cpu", structural_disc_frozen=True)
    data = precompute_clustering(data)
    with torch.no_grad():
        out = model(data)
    rec = collect_edge_telemetry(
        model,
        data,
        out,
        structure_id="SYN",
        ca_coords=prot["ca_coords"].numpy(),
    )
    return model, data, out, rec


def test_edge_epistemic_var_non_degenerate() -> None:
    _, _, _, rec = _forward_telemetry()
    assert rec.telemetry_alive, rec.telemetry_alive_reason
    assert rec.edge_epistemic_var_std > 1e-4


def test_same_expert_rate_matches_null_baseline_synthetic_collapse() -> None:
    """Null baseline = sum(p_e^2) from empirical load; same tracks null under collapse."""
    rng = np.random.default_rng(0)
    n, e = 200, 400
    load = np.array([0.07, 0.07, 0.72, 0.14])
    assign = rng.choice(4, size=n, p=load)
    edge_index = torch.stack(
        [
            torch.from_numpy(rng.integers(0, n, e)),
            torch.from_numpy(rng.integers(0, n, e)),
        ],
        dim=0,
    )
    w = torch.zeros(n, 4)
    for i, a in enumerate(assign):
        w[i, a] = 1.0
    same, null, excess, dom = same_expert_stats(edge_index, w)
    load_emp = w.mean(dim=0).numpy()
    assert abs(null - float(np.sum(load_emp * load_emp))) < 1e-5
    assert abs(same - null) < 0.08
    assert abs(excess) < 0.08
    assert dom > 0.6


def test_assess_telemetry_alive_rejects_flat_probe() -> None:
    flat = np.full(100, 0.0041)
    spread = np.linspace(0.01, 1.0, 100)
    ok_flat, reason_flat = assess_telemetry_alive(flat, spread, 0.0)
    assert not ok_flat
    assert "epistemic" in reason_flat or "aleatoric" in reason_flat
    ok, reason = assess_telemetry_alive(spread, spread * 0.5 + 0.1, 0.0)
    assert ok, reason


def test_stratify_same_expert_by_flow_high_vs_low() -> None:
    from science.training.edge_telemetry import stratify_same_expert_by_flow

    n, e = 40, 80
    ei = torch.stack([torch.randint(0, n, (e,)), torch.randint(0, n, (e,))])
    w = torch.zeros(n, 4)
    w[:, 2] = 1.0
    flow = np.concatenate([np.full(e // 2, 0.01), np.full(e - e // 2, 0.99)])
    out = stratify_same_expert_by_flow(ei, w, flow)
    assert out["high_flow"] is not None
    assert out["low_flow"] is not None
    assert out["high_flow"]["same_expert_rate"] == 1.0


def test_resistance_corr_reproducible_across_seeds() -> None:
    """Two deterministic forwards on same checkpoint must agree."""
    model, data, out, rec1 = _forward_telemetry()
    with torch.no_grad():
        out2 = model(data)
    rec2 = collect_edge_telemetry(
        model,
        data,
        out2,
        ca_coords=_synthetic_prot()["ca_coords"].numpy(),
    )
    assert math.isfinite(rec1.edge_embed_resistance_corr)
    assert abs(rec1.edge_embed_resistance_corr - rec2.edge_embed_resistance_corr) < 1e-5
    assert abs(rec1.same_expert_rate - rec2.same_expert_rate) < 1e-5


def test_collect_edge_telemetry_all_finite() -> None:
    _, _, _, rec = _forward_telemetry()
    for name in (
        "edge_embed_resistance_corr",
        "edge_epistemic_var_mean",
        "edge_epistemic_var_std",
        "edge_aleatoric_var_mean",
        "same_expert_rate",
        "same_expert_null_rate",
        "edge_flow_score_mean",
    ):
        val = getattr(rec, name)
        assert math.isfinite(val), name


@pytest.mark.skipif(
    not COLD_CKPT.is_file() or not ROUTE_CKPT.is_file(),
    reason="Requires slim MoE checkpoints on disk",
)
@pytest.mark.skipif(
    not os.environ.get("TRAINING_LOAD_FROM_PDB"),
    reason="Set TRAINING_LOAD_FROM_PDB=1 for corpus edge telemetry integration",
)
def test_edge_telemetry_no_nan_full_corpus() -> None:
    """All four field groups finite across Stage A corpus, both checkpoints."""
    from experiments.training.v6.assess_checkpoint import load_v6_model
    from experiments.training.v6.corpus import load_training_proteins
    from experiments.training.v6.train_loop import prepare_training_batch

    proteins, _ = load_training_proteins(
        Path("/tmp/dtie_pdb_cache"),
        STAGE_A_MANIFEST,
    )
    assert len(proteins) >= 8

    for ckpt in (COLD_CKPT, ROUTE_CKPT):
        model = load_v6_model(ckpt, "cpu")
        model.eval()
        with torch.no_grad():
            for prot in proteins:
                data = prepare_training_batch(
                    model, prot, "cpu", structural_disc_frozen=True
                )
                out = model(data)
                ca = prot.get("ca_coords")
                rec = collect_edge_telemetry(
                    model,
                    data,
                    out,
                    structure_id=str(prot["pdb_id"]),
                    ca_coords=ca.detach().cpu().numpy() if ca is not None else None,
                )
                assert rec.telemetry_alive, (
                    f"{ckpt.name} {prot['pdb_id']}: {rec.telemetry_alive_reason}"
                )
                assert math.isfinite(rec.edge_embed_resistance_corr)
                assert math.isfinite(rec.edge_epistemic_var_std)
                assert math.isfinite(rec.same_expert_rate)
                assert math.isfinite(rec.edge_flow_score_mean)


@pytest.mark.skipif(
    not COLD_CKPT.is_file() or not ROUTE_CKPT.is_file(),
    reason="Requires slim MoE checkpoints",
)
@pytest.mark.skipif(
    not os.environ.get("TRAINING_LOAD_FROM_PDB"),
    reason="Set TRAINING_LOAD_FROM_PDB=1",
)
def test_same_expert_rate_matches_null_baseline_checkpoints() -> None:
    """Null = sum(p_e²) from empirical load; report excess on real checkpoints."""
    from experiments.diagnostics.edge_telemetry_baseline import _run_checkpoint
    from experiments.training.v6.corpus import load_training_proteins

    proteins, _ = load_training_proteins(
        Path("/tmp/dtie_pdb_cache"),
        STAGE_A_MANIFEST,
        max_proteins=6,
    )
    for ckpt in (COLD_CKPT, ROUTE_CKPT):
        payload = _run_checkpoint(ckpt, proteins, "cpu", structural_disc_frozen=True)
        agg = payload["corpus_aggregate"]
        same = agg["corpus_same_expert_rate_mean"]
        null = agg["corpus_same_expert_null_rate_mean"]
        excess = agg.get("corpus_same_expert_excess_mean", same - null)
        dom = agg["corpus_dominant_expert_share_mean"]
        assert null <= dom + 0.05
        assert same >= null - 0.05
        assert math.isfinite(excess)


@pytest.mark.skipif(
    not ( _REPO / "checkpoints/v6/runs/edge_telemetry_mvp_baseline.json").is_file(),
    reason="Run edge_telemetry_baseline.py first for baseline artifact test",
)
def test_baseline_json_schema() -> None:
    path = _REPO / "checkpoints/v6/runs/edge_telemetry_mvp_baseline.json"
    data = json.loads(path.read_text())
    assert data["version"] == "edge_telemetry_mvp_v2"
    assert len(data["checkpoints"]) >= 1
    for ck in data["checkpoints"]:
        assert "corpus_aggregate" in ck
        assert "telemetry/track" in ck["mlflow_tags"]
