"""Tests for residue-first aleatoric diagnostics."""

from __future__ import annotations

import numpy as np

from science.training.aleatoric_residue_diagnostics import (
    DEFAULT_T_ALE_PERCENTILE,
    aleatoric_dataset_health_report,
    corpus_aleatoric_histogram,
    flag_investigation_sites,
    global_aleatoric_health_monitor,
    per_protein_aleatoric_summary,
    rank_proteins_for_active_learning,
    resolve_t_ale,
)
from science.training.uncertainty_diagnostics import extract_residue_uncertainty_rows


def _synthetic_rows(n: int = 40, *, sid: str = "1mbn") -> list[dict]:
    rng = np.random.default_rng(0)
    ale = np.clip(rng.exponential(scale=0.02, size=n) + 0.01, 0.005, 0.2)
    disc_r = rng.uniform(0.1, 0.9, size=n)
    clustering = rng.uniform(0.05, 0.8, size=n)
    rows = []
    for i in range(n):
        rows.append(
            {
                "structure_id": sid,
                "chain": "A",
                "residue_id": f"A:{i + 1}:",
                "rho": float(8 + i * 0.2),
                "tau_flag": 1.0 if i % 7 == 0 else 0.0,
                "cone_depth": float(0.2 + 0.02 * i),
                "aleatoric": float(ale[i]),
                "disc_r": float(disc_r[i]),
                "clustering": float(clustering[i]),
                "expert_routing_max": float(rng.uniform(0.2, 0.95)),
            }
        )
    return rows


def test_resolve_t_ale_percentile_default() -> None:
    rows = _synthetic_rows(100)
    t_ale, meta = resolve_t_ale(rows)
    assert meta["mode"] == "percentile"
    assert meta["percentile"] == DEFAULT_T_ALE_PERCENTILE
    ale = np.array([r["aleatoric"] for r in rows])
    expected = float(np.percentile(ale, DEFAULT_T_ALE_PERCENTILE))
    assert abs(t_ale - expected) < 1e-9
    frac = float(np.mean(ale > t_ale))
    assert 0.05 <= frac <= 0.15


def test_resolve_t_ale_absolute_override() -> None:
    rows = _synthetic_rows(50)
    t_ale, meta = resolve_t_ale(rows, t_ale=0.05)
    assert meta["mode"] == "absolute"
    assert t_ale == 0.05


def test_dataset_health_report_uses_percentile_t_ale() -> None:
    rows = _synthetic_rows(30, sid="1mbn") + _synthetic_rows(20, sid="1lyz")
    report = aleatoric_dataset_health_report(rows)
    assert report["t_ale_resolution"]["mode"] == "percentile"
    hist = report["corpus_histogram"]
    assert 0.05 <= hist["fraction_above_t_ale"] <= 0.15


def test_corpus_histogram_right_skew_on_exponential() -> None:
    rows = _synthetic_rows(200)
    hist = corpus_aleatoric_histogram(rows)
    assert hist["ok"]
    assert hist["skewness"] > 0
    assert hist["role"] == "dataset_health_monitor"
    assert len(hist["histogram"]["counts"]) == 40


def test_per_protein_summary_and_ranking() -> None:
    rows_a = _synthetic_rows(30, sid="1mbn")
    rows_b = _synthetic_rows(25, sid="4obe")
    rows_b[0]["aleatoric"] = 0.35
    summaries = [
        per_protein_aleatoric_summary(rows_a, structure_id="1mbn", chain="A", t_ale=0.05),
        per_protein_aleatoric_summary(rows_b, structure_id="4obe", chain="A", t_ale=0.05),
    ]
    ranked = rank_proteins_for_active_learning(summaries)
    assert ranked[0]["structure_id"] == "4obe"
    assert ranked[0]["active_learning_rank"] == 1


def test_investigation_sites_local_rule() -> None:
    rows = _synthetic_rows(50)
    rows[10]["aleatoric"] = 0.25
    rows[10]["disc_r"] = 0.95
    rows[10]["clustering"] = 0.02
    rows[10]["expert_routing_max"] = 0.15
    flagged = flag_investigation_sites(rows, t_ale=0.05)
    ids = {r["residue_id"] for r in flagged}
    assert rows[10]["residue_id"] in ids
    assert all(r["investigate"] for r in flagged)


def test_dataset_health_report_not_site_gate() -> None:
    rows = _synthetic_rows(30, sid="1mbn") + _synthetic_rows(20, sid="1lyz")
    report = aleatoric_dataset_health_report(rows, t_ale=0.05)
    assert report["doctrine"] == "residue_first_global_monitors_only"
    assert report["global_aleatoric_std_role"].startswith("dataset_health_monitor")
    assert len(report["per_protein"]) == 2
    assert report["n_investigation_sites"] >= 0


def test_global_health_monitor_role() -> None:
    rows = _synthetic_rows(20)
    health = global_aleatoric_health_monitor(rows)
    assert health["not_a_site_gate"] is True
    assert health["role"] == "dataset_health_monitor"


def test_extract_rows_includes_geometry_fields() -> None:
    import torch
    from torch_geometric.data import Data

    n = 5
    prot = {
        "pdb_id": "1MBN",
        "chain": "A",
        "residue_ids": [f"A:{i}:" for i in range(1, n + 1)],
        "data": Data(x=torch.zeros(n, 4)),
    }
    data = Data(clustering=torch.linspace(0.1, 0.5, n))
    out = {
        "uncertainty": {
            "epistemic": torch.ones(n, 1),
            "aleatoric": torch.linspace(0.01, 0.1, n).unsqueeze(1),
        },
        "cone_depth": torch.ones(n, 1) * 0.5,
        "hyp_projections_2d": torch.stack(
            [torch.linspace(0.2, 0.8, n), torch.zeros(n)],
            dim=1,
        ),
        "expert_weights": torch.nn.functional.softmax(
            torch.randn(n, 4),
            dim=1,
        ),
    }
    rows = extract_residue_uncertainty_rows(out, prot, graph_data=data)
    assert rows[0]["structure_id"] == "1mbn"
    assert "disc_r" in rows[0]
    assert "clustering" in rows[0]
    assert "expert_routing_max" in rows[0]
