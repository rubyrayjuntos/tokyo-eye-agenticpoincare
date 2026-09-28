"""Unit test for the read-only step loggers in scripts/wrap1_zhyp_m2_pool_decoupled_diag.py (mocked model/MLflow)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

D = pytest.importorskip("wrap1_zhyp_m2_pool_decoupled_diag")
G = D.G

TAGS = ["1MBN:A", "1LYZ:A", "1BG1:A", "1F88:A", "2Z6H:A", "1HHP:A", "1TEN:A", "1UBQ:A", "1TIM:A", "4OBE:A", "2SHP:A", "1IVO:A"]


def _fake_system():
    fe = nn.Module()
    fe._backbone = nn.Module()
    fe._backbone.w = nn.Parameter(torch.ones(4))
    fe.proj = nn.Parameter(torch.ones(4))
    fe.frozen = nn.Parameter(torch.ones(4), requires_grad=False)
    sp = nn.Module()
    sp.mechanism_head = nn.Module()
    sp.mechanism_head.w = nn.Parameter(torch.ones(4))
    sp.attn_layers = nn.Module()
    sp.attn_layers.w = nn.Parameter(torch.ones(9))
    return types.SimpleNamespace(frontend=fe, spine=sp)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    system = _fake_system()
    def fake_step(system_, optimizer, batches, **kw):
        with torch.no_grad():
            for p in list(system_.frontend.parameters()) + list(system_.spine.parameters()):
                if p.requires_grad:
                    p.add_(0.1)
        return {"loss_total": 1.0, "preclip_norm": 0.5, "bce_per_structure": [0.1 * (i + 1) for i in range(11)]}

    logged = []
    monkeypatch.setattr(G, "run_step_sdrp_only", fake_step)
    monkeypatch.setattr(G, "_all_structure_tags", lambda: list(TAGS))
    monkeypatch.setattr(D.mlflow, "active_run", lambda: object())
    monkeypatch.setattr(D.mlflow, "log_metrics", lambda payload, step=None: logged.append((step, dict(payload))))
    monkeypatch.setattr(D.M, "STEPS", 400)
    out = tmp_path / "sl.json"
    D._install_step_loggers("1IVO:A", 10, (30, 70), out, smoke=False)
    return system, logged, out


def test_per_structure_bce_named_and_ordered(harness):
    system, logged, _ = harness
    m = G.run_step_sdrp_only(system, None, [], epoch=5)
    assert m["loss_total"] == 1.0  # the step's own return value is passed through untouched
    (step, payload), = logged
    assert step == 5
    bce = {k: v for k, v in payload.items() if k.startswith("bce_struct_")}
    assert len(bce) == 11 and "bce_struct_1IVOA" not in bce  # hold tag excluded
    assert bce["bce_struct_1MBNA"] == pytest.approx(0.1) and bce["bce_struct_2SHPA"] == pytest.approx(1.1)
    assert not any(k.startswith("update_ratio_") for k in payload)  # step 5 is off-schedule


def test_update_ratio_schedule_and_value(harness):
    system, logged, out = harness
    for s in (0, 5, 10, 25, 29, 30, 50, 70, 71, 80):
        G.run_step_sdrp_only(system, None, [], epoch=s)
    with_ur = [s for s, p in logged if any(k.startswith("update_ratio_") for k in p)]
    assert with_ur == [0, 10, 30, 50, 70, 80]
    # step 0: every tracked param is ones(n) and moves +0.1 -> ratio 0.1 / 1.0 exactly
    payload0 = dict(logged)[0]
    assert payload0["update_ratio_mechanism_head"] == pytest.approx(0.1)
    assert payload0["update_ratio_attn_layers"] == pytest.approx(0.1)
    assert payload0["update_ratio_fe_backbone"] == pytest.approx(0.1)
    assert payload0["update_ratio_fe_proj"] == pytest.approx(0.1)  # requires_grad=False param is excluded, not averaged in
    assert out.is_file()


def test_snapshot_is_read_only(harness):
    system, _, _ = harness
    before = {n: p.detach().clone() for n, p in list(system.frontend.named_parameters()) + list(system.spine.named_parameters())}
    G.run_step_sdrp_only(system, None, [], epoch=0)
    for n, p in list(system.frontend.named_parameters()) + list(system.spine.named_parameters()):
        expect = before[n] + (0.1 if p.requires_grad else 0.0)
        assert torch.allclose(p, expect)  # only the step itself changed anything


def test_logger_failure_never_breaks_step(harness, monkeypatch):
    system, logged, _ = harness

    def boom():
        raise RuntimeError("logger-side failure")

    monkeypatch.setattr(G, "_all_structure_tags", boom)  # looked up inside the logger, after the step has run
    m = G.run_step_sdrp_only(system, None, [], epoch=10)
    assert m["loss_total"] == 1.0 and logged == []  # step result returned, error swallowed, nothing half-logged
