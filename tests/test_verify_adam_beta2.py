"""Tests for the --adam-beta2 option of scripts/verify_grad_flow_decoupled.py (default path must stay unchanged)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

V = pytest.importorskip("verify_grad_flow_decoupled")


def test_default_path_is_the_historical_construction(monkeypatch):
    calls = []
    real = torch.optim.Adam

    def rec(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(V.torch.optim, "Adam", rec)
    w = torch.nn.Parameter(torch.ones(3))
    V._make_optimizer([{"params": [w], "lr": 1e-3}], None)
    (args, kwargs), = calls
    assert len(args) == 1 and kwargs == {}  # exactly Adam(groups): no betas keyword at all


def test_explicit_beta2_sets_betas_and_keeps_beta1(monkeypatch):
    w = torch.nn.Parameter(torch.ones(3))
    opt = V._make_optimizer([{"params": [w], "lr": 1e-3}], 0.95)
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)


def test_explicit_0999_matches_default_numerically():
    """The control (--adam-beta2 0.999) must reproduce the default path exactly on a toy problem."""
    def run(beta2):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(5))
        opt = V._make_optimizer([{"params": [w], "lr": 1e-2}], beta2)
        for i in range(25):
            opt.zero_grad()
            ((w - torch.arange(5.0)) ** 2 * (1 + 0.1 * i)).sum().backward()
            opt.step()
        return w.detach().clone()

    assert torch.equal(run(None), run(0.999))
    assert not torch.equal(run(None), run(0.9))  # and a different beta2 does change the trajectory


def test_tag_gets_beta2_suffix_only_when_given():
    assert V._beta2_tag("", None) == ""
    assert V._beta2_tag("_dcoef0.3", None) == "_dcoef0.3"
    assert V._beta2_tag("", 0.999) == "_beta20.999"  # the control never shares the baseline's name / result file
    assert V._beta2_tag("_x", 0.95) == "_x_beta20.95"
