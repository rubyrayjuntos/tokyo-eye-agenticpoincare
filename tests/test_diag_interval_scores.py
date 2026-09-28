"""Unit tests for the continuous held-out metrics used by the diag wrapper's interval eval."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

D = pytest.importorskip("wrap1_zhyp_m2_pool_decoupled_diag")


def test_perfectly_separable_scores_give_auroc_one():
    y = np.array([0, 0, 1, 1, 4, 4, 4, 4])
    logits = np.full((8, 5), -5.0)
    logits[np.arange(8), y] = 5.0
    m = D._continuous_metrics(logits, y)
    assert set(m["auroc"]) == {0, 1, 4}  # classes 2 and 3 are absent from y and are skipped
    assert m["macro_auroc"] == pytest.approx(1.0) and m["macro_auprc"] == pytest.approx(1.0)


def test_matches_sklearn_on_random_scores():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(3)
    y = rng.choice([0, 1, 4], size=200, p=[0.1, 0.1, 0.8])
    logits = rng.normal(size=(200, 5)) + np.eye(5)[y] * 0.7
    m = D._continuous_metrics(logits, y)
    e = np.exp(logits - logits.max(1, keepdims=True))
    prob = e / e.sum(1, keepdims=True)
    for k in (0, 1, 4):
        assert m["auroc"][k] == pytest.approx(roc_auc_score(y == k, prob[:, k]))
        assert m["auprc"][k] == pytest.approx(average_precision_score(y == k, prob[:, k]))
    assert m["macro_auroc"] == pytest.approx(np.mean([m["auroc"][k] for k in (0, 1, 4)]))


def test_continuous_metrics_move_when_argmax_does_not():
    """Continuous metrics respond to small score changes that leave every argmax (and thus F1) unchanged."""
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1, 4], size=300, p=[0.1, 0.1, 0.8])
    logits = rng.normal(size=(300, 5))
    logits[:, 4] += 8.0  # majority argmax everywhere, with a margin no small perturbation crosses
    a = D._continuous_metrics(logits, y)
    logits2 = logits + rng.normal(scale=0.05, size=logits.shape)
    assert (logits.argmax(1) == logits2.argmax(1)).all()
    b = D._continuous_metrics(logits2, y)
    assert a["macro_auroc"] != b["macro_auroc"]


def test_mechanism_metrics_only_when_both_label_values_present():
    y = np.array([0, 1, 4, 4])
    logits = np.zeros((4, 5))
    m = D._continuous_metrics(logits, y, np.array([0.1, 0.9, 0.2, 0.8]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert m["mech_auroc"] == pytest.approx(1.0)
    m2 = D._continuous_metrics(logits, y, np.array([0.1, 0.9, 0.2, 0.8]), np.zeros(4))
    assert "mech_auroc" not in m2
