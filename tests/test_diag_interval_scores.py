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


# ---- hand-computed cases (expected values worked out by pairwise counting, not by sklearn) ----

def _two_class_logits(p0):
    """logits over 2 classes whose softmax gives P(class 0) = p0 exactly."""
    d = np.log(np.asarray(p0) / (1.0 - np.asarray(p0)))
    return np.stack([d / 2.0, -d / 2.0], axis=1)


def test_hand_computed_auroc_three_quarters():
    # y = [0,1,0,1], P(class0) = [.9,.8,.3,.2].
    # class 0: positives {.9,.3} vs negatives {.8,.2}: pairs .9>.8 yes, .9>.2 yes, .3>.8 no, .3>.2 yes -> 3/4
    # class 1: P(class1) = [.1,.2,.7,.8]; positives {.2,.8} vs negatives {.1,.7}: yes, no, yes, yes -> 3/4
    m = D._continuous_metrics(_two_class_logits([0.9, 0.8, 0.3, 0.2]), np.array([0, 1, 0, 1]))
    assert m["auroc"][0] == pytest.approx(0.75) and m["auroc"][1] == pytest.approx(0.75)
    assert m["macro_auroc"] == pytest.approx(0.75)


def test_reversed_ranking_gives_zero():
    y = np.array([0, 0, 1, 1])
    m = D._continuous_metrics(_two_class_logits([0.1, 0.2, 0.8, 0.9]), y)  # class-0 nodes get LOW P(class 0)
    assert m["macro_auroc"] == pytest.approx(0.0)


def test_all_tied_scores_give_half_auroc_and_prevalence_auprc():
    y = np.array([0, 0, 0, 1])
    m = D._continuous_metrics(np.zeros((4, 2)), y)  # every probability is exactly 0.5
    assert m["auroc"][0] == pytest.approx(0.5) and m["auroc"][1] == pytest.approx(0.5)
    assert m["auprc"][0] == pytest.approx(0.75) and m["auprc"][1] == pytest.approx(0.25)  # average precision of a constant score = prevalence


def test_class_with_no_positives_is_skipped_not_nan():
    y = np.array([0, 0, 1, 1])
    m = D._continuous_metrics(np.zeros((4, 5)), y)  # classes 2-4 never occur
    assert set(m["auroc"]) == {0, 1} and np.isfinite(m["macro_auroc"])


def test_hand_computed_mechanism_auroc_and_ap():
    # scores [.9,.8,.7,.1], labels [1,0,1,0]. AUROC: positives {.9,.7} vs negatives {.8,.1}: yes, yes, no, yes -> 3/4.
    # AP: ranked descending the positives sit at ranks 1 and 3 -> precision 1/1 and 2/3, AP = (1 + 2/3)/2 = 5/6.
    m = D._continuous_metrics(np.zeros((4, 5)), np.array([0, 1, 0, 1]),
                              np.array([0.9, 0.8, 0.7, 0.1]), np.array([1.0, 0.0, 1.0, 0.0]))
    assert m["mech_auroc"] == pytest.approx(0.75)
    assert m["mech_auprc"] == pytest.approx(5.0 / 6.0)


def test_variant_tags_reach_diagnostic_runs_only(monkeypatch):
    monkeypatch.setattr(D, "_EXTRA_TAGS", {"variant": "interval_scores", "results_suffix": "_interval"})
    tagged = D._diag_tags({"diagnostic": "false", "card": "m2_pool_decoupled", "gate_id": "x_prereg"})
    assert tagged["variant"] == "interval_scores" and tagged["results_suffix"] == "_interval"
    assert tagged["diagnostic"] == "true" and tagged["card"].startswith("DIAG_") and tagged["gate_id"].startswith("NOT_A_GATE")
    other = D._diag_tags({"some": "tag"})  # a set_tags call from elsewhere is left alone
    assert "variant" not in other
