"""Tests for aleatoric independence probe."""

from __future__ import annotations

import numpy as np

from science.training.aleatoric_independence_probe import aleatoric_independence_probe


def _rows_from_arrays(
    ale: np.ndarray,
    rho: np.ndarray,
    experts: np.ndarray,
) -> list[dict]:
    out: list[dict] = []
    for i in range(len(ale)):
        out.append(
            {
                "aleatoric": float(ale[i]),
                "rho": float(rho[i]),
                "expert": int(experts[i]),
            }
        )
    return out


def test_rho_proxy_detected() -> None:
    rng = np.random.default_rng(1)
    n = 200
    rho = rng.uniform(5.0, 20.0, size=n)
    ale = 0.05 * rho + rng.normal(0, 0.01, size=n)
    experts = rng.integers(0, 4, size=n)
    report = aleatoric_independence_probe(_rows_from_arrays(ale, rho, experts))
    assert report["verdict"] in ("rho_proxy", "proxy_fully_explained")


def test_expert_proxy_detected() -> None:
    rng = np.random.default_rng(2)
    n = 200
    rho = rng.uniform(5.0, 20.0, size=n)
    experts = rng.integers(0, 4, size=n)
    ale = experts.astype(np.float64) * 0.5 + rng.normal(0, 0.02, size=n)
    report = aleatoric_independence_probe(_rows_from_arrays(ale, rho, experts))
    assert report["verdict"] in ("expert_proxy", "proxy_fully_explained", "rho_proxy")


def test_independent_noise_candidate() -> None:
    rng = np.random.default_rng(3)
    n = 500
    rho = rng.uniform(5.0, 20.0, size=n)
    experts = rng.integers(0, 4, size=n)
    ale = rng.normal(2.0, 0.2, size=n)
    report = aleatoric_independence_probe(_rows_from_arrays(ale, rho, experts))
    assert report["verdict"] == "independent_signal_candidate"
    assert report["independent_signal_candidate"] is True


def test_not_yet_meaningful_flat() -> None:
    n = 100
    ale = np.full(n, 2.10) + np.linspace(-1e-5, 1e-5, n)
    rho = np.linspace(8.0, 18.0, n)
    experts = np.zeros(n, dtype=int)
    report = aleatoric_independence_probe(_rows_from_arrays(ale, rho, experts))
    assert report["verdict"] == "not_yet_meaningful"
    assert report["dynamic_range_ok"] is False
