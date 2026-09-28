"""Hinge definition v2 (see the notes section "beta2 hinge definition v2"; parameters were fixed there before this ran).

Signal x(t) = grad_l2_mechanism_head; s = trailing 5-step median; running trough m(t) = min s(u), u in [10, t-1];
hinge H = first t >= 15 with s(t') > R * m(t') for t' = t, t+1, t+2; R = 10 main (sensitivity 5, 20).
Secondary (baseline-free): first t >= 10 with clip_active == 1.  R = 10 was calibrated on the existing trajectories, so applying
it to them is a consistency check against the v1 definition, not an independent test.
"""

from __future__ import annotations

import numpy as np


def rolling_median5(x: np.ndarray) -> np.ndarray:
    s = np.full(len(x), np.nan)
    for t in range(4, len(x)):
        s[t] = np.median(x[t - 4:t + 1])
    return s


def hinge_v2(x: np.ndarray, R: float = 10.0, start_trough: int = 10, start_search: int = 15, sustain: int = 3):
    """Return the hinge step or None."""
    s = rolling_median5(np.asarray(x, dtype=float))
    n = len(s)
    for t in range(start_search, n - sustain + 1):
        ok = True
        for tp in range(t, t + sustain):
            m = np.nanmin(s[start_trough:tp])  # trough over [start_trough, tp-1]
            if not (s[tp] > R * m):
                ok = False
                break
        if ok:
            return t
    return None


def first_clip(clip_active: np.ndarray, start: int = 10):
    for t in range(start, len(clip_active)):
        if clip_active[t] >= 1.0:
            return t
    return None
