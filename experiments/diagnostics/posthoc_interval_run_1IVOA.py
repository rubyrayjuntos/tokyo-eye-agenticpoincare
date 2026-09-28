"""POST HOC, EXPLORATORY analysis of the interval run a003d87d (1IVO:A, c=0.3, seed 0). ONE run: everything is descriptive.

Not pre-registered. The pre-registered readouts (held-out F1, train mean at the eval steps) were read separately.
Questions (operator 2026-09-28): (a) do per-bucket update ratios change around the hinge (steps ~45-60) and, since the SDRP
argmax escape trails the hinge, across steps 75-150; (b) do the 11 per-structure BCE traces show the same structure
leading, and does the order in which structures' BCE falls relate to the order in which their train F1 leaves the floor?

Definitions fixed before running:
  update-ratio windows: W1 = steps 30-44 (trough), W2 = steps 45-60 (hinge), W3 = steps 61-70 (dense window ends at 70);
                        then the every-10 steps 80..150 individually.
  BCE-fall step of structure t: first step s >= 31 at which the 5-step rolling median of bce_struct_t is below
                        (median of bce_struct_t over steps 15-30) - 0.05 (sensitivity: 0.03, 0.08).
  train-F1 escape of structure t: interval_train_f1_t at step 100 (and 150) minus its value at step 75 (all structures are at their
                        all-majority floor at step 75, so this is the gain over the floor).
"""

from __future__ import annotations

import numpy as np
from mlflow.tracking import MlflowClient

RID = "a003d87def7c4be88ff2ef4277aae441"
c = MlflowClient("http://localhost:5000")
run = c.get_run(RID)


def hist(k):
    return {x.step: x.value for x in c.get_metric_history(RID, k)}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


buckets = sorted(k[len("update_ratio_"):] for k in run.data.metrics if k.startswith("update_ratio_"))
ur = {b: hist(f"update_ratio_{b}") for b in buckets}
print("=" * 100)
print("A. UPDATE RATIO ||d theta||/||theta|| per bucket (window medians, then single steps)")
wins = {"W1 30-44": range(30, 45), "W2 45-60": range(45, 61), "W3 61-70": range(61, 71)}
singles = [80, 90, 100, 110, 120, 130, 140, 150, 200, 300, 390]
hdr = ["step0"] + list(wins) + [f"s{s}" for s in singles]
print(f"  {'bucket':15s} " + " ".join(f"{h:>9s}" for h in hdr))
for b in buckets:
    h = ur[b]
    row = [h.get(0, np.nan)] + [float(np.median([h[s] for s in r if s in h])) for r in wins.values()] + [h.get(s, np.nan) for s in singles]
    print(f"  {b:15s} " + " ".join(f"{v:9.4f}" for v in row))
print("  ratios W2/W1 (hinge vs trough) and s100/W1, s150/W1:")
for b in buckets:
    h = ur[b]
    w1 = np.median([h[s] for s in wins["W1 30-44"]])
    w2 = np.median([h[s] for s in wins["W2 45-60"]])
    print(f"  {b:15s} W2/W1 = {w2 / w1:5.2f}   s100/W1 = {h[100] / w1:5.2f}   s150/W1 = {h[150] / w1:5.2f}")

print("\n" + "=" * 100)
print("B. PER-STRUCTURE BCE: which structure is lowest, and in what order do they fall")
tags = sorted(k[len("bce_struct_"):] for k in run.data.metrics if k.startswith("bce_struct_"))
bce = {t: hist(f"bce_struct_{t}") for t in tags}
top = min(len(v) for v in bce.values()) - 1
arr = {t: np.array([bce[t][s] for s in range(top + 1)]) for t in tags}
# identity of the lowest-BCE structure at each step 40..70
counts: dict[str, int] = {}
for s in range(40, 71):
    t = min(tags, key=lambda t: arr[t][s])
    counts[t] = counts.get(t, 0) + 1
print("  lowest-BCE structure at each step 40-70 (count of 31 steps):", dict(sorted(counts.items(), key=lambda kv: -kv[1])))


def fall(x, thr):
    base = np.median(x[15:31])
    for s in range(31, len(x) - 4):
        if np.median(x[s - 4:s + 1]) < base - thr:
            return s
    return None


falls = {thr: {t: fall(arr[t], thr) for t in tags} for thr in (0.03, 0.05, 0.08)}
print("  BCE-fall step per structure (threshold 0.03 / 0.05 / 0.08):")
for t in tags:
    print(f"   {t:7s} {falls[0.03][t]!s:>5s} {falls[0.05][t]!s:>5s} {falls[0.08][t]!s:>5s}")

print("\n" + "=" * 100)
print("C. TRAIN-STRUCTURE F1 ESCAPE vs BCE-FALL ORDER (n = 11 structures, one run)")
f1 = {t: hist(f"interval_train_f1_{t}") for t in tags}
gain = {s: {t: f1[t][s] - f1[t][75] for t in tags} for s in (100, 150)}
print(f"  {'struct':7s} {'F1@75':>7s} {'gain@100':>9s} {'gain@150':>9s} {'BCEfall.05':>10s}")
for t in tags:
    print(f"  {t:7s} {f1[t][75]:7.4f} {gain[100][t]:9.4f} {gain[150][t]:9.4f} {falls[0.05][t]!s:>10s}")
for thr in (0.03, 0.05, 0.08):
    ok = [t for t in tags if falls[thr][t] is not None]
    if len(ok) >= 5:
        print(f"  Spearman(BCE-fall step, F1 gain@100), thr {thr}: {spearman([falls[thr][t] for t in ok], [gain[100][t] for t in ok]):+.2f}; "
              f"gain@150: {spearman([falls[thr][t] for t in ok], [gain[150][t] for t in ok]):+.2f} (n={len(ok)}; negative = earlier BCE fall, larger F1 gain)")

print("\n" + "=" * 100)
print("D. SDRP CE and dehydron BCE over time (window means)")
for k in ("loss_sdrp_ce", "loss_dehydron_bce"):
    h = hist(k)
    print(f"  {k}: " + ", ".join(f"{a}-{b}: {np.mean([h[s] for s in range(a, b + 1) if s in h]):.3f}" for a, b in
                                  ((0, 24), (25, 49), (50, 74), (75, 99), (100, 149), (150, 199), (200, 299), (300, 399))))
