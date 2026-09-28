"""Trajectory ("hinge") analysis over logged MLflow data. Reproducible; reads only the MLflow API.

Question (operator, 2026-09-28): is the clipping burst preceded by earlier behaviour? The run looks like a plateau
(~steps 15-45) followed by an "escape" (hinge ~45-55). Two candidate explanations: (L) a landscape plateau, or (A) an
Adam second-moment artifact (beta2=0.999: the denominator is ~a uniform average of squared gradients over the first
hundreds of steps, so a large step-0 gradient shrinks the effective step size during the plateau).

OPERATIONAL DEFINITIONS, FIXED BEFORE LOOKING AT ANY ALIGNMENT (thresholds are arbitrary, so sensitivity is reported):
  plateau window P   = steps 15..30 (median of the metric over P is the run's own baseline)
  onset of metric m  = first step s > 30 such that the condition holds for 2 consecutive steps:
      gradient / norm metrics : x(s) > K * median_P(x)                   (K = 3; sensitivity 2 and 5)
      bce_per_structure_min   : x(s) < median_P(x) - D                   (D = 0.02; sensitivity 0.01 and 0.05)
      bce_per_structure_max   : x(s) > median_P(x) + U                   (U = 0.05; sensitivity 0.02 and 0.10)
      euc_skip_share (M2 only): x(s) > median_P(x) + 0.20                (sensitivity 0.10 and 0.30)
      first clip              : first step s > 30 with clip_active == 1
Independent trajectories only: the 400-step verify runs repeat the first 100 steps of the 100-step runs at the same
seed/coefficient (Spearman 0.96-0.99), so they are excluded. n = 11 runs (6 "M2" pool runs, 5 "V" verify runs).
Any number here is DESCRIPTIVE; with n = 11 and coefficient co-varying with early gradient scale, nothing is a test.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from mlflow.tracking import MlflowClient

RUNS = {
    "M2 fold0 c1.0": ("02f77ba040884f76b02005ee1e67fe74", "M2", 1.0),
    "M2 fold0 c0.3": ("a38122c791c54f3cbdb926f8b019e1c6", "M2", 0.3),
    "M2 1IVO c1.0": ("b36646842eb04134ada42a9cb1b31fc3", "M2", 1.0),
    "M2 1IVO c0.3": ("ba58175a73c3490688008775ac6424fa", "M2", 0.3),
    "M2 2Z6H c1.0": ("3cafaff898f340a7b26d19cc44e0f5e1", "M2", 1.0),
    "M2 2Z6H c0.3": ("1212cdf8f81d45caaf2630262996f4dd", "M2", 0.3),
    "V s0 c1.0": ("9352afc6cc7d4afd8fa9377d8576d57f", "V", 1.0),
    "V s0 c0.3": ("98daaf50994347d0acde10701d1ba456", "V", 0.3),
    "V s0 c1.0 mechLR0.3": ("726418c53bb547898f80b67ad4f9739e", "V", 1.0),
    "V s1 c1.0": ("f30b8567dda049799cd1efd4cc0b94a2", "V", 1.0),
    "V s1 c0.3": ("809ed49de1e84a1b99d44e91e0326bfb", "V", 0.3),
}
NSTEP = 100  # every run is logged at every step through 99 (M2: through 149)
P0, P1 = 15, 30
client = MlflowClient("http://localhost:5000")


def series(rid, key):
    h = {x.step: x.value for x in client.get_metric_history(rid, key)}
    if any(s not in h for s in range(NSTEP)):
        return None
    return np.array([h[s] for s in range(NSTEP)], dtype=float)


def load():
    keys = ["grad_l2_mechanism_head", "grad_l2_attn_layers", "grad_l2_moe", "grad_l2_sdrp_head", "preclip_norm",
            "bce_per_structure_min", "bce_per_structure_max", "loss_dehydron_bce", "clip_active",
            "euc_skip_share", "grad_l2_euc_skip", "postclip_norm"]
    data = {}
    for name, (rid, fam, _) in RUNS.items():
        data[name] = {k: series(rid, k) for k in keys}
    return data


def onset(x, kind, thr, start=P1 + 1, sustain=2):
    if x is None:
        return None
    base = float(np.median(x[P0:P1 + 1]))
    if kind == "ratio":
        cond = x > thr * base
    elif kind == "drop":
        cond = x < base - thr
    elif kind == "rise":
        cond = x > base + thr
    elif kind == "clip":
        cond = x >= 1.0
        sustain = 1
    else:
        raise ValueError(kind)
    for s in range(start, len(x) - sustain + 1):
        if bool(np.all(cond[s:s + sustain])):
            return s
    return None


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def cv(v):
    v = np.asarray(v, float)
    return float(np.std(v) / np.mean(v)) if np.mean(v) else float("nan")


ONSETS = {  # metric -> (kind, main threshold, sensitivity thresholds)
    "grad_mech": ("grad_l2_mechanism_head", "ratio", 3.0, (2.0, 5.0)),
    "grad_attn": ("grad_l2_attn_layers", "ratio", 3.0, (2.0, 5.0)),
    "grad_moe": ("grad_l2_moe", "ratio", 3.0, (2.0, 5.0)),
    "grad_sdrp": ("grad_l2_sdrp_head", "ratio", 3.0, (2.0, 5.0)),
    "preclip": ("preclip_norm", "ratio", 3.0, (2.0, 5.0)),
    "bce_min_drop": ("bce_per_structure_min", "drop", 0.02, (0.01, 0.05)),
    "bce_max_rise": ("bce_per_structure_max", "rise", 0.05, (0.02, 0.10)),
    "euc_share_rise": ("euc_skip_share", "rise", 0.20, (0.10, 0.30)),
    "grad_euc_skip": ("grad_l2_euc_skip", "ratio", 3.0, (2.0, 5.0)),
    "first_clip": ("clip_active", "clip", 1.0, ()),
}


def main():
    data = load()
    out = {}
    print("=" * 100)
    print("0. Plateau stationarity check (max/min of each metric over steps 15-30; large = weak plateau assumption)")
    for name in RUNS:
        d = data[name]
        g = d["grad_l2_mechanism_head"][P0:P1 + 1]
        bmin = d["bce_per_structure_min"][P0:P1 + 1]
        print(f"  {name:22s} grad_mech max/min over P = {g.max() / g.min():5.2f} | bce_min over P = {bmin.min():.3f}..{bmin.max():.3f} "
              f"| bce_max over P = {d['bce_per_structure_max'][P0:P1 + 1].min():.3f}..{d['bce_per_structure_max'][P0:P1 + 1].max():.3f}")

    print("\n" + "=" * 100)
    print("1. ONSET STEPS per run (main thresholds); None = not detected within the logged window")
    hdr = ["grad_mech", "bce_min_drop", "bce_max_rise", "preclip", "first_clip", "grad_attn", "grad_moe", "grad_sdrp", "euc_share_rise", "grad_euc_skip"]
    print(f"  {'run':22s} " + " ".join(f"{h[:12]:>12s}" for h in hdr))
    table = {}
    for name in RUNS:
        row = {}
        for h in hdr:
            key, kind, thr, _ = ONSETS[h]
            row[h] = onset(data[name][key], kind, thr)
        table[name] = row
        print(f"  {name:22s} " + " ".join(f"{str(row[h]):>12s}" for h in hdr))
    out["onsets_main"] = table

    print("\n  Sensitivity of the mechanism-head onset and bce_min drop onset to thresholds:")
    for name in RUNS:
        gm = [onset(data[name]["grad_l2_mechanism_head"], "ratio", t) for t in (2.0, 3.0, 5.0)]
        bm = [onset(data[name]["bce_per_structure_min"], "drop", t) for t in (0.01, 0.02, 0.05)]
        print(f"  {name:22s} grad_mech K=2/3/5: {gm} | bce_min D=.01/.02/.05: {bm}")

    print("\n" + "=" * 100)
    print("2a. WHICH MOVES FIRST: onset(metric) - onset(grad_mech), steps (negative = metric leads the mechanism-head jump)")
    others = ["bce_min_drop", "bce_max_rise", "preclip", "first_clip", "grad_attn", "grad_moe", "grad_sdrp", "euc_share_rise", "grad_euc_skip"]
    for h in others:
        lags = []
        for name in RUNS:
            a, b = table[name][h], table[name]["grad_mech"]
            if a is not None and b is not None:
                lags.append(a - b)
        if lags:
            lead = sum(1 for l in lags if l < -1)
            tie = sum(1 for l in lags if -1 <= l <= 1)
            lag = sum(1 for l in lags if l > 1)
            print(f"  {h:15s} n={len(lags):2d} median lag {np.median(lags):+5.1f} (min {min(lags):+d}, max {max(lags):+d}) | leads(<-1): {lead}, within +-1: {tie}, lags(>+1): {lag}")

    print("\n2b. Cross-correlation over steps 35-60 (z-scored, log for gradients), lag k: corr(x[t], y[t+k]); positive peak lag = y lags x")
    w0, w1 = 35, 60

    def z(v, log=False):
        v = np.log(np.maximum(v[w0:w1 + 1], 1e-12)) if log else v[w0:w1 + 1]
        return (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)

    pairs = [("grad_l2_mechanism_head", True, "bce_per_structure_min", False),
             ("grad_l2_mechanism_head", True, "bce_per_structure_max", False),
             ("bce_per_structure_max", False, "bce_per_structure_min", False)]
    for xk, xl, yk, yl in pairs:
        peaks = []
        for name in RUNS:
            x, y = z(data[name][xk], xl), z(data[name][yk], yl)
            best = max(range(-8, 9), key=lambda k: abs(np.mean(x[max(0, -k):len(x) - max(0, k)] * y[max(0, k):len(y) - max(0, -k)])))
            xs = x[max(0, -best):len(x) - max(0, best)]
            ys = y[max(0, best):len(y) - max(0, -best)]
            peaks.append((best, float(np.mean(xs * ys))))
        print(f"  x={xk[:24]:24s} y={yk[:22]:22s} peak lags {[p[0] for p in peaks]} | corr at peak {[round(p[1], 2) for p in peaks]}")

    print("\n" + "=" * 100)
    print("3. EARLY GRADIENT ENERGY vs HINGE STEP (hinge = mechanism-head onset, main threshold)")
    rows = []
    for name in RUNS:
        d = data[name]
        e_pre = float(np.sum(d["preclip_norm"][0:11] ** 2))
        e_mech = float(np.sum(d["grad_l2_mechanism_head"][0:11] ** 2))
        gpl = float(np.median(d["grad_l2_mechanism_head"][P0:P1 + 1]))
        h = table[name]["grad_mech"]
        rows.append((name, RUNS[name][1], RUNS[name][2], e_pre, e_mech, gpl, h))
        print(f"  {name:22s} coeff {RUNS[name][2]:.1f}  E0_preclip={e_pre:7.3f}  E0_mech={e_mech:7.3f}  g_plateau_mech={gpl:.4f}  hinge={h}")
    ok = [r for r in rows if r[6] is not None]
    hs = [r[6] for r in ok]
    print(f"  Spearman(E0_preclip, hinge) = {spearman([r[3] for r in ok], hs):+.2f} | Spearman(E0_mech, hinge) = {spearman([r[4] for r in ok], hs):+.2f} (n={len(ok)})")
    for c in (1.0, 0.3):
        sub = [r for r in ok if r[2] == c]
        if len(sub) >= 3:
            print(f"    within coefficient {c}: Spearman(E0_preclip, hinge) = {spearman([r[3] for r in sub], [r[6] for r in sub]):+.2f}, "
                  f"Spearman(E0_mech, hinge) = {spearman([r[4] for r in sub], [r[6] for r in sub]):+.2f} (n={len(sub)})")
    # Adam-artifact quantitative prediction: hinge ~ E_early / g_plateau^2 (log-log slope ~ 1 expected if (A) holds)
    pred = np.log([r[4] / r[5] ** 2 for r in ok])
    lh = np.log(hs)
    slope = float(np.polyfit(pred, lh, 1)[0])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(ok), len(ok))
        if len(set(idx)) > 2:
            bs.append(np.polyfit(pred[idx], lh[idx], 1)[0])
    print(f"  Adam-artifact prediction hinge ~ E0_mech/g_plateau^2: Spearman = {spearman(pred, lh):+.2f}; log-log slope = {slope:+.2f} "
          f"(bootstrap 95% CI {np.percentile(bs, 2.5):+.2f}..{np.percentile(bs, 97.5):+.2f}; (A) predicts ~ +1)")
    out["hinge"] = {r[0]: r[6] for r in rows}

    print("\n" + "=" * 100)
    print("4. OPTIMIZER + effective-step proxy")
    print("  Both runners: torch.optim.Adam(groups) -> betas=(0.9, 0.999), eps=1e-8, weight_decay=0 (PyTorch defaults); gradient clipping (max_grad_norm 1.0) is applied BEFORE Adam.")
    print(f"  beta2^40 = {0.999 ** 40:.3f}: at step 40 the step-0 gradient still carries {0.999 ** 40:.1%} of its original weight in v_hat (i.e. ~uniform average).")
    print("  Norm-level proxy r_t = |g_t| / sqrt(mean_{i<=t} g_i^2), using the clipped total-gradient norm (postclip_norm for M2; min(preclip,1) for V).")
    print("  Adam is per-parameter, so this is only an order-of-magnitude proxy. r_t ~ effective step / nominal step.")
    rat = {}
    for name in RUNS:
        d = data[name]
        g = d["postclip_norm"][:NSTEP] if d["postclip_norm"] is not None else np.minimum(d["preclip_norm"], 1.0)
        m = np.sqrt(np.cumsum(g ** 2) / np.arange(1, NSTEP + 1))
        r = g / m
        rat[name] = r
        h = table[name]["grad_mech"]
        print(f"  {name:22s} median r over steps 15-45 = {np.median(r[15:46]):.3f} | r at step 20/30/40/50 = {r[20]:.2f}/{r[30]:.2f}/{r[40]:.2f}/{r[50]:.2f}"
              + (f" | r at hinge({h}) = {r[h]:.2f}" if h is not None else ""))
    hr = [rat[n][table[n]['grad_mech']] for n in RUNS if table[n]['grad_mech'] is not None]
    r40 = [rat[n][40] for n in RUNS]
    print(f"  CV of hinge step = {cv(hs):.2f} | CV of r at the hinge = {cv(hr):.2f} | CV of r at fixed step 40 = {cv(r40):.2f}")
    print("  (a run's r at the hinge being MORE similar across runs than the hinge step is weak evidence that a threshold in effective step matters)")

    print("\n" + "=" * 100)
    print("5. PROGRESS-VARIABLE ALIGNMENT (bce_per_structure_min and loss_dehydron_bce at the hinge, vs the same quantity at fixed plateau steps)")
    vals_h_min = [data[n]["bce_per_structure_min"][table[n]["grad_mech"]] for n in RUNS if table[n]["grad_mech"] is not None]
    vals_h_loss = [data[n]["loss_dehydron_bce"][table[n]["grad_mech"]] for n in RUNS if table[n]["grad_mech"] is not None]
    vals_p_min = [float(np.median(data[n]["bce_per_structure_min"][P0:P1 + 1])) for n in RUNS]
    vals_p_loss = [float(np.median(data[n]["loss_dehydron_bce"][P0:P1 + 1])) for n in RUNS]
    print(f"  bce_min at hinge: mean {np.mean(vals_h_min):.3f}, CV {cv(vals_h_min):.3f} | bce_min median over plateau P: mean {np.mean(vals_p_min):.3f}, CV {cv(vals_p_min):.3f}")
    print(f"  loss_dehydron_bce at hinge: mean {np.mean(vals_h_loss):.3f}, CV {cv(vals_h_loss):.3f} | over plateau P: mean {np.mean(vals_p_loss):.3f}, CV {cv(vals_p_loss):.3f}")
    print("  CAVEAT: on a flat plateau the loss is ~constant, so alignment 'at the hinge' in loss space is near-guaranteed and cannot discriminate (L) from (A).")

    print("\n" + "=" * 100)
    print("6. NOISE-SCALED ONSETS (POST HOC: added after section 0 showed plateau noise ~ the fixed bce_min thresholds; z chosen after seeing that z>=4 rarely triggers for gradients)")
    print("  onset = first step > 30 where the metric leaves its own plateau median by z robust sigmas (1.4826*MAD over steps 15-30;")
    print("  gradients on log scale, bce_min downward), sustained 2 steps. Same z applied to both metrics.")

    def zonset(x, z, direction, log=False):
        v = np.log(np.maximum(x, 1e-12)) if log else x
        base = np.median(v[P0:P1 + 1])
        sig = 1.4826 * np.median(np.abs(v[P0:P1 + 1] - base))
        cond = (v > base + z * sig) if direction == "up" else (v < base - z * sig)
        for s in range(P1 + 1, len(v) - 1):
            if cond[s] and cond[s + 1]:
                return s
        return None

    for z in (2.0, 3.0, 4.0):
        lags, gs, bs_ = [], [], []
        for name in RUNS:
            g = zonset(data[name]["grad_l2_mechanism_head"], z, "up", log=True)
            b = zonset(data[name]["bce_per_structure_min"], z, "down")
            gs.append(g); bs_.append(b)
            if g is not None and b is not None:
                lags.append(b - g)
        lead = sum(1 for l in lags if l < 0); tie = sum(1 for l in lags if l == 0); lag = sum(1 for l in lags if l > 0)
        if lags:
            print(f"  z={z:.0f}: bce_min onset - grad_mech onset: n={len(lags)} median {np.median(lags):+.1f} range {min(lags):+d}..{max(lags):+d} | earlier: {lead}, same: {tie}, later: {lag}")
        else:
            print(f"  z={z:.0f}: no run has both onsets detected")
        print(f"        grad onsets {gs}\n        bce_min onsets {bs_}")
    print("  Hinge by coefficient (main K=3 definition): "
          + ", ".join(f"c{c}: {sorted(h for n, h in out['hinge'].items() if RUNS[n][2] == c)}" for c in (1.0, 0.3)))
    for c in (1.0, 0.3):
        v = [data[n]["bce_per_structure_min"][out['hinge'][n]] for n in RUNS if RUNS[n][2] == c]
        print(f"  bce_min value at hinge, coefficient {c}: {[round(x, 3) for x in v]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(17, 9))
        cols = {("M2", 1.0): "tab:red", ("M2", 0.3): "tab:blue", ("V", 1.0): "tab:orange", ("V", 0.3): "tab:cyan"}
        for name, (rid, fam, c) in RUNS.items():
            d = data[name]
            col = cols[(fam, c)]
            ls = "--" if "mechLR" in name else "-"
            s = np.arange(NSTEP)
            ax[0, 0].plot(s, d["grad_l2_mechanism_head"], color=col, ls=ls, alpha=.7, lw=1)
            ax[0, 1].plot(s, d["bce_per_structure_min"], color=col, ls=ls, alpha=.7, lw=1)
            ax[0, 2].plot(s, d["bce_per_structure_max"], color=col, ls=ls, alpha=.7, lw=1)
            ax[1, 0].plot(d["bce_per_structure_min"], d["grad_l2_mechanism_head"], color=col, ls=ls, alpha=.6, lw=1)
            ax[1, 1].plot(s, rat[name], color=col, ls=ls, alpha=.7, lw=1)
            ax[1, 2].plot(s, d["preclip_norm"], color=col, ls=ls, alpha=.7, lw=1)
        ax[0, 0].set_yscale("log"); ax[0, 0].set_title("grad_l2_mechanism_head vs step")
        ax[0, 1].set_title("bce_per_structure_min vs step"); ax[0, 2].set_title("bce_per_structure_max vs step")
        ax[1, 0].set_yscale("log"); ax[1, 0].set_title("grad_mech vs bce_per_structure_min (progress-variable x)"); ax[1, 0].invert_xaxis()
        ax[1, 1].set_title("Adam effective-step proxy r_t (norm level)"); ax[1, 2].set_title("preclip_norm vs step"); ax[1, 2].set_yscale("log")
        for a in ax.flat[[0, 1, 2, 4, 5]]:
            a.axvspan(P0, P1, color="grey", alpha=.1); a.set_xlabel("step")
        for k, v in {"M2 c1.0": "tab:red", "M2 c0.3": "tab:blue", "V c1.0": "tab:orange", "V c0.3": "tab:cyan"}.items():
            ax[0, 0].plot([], [], color=v, label=k)
        ax[0, 0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig("/workspace/experiments/trajectory_overlay.png", dpi=110)
        print("\nSaved overlay figure: experiments/trajectory_overlay.png (untracked)")
    except Exception as exc:  # noqa: BLE001
        print(f"\n(figure skipped: {type(exc).__name__}: {exc})")
    json.dump(out, open("/tmp/trajectory_hinge_results.json", "w"), default=str)


if __name__ == "__main__":
    sys.exit(main())
