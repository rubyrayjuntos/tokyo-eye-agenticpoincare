"""
epistemic_decoupling_baseline.py
================================
Baseline + go/no-go diagnostic for STEP4_EPISTEMIC_DEPTH_DECOUPLING_PREREG.md.
This script answers three questions BEFORE any GPU run, and it is itself a
GATE on whether Step 4 is worth running:
  Q1 (GO/NO-GO on the referent): after partialling depth out of B-factor, is
     there any residual B-factor variance left for a decoupled epistemic head
     to chase? If depth already explains ~all of B-factor, the referent is dead
     and Step 4 must pick a different one.
  Q2 (baseline to beat): what is corr(e_resid, b_resid) for the CURRENT
     (collinear) epistemic head? Step 4's pass = improvement over THIS number.
  Q3 (noise band): what is the run-to-run / sampling noise on that baseline, so
     the Step 4 pass threshold is "baseline + margin beyond noise," not a round
     number like 0.15.
CRITICAL correctness requirements (each is a way the endpoint gets silently
compromised if omitted):
  * B-factors are NOT comparable across structures (resolution/refinement scale
    differ). All correlations are computed on PER-STRUCTURE z-scored B-factors
    and aggregated across structures — never pooled raw. Pooling raw B-factors
    measures "which crystal was better," not per-residue uncertainty.
  * Missing-residue bias: crystal gaps fall on disordered loops / flexible
    termini — exactly the high-uncertainty residues epistemic should care about.
    A structure can pass a 90% coverage gate while lacking B-factors for the
    residues that matter most. This script reports WHERE the gaps fall, not just
    how many.
  * The partial correlation removes depth's LINEAR contribution from both
    variables. If epistemic is ~collinear with depth (the premise: r~0.99), its
    depth-residual is mostly noise, so the baseline corr(e_resid,b_resid) should
    be near zero — that near-zero is the thing Step 4 tries to raise.
Data loaders are stubs (marked # WIRE). Statistics are verified by --selftest.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_CHECKPOINT = Path(
    "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"
)
DEFAULT_PDB_DIR = Path("/tmp/dtie_pdb_cache")
DEFAULT_STRUCTURES = ["11QE", "4OBE", "1IVO", "4MNE"]

# ---------------------------------------------------------------------------
# GO/NO-GO thresholds — documented, not magic. Calibrate/confirm, don't trust.
# ---------------------------------------------------------------------------
# Residual B-factor signal is "usable" if, after regressing out depth, the
# normalized B-factor still has at least this fraction of its variance left.
# 0.25 is a starting proposal: if depth explains >75% of normalized B-factor,
# there is little independent signal for epistemic to add. CONFIRM against the
# corpus output before locking (same discipline as NMI_stop = p90).
MIN_RESIDUAL_BFACTOR_VAR_FRAC = 0.25
# Coverage gate (from the pre-reg). Reported alongside gap-location bias.
MIN_BFACTOR_COVERAGE = 0.90
# Bootstrap resamples for the noise band on the baseline partial correlation.
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# ===========================================================================
# Verified statistics core
# ===========================================================================
def zscore(x: np.ndarray) -> np.ndarray:
    """Per-structure normalization. Ignores NaN in mean/std."""
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return np.full_like(x, np.nan, dtype=float)
    return (x - m) / s


def _ols_residual(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residual of y after removing the best linear fit on z (with intercept).
    z may be 1D (single covariate) or 2D (n, k)."""
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    X = np.column_stack([np.ones(len(y)), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """corr(x, y | z): correlation of x and y after linearly removing z from
    both. Returns NaN if either residual has no variance."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 5:
        return float("nan")
    rx = _ols_residual(x[mask], z[mask])
    ry = _ols_residual(y[mask], z[mask])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def residual_variance_fraction(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Fraction of var(y) that remains after regressing out z, and R²(z->y).
    This is the GO/NO-GO quantity for the B-factor referent: 'is there anything
    left once depth is removed?'"""
    mask = np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    yv = y[mask]
    resid = _ols_residual(yv, z[mask])
    total_var = np.var(yv)
    if total_var == 0:
        return float("nan"), float("nan")
    resid_frac = float(np.var(resid) / total_var)
    r2 = 1.0 - resid_frac
    return resid_frac, r2


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | list[float]]:
    """Percentile CI + std of partial_corr(x,y|z) by resampling residues.
    Gives the noise band the Step 4 threshold must clear."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 10:
        return {
            "point": float("nan"),
            "ci95": [float("nan")] * 2,
            "std": float("nan"),
        }
    rng = np.random.default_rng(seed)
    point = partial_corr(x, y, z)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(partial_corr(x[idx], y[idx], z[idx]))
    boots = np.array([b for b in boots if np.isfinite(b)])
    return {
        "point": float(point),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "std": float(np.std(boots)),
    }


def gap_bias(
    present_mask: np.ndarray,
    sasa: np.ndarray,
    depth: np.ndarray,
    ss_is_coil: np.ndarray | None,
) -> dict[str, float | int | str]:
    """Where do missing-B-factor residues fall? Compares gap residues to present
    residues on SASA / depth / coil-fraction. Large positive SASA diff => gaps
    are surface/flexible => referent biased toward easy (buried) residues."""
    gap = ~present_mask
    out: dict[str, float | int | str] = {
        "coverage": float(present_mask.mean()),
        "n_gap": int(gap.sum()),
    }
    if gap.sum() == 0:
        out["note"] = "no gaps"
        return out

    def safe_mean(a: np.ndarray, m: np.ndarray) -> float:
        v = a[m]
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    out["sasa_gap_minus_present"] = safe_mean(sasa, gap) - safe_mean(sasa, present_mask)
    out["depth_gap_minus_present"] = safe_mean(depth, gap) - safe_mean(depth, present_mask)
    if ss_is_coil is not None:
        out["coil_frac_gap"] = safe_mean(ss_is_coil.astype(float), gap)
        out["coil_frac_present"] = safe_mean(ss_is_coil.astype(float), present_mask)
    return out


# ===========================================================================
# Per-structure baseline
# ===========================================================================
@dataclass
class StructureBaseline:
    structure_id: str
    n_residues: int
    coverage: float
    gap_bias: dict[str, float | int | str]
    r_epi_depth_raw: float
    r_epi_sasa_raw: float
    partial_epi_sasa_given_depth: float
    # corr(bf_resid, sasa_resid | depth) — same as partial_corr(bf_z, sasa, depth)
    corr_bf_resid_sasa_resid_given_depth: float
    r2_depth_to_bfactor: float  # how much depth explains B-factor
    residual_bfactor_var_frac: float  # GO/NO-GO: signal left after depth
    corr_depth_bfactor_raw: float
    corr_epi_bfactor_raw: float  # trivially high if epi~depth~bfactor
    baseline_partial_corr: dict[str, float | list[float]]
    referent_usable: bool
    structure_note: str = ""


def compute_structure_baseline(
    structure_id: str,
    epi: np.ndarray,
    depth: np.ndarray,
    bfactor_raw: np.ndarray,
    sasa: np.ndarray,
    ss_is_coil: np.ndarray | None,
    present_mask: np.ndarray,
) -> StructureBaseline:
    bf = zscore(np.where(present_mask, bfactor_raw, np.nan))  # per-structure norm
    resid_frac, r2 = residual_variance_fraction(bf, depth)
    band = bootstrap_ci(epi, bf, depth)  # corr(epi,bf|depth)
    usable = (
        np.isfinite(resid_frac)
        and resid_frac >= MIN_RESIDUAL_BFACTOR_VAR_FRAC
        and present_mask.mean() >= MIN_BFACTOR_COVERAGE
    )

    def raw_corr(a: np.ndarray, b: np.ndarray) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 5 else float("nan")

    bf_sasa_partial = partial_corr(bf, sasa, depth)

    return StructureBaseline(
        structure_id=structure_id,
        n_residues=int(len(epi)),
        coverage=float(present_mask.mean()),
        gap_bias=gap_bias(present_mask, sasa, depth, ss_is_coil),
        r_epi_depth_raw=raw_corr(epi, depth),
        r_epi_sasa_raw=raw_corr(epi, sasa),
        partial_epi_sasa_given_depth=partial_corr(epi, sasa, depth),
        corr_bf_resid_sasa_resid_given_depth=bf_sasa_partial,
        r2_depth_to_bfactor=r2,
        residual_bfactor_var_frac=resid_frac,
        corr_depth_bfactor_raw=raw_corr(depth, bf),
        corr_epi_bfactor_raw=raw_corr(epi, bf),
        baseline_partial_corr=band,
        referent_usable=bool(usable),
        structure_note=_structure_note(structure_id),
    )


def _structure_note(structure_id: str) -> str:
    notes = {
        "11QE": "KRAS G12D/I55E suppressor (PDB TITLE); allosteric constraint redistribution — expect elevated mobility coupling vs WT KRAS",
    }
    return notes.get(structure_id.upper(), "")


def _suggest_lambda_schedule(mean_bf_sasa_resid_corr: float) -> dict[str, float | str]:
    """λ₁/λ₂ hint for Step 4 — **not authoritative** (see INVALID note below).

    Empirical v2/v3 runs showed that when bf_resid and sasa_resid co-vary and both
    route through a single ν, raising λ₂ above λ₁ destroys B-factor alignment; do not
    use the legacy collinearity branch for scheduling.
    """
    if not np.isfinite(mean_bf_sasa_resid_corr):
        lam1, lam2 = 0.25, 0.05
        rationale = (
            "bf-sasa residual correlation unavailable; default λ₁=0.25, λ₂=0.05 "
            "(conservative — do not use symmetric 0.25/0.25)"
        )
    elif mean_bf_sasa_resid_corr > 0.4:
        # INVALID legacy branch — retained for audit only; inverted by v2/v3 evidence.
        lam1, lam2 = 0.22, 0.05
        rationale = (
            f"INVALID_SCHEDULE_DO_NOT_USE: mean corr(bf_resid,sasa_resid|depth)="
            f"{mean_bf_sasa_resid_corr:.3f} > 0.4 triggered legacy λ₂>λ₁ rule, but v2 "
            f"(λ₂=0.246) and v3 (λ₂=0.10) showed high bf–sasa overlap makes λ₂ dominate ν "
            f"and erode B-factor alignment. Use staged λ₁-only then λ₂≤0.05 instead."
        )
    elif mean_bf_sasa_resid_corr < 0.3:
        lam1, lam2 = 0.22, 0.10
        rationale = (
            f"mean corr(bf_resid,sasa_resid|depth)={mean_bf_sasa_resid_corr:.3f} < 0.3: "
            "λ₁=0.22, λ₂=0.10 with staged λ₁-only phase (empirical default)"
        )
    else:
        lam1 = 0.22
        lam2 = 0.05
        rationale = (
            f"mean corr(bf_resid,sasa_resid|depth)={mean_bf_sasa_resid_corr:.3f} in [0.3,0.4]: "
            "λ₁=0.22, λ₂≤0.05 after λ₁-only window (do not use λ₂≈λ₁/(1-r²); v3 disproved)"
        )
    out: dict[str, float | str] = {
        "lambda_1_bf_align": lam1,
        "lambda_2_sasa_pen": lam2,
        "lambda_3_anticollapse": 0.05,
        "mean_bf_sasa_resid_corr": round(mean_bf_sasa_resid_corr, 4),
        "rationale": rationale,
        "authoritative": False,
        "scheduling_note": (
            "NOT AUTHORITATIVE — derive λ from staged Option B + per-epoch r(epi,bf_resid) "
            "gate. Legacy λ₂>λ₁ when corr>0.4 is empirically wrong for single-ν head."
        ),
    }
    if mean_bf_sasa_resid_corr > 0.4:
        out["INVALID_SCHEDULE_DO_NOT_USE"] = True
    return out


def cohort_verdict(structs: list[StructureBaseline]) -> dict[str, float | int | str]:
    usable = [s for s in structs if s.referent_usable]
    resid_fracs = [
        s.residual_bfactor_var_frac for s in structs if np.isfinite(s.residual_bfactor_var_frac)
    ]
    baselines = [
        float(s.baseline_partial_corr["point"])
        for s in structs
        if np.isfinite(float(s.baseline_partial_corr["point"]))
    ]
    sasa_partials = [
        s.partial_epi_sasa_given_depth
        for s in structs
        if np.isfinite(s.partial_epi_sasa_given_depth)
    ]
    bf_sasa_resid = [
        s.corr_bf_resid_sasa_resid_given_depth
        for s in structs
        if np.isfinite(s.corr_bf_resid_sasa_resid_given_depth)
    ]
    noise = [
        float(s.baseline_partial_corr["std"])
        for s in structs
        if np.isfinite(float(s.baseline_partial_corr["std"]))
    ]
    go = len(usable) >= max(1, (len(structs) + 1) // 2)  # majority usable
    v: dict[str, float | int | str] = {
        "n_structures": len(structs),
        "n_referent_usable": len(usable),
        "GO_NO_GO": "GO" if go else "NO-GO",
        "mean_residual_bfactor_var_frac": float(np.mean(resid_fracs))
        if resid_fracs
        else float("nan"),
        "mean_baseline_partial_corr": float(np.mean(baselines)) if baselines else float("nan"),
        "mean_partial_epi_sasa_given_depth": float(np.mean(sasa_partials))
        if sasa_partials
        else float("nan"),
        "mean_corr_bf_resid_sasa_resid_given_depth": float(np.mean(bf_sasa_resid))
        if bf_sasa_resid
        else float("nan"),
        "mean_bootstrap_std": float(np.mean(noise)) if noise else float("nan"),
    }
    if bf_sasa_resid:
        v["suggested_lambda_schedule"] = _suggest_lambda_schedule(float(np.mean(bf_sasa_resid)))
    # Derived Step-4 pass threshold: baseline + 2*noise (clears the band), not 0.15.
    if baselines and noise:
        v["suggested_step4_pass_delta"] = round(2 * float(np.mean(noise)), 3)
        v["note_threshold"] = (
            "DETECTION FLOOR ONLY: Step 4 B-factor improvement must exceed baseline by >= "
            "this delta (2x bootstrap std). This is measurability, not biological significance. "
            "Also require partial(epi,sasa|depth) does not rise (SASA confound guard)."
        )
    if not go:
        v["reason"] = (
            "Depth explains most normalized B-factor variance "
            "(residual < %.2f) on too many structures — B-factor is a "
            "weak referent; pick another before Step 4." % MIN_RESIDUAL_BFACTOR_VAR_FRAC
        )
    return v


# ===========================================================================
# Data loader — checkpoint inference + PDB B-factors (aligned to training graph)
# ===========================================================================
def _bfactors_for_training_graph(
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Cα B-factors aligned to load_protein_graph residue order and filters."""
    from Bio.PDB import PDBParser

    from experiments.training.v6._data import (
        _chain_cache_dir,
        _compute_rho,
        _download_pdb,
        _extract_chain,
    )

    pdb_path = _download_pdb(pdb_id.upper(), pdb_dir)
    chain_path = _extract_chain(pdb_path, chain, _chain_cache_dir(pdb_dir))
    structure = PDBParser(QUIET=True).get_structure(pdb_id, str(chain_path))
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    all_atoms = [a for r in residues for a in r.get_atoms()]

    bfactor_raw: list[float] = []
    present_mask: list[bool] = []
    for res in residues:
        rho = _compute_rho(res, all_atoms)
        if rho < 0 or "CA" not in res:
            continue
        ca = res["CA"]
        bf = float(ca.get_bfactor())
        has_bf = np.isfinite(bf)
        bfactor_raw.append(bf if has_bf else float("nan"))
        present_mask.append(has_bf)

    if len(bfactor_raw) < 10:
        raise RuntimeError(f"{pdb_id}:{chain} — fewer than 10 B-factor residues after filtering")

    return np.asarray(bfactor_raw, dtype=np.float64), np.asarray(present_mask, dtype=bool)


@torch.inference_mode()
def load_per_residue(
    structure_id: str,
    chain: str = "A",
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Return per-residue arrays aligned to the v6 training graph.

    epi / depth: warm-start checkpoint forward pass (cone_depth, epistemic).
    bfactor_raw / present_mask: Cα ATOM records from PDB (same filters as _data.py).
    sasa / ss_is_coil: node features from the training graph.
    """
    from experiments.diagnostics.embedding_occupancy_audit import load_audit_model
    from experiments.training.v6._data import load_protein_graph
    from experiments.training.v6.train_loop import attach_v6_features

    pdb_id = structure_id.upper()
    prot = load_protein_graph(pdb_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Could not load protein graph for {pdb_id}:{chain}")

    data = attach_v6_features(prot["data"])
    model, version = load_audit_model(checkpoint, device)
    if version != "v6":
        raise RuntimeError(f"Expected v6 checkpoint, got {version}")

    out = model(data.to(device))
    epi = out["uncertainty"]["epistemic"].detach().cpu().numpy().reshape(-1)
    depth = out["cone_depth"].detach().cpu().numpy().reshape(-1)
    sasa = data.x[:, 3].detach().cpu().numpy()
    ss_labels = data.ss_onehot.argmax(dim=1).detach().cpu().numpy()
    ss_is_coil = ss_labels == 2

    bfactor_raw, present_mask = _bfactors_for_training_graph(pdb_id, chain, pdb_dir)
    n = epi.shape[0]
    if bfactor_raw.shape[0] != n:
        raise RuntimeError(
            f"{pdb_id}:{chain} B-factor count {bfactor_raw.shape[0]} != graph nodes {n}"
        )

    return {
        "epi": epi,
        "depth": depth,
        "bfactor_raw": bfactor_raw,
        "sasa": sasa,
        "ss_is_coil": ss_is_coil,
        "present_mask": present_mask,
    }


def run(
    structure_ids: list[str],
    chain: str = "A",
    *,
    checkpoint: Path,
    pdb_dir: Path,
    device: str = "cpu",
) -> dict[str, object]:
    results = []
    for sid in structure_ids:
        d = load_per_residue(
            sid,
            chain=chain,
            checkpoint=checkpoint,
            pdb_dir=pdb_dir,
            device=device,
        )
        results.append(
            compute_structure_baseline(
                sid,
                d["epi"],
                d["depth"],
                d["bfactor_raw"],
                d["sasa"],
                d.get("ss_is_coil"),
                d["present_mask"],
            )
        )
    return {
        "checkpoint": str(checkpoint),
        "pdb_dir": str(pdb_dir),
        "chain": chain,
        "structures": structure_ids,
        "per_structure": [asdict(r) for r in results],
        "cohort": cohort_verdict(results),
        "thresholds": {
            "MIN_RESIDUAL_BFACTOR_VAR_FRAC": MIN_RESIDUAL_BFACTOR_VAR_FRAC,
            "MIN_BFACTOR_COVERAGE": MIN_BFACTOR_COVERAGE,
        },
    }


# ===========================================================================
# Self-tests: verify the statistics on synthetic data with KNOWN ground truth
# ===========================================================================
def _selftest() -> int:
    rng = np.random.default_rng(0)
    n = 4000
    # 1) partial_corr recovers a known partial correlation independent of z.
    z = rng.normal(size=n)
    # build residuals rx, ry with target correlation 0.5, independent of z
    r_target = 0.5
    a = rng.normal(size=n)
    b = r_target * a + np.sqrt(1 - r_target**2) * rng.normal(size=n)
    x = 2.0 * z + a  # depends on z + residual a
    y = -1.5 * z + b  # depends on z + residual b (corr r_target with a)
    pc = partial_corr(x, y, z)
    assert abs(pc - r_target) < 0.05, f"partial_corr {pc} != {r_target}"
    # and the raw correlation is contaminated by z (should differ from partial)
    raw = np.corrcoef(x, y)[0, 1]
    assert abs(raw - r_target) > 0.1, "raw corr should be confounded by z"

    # 2) residual_variance_fraction recovers a known depth->y R².
    #    y = depth + noise with var ratio controlling R².
    depth = rng.normal(size=n)
    for true_r2 in (0.2, 0.75, 0.95):
        sig = np.sqrt(true_r2)
        noi = np.sqrt(1 - true_r2)
        y = sig * depth + noi * rng.normal(size=n)
        frac, r2 = residual_variance_fraction(y, depth)
        assert abs(r2 - true_r2) < 0.05, f"R² {r2} != {true_r2}"
        assert abs(frac - (1 - true_r2)) < 0.05

    # 3) the collinear-epistemic premise: epi ~= depth => baseline partial corr ~ 0
    #    even when B-factor HAS depth-independent signal.
    latent = rng.normal(size=n)  # depth-independent uncertainty
    bfactor = 0.6 * depth + 0.6 * latent + 0.3 * rng.normal(size=n)
    epi_collinear = depth + 0.02 * rng.normal(size=n)  # r(epi,depth)~0.999
    bf_norm = zscore(bfactor)
    pc_collinear = partial_corr(epi_collinear, bf_norm, depth)
    assert abs(pc_collinear) < 0.1, f"collinear epi should give ~0 partial, got {pc_collinear}"
    # ...and a DECOUPLED epistemic that carries the latent lifts the partial corr
    epi_decoupled = depth + 0.8 * latent + 0.1 * rng.normal(size=n)
    pc_decoupled = partial_corr(epi_decoupled, bf_norm, depth)
    assert pc_decoupled > 0.4, f"decoupled epi should lift partial corr, got {pc_decoupled}"
    # and the referent is 'usable': residual B-factor var after depth is substantial
    frac, _r2 = residual_variance_fraction(bf_norm, depth)
    assert frac > MIN_RESIDUAL_BFACTOR_VAR_FRAC, f"resid frac {frac} should be usable"

    # 4) zscore normalizes; bootstrap CI brackets the point estimate.
    zz = zscore(bfactor)
    assert abs(np.nanmean(zz)) < 1e-9 and abs(np.nanstd(zz) - 1) < 1e-9
    band = bootstrap_ci(epi_decoupled, bf_norm, depth, n_boot=300)
    assert band["ci95"][0] <= band["point"] <= band["ci95"][1]

    # 5) gap_bias flags surface-biased gaps.
    sasa = rng.normal(size=n)
    present = sasa < 1.0  # drop high-SASA residues
    gb = gap_bias(present, sasa, depth, None)
    assert gb["sasa_gap_minus_present"] > 0, "gaps should be higher-SASA here"
    print(
        "self-tests PASSED: partial_corr, residual_var_frac, collinear-vs-decoupled "
        "premise, zscore, bootstrap band, gap_bias"
    )
    return 0


def _parse_structures(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", nargs="+", metavar="STRUCTURE_ID")
    ap.add_argument(
        "--structures",
        default=",".join(DEFAULT_STRUCTURES),
        help="Comma-separated PDB IDs (default: step-2 battery)",
    )
    ap.add_argument("--chain", default="A")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/v6/diagnostics/epistemic_decoupling_baseline.json"),
    )
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    structure_ids = args.run if args.run else _parse_structures(args.structures)
    if structure_ids:
        report = run(
            structure_ids,
            chain=args.chain,
            checkpoint=args.checkpoint,
            pdb_dir=args.pdb_dir,
            device=args.device,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report["cohort"], indent=2))
        sys.exit(0)
    ap.print_help()
