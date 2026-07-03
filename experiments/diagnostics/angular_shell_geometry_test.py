"""Radius-conditioned angular geometry test (step 2 pre-registered).

Tests whether pre-routing disc angle carries structural signal at fixed radius.

Usage:
  python -m experiments.diagnostics.angular_shell_geometry_test \\
    --checkpoint checkpoints/v6/runs/shell_p2_hypmix_final/v6_best.pt \\
    --structures 11QE:A,4OBE:A,1IVO:A,4MNE:A \\
    --pdb-dir /tmp/dtie_pdb_cache
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from experiments.diagnostics.embedding_occupancy_audit import load_audit_model
from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.train_loop import attach_v6_features

DEFAULT_STRUCTURES = "11QE:A,4OBE:A,1IVO:A,4MNE:A"
MIN_SHELL_SIZE = 12
N_RADIAL_SHELLS = 5
N_PERMUTATIONS = 1000
N_SEQ_BLOCKS = 8
ALPHA = 0.05
PASS_MIN_STRUCTURES = 2
PASS_MIN_STRUCTURES_TOTAL = 3
PASS_MIN_SHELLS = 2


def _parse_structures(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            pdb, chain = item.split(":", 1)
        else:
            pdb, chain = item, "A"
        out.append((pdb.upper(), chain))
    return out


def _ss_labels_from_data(data: Any) -> np.ndarray:
    oh = data.ss_onehot.detach().cpu().numpy()
    return np.argmax(oh, axis=1).astype(np.int64)


def _circular_mi_discrete(theta: np.ndarray, labels: np.ndarray, n_bins: int = 8) -> float:
    theta = np.asarray(theta, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if theta.size < 3:
        return 0.0
    span = theta.max() - theta.min()
    if span < 1e-12:
        return 0.0
    bins = np.floor((theta - theta.min()) / span * n_bins).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)
    mi = 0.0
    for b in range(n_bins):
        for c in np.unique(labels):
            p_bc = np.mean((bins == b) & (labels == c))
            if p_bc <= 0:
                continue
            p_b = np.mean(bins == b)
            p_c = np.mean(labels == c)
            mi += p_bc * math.log(p_bc / (p_b * p_c + 1e-12) + 1e-12)
    return float(max(mi, 0.0))


def _perm_pvalue(observed: float, nulls: np.ndarray) -> float:
    nulls = np.asarray(nulls, dtype=np.float64)
    if nulls.size == 0:
        return 1.0
    return float((1 + np.sum(nulls >= observed)) / (1 + nulls.size))


def _bh_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [1.0] * m
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        raw_rank = m - rank + 1
        val = p_values[idx] * m / raw_rank
        prev = min(prev, val)
        adjusted[idx] = min(prev, 1.0)
    return adjusted


def _assign_radial_shells(r: np.ndarray, n_shells: int) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    quantiles = np.linspace(0, 1, n_shells + 1)
    edges = np.quantile(r, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(r.size, dtype=np.int64)
    return np.digitize(r, edges[1:-1], right=True).astype(np.int64)


def _seq_blocks(seq_index: np.ndarray, n_blocks: int) -> np.ndarray:
    seq_index = np.asarray(seq_index, dtype=np.float64)
    edges = np.quantile(seq_index, np.linspace(0, 1, n_blocks + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(seq_index.size, dtype=np.int64)
    return np.digitize(seq_index, edges[1:-1], right=True).astype(np.int64)


def _permute_within_seq_blocks(labels: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = labels.copy()
    for b in np.unique(blocks):
        mask = blocks == b
        idx = np.where(mask)[0]
        if idx.size < 2:
            continue
        out[idx] = rng.permutation(out[idx])
    return out


@dataclass
class ShellTestResult:
    shell: int
    n: int
    r_min: float
    r_max: float
    covariate: str
    mi_observed: float
    mi_perm_mean: float
    mi_perm_p: float
    mi_perm_p_bh: float | None = None
    seq_blocked_mi_observed: float | None = None
    seq_blocked_mi_perm_p: float | None = None


@dataclass
class StructureAngularReport:
    structure_id: str
    chain: str
    n_residues: int
    disc_sigma2_sigma1: float
    disc_line_thickness: float
    shell_tests: list[ShellTestResult] = field(default_factory=list)
    ss_shells_significant_raw: int = 0
    ss_shells_significant_seq_blocked: int = 0
    sasa_shells_significant: int = 0
    routing_ss_correlation: float | None = None
    verdict: Literal["pass", "partial", "fail"] = "fail"
    notes: list[str] = field(default_factory=list)


def _disc_occupancy_metrics(xy: np.ndarray) -> tuple[float, float]:
    from science.dtie.v6.gnn.slice_diagnostics import disc_2d_stats

    stats = disc_2d_stats(xy)
    sr = stats.get("sigma_ratio") or []
    s2s1 = float(sr[1]) if len(sr) > 1 else 0.0
    thick = float(stats.get("line_thickness_rms") or 0.0)
    return s2s1, thick


def _test_shell_covariate(
    theta: np.ndarray,
    values: np.ndarray,
    *,
    covariate: str,
    shell_id: int,
    r_vals: np.ndarray,
    seq_blocks: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> ShellTestResult:
    mi_obs = _circular_mi_discrete(theta, values)
    nulls = np.empty(n_perm, dtype=np.float64)
    for _ in range(n_perm):
        nulls[_] = _circular_mi_discrete(theta, rng.permutation(values))
    p_raw = _perm_pvalue(mi_obs, nulls)

    seq_shuffled = _permute_within_seq_blocks(values, seq_blocks, rng)
    mi_seq = _circular_mi_discrete(theta, seq_shuffled)
    nulls_seq = np.empty(n_perm, dtype=np.float64)
    for _ in range(n_perm):
        nulls_seq[_] = _circular_mi_discrete(
            theta, _permute_within_seq_blocks(values, seq_blocks, rng)
        )
    p_seq = _perm_pvalue(mi_seq, nulls_seq)

    return ShellTestResult(
        shell=shell_id,
        n=int(theta.size),
        r_min=float(np.min(r_vals)),
        r_max=float(np.max(r_vals)),
        covariate=covariate,
        mi_observed=mi_obs,
        mi_perm_mean=float(np.mean(nulls)),
        mi_perm_p=p_raw,
        seq_blocked_mi_observed=mi_seq,
        seq_blocked_mi_perm_p=p_seq,
    )


def _routing_ss_alignment(expert_weights: np.ndarray, ss_labels: np.ndarray) -> float:
    if expert_weights.shape[0] < 5:
        return float("nan")
    helix_ind = (ss_labels == 0).astype(np.float64)
    e0 = expert_weights[:, 0]
    if np.std(helix_ind) < 1e-8 or np.std(e0) < 1e-8:
        return float("nan")
    return float(np.corrcoef(helix_ind, e0)[0, 1])


@torch.inference_mode()
def analyze_structure(
    model: torch.nn.Module,
    version: str,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    *,
    n_shells: int,
    n_perm: int,
    min_shell_size: int,
    seed: int,
) -> StructureAngularReport | None:
    prot = load_protein_graph(pdb_id, chain, pdb_dir)
    if prot is None or version != "v6":
        return None

    data = attach_v6_features(prot["data"])
    out = model(data)

    hyp2d = out.get("hyp_projections_2d")
    if hyp2d is None:
        hyp2d = out.get("hyp_proj_2d")
    if hyp2d is None:
        return None

    xy = hyp2d.detach().cpu().numpy()
    r = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    ss = _ss_labels_from_data(data)
    sasa = data.x[:, 3].detach().cpu().numpy()
    seq_index = np.arange(xy.shape[0], dtype=np.float64)
    expert_w = out["expert_weights"].detach().cpu().numpy()

    s2s1, thick = _disc_occupancy_metrics(xy)
    shells = _assign_radial_shells(r, n_shells)
    seq_blk = _seq_blocks(seq_index, N_SEQ_BLOCKS)
    rng = np.random.default_rng(seed)

    report = StructureAngularReport(
        structure_id=pdb_id,
        chain=chain,
        n_residues=int(xy.shape[0]),
        disc_sigma2_sigma1=s2s1,
        disc_line_thickness=thick,
        routing_ss_correlation=_routing_ss_alignment(expert_w, ss),
    )

    covariates: list[tuple[str, np.ndarray]] = [
        ("ss_class", ss),
        ("sasa", np.digitize(sasa, np.quantile(sasa, [0.33, 0.66])).astype(np.int64)),
        ("seq_index", np.digitize(seq_index, np.quantile(seq_index, [0.33, 0.66])).astype(np.int64)),
    ]

    for shell_id in range(int(shells.max()) + 1):
        mask = shells == shell_id
        if int(mask.sum()) < min_shell_size:
            continue
        for cov_name, cov_vals in covariates:
            report.shell_tests.append(
                _test_shell_covariate(
                    theta[mask],
                    cov_vals[mask],
                    covariate=cov_name,
                    shell_id=shell_id,
                    r_vals=r[mask],
                    seq_blocks=seq_blk[mask],
                    rng=rng,
                    n_perm=n_perm,
                )
            )

    pvals = [t.mi_perm_p for t in report.shell_tests]
    bh = _bh_adjust(pvals)
    for t, adj in zip(report.shell_tests, bh, strict=True):
        t.mi_perm_p_bh = adj

    ss_tests = [t for t in report.shell_tests if t.covariate == "ss_class"]
    sasa_tests = [t for t in report.shell_tests if t.covariate == "sasa"]

    report.ss_shells_significant_raw = sum(1 for t in ss_tests if t.mi_perm_p < ALPHA)
    report.ss_shells_significant_seq_blocked = sum(
        1 for t in ss_tests if (t.seq_blocked_mi_perm_p or 1.0) < ALPHA
    )
    report.sasa_shells_significant = sum(1 for t in sasa_tests if t.mi_perm_p < ALPHA)

    sig_shells = sorted({t.shell for t in ss_tests if (t.seq_blocked_mi_perm_p or 1.0) < ALPHA})
    non_adjacent_pair = any(abs(a - b) >= 2 for i, a in enumerate(sig_shells) for b in sig_shells[i + 1 :])
    has_crescent = thick >= 0.02 and s2s1 >= 0.15

    if (
        len(sig_shells) >= PASS_MIN_SHELLS
        and non_adjacent_pair
        and report.ss_shells_significant_seq_blocked >= PASS_MIN_SHELLS
    ):
        report.verdict = "pass"
        report.notes.append("SS segregates by theta in >=2 non-adjacent shells (seq-blocked perm).")
    elif has_crescent and report.ss_shells_significant_seq_blocked == 0:
        report.verdict = "partial"
        report.notes.append("Crescent occupancy; within-shell SS-theta flat after seq block.")
    else:
        report.verdict = "fail"
        report.notes.append("No seq-blocked SS angular segregation at fixed radius.")

    if report.sasa_shells_significant > report.ss_shells_significant_seq_blocked:
        report.notes.append("SASA exceeds SS — likely radial confound.")

    return report


def _cohort_verdict(reports: list[StructureAngularReport]) -> Literal["pass", "partial", "fail"]:
    passes = sum(1 for r in reports if r.verdict == "pass")
    partials = sum(1 for r in reports if r.verdict == "partial")
    if passes >= PASS_MIN_STRUCTURES and len(reports) >= PASS_MIN_STRUCTURES_TOTAL:
        return "pass"
    if passes + partials >= PASS_MIN_STRUCTURES:
        return "partial"
    return "fail"


def run_angular_shell_test(
    checkpoint: Path,
    structures: list[tuple[str, str]],
    pdb_dir: Path,
    *,
    n_shells: int = N_RADIAL_SHELLS,
    n_perm: int = N_PERMUTATIONS,
    min_shell_size: int = MIN_SHELL_SIZE,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    model, version = load_audit_model(checkpoint, device)

    reports: list[StructureAngularReport] = []
    for i, (pdb, chain) in enumerate(structures):
        rep = analyze_structure(
            model, version, pdb, chain, pdb_dir,
            n_shells=n_shells, n_perm=n_perm, min_shell_size=min_shell_size,
            seed=seed + i * 997,
        )
        if rep is not None:
            reports.append(rep)

    return {
        "checkpoint": str(checkpoint),
        "structures_requested": [f"{p}:{c}" for p, c in structures],
        "cohort_verdict": _cohort_verdict(reports),
        "pass_rule": (
            f">={PASS_MIN_STRUCTURES}/{PASS_MIN_STRUCTURES_TOTAL} structures pass; "
            f">={PASS_MIN_SHELLS} non-adjacent shells; seq-blocked permutation; BH per structure"
        ),
        "structures": [asdict(r) for r in reports],
    }


def _print_report(payload: dict[str, Any]) -> None:
    print(f"checkpoint: {payload['checkpoint']}")
    print(f"cohort_verdict: {payload['cohort_verdict']}")
    for s in payload["structures"]:
        print(
            f"  {s['structure_id']}:{s['chain']} verdict={s['verdict']} "
            f"thick={s['disc_line_thickness']:.4f} ss_blk={s['ss_shells_significant_seq_blocked']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--structures", type=str, default=DEFAULT_STRUCTURES)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-shells", type=int, default=N_RADIAL_SHELLS)
    parser.add_argument("--n-perm", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--min-shell-size", type=int, default=MIN_SHELL_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = run_angular_shell_test(
        args.checkpoint,
        _parse_structures(args.structures),
        args.pdb_dir,
        n_shells=args.n_shells,
        n_perm=args.n_perm,
        min_shell_size=args.min_shell_size,
        seed=args.seed,
        device=args.device,
    )
    _print_report(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
