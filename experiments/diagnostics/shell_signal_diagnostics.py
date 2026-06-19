"""
shell_signal_diagnostics.py — Outer Shell Signal Diagnostics for GNNv6 / DTIE v5
==================================================================================
Eidetix Bio | 2026-06-19

Tests whether GNNv5 (the "v6" manifold) is still inferring on the outer shell
— the high-SASA, high-uncertainty, solvent-exposed communication layer that was
explicitly rendered in GNNv2 but has since become implicit.

Four diagnostic probes, each targeting a different failure mode:

  PROBE 1 — Uncertainty vs. Solvent Accessibility Correlation
    If the shell signal is alive, high epistemic uncertainty + high cone_depth
    should correlate positively with high SASA (node feature index 3).
    Weak or inverted correlation = shell signal diluted.

  PROBE 2 — Core-Only Ablation
    Strip out all residues with SASA > 0.25 or cone_depth above the 75th
    percentile, then compare inference quality metrics.  A measurable drop in
    uncertainty calibration (epistemic spread, depth variance) when the shell
    is removed confirms the model was relying on it.

  PROBE 3 — Surface Hotspot Alignment
    Take the top-N residues by composite peripheral score (high epistemic *
    high cone_depth, i.e. the "persistent-leak" proxy) and check what fraction
    are actually surface-exposed (SASA > 0.20) and what fraction fall in known
    functional surface regions (Switch-I/II, P-loop for KRAS).

  PROBE 4 — Radial Gradient Strength
    Color residues by cone_depth on the 2D disc.  The shell-alive signal
    is a clear radial gradient: low cone_depth at the disc centre (core) and
    high cone_depth at the periphery (shell).  Quantify this as Pearson r
    between |hyp_proj_2d| (disc radius) and cone_depth.  r > 0.6 = shell
    visible; r < 0.3 = gradient collapsed.

Usage:
    python shell_signal_diagnostics.py \\
        --checkpoint checkpoints/v5/tokyo_eyes_v5.pt \\
        --pdb_dir /tmp/dtie_pdb_cache \\
        [--structure_id 9O0R] [--chain A] [--device cpu]

    The script will also accept a JSON file of pre-computed outputs:
        --precomputed_json /path/to/gnn_result.json
    in which case PDB loading and model inference are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# KRAS domain annotations — used for Probe 3 surface alignment check.
KRAS_DOMAINS = {
    "P-loop":    list(range(10, 18)),
    "Switch-I":  list(range(25, 41)),
    "Switch-II": list(range(57, 76)),
}
KRAS_FUNCTIONAL_SURFACE = {r for rng in KRAS_DOMAINS.values() for r in rng}

SASA_SURFACE_THRESHOLD = 0.20   # residues with SASA ≥ this are "surface-exposed"
SASA_OUTER_THRESHOLD   = 0.25   # ablation: mask SASA ≥ this as "outer shell"
DEPTH_OUTER_PERCENTILE = 75     # ablation: also mask cone_depth above this percentile

PROBE_SEPARATOR = "=" * 72


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_via_model(
    structure_id: str,
    chain: str,
    checkpoint_path: str,
    pdb_dir: Path,
    device: str,
) -> Tuple[dict, dict]:
    """Run full inference with the v5 model and return (raw_output, prot_dict)."""
    import torch

    # Add the v4 training directory to sys.path so load_protein_graph is importable.
    _repo_root = Path(__file__).resolve().parents[2]
    _v4_train = _repo_root / "experiments" / "training" / "v4"
    _v5_model = _repo_root / "science" / "dtie" / "v5" / "gnn"
    for p in [str(_v4_train), str(_v5_model)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from train_v4 import load_protein_graph  # type: ignore
    from model import GOSPConeMapper, precompute_clustering  # type: ignore

    prot = load_protein_graph(structure_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Could not load structure {structure_id}:{chain}")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Support both raw state_dict and wrapped {"model_state_dict": ...} formats.
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    data = precompute_clustering(prot["data"]).to(device)
    with torch.no_grad():
        raw = model(data)

    # Convert tensors to numpy for downstream probes.
    out: dict = {
        "cone_depth":      raw["cone_depth"].squeeze().cpu().numpy(),
        "cone_width":      raw["cone_width"].squeeze().cpu().numpy(),
        "epistemic":       raw["uncertainty"]["epistemic"].cpu().numpy(),
        "aleatoric":       raw["uncertainty"]["aleatoric"].cpu().numpy(),
        "hyp_proj_2d":     raw["hyp_projections_2d"].cpu().numpy(),
        "x_routed_hyp":    raw["x_routed_hyp"].cpu().numpy(),
        # Node features: [rho, tau_flag, ss_type, sasa]
        "sasa":            data.x[:, 3].cpu().numpy(),
        "rho":             data.x[:, 0].cpu().numpy(),
        "residue_ids":     prot.get("residue_ids", []),
    }

    # Store expert weights if present.
    if "expert_weights" in raw:
        out["expert_weights"] = raw["expert_weights"].cpu().numpy()

    return out, prot


def _load_from_json(json_path: str) -> dict:
    """Load pre-computed GNN outputs from a JSON export (export_for_viewer format)."""
    with open(json_path) as f:
        data = json.load(f)

    nodes = data.get("nodes", data)
    n = len(nodes)

    out = {
        "cone_depth":  np.array([nd.get("cone_depth", 0.0)        for nd in nodes]),
        "cone_width":  np.array([nd.get("cone_width", 0.0)         for nd in nodes]),
        "epistemic":   np.array([nd.get("epistemic_uncertainty", 0.0) for nd in nodes]),
        "aleatoric":   np.array([nd.get("aleatoric_uncertainty", 0.0) for nd in nodes]),
        "hyp_proj_2d": np.array([nd.get("disc_pos", [0.0, 0.0])    for nd in nodes]),
        "sasa":        np.array([nd.get("sasa", 0.0)               for nd in nodes]),
        "rho":         np.array([nd.get("rho",  0.0)               for nd in nodes]),
        "residue_ids": [nd.get("residue_id", f"?:{i}:") for i, nd in enumerate(nodes)],
    }
    return out


# ---------------------------------------------------------------------------
# PROBE 1 — Uncertainty vs. Solvent Accessibility Correlation
# ---------------------------------------------------------------------------

def probe1_uncertainty_sasa_correlation(out: dict) -> None:
    print(f"\n{PROBE_SEPARATOR}")
    print("PROBE 1: Epistemic Uncertainty & cone_depth vs. SASA")
    print(PROBE_SEPARATOR)
    print(
        "Hypothesis: if the outer shell is still alive in the model,\n"
        "high epistemic uncertainty and high cone_depth should co-occur\n"
        "with high SASA (node feature 3 — solvent accessibility).\n"
    )

    sasa      = out["sasa"]
    depth     = out["cone_depth"]
    epistemic = out["epistemic"]
    aleatoric = out["aleatoric"]

    n = len(sasa)
    if n == 0:
        print("  ERROR: empty output")
        return

    # Pearson r for each pair.
    def pearson(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) < 3:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    r_epi_sasa   = pearson(epistemic, sasa)
    r_depth_sasa = pearson(depth, sasa)
    r_ale_sasa   = pearson(aleatoric, sasa)
    r_epi_depth  = pearson(epistemic, depth)

    print(f"  {'Pair':<34} {'Pearson r':>10}  Interpretation")
    print(f"  {'-'*70}")
    _row("epistemic_uncertainty × SASA",  r_epi_sasa,
         "> 0.5 = shell alive; < 0.2 = signal diluted")
    _row("cone_depth × SASA",             r_depth_sasa,
         "> 0.5 = radial gradient intact; < 0.2 = collapsed")
    _row("aleatoric_uncertainty × SASA",  r_ale_sasa,
         "> 0.4 = model sees surface noise; < 0.1 = not surface-aware")
    _row("epistemic × cone_depth",        r_epi_depth,
         "> 0.5 = uncertainty tracks depth (good joint calibration)")
    print()

    # Stratified statistics: core vs. shell vs. surface.
    labels = _shell_labels(sasa, depth)
    _print_stratum_stats(out, labels,
                         strata=["core", "intermediate", "shell"],
                         fields=["epistemic", "aleatoric", "cone_depth"])

    # Verdict.
    shell_alive = (r_epi_sasa > 0.40) and (r_depth_sasa > 0.40)
    print(f"\n  VERDICT: outer shell signal is "
          f"{'ALIVE ✓' if shell_alive else 'WEAK or ABSENT ✗'} "
          f"(r_epi_sasa={r_epi_sasa:.3f}, r_depth_sasa={r_depth_sasa:.3f})")


# ---------------------------------------------------------------------------
# PROBE 2 — Core-Only Ablation
# ---------------------------------------------------------------------------

def probe2_core_ablation(
    out: dict,
    checkpoint_path: Optional[str],
    prot: Optional[dict],
    device: str,
) -> None:
    print(f"\n{PROBE_SEPARATOR}")
    print("PROBE 2: Core-Only Ablation (mask outer shell residues)")
    print(PROBE_SEPARATOR)
    print(
        "Strips residues with SASA > {:.2f} OR cone_depth > {}th percentile,\n"
        "then re-runs inference and compares uncertainty calibration metrics.\n"
        "A drop confirms the model was relying on shell information.\n".format(
            SASA_OUTER_THRESHOLD, DEPTH_OUTER_PERCENTILE
        )
    )

    sasa  = out["sasa"]
    depth = out["cone_depth"]
    n = len(sasa)

    depth_cutoff = float(np.percentile(depth, DEPTH_OUTER_PERCENTILE))
    shell_mask = (sasa >= SASA_OUTER_THRESHOLD) | (depth >= depth_cutoff)
    core_mask  = ~shell_mask

    n_shell = int(shell_mask.sum())
    n_core  = int(core_mask.sum())

    print(f"  Total residues : {n}")
    print(f"  Outer shell    : {n_shell} ({100*n_shell/n:.1f}%)"
          f"  [SASA≥{SASA_OUTER_THRESHOLD} or depth≥p{DEPTH_OUTER_PERCENTILE}={depth_cutoff:.3f}]")
    print(f"  Core           : {n_core} ({100*n_core/n:.1f}%)")

    # --- Metrics on full inference output (already computed) ---
    full_epi_spread  = float(out["epistemic"].std())
    full_depth_var   = float(out["cone_depth"].std())
    full_epi_mean    = float(out["epistemic"].mean())

    print(f"\n  Full-graph metrics:")
    print(f"    epistemic spread (σ) : {full_epi_spread:.4f}")
    print(f"    epistemic mean       : {full_epi_mean:.4f}")
    print(f"    cone_depth spread (σ): {full_depth_var:.4f}")

    # --- Core-only metrics (from the existing output, restricted to core nodes) ---
    # Without re-running inference (which requires a live model), we can still
    # assess how the model's outputs *on the core* compare with the full outputs.
    # If the shell was informative, we'd expect the core-only signal to be weaker.
    core_epi_mean   = float(out["epistemic"][core_mask].mean())
    core_epi_spread = float(out["epistemic"][core_mask].std())
    core_depth_std  = float(out["cone_depth"][core_mask].std())

    print(f"\n  Core-only node metrics (no re-inference, restricted view):")
    print(f"    epistemic spread (σ) : {core_epi_spread:.4f}")
    print(f"    epistemic mean       : {core_epi_mean:.4f}")
    print(f"    cone_depth spread (σ): {core_depth_std:.4f}")

    epi_drop   = (full_epi_spread - core_epi_spread) / (full_epi_spread + 1e-9)
    depth_drop = (full_depth_var  - core_depth_std)  / (full_depth_var  + 1e-9)

    print(f"\n  Δ epistemic spread (full→core): {epi_drop:+.1%}")
    print(f"  Δ depth spread     (full→core): {depth_drop:+.1%}")

    # --- Live re-inference (only if model is available) ---
    if checkpoint_path and prot is not None:
        try:
            _probe2_live_ablation(out, prot, checkpoint_path, core_mask, device)
        except Exception as exc:
            print(f"\n  [live re-inference skipped: {exc}]")
    else:
        print(
            "\n  NOTE: pass --checkpoint to enable live re-inference on a masked graph."
        )

    epi_drop_significant = abs(epi_drop) > 0.10
    print(
        f"\n  VERDICT: shell contribution is "
        f"{'SIGNIFICANT ✓' if epi_drop_significant else 'MARGINAL ✗'} "
        f"(epistemic spread drops {epi_drop:+.1%} when shell is excluded)"
    )


def _probe2_live_ablation(
    out: dict, prot: dict, checkpoint_path: str, core_mask: np.ndarray, device: str
) -> None:
    """Re-run inference with shell nodes zeroed to isolate the causal effect."""
    import torch

    _repo_root = Path(__file__).resolve().parents[2]
    _v5_model  = _repo_root / "science" / "dtie" / "v5" / "gnn"
    if str(_v5_model) not in sys.path:
        sys.path.insert(0, str(_v5_model))

    from model import GOSPConeMapper, precompute_clustering  # type: ignore

    model = GOSPConeMapper(node_dim=4, hidden=128, num_experts=4)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    data_core = precompute_clustering(prot["data"].clone()).to(device)
    # Zero out all shell node features — simulates "core-only" graph.
    shell_t = torch.tensor(~core_mask, dtype=torch.bool, device=device)
    data_core.x[shell_t] = 0.0

    with torch.no_grad():
        raw_core = model(data_core)

    ablated_epi    = raw_core["uncertainty"]["epistemic"].cpu().numpy()
    ablated_spread = float(ablated_epi.std())
    ablated_mean   = float(ablated_epi.mean())
    ablated_depth  = float(raw_core["cone_depth"].squeeze().cpu().numpy().std())

    print(f"\n  Live-ablated metrics (shell nodes zeroed):")
    print(f"    epistemic spread (σ) : {ablated_spread:.4f}")
    print(f"    epistemic mean       : {ablated_mean:.4f}")
    print(f"    cone_depth spread (σ): {ablated_depth:.4f}")

    full_epi_spread = float(out["epistemic"].std())
    full_depth_var  = float(out["cone_depth"].std())
    live_epi_drop   = (full_epi_spread - ablated_spread) / (full_epi_spread + 1e-9)
    live_depth_drop = (full_depth_var  - ablated_depth)  / (full_depth_var  + 1e-9)

    print(f"  Δ epistemic spread (causal, shell→0): {live_epi_drop:+.1%}")
    print(f"  Δ depth spread     (causal, shell→0): {live_depth_drop:+.1%}")


# ---------------------------------------------------------------------------
# PROBE 3 — Surface Hotspot Alignment
# ---------------------------------------------------------------------------

def probe3_surface_hotspot_alignment(out: dict, top_n: int = 30) -> None:
    print(f"\n{PROBE_SEPARATOR}")
    print(f"PROBE 3: Surface Hotspot Alignment (top-{top_n} peripheral residues)")
    print(PROBE_SEPARATOR)
    print(
        "Composites the 'persistent-leak' proxy score:\n"
        "    score = epistemic_uncertainty × cone_depth\n"
        "then checks whether the top residues are actually surface-exposed\n"
        "and whether they overlap known functional surface regions.\n"
    )

    sasa      = out["sasa"]
    depth     = out["cone_depth"]
    epistemic = out["epistemic"]
    res_ids   = out.get("residue_ids", [])
    n         = len(sasa)

    # Composite peripheral score: high-uncertainty, high-depth = outer shell leaker.
    score = epistemic * depth
    ranked = np.argsort(score)[::-1]
    top_idx = ranked[:top_n]

    n_surface  = int((sasa[top_idx] >= SASA_SURFACE_THRESHOLD).sum())
    n_total    = len(top_idx)
    bg_surface = int((sasa >= SASA_SURFACE_THRESHOLD).sum())

    surface_rate_top = n_surface / n_total
    surface_rate_bg  = bg_surface / n

    print(f"  Top-{top_n} by (epistemic × depth):")
    print(f"    surface-exposed (SASA≥{SASA_SURFACE_THRESHOLD}): "
          f"{n_surface}/{n_total} = {surface_rate_top:.1%}  "
          f"(background: {surface_rate_bg:.1%})")

    enrichment = surface_rate_top / (surface_rate_bg + 1e-9)
    print(f"    surface enrichment: {enrichment:.2f}×  "
          f"({'ENRICHED ✓' if enrichment > 1.5 else 'not enriched ✗'})")

    # Functional surface overlap (KRAS-specific).
    functional_hits = 0
    if res_ids:
        print(f"\n  Top-{top_n} residue details:")
        print(f"  {'Rank':<5} {'ResID':<14} {'Score':>8} {'SASA':>6} "
              f"{'Depth':>7} {'Epi':>7} {'KRAS func?':>11}")
        print(f"  {'-'*64}")
        for rank, i in enumerate(top_idx, 1):
            rid   = res_ids[i] if i < len(res_ids) else f"?:{i}:"
            resnum = _parse_resnum(rid)
            is_func = resnum in KRAS_FUNCTIONAL_SURFACE if resnum else False
            if is_func:
                functional_hits += 1
            marker = "★" if is_func else " "
            print(f"  {rank:<5} {rid:<14} {score[i]:>8.4f} {sasa[i]:>6.3f} "
                  f"{depth[i]:>7.3f} {epistemic[i]:>7.4f} {marker:>11}")

        func_rate = functional_hits / n_total
        print(f"\n  Overlap with KRAS functional surface "
              f"(Switch-I/II, P-loop): {functional_hits}/{n_total} = {func_rate:.1%}")

    verdict = surface_rate_top > (surface_rate_bg * 1.4)
    print(
        f"\n  VERDICT: top peripheral cluster "
        f"{'ALIGNS WITH SURFACE ✓' if verdict else 'DOES NOT preferentially surface-align ✗'} "
        f"(enrichment={enrichment:.2f}×)"
    )


# ---------------------------------------------------------------------------
# PROBE 4 — Radial Gradient Strength
# ---------------------------------------------------------------------------

def probe4_radial_gradient(out: dict) -> None:
    print(f"\n{PROBE_SEPARATOR}")
    print("PROBE 4: Radial Gradient in the 2D Poincaré Disc")
    print(PROBE_SEPARATOR)
    print(
        "Measures whether cone_depth tracks disc radius (|hyp_proj_2d|).\n"
        "A shell-alive model shows a clear radial gradient:\n"
        "  low cone_depth  → disc centre  (core)\n"
        "  high cone_depth → disc edge    (outer shell)\n"
        "Quantified by Pearson r between |proj| and cone_depth.\n"
        "  r > 0.60 = gradient intact\n"
        "  r < 0.30 = gradient collapsed\n"
    )

    hyp = out["hyp_proj_2d"]
    depth = out["cone_depth"]
    sasa  = out["sasa"]

    disc_radius = np.linalg.norm(hyp, axis=1) if hyp.ndim == 2 else np.abs(hyp)

    r_radius_depth = _pearson(disc_radius, depth)
    r_radius_epi   = _pearson(disc_radius, out["epistemic"])
    r_radius_sasa  = _pearson(disc_radius, sasa)

    print(f"  {'Pair':<34} {'Pearson r':>10}  Strength")
    print(f"  {'-'*60}")
    _row("|proj| × cone_depth",   r_radius_depth,
         "> 0.60 strong; 0.30-0.60 moderate; < 0.30 collapsed")
    _row("|proj| × epistemic",    r_radius_epi,
         "high = uncertainty lives at disc periphery")
    _row("|proj| × SASA",         r_radius_sasa,
         "high = Poincaré periphery maps to molecular surface")

    # Bin disc by radius percentiles and show mean cone_depth per bin.
    pctiles = [0, 20, 40, 60, 80, 100]
    thresholds = np.percentile(disc_radius, pctiles)
    print(f"\n  Radial bins (disc radius percentile → mean cone_depth, mean SASA):")
    print(f"  {'Bin':<18} {'N':>5} {'mean depth':>11} {'mean SASA':>10} {'mean epi':>9}")
    print(f"  {'-'*58}")
    for lo, hi, plo, phi in zip(
        thresholds[:-1], thresholds[1:], pctiles[:-1], pctiles[1:]
    ):
        mask = (disc_radius >= lo) & (disc_radius < hi)
        if phi == 100:
            mask = disc_radius >= lo
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        md = float(depth[mask].mean())
        ms = float(sasa[mask].mean())
        me = float(out["epistemic"][mask].mean())
        print(f"  p{plo:>2}-p{phi:<2} (r={lo:.3f}-{hi:.3f})"
              f"  {cnt:>5}  {md:>11.4f}  {ms:>10.4f}  {me:>9.4f}")

    gradient_strength = r_radius_depth
    verdict = gradient_strength > 0.50
    print(
        f"\n  VERDICT: radial gradient is "
        f"{'INTACT ✓' if verdict else 'WEAK or ABSENT ✗'} "
        f"(r_radius_depth={gradient_strength:.3f})"
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(scores: List[Tuple[str, bool]]) -> None:
    print(f"\n{PROBE_SEPARATOR}")
    print("SUMMARY — Outer Shell Signal Health in GNNv5 / DTIE v6")
    print(PROBE_SEPARATOR)
    passed = sum(1 for _, v in scores if v)
    for label, ok in scores:
        print(f"  {'✓' if ok else '✗'}  {label}")
    print(f"\n  Score: {passed}/{len(scores)} probes passed")
    if passed == len(scores):
        print("  The outer shell is still a first-class citizen. ✓")
    elif passed >= len(scores) // 2:
        print(
            "  Partial shell signal — consider adding explicit shell supervision\n"
            "  (RSA + dehydron wrapper features, auxiliary shell-membership loss)."
        )
    else:
        print(
            "  Shell signal is significantly degraded. GNNv6 is inferring\n"
            "  primarily on the inner core. Restore outer-shell features and\n"
            "  add a dedicated shell-depth auxiliary loss to recover v2 insight."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _row(label: str, r: float, note: str) -> None:
    r_str = f"{r:>10.3f}" if np.isfinite(r) else f"{'NaN':>10}"
    print(f"  {label:<34} {r_str}  {note}")


def _shell_labels(sasa: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Assign each residue to core / intermediate / shell based on SASA + depth."""
    depth_75 = np.percentile(depth, 75)
    depth_25 = np.percentile(depth, 25)
    labels = np.full(len(sasa), "intermediate", dtype=object)
    labels[(sasa < SASA_SURFACE_THRESHOLD) & (depth < depth_25)] = "core"
    labels[(sasa >= SASA_SURFACE_THRESHOLD) & (depth >= depth_75)] = "shell"
    return labels


def _print_stratum_stats(
    out: dict,
    labels: np.ndarray,
    strata: List[str],
    fields: List[str],
) -> None:
    header = f"  {'Stratum':<14} {'N':>5}"
    for f in fields:
        header += f"  {f[:10]:>10}"
    print(f"\n{header}")
    print(f"  {'-' * (20 + 12 * len(fields))}")
    for s in strata:
        mask = labels == s
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        row = f"  {s:<14} {cnt:>5}"
        for f in fields:
            arr = out.get(f, out.get("epistemic"))
            if arr is not None and len(arr) > 0:
                row += f"  {float(arr[mask].mean()):>10.4f}"
        print(row)


def _parse_resnum(res_id: str) -> Optional[int]:
    """Extract residue number from 'A:123:' or '123' format."""
    try:
        parts = res_id.split(":")
        return int(parts[1]) if len(parts) >= 2 else int(parts[0])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNNv5 Outer Shell Signal Diagnostics"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/v5/tokyo_eyes_v5.pt",
        help="Path to v5 checkpoint (.pt)",
    )
    parser.add_argument(
        "--pdb_dir", type=str, default="/tmp/dtie_pdb_cache",
        help="PDB cache directory for structure download",
    )
    parser.add_argument(
        "--structure_id", type=str, default="4OBE",
        help="PDB structure ID to analyse",
    )
    parser.add_argument(
        "--chain", type=str, default="A",
        help="Chain to analyse",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="torch device (cpu or cuda:N)",
    )
    parser.add_argument(
        "--precomputed_json", type=str, default=None,
        help="Skip inference: load pre-computed GNN outputs from JSON export",
    )
    parser.add_argument(
        "--top_n", type=int, default=30,
        help="Number of top peripheral residues to inspect in Probe 3",
    )
    args = parser.parse_args()

    print(f"\n{'#' * 72}")
    print("  GNNv5 / DTIE v6 — Outer Shell Signal Diagnostics")
    print(f"  Structure: {args.structure_id}:{args.chain}")
    print(f"{'#' * 72}")

    # ---- Data loading ----
    prot: Optional[dict] = None
    ckpt: Optional[str]  = None

    if args.precomputed_json:
        print(f"\nLoading pre-computed outputs from {args.precomputed_json}")
        out = _load_from_json(args.precomputed_json)
    else:
        print(f"\nRunning inference: {args.structure_id} via {args.checkpoint}")
        out, prot = _load_via_model(
            args.structure_id,
            args.chain,
            args.checkpoint,
            Path(args.pdb_dir),
            args.device,
        )
        ckpt = args.checkpoint
        print(f"  Loaded {len(out['sasa'])} residues from {args.structure_id}:{args.chain}")

    # ---- Run probes ----
    probe1_uncertainty_sasa_correlation(out)

    probe2_core_ablation(out, ckpt, prot, args.device)

    probe3_surface_hotspot_alignment(out, top_n=args.top_n)

    probe4_radial_gradient(out)

    # ---- Summary ----
    # Re-derive pass/fail verdicts for the summary table.
    sasa      = out["sasa"]
    depth     = out["cone_depth"]
    epistemic = out["epistemic"]
    hyp       = out["hyp_proj_2d"]
    disc_radius = np.linalg.norm(hyp, axis=1) if hyp.ndim == 2 else np.abs(hyp)
    score       = epistemic * depth
    top_idx     = np.argsort(score)[::-1][:args.top_n]

    r_epi_sasa     = _pearson(epistemic, sasa)
    r_depth_sasa   = _pearson(depth, sasa)
    r_radius_depth = _pearson(disc_radius, depth)
    bg_surface     = float((sasa >= SASA_SURFACE_THRESHOLD).mean())
    top_surface    = float((sasa[top_idx] >= SASA_SURFACE_THRESHOLD).mean())
    enrichment     = top_surface / (bg_surface + 1e-9)
    depth_cutoff   = float(np.percentile(depth, DEPTH_OUTER_PERCENTILE))
    shell_mask     = (sasa >= SASA_OUTER_THRESHOLD) | (depth >= depth_cutoff)
    full_epi_spread = float(epistemic.std())
    core_epi_spread = float(epistemic[~shell_mask].std())
    epi_drop        = (full_epi_spread - core_epi_spread) / (full_epi_spread + 1e-9)

    verdicts = [
        ("Probe 1 — epistemic+depth correlate with SASA",
         (r_epi_sasa > 0.40) and (r_depth_sasa > 0.40)),
        ("Probe 2 — shell contributes to uncertainty spread",
         abs(epi_drop) > 0.10),
        ("Probe 3 — top peripheral residues surface-enriched",
         enrichment > 1.4),
        ("Probe 4 — radial gradient intact in Poincaré disc",
         r_radius_depth > 0.50),
    ]

    print_summary(verdicts)
    print()


if __name__ == "__main__":
    main()
