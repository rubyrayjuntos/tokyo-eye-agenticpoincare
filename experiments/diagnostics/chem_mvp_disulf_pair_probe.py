#!/usr/bin/env python3
"""Per-structure disulfide chem-pair probe for Chem-MVP vs matched baseline.

Pre-registered in ``docs/specs/struct-conn-typed-edges/ablation.md``:

- Physics gates first (caller should stop on rim / cone·τ regression).
- Report ``1LYZ`` / ``1F88`` / ``1IVO`` individually before any pool.
- Trunk (``encoder_h`` Euclidean) and disc (Poincaré) separately.
- Dominance guard: refuse pooled win if any structure >40% of pairs.
- Win needs sign-correct trunk compaction on all three structures, not token Δ.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.common.hyperbolic_lorentz_ops import poincare_distance
from science.dtie.v66.chem_edge_graph import map_bond_endpoints_to_nodes
from science.training.gnn_lineage import load_model_from_checkpoint

DISULF_PROBE_STRUCTURES = ("1LYZ", "1F88", "1IVO")
DOMINANCE_PAIR_SHARE_MAX = 0.40
# Same floor as KRAS OOD ``ambiguous_small`` — below this is "technically nonzero."
NONTRIVIAL_ABS_NORM_DELTA = 0.02


def mapped_disulf_pairs(
    prot: Mapping[str, Any],
    *,
    bond_type: str = "disulf",
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Return undirected in-graph pairs for ``bond_type`` plus raw/mapped counts."""
    bonds = list(prot.get("covalent_bonds") or [])
    structure_id = str(prot.get("structure_id") or prot.get("pdb_id") or "").lower()
    chain = str(prot.get("chain") or "A")
    residue_ids = prot.get("residue_ids")
    if residue_ids is None:
        raise ValueError("prot requires residue_ids")
    mapped, skipped = map_bond_endpoints_to_nodes(
        bonds,
        residue_ids,
        structure_id=structure_id,
        chain_label=chain,
    )
    raw_type = sum(1 for b in bonds if str(b.get("bond_type", "")).lower() == bond_type)
    pairs = [(i, j) for i, j, bt in mapped if bt == bond_type]
    return pairs, {
        "raw_bonds_total": len(bonds),
        "raw_bonds_type": int(raw_type),
        "mapped_pairs_type": len(pairs),
        "map_skipped": int(skipped),
    }


def summarize_pair_distances_disc(
    disc: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
    *,
    curvature: float,
) -> dict[str, Any]:
    """Poincaré chem-pair mean normalized by all-residue pair median."""
    disc = np.asarray(disc, dtype=np.float64)
    if disc.ndim != 2:
        raise ValueError(f"disc must be [N, 2], got {disc.shape}")
    if not pair_indices:
        raise ValueError("pair_indices must be non-empty")

    all_distances = [
        poincare_distance(disc[i], disc[j], curvature)
        for i, j in combinations(range(disc.shape[0]), 2)
    ]
    all_pair_median = float(np.median(all_distances))
    if not np.isfinite(all_pair_median) or all_pair_median <= 0.0:
        raise ValueError("All-residue Poincare distance median is not positive")

    pair_rows: dict[str, dict[str, float]] = {}
    dists: list[float] = []
    for i, j in pair_indices:
        d = float(poincare_distance(disc[i], disc[j], curvature))
        pair_rows[f"{i}-{j}"] = {"poincare": d}
        dists.append(d)
    mean = float(np.mean(dists))
    return {
        "layer": "hyp_projections_2d",
        "metric": "poincare",
        "n_pairs": len(pair_indices),
        "pairs": pair_rows,
        "chem_mean_poincare": mean,
        "all_pair_median_poincare": all_pair_median,
        "chem_mean_poincare_normalized": mean / all_pair_median,
    }


def summarize_pair_distances_trunk(
    trunk: np.ndarray,
    pair_indices: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Euclidean chem-pair mean on ``encoder_h``, normalized by all-pair median."""
    trunk = np.asarray(trunk, dtype=np.float64)
    if trunk.ndim != 2:
        raise ValueError(f"encoder_h must be [N, D], got {trunk.shape}")
    if not pair_indices:
        raise ValueError("pair_indices must be non-empty")

    all_distances = [
        float(np.linalg.norm(trunk[i] - trunk[j]))
        for i, j in combinations(range(trunk.shape[0]), 2)
    ]
    all_pair_median = float(np.median(all_distances))
    if not np.isfinite(all_pair_median) or all_pair_median <= 0.0:
        raise ValueError("All-residue Euclidean distance median is not positive")

    pair_rows: dict[str, dict[str, float]] = {}
    dists: list[float] = []
    for i, j in pair_indices:
        d = float(np.linalg.norm(trunk[i] - trunk[j]))
        pair_rows[f"{i}-{j}"] = {"euclidean": d}
        dists.append(d)
    mean = float(np.mean(dists))
    return {
        "layer": "encoder_h",
        "metric": "euclidean_l2",
        "n_pairs": len(pair_indices),
        "pairs": pair_rows,
        "chem_mean_euclidean": mean,
        "all_pair_median_euclidean": all_pair_median,
        "chem_mean_euclidean_normalized": mean / all_pair_median,
    }


def layer_verdict(trunk_norm_delta: float, disc_norm_delta: float) -> str:
    """Classify trunk vs disc compaction (negative = closer under chem)."""
    if (
        abs(trunk_norm_delta) < NONTRIVIAL_ABS_NORM_DELTA
        and abs(disc_norm_delta) >= NONTRIVIAL_ABS_NORM_DELTA
    ):
        return "projection_only"
    if (
        abs(trunk_norm_delta) >= NONTRIVIAL_ABS_NORM_DELTA
        and np.sign(trunk_norm_delta) == np.sign(disc_norm_delta)
    ):
        return "trunk_and_disc_same_sign"
    if (
        abs(trunk_norm_delta) >= NONTRIVIAL_ABS_NORM_DELTA
        and np.sign(trunk_norm_delta) != np.sign(disc_norm_delta)
    ):
        return "trunk_disc_disagree"
    return "ambiguous_small"


def _capture_encoder_h(model: torch.nn.Module) -> tuple[list[Any], dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _radial_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()

    def _angular_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
        if "encoder_h" not in captured and inputs and torch.is_tensor(inputs[0]):
            captured["encoder_h"] = inputs[0].detach()

    handles.append(model.radial_head.register_forward_hook(_radial_hook))
    if hasattr(model, "angular_head"):
        handles.append(model.angular_head.register_forward_hook(_angular_hook))
    return handles, captured


def _forward(
    checkpoint: Path,
    prot: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = load_model_from_checkpoint(checkpoint, device)
    model.eval()
    handles, captured = _capture_encoder_h(model)
    try:
        data = prepare_training_batch(model, prot, device)
        expected_width = int(model.node_emb.in_features)
        actual_width = int(data.x.size(1))
        if actual_width != expected_width:
            raise ValueError(
                f"{checkpoint}: model expects node width {expected_width}, got {actual_width}"
            )
        with torch.no_grad():
            out = model(data)
    finally:
        for handle in handles:
            handle.remove()
    if "encoder_h" not in captured:
        raise RuntimeError(f"{checkpoint}: failed to capture encoder_h")
    disc = out["hyp_projections_2d"].detach().cpu().numpy()
    encoder_h = captured["encoder_h"].cpu().numpy()
    curvature = float(model.curvature.detach().cpu().item())
    return disc, encoder_h, curvature


def evaluate_structure(
    *,
    prot: dict[str, Any],
    baseline_checkpoint: Path,
    chem_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    pdb_id = str(prot.get("pdb_id") or prot.get("structure_id") or "").upper()
    pairs, counts = mapped_disulf_pairs(prot)
    if not pairs:
        return {
            "pdb_id": pdb_id,
            "chain": prot.get("chain"),
            "error": "no_mapped_disulf_pairs",
            "pair_counts": counts,
        }

    base_disc, base_trunk, base_c = _forward(baseline_checkpoint, prot, device)
    chem_disc, chem_trunk, chem_c = _forward(chem_checkpoint, prot, device)

    baseline_disc = summarize_pair_distances_disc(base_disc, pairs, curvature=base_c)
    chem_disc_s = summarize_pair_distances_disc(chem_disc, pairs, curvature=chem_c)
    baseline_trunk = summarize_pair_distances_trunk(base_trunk, pairs)
    chem_trunk_s = summarize_pair_distances_trunk(chem_trunk, pairs)

    trunk_norm_delta = (
        chem_trunk_s["chem_mean_euclidean_normalized"]
        - baseline_trunk["chem_mean_euclidean_normalized"]
    )
    disc_norm_delta = (
        chem_disc_s["chem_mean_poincare_normalized"]
        - baseline_disc["chem_mean_poincare_normalized"]
    )
    trunk_compacts = trunk_norm_delta <= -NONTRIVIAL_ABS_NORM_DELTA
    disc_compacts = disc_norm_delta <= -NONTRIVIAL_ABS_NORM_DELTA
    trunk_token_only = (
        trunk_norm_delta < 0.0 and abs(trunk_norm_delta) < NONTRIVIAL_ABS_NORM_DELTA
    )

    return {
        "pdb_id": pdb_id,
        "chain": prot.get("chain"),
        "n_residues": int(prot["data"].x.shape[0]),
        "pair_counts": counts,
        "mapped_disulf_pairs": [f"{i}-{j}" for i, j in pairs],
        "baseline": {
            "curvature": base_c,
            "disc": baseline_disc,
            "trunk": baseline_trunk,
        },
        "chem": {
            "curvature": chem_c,
            "disc": chem_disc_s,
            "trunk": chem_trunk_s,
        },
        "delta_chem_minus_baseline": {
            "chem_mean_poincare": (
                chem_disc_s["chem_mean_poincare"] - baseline_disc["chem_mean_poincare"]
            ),
            "chem_mean_poincare_normalized": disc_norm_delta,
            "chem_mean_euclidean_trunk": (
                chem_trunk_s["chem_mean_euclidean"] - baseline_trunk["chem_mean_euclidean"]
            ),
            "chem_mean_euclidean_trunk_normalized": trunk_norm_delta,
            "layer_verdict": layer_verdict(trunk_norm_delta, disc_norm_delta),
            "trunk_compacts_nontrivial": trunk_compacts,
            "disc_compacts_nontrivial": disc_compacts,
            "trunk_token_compaction_only": trunk_token_only,
        },
    }


def aggregate_verdict(structures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply ablation win / partial / fail locks (thin-N skeptical)."""
    rows = [s for s in structures if "delta_chem_minus_baseline" in s]
    by_id = {str(s["pdb_id"]).upper(): s for s in rows}
    missing = [p for p in DISULF_PROBE_STRUCTURES if p not in by_id]
    pair_counts = {
        pid: int(by_id[pid]["pair_counts"]["mapped_pairs_type"])
        for pid in DISULF_PROBE_STRUCTURES
        if pid in by_id
    }
    total_pairs = sum(pair_counts.values())
    pair_share = {
        pid: (pair_counts[pid] / total_pairs if total_pairs else 0.0)
        for pid in pair_counts
    }
    dominance_ok = all(share <= DOMINANCE_PAIR_SHARE_MAX for share in pair_share.values())

    trunk_ok = {
        pid: bool(by_id[pid]["delta_chem_minus_baseline"]["trunk_compacts_nontrivial"])
        for pid in DISULF_PROBE_STRUCTURES
        if pid in by_id
    }
    disc_ok = {
        pid: bool(by_id[pid]["delta_chem_minus_baseline"]["disc_compacts_nontrivial"])
        for pid in DISULF_PROBE_STRUCTURES
        if pid in by_id
    }
    token_only = {
        pid: bool(by_id[pid]["delta_chem_minus_baseline"]["trunk_token_compaction_only"])
        for pid in DISULF_PROBE_STRUCTURES
        if pid in by_id
    }
    layer = {
        pid: by_id[pid]["delta_chem_minus_baseline"]["layer_verdict"]
        for pid in DISULF_PROBE_STRUCTURES
        if pid in by_id
    }

    n_trunk = sum(1 for v in trunk_ok.values() if v)
    n_disc = sum(1 for v in disc_ok.values() if v)
    any_measurable = any(
        abs(by_id[pid]["delta_chem_minus_baseline"]["chem_mean_euclidean_trunk_normalized"])
        >= NONTRIVIAL_ABS_NORM_DELTA
        or abs(by_id[pid]["delta_chem_minus_baseline"]["chem_mean_poincare_normalized"])
        >= NONTRIVIAL_ABS_NORM_DELTA
        for pid in by_id
    )

    if missing:
        outcome = "fail"
        reason = f"missing_structures:{','.join(missing)}"
    elif n_trunk == 3 and all(not token_only[p] for p in DISULF_PROBE_STRUCTURES):
        # Ablation also requires dominance for any pooled win claim; per-structure
        # win still needs all three nontrivial trunk compactors.
        outcome = "win" if dominance_ok else "partial"
        reason = (
            "trunk_compaction_all_three"
            if dominance_ok
            else "trunk_ok_but_pooled_dominance_guard_fails"
        )
    elif n_trunk >= 1 and n_trunk < 3:
        outcome = "partial"
        reason = "trunk_compaction_incomplete_across_structures"
    elif n_disc >= 1 and n_trunk == 0:
        outcome = "partial"
        reason = "disc_only_compaction"
    elif any(token_only.values()) and n_trunk == 0:
        outcome = "partial"
        reason = "token_trunk_deltas_only"
    elif not any_measurable:
        outcome = "fail"
        reason = "no_measurable_pairwise_effect"
    else:
        outcome = "partial"
        reason = "mixed_or_non_win"

    return {
        "outcome": outcome,
        "reason": reason,
        "nontrivial_abs_norm_delta": NONTRIVIAL_ABS_NORM_DELTA,
        "dominance_pair_share_max": DOMINANCE_PAIR_SHARE_MAX,
        "mapped_pair_counts": pair_counts,
        "mapped_pair_share": pair_share,
        "dominance_guard_passes": dominance_ok,
        "trunk_compacts_nontrivial": trunk_ok,
        "disc_compacts_nontrivial": disc_ok,
        "trunk_token_compaction_only": token_only,
        "layer_verdict": layer,
        "n_structures_trunk_ok": n_trunk,
        "n_structures_disc_ok": n_disc,
    }


def load_probe_proteins(cache_path: Path) -> dict[str, dict[str, Any]]:
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    proteins = blob.get("proteins") or []
    out: dict[str, dict[str, Any]] = {}
    for prot in proteins:
        pid = str(prot.get("pdb_id") or prot.get("structure_id") or "").upper()
        if pid in DISULF_PROBE_STRUCTURES:
            out[pid] = prot
    missing = [p for p in DISULF_PROBE_STRUCTURES if p not in out]
    if missing:
        raise RuntimeError(f"{cache_path}: missing {missing}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("pdb_cache/corpus_cache/graphs_38a6993d7a439aa4.pt"),
        help="Stage A-12 cache with covalent_bonds (covbond_v1)",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/chem_mvp_baseline_role_stage_a12_cold_v1/v66_best.pt"
        ),
    )
    parser.add_argument(
        "--chem-checkpoint",
        type=Path,
        default=Path("checkpoints/v66/runs/chem_mvp_stage_a12_cold_v1/v66_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "checkpoints/v66/diagnostics/chem_mvp_stage_a12/disulf_pair_probe.json"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.environ.setdefault("GNN_INPUT_MODE", "topology_three_vector")

    proteins = load_probe_proteins(args.corpus_cache)
    results: list[dict[str, Any]] = []
    # Explicit order: small counts first, then 1IVO — not pooled-first.
    for pdb_id in DISULF_PROBE_STRUCTURES:
        print(f"evaluating {pdb_id} ...", flush=True)
        results.append(
            evaluate_structure(
                prot=proteins[pdb_id],
                baseline_checkpoint=args.baseline_checkpoint,
                chem_checkpoint=args.chem_checkpoint,
                device=args.device,
            )
        )

    verdict = aggregate_verdict(results)
    report = {
        "schema_version": 1,
        "interpretation": (
            "Negative chem−baseline normalized deltas mean Chem-MVP places "
            "disulfide cysteines closer. Trunk = encoder_h L2; disc = Poincare. "
            "Pooled numbers are secondary; dominance guard and per-structure "
            "trunk compaction decide win vs partial."
        ),
        "corpus_cache": str(args.corpus_cache),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "chem_checkpoint": str(args.chem_checkpoint),
        "structures": results,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"verdict": verdict, "per_structure": [
        {
            "pdb_id": s["pdb_id"],
            "mapped_disulf": s.get("pair_counts", {}).get("mapped_pairs_type"),
            "trunk_norm_delta": s.get("delta_chem_minus_baseline", {}).get(
                "chem_mean_euclidean_trunk_normalized"
            ),
            "disc_norm_delta": s.get("delta_chem_minus_baseline", {}).get(
                "chem_mean_poincare_normalized"
            ),
            "layer_verdict": s.get("delta_chem_minus_baseline", {}).get("layer_verdict"),
            "trunk_ok": s.get("delta_chem_minus_baseline", {}).get(
                "trunk_compacts_nontrivial"
            ),
        }
        for s in results
    ]}, indent=2))


if __name__ == "__main__":
    main()
