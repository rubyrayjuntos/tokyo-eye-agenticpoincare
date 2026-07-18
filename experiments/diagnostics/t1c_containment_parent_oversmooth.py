"""T1c / MP null on SSE parent nodes (design §5.2 oversmoothing-at-root).

Compares relative rank loss on the parent-node ``encoder_h`` slice
``[n_residue_nodes:]`` against a synthetic pre-MP control with rank matched
to mean children-per-parent fan-in, pushed through the same trained MP stack.

Usage (after containment cold arm):
  python -m experiments.diagnostics.t1c_containment_parent_oversmooth \\
    --checkpoint checkpoints/v66/runs/containment_pathb_stage_a12_cold_v1/v66_best.pt \\
    --out checkpoints/v66/diagnostics/t1c_containment_parent_oversmooth/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.t1c_mp_relative_loss_null import (
    _rel_drop,
    _run_mp_from_pre,
    _synth_embedding,
)
from experiments.diagnostics.trunk_hidden_occupancy import _match_x, _svd_pack
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint


def parent_slice_embeddings(
    embeddings: np.ndarray | torch.Tensor,
    n_residue_nodes: int,
) -> np.ndarray:
    """Return parent-row slice ``embeddings[n_residue_nodes:]`` only."""
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D embeddings, got shape {arr.shape}")
    if arr.shape[0] <= n_residue_nodes:
        return arr[0:0].reshape(0, arr.shape[1])
    return arr[n_residue_nodes:]


def _mean_fan_in(data: Any) -> float:
    """Mean children per parent from ``contain_down`` edge counts."""
    n_parent = int(getattr(data, "n_parent_nodes", 0) or 0)
    if n_parent <= 0:
        return 1.0
    counts = getattr(data, "containment_edge_counts", None) or {}
    down = int(counts.get("contain_down", 0))
    return max(1.0, float(down) / float(n_parent))


def _run_mp_parent_slice(
    model: torch.nn.Module,
    prot: dict[str, Any],
    device: str,
    *,
    pre_mp_parent_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (pre_mp_parent, encoder_h_parent, meta) for one protein."""
    structural = bool(getattr(model, "structural_disc_frozen", False))
    data = prepare_training_batch(
        model, prot, device, structural_disc_frozen=structural
    )
    data = _match_x(model, data)
    n_res = int(getattr(data, "n_residue_nodes", prot.get("n_residues", data.x.size(0))))
    n_parent = int(getattr(data, "n_parent_nodes", 0) or 0)
    fan_in = _mean_fan_in(data)

    override: torch.Tensor | None = None
    if pre_mp_parent_override is not None:
        natural_pre, _ = _run_mp_from_pre(
            model, prot, device, pre_mp_override=None, znorm_inputs=False, mu=None, sd=None
        )
        full = natural_pre.copy()
        parent_override = np.asarray(pre_mp_parent_override, dtype=np.float64)
        if parent_override.shape[0] != n_parent:
            raise ValueError(
                f"parent override rows {parent_override.shape[0]} != n_parent {n_parent}"
            )
        full[n_res : n_res + n_parent] = parent_override
        override = torch.as_tensor(full, device=device, dtype=torch.float32)

    pre, enc = _run_mp_from_pre(
        model,
        prot,
        device,
        pre_mp_override=override,
        znorm_inputs=False,
        mu=None,
        sd=None,
    )
    meta = {
        "n_residue_nodes": n_res,
        "n_parent_nodes": n_parent,
        "mean_fan_in": fan_in,
        "total_nodes": int(pre.shape[0]),
    }
    return (
        parent_slice_embeddings(pre, n_res),
        parent_slice_embeddings(enc, n_res),
        meta,
    )


def _parent_pooled(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    mode: str,
    rank: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate per-protein parent-slice relative rank drops."""
    pre_ers: list[float] = []
    enc_ers: list[float] = []
    rel_drops: list[float] = []
    fan_ins: list[float] = []
    n_parent_total = 0
    hidden = int(model.hidden)

    for i, prot in enumerate(proteins):
        structural = bool(getattr(model, "structural_disc_frozen", False))
        data = prepare_training_batch(
            model, prot, device, structural_disc_frozen=structural
        )
        data = _match_x(model, data)
        n_res = int(getattr(data, "n_residue_nodes", data.x.size(0)))
        n_parent = int(getattr(data, "n_parent_nodes", 0) or 0)
        fan_in = _mean_fan_in(data)
        fan_ins.append(fan_in)

        parent_override: np.ndarray | None = None
        if mode == "baseline":
            pass
        elif mode.startswith("synth_rank_"):
            if n_parent <= 0:
                continue
            target_rank = int(rank or max(1, round(fan_in)))
            parent_override = _synth_embedding(
                n_parent, hidden, target_rank, seed=seed + 17 * i
            )
            parent_override = parent_override * 0.5
        else:
            raise ValueError(mode)

        pre_p, enc_p, meta = _run_mp_parent_slice(
            model,
            prot,
            device,
            pre_mp_parent_override=parent_override,
        )
        n_parent_total += int(meta["n_parent_nodes"])
        if pre_p.shape[0] == 0:
            continue

        pre_s = _svd_pack(pre_p, name="pre_mp_parent")
        enc_s = _svd_pack(enc_p, name="encoder_h_parent")
        pre_er = float(pre_s["effective_rank"])
        enc_er = float(enc_s["effective_rank"])
        pre_ers.append(pre_er)
        enc_ers.append(enc_er)
        rel_drops.append(_rel_drop(pre_er, enc_er))

    if not rel_drops:
        return {
            "mode": mode,
            "n_parent_nodes": 0,
            "n_proteins": len(proteins),
            "error": "no_parent_nodes",
        }

    pre_mean = float(np.mean(pre_ers))
    enc_mean = float(np.mean(enc_ers))
    rel_mean = float(np.mean(rel_drops))
    return {
        "mode": mode,
        "n_parent_nodes": int(n_parent_total),
        "n_proteins": len(proteins),
        "mean_fan_in": float(np.mean(fan_ins)) if fan_ins else float("nan"),
        "pre_mp": {
            "effective_rank_mean": pre_mean,
            "effective_rank_std": float(np.std(pre_ers)),
        },
        "encoder_h": {
            "effective_rank_mean": enc_mean,
            "effective_rank_std": float(np.std(enc_ers)),
        },
        "relative_rank_drop": rel_mean,
        "relative_rank_drop_std": float(np.std(rel_drops)),
        "absolute_rank_drop": float(pre_mean - enc_mean),
    }


def _parent_verdict(
    baseline: dict[str, Any],
    synth_fanin: dict[str, Any],
    synth_r4: dict[str, Any],
) -> dict[str, Any]:
    """Same qualitative read as residue T1c (``MP_LOSS_*`` family)."""
    parent_drop = float(baseline.get("relative_rank_drop", float("nan")))
    null_drops = [
        float(synth_fanin.get("relative_rank_drop", float("nan"))),
        float(synth_r4.get("relative_rank_drop", float("nan"))),
    ]
    null_mean = float(np.nanmean(null_drops))
    excess = parent_drop - null_mean
    if excess <= 0.10:
        mp_verdict = "MP_LOSS_CONSISTENT_WITH_NULL"
    elif excess >= 0.20:
        mp_verdict = "MP_INDEPENDENT_SQUEEZE_T1C"
    else:
        mp_verdict = "MP_LOSS_BORDERLINE"

    reads = {
        "MP_LOSS_CONSISTENT_WITH_NULL": (
            "Parent relative rank loss matches synthetic fan-in controls — "
            "expected MP aggregation, not parent collapse beyond null."
        ),
        "MP_INDEPENDENT_SQUEEZE_T1C": (
            "Parent MP destroys more rank than synthetic controls — "
            "oversmoothing-at-root risk; do not interpret flow-influence."
        ),
        "MP_LOSS_BORDERLINE": (
            "Mild parent excess over null — watch on cold grade; "
            "no separate commitment yet."
        ),
    }
    return {
        "parent_relative_drop": parent_drop,
        "null_relative_drop_mean": null_mean,
        "excess_over_null": excess,
        "mp_verdict": mp_verdict,
        "read": reads[mp_verdict],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="T1c parent-node oversmoothing gate (containment §5.2)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("pdb_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/t1c_containment_parent_oversmooth"),
    )
    args = parser.parse_args(argv)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    proteins, _ = load_training_proteins(args.pdb_dir, args.corpus, max_residues=1200)
    model = load_model_from_checkpoint(args.checkpoint, device)
    model.eval()
    if not bool(getattr(model, "containment_edge_mp", False)):
        raise SystemExit(
            "checkpoint must have containment_edge_mp=True for parent oversmooth gate"
        )

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tc = raw.get("training_config") if isinstance(raw, dict) else {}
    if isinstance(tc, dict):
        model.structural_disc_frozen = bool(
            tc.get("structural_disc_frozen", False)
            or tc.get("slim_moe_structural_ssot", False)
        )
        model.hyperbolic_mp_graph = bool(tc.get("hyperbolic_mp_graph", False))

    # Fan-in-matched rank: round mean children per parent across corpus.
    fan_ins: list[float] = []
    for prot in proteins:
        structural = bool(getattr(model, "structural_disc_frozen", False))
        data = prepare_training_batch(
            model, prot, device, structural_disc_frozen=structural
        )
        fan_ins.append(_mean_fan_in(data))
    matched_rank = max(1, int(round(float(np.mean(fan_ins))))) if fan_ins else 2

    baseline = _parent_pooled(model, proteins, device, mode="baseline", seed=args.seed)
    synth_fanin = _parent_pooled(
        model,
        proteins,
        device,
        mode=f"synth_rank_{matched_rank}",
        rank=matched_rank,
        seed=args.seed,
    )
    synth_r4 = _parent_pooled(
        model, proteins, device, mode="synth_rank_4", rank=4, seed=args.seed
    )
    verdict = _parent_verdict(baseline, synth_fanin, synth_r4)

    report = {
        "checkpoint": str(args.checkpoint),
        "slice": "encoder_h[n_residue_nodes:]",
        "matched_fan_in_rank": matched_rank,
        "conditions": {
            "baseline_parent": baseline,
            f"synth_rank_{matched_rank}_parent": synth_fanin,
            "synth_rank_4_parent": synth_r4,
        },
        "verdict": verdict,
    }

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    print(
        f"parent drop={verdict['parent_relative_drop']:.3f} "
        f"null≈{verdict['null_relative_drop_mean']:.3f} "
        f"excess={verdict['excess_over_null']:.3f} → {verdict['mp_verdict']}"
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
