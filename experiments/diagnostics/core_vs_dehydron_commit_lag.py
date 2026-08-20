"""Core vs dehydron max-p trajectory + board-gate logit-gap audit.

Pre-reg branch diagnostic for dehydron commitment lag on the core-quota lineage.

Checks (topology_only HyperbolicPrototypeGate — board→prototype logits, not raw hyp embeds):

1. **Magnitude / peakedness gap:** per-residue top1−top2 logit gap and logit range
   on soft gate scores (quota off), core vs dehydron populations.
2. **Gradient starvation shape:** early (ge5→15) vs late (ge15→30) Δ mean-max-p
   for each population. Starvation = similar early Δ, dehydron flattens late while
   core keeps climbing.

Usage:
  python -m experiments.diagnostics.core_vs_dehydron_commit_lag \\
    --run-dir checkpoints/v66/runs/fix1_s4_stack_core_quota_stage_a12_cold_v1 \\
    --corpus manifests/v6_corpus_stage_a_small_v1.json \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint


def _match_x(model: torch.nn.Module, data: Any) -> Any:
    in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
    if data.x.size(-1) > in_f:
        data.x = data.x[:, :in_f].contiguous()
    return data


def _dehydron_mask(prot: dict[str, Any], data: Any, n: int) -> np.ndarray:
    if prot.get("target_dehydron") is not None:
        dh = np.asarray(prot["target_dehydron"], dtype=float).reshape(-1)[:n]
    elif data.x.size(1) > 1:
        dh = data.x[:, 1].detach().float().cpu().numpy()[:n]
    else:
        dh = np.zeros(n, dtype=float)
    return dh > 0.5


def _pop_stats(
    max_p: np.ndarray,
    logit_range: np.ndarray,
    top12_gap: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    if int(mask.sum()) == 0:
        return {
            "n": 0,
            "mean_max_p": float("nan"),
            "frac_max_p_ge_0_60": float("nan"),
            "mean_logit_range": float("nan"),
            "mean_top1_top2_gap": float("nan"),
            "median_top1_top2_gap": float("nan"),
        }
    mp = max_p[mask]
    lr = logit_range[mask]
    g = top12_gap[mask]
    return {
        "n": int(mask.sum()),
        "mean_max_p": float(mp.mean()),
        "frac_max_p_ge_0_60": float(np.mean(mp >= 0.60)),
        "mean_logit_range": float(lr.mean()),
        "mean_top1_top2_gap": float(g.mean()),
        "median_top1_top2_gap": float(np.median(g)),
    }


@torch.no_grad()
def measure_epoch(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    apply_quota: bool = False,
    tau_cap: float = 0.40,
    dominant_by_pdb: dict[str, list[bool]] | None = None,
) -> dict[str, Any]:
    """Soft max-p + pre-softmax peakedness by population.

    When ``apply_quota`` is False (default), forces ``core_capacity_quota_tau=0``
    so expert_weights match gate softmax (board→prototype). When True, restores
    frozen dominant masks + τ_cap so mix matches training companions.
    """
    was_tau = float(getattr(model, "core_capacity_quota_tau", 0.0) or 0.0)
    if apply_quota:
        model.core_capacity_quota_tau = float(tau_cap)
        if dominant_by_pdb is not None:
            model._core_quota_dominant = {
                k: torch.tensor(v, dtype=torch.bool)
                for k, v in dominant_by_pdb.items()
            }
    else:
        model.core_capacity_quota_tau = 0.0
    model.eval()

    max_ps: list[float] = []
    ranges: list[float] = []
    gaps: list[float] = []
    is_dh: list[bool] = []
    hard_expert: list[int] = []
    gate_max_ps: list[float] = []

    for prot in proteins:
        data = prepare_training_batch(model, prot, device)
        data = _match_x(model, data)
        out = model(data)
        w = out["expert_weights"].float()
        pre = getattr(model.gate, "_last_pre_softmax", None) or {}
        logits = pre.get("adjusted_logits")
        if logits is None:
            logits = pre.get("raw_logits")
        if logits is None:
            logits = w.clamp_min(1e-8).log()
        logits = logits.float()
        gate_soft = F.softmax(logits, dim=-1)
        # Mix weights (post-quota when enabled); gate soft always from logits.
        max_p = w.max(dim=-1).values
        gate_max_p = gate_soft.max(dim=-1).values
        row_range = logits.max(dim=-1).values - logits.min(dim=-1).values
        top2 = torch.topk(logits, k=2, dim=-1).values
        gap = top2[:, 0] - top2[:, 1]
        n = w.shape[0]
        dh = _dehydron_mask(prot, data, n)
        max_ps.extend(max_p.cpu().tolist())
        gate_max_ps.extend(gate_max_p.cpu().tolist())
        ranges.extend(row_range.cpu().tolist())
        gaps.extend(gap.cpu().tolist())
        is_dh.extend(dh.tolist())
        hard_expert.extend(w.argmax(dim=-1).cpu().tolist())

    model.core_capacity_quota_tau = was_tau

    max_ps_a = np.asarray(max_ps, dtype=float)
    gate_max_a = np.asarray(gate_max_ps, dtype=float)
    ranges_a = np.asarray(ranges, dtype=float)
    gaps_a = np.asarray(gaps, dtype=float)
    dh_a = np.asarray(is_dh, dtype=bool)
    hard_a = np.asarray(hard_expert, dtype=int)

    core = ~dh_a
    deh = dh_a
    if int(deh.sum()) > 0:
        vals, counts = np.unique(hard_a[deh], return_counts=True)
        preferred = int(vals[int(np.argmax(counts))])
    else:
        preferred = -1
    on_pole = deh & (hard_a == preferred) if preferred >= 0 else np.zeros_like(deh)
    off_pole = deh & (hard_a != preferred) if preferred >= 0 else np.zeros_like(deh)

    def _gate_pop(mask: np.ndarray) -> dict[str, float]:
        if int(mask.sum()) == 0:
            return {"n": 0, "mean_max_p": float("nan"), "frac_max_p_ge_0_60": float("nan")}
        mp = gate_max_a[mask]
        return {
            "n": int(mask.sum()),
            "mean_max_p": float(mp.mean()),
            "frac_max_p_ge_0_60": float(np.mean(mp >= 0.60)),
        }

    return {
        "apply_quota": bool(apply_quota),
        "n_total": int(len(max_ps_a)),
        "n_core": int(core.sum()),
        "n_dehydron": int(deh.sum()),
        "dehydron_preferred_expert": preferred,
        "mix": {
            "core": _pop_stats(max_ps_a, ranges_a, gaps_a, core),
            "dehydron": _pop_stats(max_ps_a, ranges_a, gaps_a, deh),
            "dehydron_on_preferred_pole": _pop_stats(
                max_ps_a, ranges_a, gaps_a, on_pole
            ),
            "dehydron_off_preferred_pole": _pop_stats(
                max_ps_a, ranges_a, gaps_a, off_pole
            ),
        },
        "gate_soft": {
            "core": _gate_pop(core),
            "dehydron": _gate_pop(deh),
        },
        "logit_peakedness": {
            "core_mean_top1_top2_gap": float(
                _pop_stats(max_ps_a, ranges_a, gaps_a, core)["mean_top1_top2_gap"]
            ),
            "dehydron_mean_top1_top2_gap": float(
                _pop_stats(max_ps_a, ranges_a, gaps_a, deh)["mean_top1_top2_gap"]
            ),
            "core_mean_logit_range": float(
                _pop_stats(max_ps_a, ranges_a, gaps_a, core)["mean_logit_range"]
            ),
            "dehydron_mean_logit_range": float(
                _pop_stats(max_ps_a, ranges_a, gaps_a, deh)["mean_logit_range"]
            ),
            "gap_ratio_core_over_deh": (
                float(
                    _pop_stats(max_ps_a, ranges_a, gaps_a, core)["mean_top1_top2_gap"]
                    / max(
                        _pop_stats(max_ps_a, ranges_a, gaps_a, deh)[
                            "mean_top1_top2_gap"
                        ],
                        1e-12,
                    )
                )
                if int(deh.sum()) and int(core.sum())
                else float("nan")
            ),
        },
    }


def _branch(
    early_core: float,
    early_deh: float,
    late_core: float,
    late_deh: float,
    gap_ratio_ge15: float,
    *,
    magnitude_gap_floor: float = 1.25,
    starvation_late_ratio: float = 0.40,
) -> dict[str, Any]:
    """Three-way branch from magnitude gap × starvation shape."""
    mag = (
        gap_ratio_ge15 == gap_ratio_ge15
        and gap_ratio_ge15 >= magnitude_gap_floor
    )
    # Similar early trajectories: |early_deh - early_core| small relative to scales
    early_similar = (
        early_core == early_core
        and early_deh == early_deh
        and abs(early_deh - early_core) <= max(0.05, 0.5 * abs(early_core))
    )
    late_starve = (
        late_core == late_core
        and late_deh == late_deh
        and late_core > 0.02
        and late_deh < starvation_late_ratio * late_core
    )
    starvation = early_similar and late_starve

    if mag and starvation:
        branch = "BOTH_MAGNITUDE_GAP_AND_STARVATION"
        lever = "two_pool_masking + reverse_commit_hinge"
    elif mag and not starvation:
        branch = "MAGNITUDE_GAP_ONLY"
        lever = "reverse_commit_hinge (STE ReLU(0.60-p_e*)^2 on dh=1)"
    elif starvation and not mag:
        branch = "STARVATION_ONLY"
        lever = "two_pool_masking / separate pool temperature (likely sufficient)"
    else:
        branch = "NEITHER_CLEAR"
        lever = "re-inspect; do not assume either lever"

    return {
        "magnitude_gap_present": bool(mag),
        "starvation_pattern_present": bool(starvation),
        "early_deltas_similar": bool(early_similar),
        "late_dehydron_flat_vs_core": bool(late_starve),
        "branch": branch,
        "recommended_lever": lever,
        "floors": {
            "magnitude_gap_core_over_deh_min": magnitude_gap_floor,
            "starvation_late_deh_over_core_max": starvation_late_ratio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "checkpoints/v66/runs/fix1_s4_stack_core_quota_stage_a12_cold_v1"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--epochs",
        type=str,
        default="5,15,30",
        help="Comma-separated epoch checkpoints to score (default 5,15,30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <run-dir>/core_vs_dehydron_commit_lag.json",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.run_dir / "core_vs_dehydron_commit_lag.json"

    epochs = [int(x.strip()) for x in str(args.epochs).split(",") if x.strip()]
    proteins, _failed = load_training_proteins(
        args.pdb_dir,
        args.corpus,
        max_proteins=12,
    )

    freeze_path = args.run_dir / "core_quota_dominant_ge0.json"
    dominant_by_pdb: dict[str, list[bool]] | None = None
    if freeze_path.exists():
        dominant_by_pdb = json.loads(freeze_path.read_text()).get("dominant_by_pdb")

    per_epoch_gate: dict[str, Any] = {}
    per_epoch_quota: dict[str, Any] = {}
    for ep in epochs:
        ckpt = args.run_dir / "epochs" / f"epoch_{ep:03d}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = load_model_from_checkpoint(ckpt, args.device)
        model.to(args.device)
        gate_stats = measure_epoch(
            model, proteins, args.device, apply_quota=False
        )
        quota_stats = measure_epoch(
            model,
            proteins,
            args.device,
            apply_quota=True,
            tau_cap=0.40,
            dominant_by_pdb=dominant_by_pdb,
        )
        per_epoch_gate[str(ep)] = gate_stats
        per_epoch_quota[str(ep)] = quota_stats
        g = gate_stats
        lp = g["logit_peakedness"]
        print(
            f"ge{ep} GATE: core mean_max_p={g['gate_soft']['core']['mean_max_p']:.4f} "
            f"frac≥0.60={g['gate_soft']['core']['frac_max_p_ge_0_60']:.3f} "
            f"gap={lp['core_mean_top1_top2_gap']:.4f} | "
            f"deh mean_max_p={g['gate_soft']['dehydron']['mean_max_p']:.4f} "
            f"frac≥0.60={g['gate_soft']['dehydron']['frac_max_p_ge_0_60']:.3f} "
            f"gap={lp['dehydron_mean_top1_top2_gap']:.4f} "
            f"pole=e{g['dehydron_preferred_expert']}"
        )
        qm = quota_stats["mix"]
        print(
            f"ge{ep} QUOTA-mix: core mean_max_p={qm['core']['mean_max_p']:.4f} "
            f"frac≥0.60={qm['core']['frac_max_p_ge_0_60']:.3f} | "
            f"deh mean_max_p={qm['dehydron']['mean_max_p']:.4f} "
            f"frac≥0.60={qm['dehydron']['frac_max_p_ge_0_60']:.3f}"
        )

    def _gate_mp(ep: int, pop: str) -> float:
        return float(per_epoch_gate[str(ep)]["gate_soft"][pop]["mean_max_p"])

    def _gap(ep: int, pop: str) -> float:
        key = (
            "core_mean_top1_top2_gap"
            if pop == "core"
            else "dehydron_mean_top1_top2_gap"
        )
        return float(per_epoch_gate[str(ep)]["logit_peakedness"][key])

    def _quota_mp(ep: int, pop: str) -> float:
        return float(per_epoch_quota[str(ep)]["mix"][pop]["mean_max_p"])

    def _quota_frac(ep: int, pop: str) -> float:
        return float(per_epoch_quota[str(ep)]["mix"][pop]["frac_max_p_ge_0_60"])

    if 5 in epochs and 15 in epochs and 30 in epochs:
        early_core = _gate_mp(15, "core") - _gate_mp(5, "core")
        early_deh = _gate_mp(15, "dehydron") - _gate_mp(5, "dehydron")
        late_core = _gate_mp(30, "core") - _gate_mp(15, "core")
        late_deh = _gate_mp(30, "dehydron") - _gate_mp(15, "dehydron")
        gap_ratio_15 = _gap(15, "core") / max(_gap(15, "dehydron"), 1e-12)
        gap_ratio_30 = _gap(30, "core") / max(_gap(30, "dehydron"), 1e-12)
        # Inverted: dehydron more peaked than core
        deh_stronger = gap_ratio_15 < 1.0
    else:
        raise SystemExit("Need epochs 5,15,30 for early/late starvation split")

    branch = _branch(early_core, early_deh, late_core, late_deh, gap_ratio_15)
    if deh_stronger and not branch["magnitude_gap_present"]:
        branch = {
            **branch,
            "branch": "DEHYDRON_STRONGER_PEAKEDNESS_NEITHER_GAP_NOR_STARVATION",
            "recommended_lever": (
                "Do NOT treat as magnitude-gap reverse-hinge. "
                "Investigate quota STE artifact: core gets hard one-hots while "
                "dehydron stays soft — committed purity may lag without gate lag."
            ),
            "dehydron_stronger_peakedness": True,
        }

    report = {
        "run_dir": str(args.run_dir),
        "note": (
            "Gate path: core_capacity_quota_tau=0 during measure. "
            "Quota-mix path: frozen ge0 dominant masks + τ_cap=0.40 (training companions). "
            "Logit gaps from adjusted_logits (board→prototype), not hyp node embeds."
        ),
        "per_epoch_gate": per_epoch_gate,
        "per_epoch_quota_mix": per_epoch_quota,
        "deltas_gate_soft": {
            "early_ge5_to_ge15": {
                "core_mean_max_p": early_core,
                "dehydron_mean_max_p": early_deh,
            },
            "late_ge15_to_ge30": {
                "core_mean_max_p": late_core,
                "dehydron_mean_max_p": late_deh,
            },
        },
        "magnitude_gap_gate": {
            "ge15_core_over_deh_top1_top2_gap": gap_ratio_15,
            "ge30_core_over_deh_top1_top2_gap": gap_ratio_30,
            "ge15_core_mean_top1_top2_gap": _gap(15, "core"),
            "ge15_dehydron_mean_top1_top2_gap": _gap(15, "dehydron"),
            "ge30_core_mean_top1_top2_gap": _gap(30, "core"),
            "ge30_dehydron_mean_top1_top2_gap": _gap(30, "dehydron"),
            "dehydron_stronger_than_core": bool(deh_stronger),
        },
        "quota_mix_commit_asymmetry": {
            "ge30_core_frac_ge_0_60": _quota_frac(30, "core"),
            "ge30_dehydron_frac_ge_0_60": _quota_frac(30, "dehydron"),
            "ge30_core_mean_max_p": _quota_mp(30, "core"),
            "ge30_dehydron_mean_max_p": _quota_mp(30, "dehydron"),
            "note": (
                "If core frac≫dehydron frac under quota but gate soft is comparable "
                "(or dehydron stronger), the committed-population lag is an STE "
                "hardening artifact on core, not a gate magnitude/starvation failure."
            ),
        },
        "branch": branch,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\nEarly (ge5→15) GATE: core {:+.4f} | dehydron {:+.4f}".format(early_core, early_deh))
    print("Late  (ge15→30) GATE: core {:+.4f} | dehydron {:+.4f}".format(late_core, late_deh))
    print(
        "Gap ratio core/deh @ge15={:.3f} @ge30={:.3f} (dehydron_stronger={})".format(
            gap_ratio_15, gap_ratio_30, deh_stronger
        )
    )
    print(
        "QUOTA-mix ge30 frac≥0.60: core={:.3f} deh={:.3f}".format(
            _quota_frac(30, "core"), _quota_frac(30, "dehydron")
        )
    )
    print("BRANCH:", branch["branch"], "→", branch["recommended_lever"])
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
