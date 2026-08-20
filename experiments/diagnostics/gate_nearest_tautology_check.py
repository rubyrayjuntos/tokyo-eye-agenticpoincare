"""Check whether frac_maj_nearest_is_maj_proto=1.0 is tautological given gate math.

Compares per-residue routing argmax (softmax weights / adjusted logits) against
argmin hyperbolic prototype distance, and reports expert_bias + overload_penalty
magnitudes relative to distance-induced logit spread.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.diagnostics.prototype_repulsion_epoch import committed_majority_partition
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.training.gnn_lineage import load_model_from_checkpoint

DEVICE = "cuda"


def _bootstrap_corr_ci(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 5000, seed: int = 0
) -> dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return {"n": int(len(x)), "r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(seed)
    rs = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if x[idx].std() == 0 or y[idx].std() == 0:
            continue
        rs.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    if not rs:
        return {"n": n, "r": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    rs_a = np.array(rs)
    return {
        "n": n,
        "r": float(np.corrcoef(x, y)[0, 1]),
        "ci_lo": float(np.percentile(rs_a, 2.5)),
        "ci_hi": float(np.percentile(rs_a, 97.5)),
    }


def audit_gate_tautology(ckpt: Path, label: str, input_mode: str) -> dict[str, Any]:
    os.environ["GNN_INPUT_MODE"] = input_mode
    proteins, _ = load_training_proteins(
        Path("/tmp/dtie_pdb_cache"),
        Path("/app/manifests/v6_corpus_stage_a_small_v1.json"),
        max_proteins=12,
    )
    model = load_model_from_checkpoint(ckpt, DEVICE)
    model.to(DEVICE)
    model.core_capacity_quota_tau = 0.0
    model.eval()

    per_structure: list[dict[str, Any]] = []
    all_soft_eq_nearest: list[float] = []
    all_adj_eq_nearest: list[float] = []
    all_raw_eq_nearest: list[float] = []
    all_audit_frac_maj: list[float] = []
    bias_rows: list[list[float]] = []
    penalty_rows: list[list[float]] = []
    scale_vals: list[float] = []
    logit_spread_vs_bias: list[float] = []

    gate = model.gate
    bias = gate.expert_bias.detach().float().cpu().numpy().tolist()
    bias_rows.append(bias)

    with torch.no_grad():
        for prot in proteins:
            data = prepare_training_batch(model, prot, DEVICE)
            in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
            if data.x.size(-1) > in_f:
                data.x = data.x[:, :in_f].contiguous()
            out = model(data)
            w = out["expert_weights"].float()
            pre = getattr(gate, "_last_pre_softmax", {}) or {}
            dists = pre.get("dists")
            raw_logits = pre.get("raw_logits")
            adjusted_logits = pre.get("adjusted_logits")
            if dists is None or raw_logits is None or adjusted_logits is None:
                continue
            dists = dists.float()
            raw_logits = raw_logits.float()
            adjusted_logits = adjusted_logits.float()

            nearest = dists.argmin(dim=-1)
            soft_hard = w.argmax(dim=-1)
            raw_hard = raw_logits.argmax(dim=-1)
            adj_hard = adjusted_logits.argmax(dim=-1)

            frac_soft_nearest = float((soft_hard == nearest).float().mean())
            frac_raw_nearest = float((raw_hard == nearest).float().mean())
            frac_adj_nearest = float((adj_hard == nearest).float().mean())
            all_soft_eq_nearest.append(frac_soft_nearest)
            all_raw_eq_nearest.append(frac_raw_nearest)
            all_adj_eq_nearest.append(frac_adj_nearest)

            # Recompute overload penalty (same as gate forward).
            scores_initial = torch.softmax(raw_logits, dim=-1)
            expected_load = scores_initial.mean(dim=0)
            overload_penalty = torch.relu(expected_load - gate.capacity_threshold) ** 2
            penalty_rows.append(overload_penalty.cpu().numpy().tolist())
            scale = float(pre.get("logit_scale", float("nan")))
            scale_vals.append(scale)

            # Typical per-residue distance-induced logit spread vs bias/penalty.
            d_spread = (dists.max(dim=-1).values - dists.min(dim=-1).values) * scale
            bp_spread = float(
                (gate.expert_bias - 2.0 * overload_penalty).max()
                - (gate.expert_bias - 2.0 * overload_penalty).min()
            )
            if d_spread.numel():
                logit_spread_vs_bias.append(float((d_spread / (bp_spread + 1e-8)).median()))

            max_p = w.max(dim=-1).values
            committed = max_p >= 0.60
            dh = prot.get("target_dehydron")
            if dh is None and data.x.size(1) > 1:
                dh = data.x[:, 1]
            if dh is None:
                dh = torch.zeros(w.shape[0], device=w.device)
            dh = dh.float().reshape(-1)[: w.shape[0]]
            part = committed_majority_partition(w.cpu().numpy(), dh.cpu().numpy())
            maj_e = part["majority_expert"]
            frac_audit = float("nan")
            if maj_e is not None and part["n_committed"] >= 20:
                maj_mask = committed & (soft_hard == int(maj_e))
                if maj_mask.any():
                    frac_audit = float((nearest[maj_mask] == int(maj_e)).float().mean())
                    all_audit_frac_maj.append(frac_audit)

            per_structure.append(
                {
                    "pdb_id": str(prot.get("pdb_id") or "?").upper(),
                    "frac_soft_hard_eq_nearest": frac_soft_nearest,
                    "frac_raw_hard_eq_nearest": frac_raw_nearest,
                    "frac_adj_hard_eq_nearest": frac_adj_nearest,
                    "frac_audit_maj_nearest_eq_maj": frac_audit,
                    "n_residues": int(w.shape[0]),
                    "expert_bias": bias,
                    "overload_penalty": overload_penalty.cpu().numpy().tolist(),
                    "logit_scale_softplus": scale,
                    "median_dist_logit_spread_over_bias_penalty_spread": (
                        float(d_spread.median() / (bp_spread + 1e-8)) if d_spread.numel() else None
                    ),
                }
            )

    # Structure-level corr bootstrap (same n as monopole audit).
    struct_rows = [r for r in per_structure if r["frac_audit_maj_nearest_eq_maj"] == r["frac_audit_maj_nearest_eq_maj"]]
    shares = []
    d_maj = []
    for prot in proteins:
        data = prepare_training_batch(model, prot, DEVICE)
        in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
        if data.x.size(-1) > in_f:
            data.x = data.x[:, :in_f].contiguous()
        with torch.no_grad():
            out = model(data)
        w = out["expert_weights"].float()
        pre = getattr(gate, "_last_pre_softmax", {}) or {}
        dists = pre.get("dists")
        if dists is None:
            continue
        dists = dists.float()
        dh = prot.get("target_dehydron")
        if dh is None and data.x.size(1) > 1:
            dh = data.x[:, 1]
        if dh is None:
            dh = torch.zeros(w.shape[0], device=w.device)
        dh = dh.float().reshape(-1)[: w.shape[0]]
        part = committed_majority_partition(w.cpu().numpy(), dh.cpu().numpy())
        maj_e = part["majority_expert"]
        if maj_e is None or part["n_committed"] < 20:
            continue
        max_p = w.max(dim=-1).values
        committed = max_p >= 0.60
        maj_mask = committed & (w.argmax(dim=-1) == int(maj_e))
        if not maj_mask.any():
            continue
        shares.append(float(part["majority_share"]))
        d_maj.append(float(dists[:, int(maj_e)][maj_mask].mean()))

    shares_a = np.array(shares)
    d_maj_a = np.array(d_maj)

    def verdict_tautology(mean_soft_eq: float, mean_audit: float) -> str:
        if mean_soft_eq >= 0.999 and mean_audit >= 0.999:
            return "TAUTOLOGICAL_OR_NEAR_TAUTOLOGICAL"
        if mean_soft_eq < 0.95:
            return "NOT_TAUTOLOGICAL_SOFT_NEAREST_DIVERGES"
        return "AUDIT_FRAC_MAY_DIFFER_FROM_GLOBAL"

    mean_soft = float(np.mean(all_soft_eq_nearest)) if all_soft_eq_nearest else float("nan")
    mean_audit = float(np.mean(all_audit_frac_maj)) if all_audit_frac_maj else float("nan")

    return {
        "run": label,
        "gnn_input_mode": input_mode,
        "checkpoint": str(ckpt),
        "gate_routing_rule": "adjusted_logits = -softplus(scale)*d_H + expert_bias - 2*overload_penalty; scores=softmax(adjusted_logits); hard=argmax(scores)",
        "tautology_conditions": {
            "nearest_equals_hard_requires": "argmax_e(-scale*d_e + b_e - 2p_e) == argmin_e(d_e); guaranteed only if bias and overload_penalty equal across experts (penalty is generally non-uniform)",
            "audit_frac_maj_subset": "mean(nearest==maj_e | committed & hard==maj_e); equals 1.0 if hard==nearest always on that subset",
        },
        "corpus_wide": {
            "mean_frac_soft_hard_eq_nearest": mean_soft,
            "mean_frac_raw_hard_eq_nearest": float(np.mean(all_raw_eq_nearest)) if all_raw_eq_nearest else None,
            "mean_frac_adj_hard_eq_nearest": float(np.mean(all_adj_eq_nearest)) if all_adj_eq_nearest else None,
            "mean_frac_audit_maj_nearest_eq_maj": mean_audit,
            "verdict": verdict_tautology(mean_soft, mean_audit),
        },
        "gate_params": {
            "expert_bias": bias,
            "expert_bias_range": float(max(bias) - min(bias)),
            "mean_overload_penalty": (
                [float(x) for x in np.mean(penalty_rows, axis=0).tolist()]
                if penalty_rows
                else None
            ),
            "mean_logit_scale_softplus": float(np.mean(scale_vals)) if scale_vals else None,
            "median_dist_logit_spread_over_bias_penalty_spread": (
                float(np.median(logit_spread_vs_bias)) if logit_spread_vs_bias else None
            ),
        },
        "bootstrap_corr_share_vs_d_to_maj": _bootstrap_corr_ci(shares_a, d_maj_a),
        "per_structure": per_structure,
    }


def main() -> None:
    runs = [
        (
            "three_vector_seed2",
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_stack_three_vector_cold_seed2_v1/"
                "epochs/epoch_030.pt"
            ),
            "topology_three_vector",
        ),
        (
            "legacy4d_seed2",
            Path(
                "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1/"
                "epochs/epoch_030.pt"
            ),
            "legacy_four_vector",
        ),
    ]
    out = Path(
        "/app/checkpoints/v66/diagnostics/three_vector_stack_battery_reverify/"
        "gate_nearest_tautology_check.json"
    )
    report = {
        "tag": "GATE_NEAREST_TAUTOLOGY_CHECK",
        "runs": {label: audit_gate_tautology(ckpt, label, mode) for label, ckpt, mode in runs},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", out)
    for label, r in report["runs"].items():
        cw = r["corpus_wide"]
        gp = r["gate_params"]
        boot = r["bootstrap_corr_share_vs_d_to_maj"]
        print(
            label,
            "soft==nearest",
            cw["mean_frac_soft_hard_eq_nearest"],
            "audit_frac",
            cw["mean_frac_audit_maj_nearest_eq_maj"],
            "verdict",
            cw["verdict"],
            "bias_range",
            gp["expert_bias_range"],
            "corr_d_bootstrap",
            boot,
        )


if __name__ == "__main__":
    main()
