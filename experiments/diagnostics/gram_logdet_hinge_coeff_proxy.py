"""Coeff proxy for saturating Gram logdet hinge (before freeze/train).

Measures unit-λ_g gate pressure vs primary / nearest-pair repulsion on
stack ge30 checkpoints (seed1 + seed2).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.v6.loss import gosp_loss_v6
from science.training.gnn_lineage import load_model_from_checkpoint

DEVICE = "cuda"
EPS = 1e-4
# Saturates when logdet ≥ τ (≈ floor-edge equal-eig at λ_min=0.15).
TAU_LOGDET = -1.15

RUNS = {
    "seed1": Path(
        "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_v1"
    ),
    "seed2": Path(
        "/app/checkpoints/v66/runs/fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1"
    ),
}

proteins, _ = load_training_proteins(
    Path("/tmp/dtie_pdb_cache"),
    Path("/app/manifests/v6_corpus_stage_a_small_v1.json"),
    max_proteins=12,
)


def gram_stats(model: torch.nn.Module) -> dict:
    tan = model.gate.prototype_bank.prototype_tangent
    a = F.normalize(tan.float(), dim=-1)
    g = a @ a.T
    eigs = torch.linalg.eigvalsh(g)
    logdet = torch.logdet(g + EPS * torch.eye(g.shape[0], device=g.device, dtype=g.dtype))
    return {
        "eig_min": float(eigs.min().detach()),
        "eig_max": float(eigs.max().detach()),
        "gram_condition": float((eigs.max() / (eigs.abs().min() + 1e-12)).detach()),
        "logdet": float(logdet.detach()),
        "hinge_raw": float(F.relu(torch.tensor(TAU_LOGDET, device=logdet.device) - logdet).detach()),
    }


def gram_hinge_loss(model: torch.nn.Module) -> torch.Tensor:
    tan = model.gate.prototype_bank.prototype_tangent
    a = F.normalize(tan.float(), dim=-1)
    g = a @ a.T
    e = g.shape[0]
    logdet = torch.logdet(g + EPS * torch.eye(e, device=g.device, dtype=g.dtype))
    return F.relu(torch.as_tensor(TAU_LOGDET, device=logdet.device, dtype=logdet.dtype) - logdet).pow(2)


def _grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().float().pow(2).sum().item())
    return total**0.5


def proxy_one(ckpt: Path, name: str) -> dict:
    model = load_model_from_checkpoint(ckpt, DEVICE)
    model.to(DEVICE)
    model.core_capacity_quota_tau = 0.0
    model.train()
    model.gate.expert_dropout_p = 0.0

    stats0 = gram_stats(model)
    gate_params = [p for p in model.gate.parameters() if p.requires_grad]
    proto = model.gate.prototype_bank.prototype_tangent

    primary_gate = 0.0
    primary_proto = 0.0
    hinge_gate = 0.0
    hinge_proto = 0.0
    rep_gate = 0.0
    rep_proto = 0.0
    n = 0
    hinge_vals = []
    primary_vals = []
    rep_vals = []

    for prot in proteins[:6]:
        data = prepare_training_batch(model, prot, DEVICE)
        in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
        if data.x.size(-1) > in_f:
            data.x = data.x[:, :in_f].contiguous()

        target_rho = prot["target_rho"].to(DEVICE)
        ca_coords = prot["ca_coords"].to(DEVICE)
        target_dehydron = prot.get("target_dehydron")
        if target_dehydron is not None and torch.is_tensor(target_dehydron):
            target_dehydron = target_dehydron.to(DEVICE)
        else:
            target_dehydron = None
        sasa = prot.get("sasa")
        if sasa is not None and torch.is_tensor(sasa):
            sasa = sasa.to(DEVICE)

        model.zero_grad(set_to_none=True)
        out = model(data)
        losses = gosp_loss_v6(
            output=out,
            target_rho=target_rho.squeeze(-1) if target_rho.dim() > 1 else target_rho,
            ca_coords=ca_coords,
            sasa=sasa,
            target_dehydron=(
                target_dehydron.squeeze(-1)
                if target_dehydron is not None and target_dehydron.dim() > 1
                else target_dehydron
            ),
            prototype_repulsion_coeff=1.0,
            prototype_repulsion_margin=0.25,
        )
        total = losses["total"]
        rep = losses.get("prototype_repulsion")
        if rep is None or not torch.is_tensor(rep):
            dmin = out.get("prototype_pair_min_dist")
            rep = (
                F.relu(0.25 - dmin).pow(2)
                if dmin is not None and torch.is_tensor(dmin)
                else torch.zeros((), device=DEVICE)
            )
        # Primary = total without repulsion (and without gram, which isn't in loss yet)
        primary = total - rep

        primary.backward(retain_graph=True)
        primary_gate += _grad_norm(gate_params)
        primary_proto += _grad_norm([proto])
        primary_vals.append(float(primary.detach()))

        model.zero_grad(set_to_none=True)
        out = model(data)
        dmin = out.get("prototype_pair_min_dist")
        if dmin is None:
            audit = out.get("audit_trail") or {}
            dmin = audit.get("prototype_pair_min_dist") if isinstance(audit, dict) else None
        rep_l = (
            F.relu(0.25 - dmin)
            if dmin is not None and torch.is_tensor(dmin)
            else torch.zeros((), device=DEVICE)
        )
        # Match loss.py form: coeff * relu (not squared) — use unit coeff already in form
        rep_l.backward(retain_graph=True)
        rep_gate += _grad_norm(gate_params)
        rep_proto += _grad_norm([proto])
        rep_vals.append(float(rep_l.detach()))

        model.zero_grad(set_to_none=True)
        h = gram_hinge_loss(model)
        h.backward()
        hinge_gate += _grad_norm(gate_params)
        hinge_proto += _grad_norm([proto])
        hinge_vals.append(float(h.detach()))
        n += 1

    n = max(n, 1)
    pg, pp = primary_gate / n, primary_proto / n
    hg, hp = hinge_gate / n, hinge_proto / n
    rg, rp = rep_gate / n, rep_proto / n

    def safe_ratio(a, b):
        return float(a / b) if b > 1e-12 else float("nan")

    lam_match_proto = safe_ratio(pp, hp) if hp > 1e-12 else float("nan")
    lam_match_gate = safe_ratio(pg, hg) if hg > 1e-12 else float("nan")
    recommended = lam_match_proto
    if recommended == recommended and recommended > 0:
        # Keep true 1× match; do not floor at 0.05 (unit hinge grads are large).
        recommended_reg = float(round(recommended, 4))
        if recommended_reg < 1e-5:
            recommended_reg = 1e-5
    else:
        recommended_reg = None

    return {
        "run": name,
        "tau_logdet": TAU_LOGDET,
        "eps": EPS,
        "gram_stats_ge30": stats0,
        "mean_primary_loss": float(sum(primary_vals) / n),
        "mean_repulsion_loss_unit": float(sum(rep_vals) / n),
        "mean_hinge_sq": float(sum(hinge_vals) / n),
        "grad_primary_gate": pg,
        "grad_primary_proto": pp,
        "grad_repulsion_gate": rg,
        "grad_repulsion_proto": rp,
        "grad_unit_hinge_gate": hg,
        "grad_unit_hinge_proto": hp,
        "lambda_for_1x_primary_proto": lam_match_proto,
        "lambda_for_1x_primary_gate": lam_match_gate,
        "lambda_for_1x_repulsion_proto": safe_ratio(rp, hp),
        "recommended_lambda_g": recommended_reg,
    }


def main() -> None:
    results = {}
    for name, run in RUNS.items():
        results[name] = proxy_one(run / "epochs" / "epoch_030.pt", name)

    # Register from seed1 (ill-conditioned); seed2 should be near-saturated (hinge~0)
    s1 = results["seed1"]
    s2 = results["seed2"]
    # If seed2 hinge already ~0, its grad may be ~0 — expected
    report = {
        "tau_logdet": TAU_LOGDET,
        "loss_form": "ReLU(tau_logdet - logdet(G+eps I))^2 on unit-normalized prototype tangents",
        "seed1": s1,
        "seed2": s2,
        "registered_lambda_g": s1.get("recommended_lambda_g"),
        "rationale": (
            "Match unit-hinge prototype_tangent grad to primary proto grad on seed1 ge30 "
            "(ill-conditioned bank). Seed2 may already be near hinge saturation."
        ),
        "saturation_note": (
            "Unit-row Gram ⇒ raw -logdet bounded below at orthogonality (no radius runaway). "
            "Hinge still required so pressure stops at registered floor rather than full orthogonality."
        ),
    }
    out = Path("/tmp/gram_logdet_hinge_coeff_proxy.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("REPORT_JSON_BEGIN")
    print(json.dumps(report))
    print("REPORT_JSON_END")
    print(json.dumps(report, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
