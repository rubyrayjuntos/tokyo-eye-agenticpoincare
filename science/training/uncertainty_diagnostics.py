"""Residue-level epistemic / aleatoric sanity checks (read-only)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from science.dtie.common.residue_features import TAU

NODE_EPI_STD_FLOOR = 1e-4
NODE_ALE_STD_FLOOR = 1e-4
# Coupled-head checkpoints can pass alive floor (~0.01) but fail save/informative gates.
NODE_ALE_INFORMATIVE_FLOOR = 0.05
TOTAL_TOLERANCE = 1e-3


def extract_residue_uncertainty_rows(
    output: dict[str, Any],
    prot: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per residue: governed features + ν_epi, ν_ale, ν_total + evidence."""
    unc = output["uncertainty"]
    epi = unc["epistemic"].detach().cpu().numpy().reshape(-1)
    ale = unc["aleatoric"].detach().cpu().numpy().reshape(-1)
    total = unc.get("total")
    if total is not None:
        total_np = total.detach().cpu().numpy().reshape(-1)
    else:
        total_np = epi + ale

    depth = output["cone_depth"].detach().cpu().numpy().reshape(-1)
    evidence = output.get("evidence") or {}
    nu = evidence.get("nu")
    alpha = evidence.get("alpha")
    beta = evidence.get("beta")
    mu = evidence.get("mu")
    nu_np = nu.detach().cpu().numpy().reshape(-1) if nu is not None else None
    alpha_np = alpha.detach().cpu().numpy().reshape(-1) if alpha is not None else None
    beta_np = beta.detach().cpu().numpy().reshape(-1) if beta is not None else None
    mu_np = mu.detach().cpu().numpy().reshape(-1) if mu is not None else None

    x = prot["data"].x.detach().cpu().numpy()
    rho = x[:, 0]
    tau_flag = x[:, 1]
    ss = x[:, 2] if x.shape[1] > 2 else np.zeros_like(rho)
    sasa = x[:, 3] if x.shape[1] > 3 else np.full_like(rho, np.nan)

    residue_ids = list(prot.get("residue_ids") or [f"idx:{i}" for i in range(len(epi))])
    expert_w = output.get("expert_weights")
    if expert_w is not None:
        ew = expert_w.detach().cpu().numpy()
        if ew.ndim == 1:
            expert_assign = ew.astype(int)
        else:
            expert_assign = ew.argmax(axis=1)
    else:
        expert_assign = np.full(len(epi), -1, dtype=int)

    rows: list[dict[str, Any]] = []
    for i, rid in enumerate(residue_ids):
        row: dict[str, Any] = {
            "residue_id": str(rid),
            "index": i,
            "rho": float(rho[i]),
            "tau_flag": float(tau_flag[i]),
            "ss_type": float(ss[i]),
            "sasa": float(sasa[i]) if np.isfinite(sasa[i]) else None,
            "cone_depth": float(depth[i]),
            "epistemic": float(epi[i]),
            "aleatoric": float(ale[i]),
            "total": float(total_np[i]),
            "total_matches_sum": bool(
                abs(float(total_np[i]) - float(epi[i] + ale[i])) < TOTAL_TOLERANCE
            ),
            "near_tau_boundary": bool(abs(float(rho[i]) - TAU) <= 1.0),
            "expert": int(expert_assign[i]) if i < len(expert_assign) else -1,
        }
        if nu_np is not None:
            row["evidence_nu"] = float(nu_np[i])
            row["epistemic_from_nu"] = float(1.0 / max(nu_np[i], 1e-12))
        if alpha_np is not None and beta_np is not None:
            row["evidence_alpha"] = float(alpha_np[i])
            row["evidence_beta"] = float(beta_np[i])
            ale_raw = beta_np[i] / max(alpha_np[i] - 1.0, 1e-6)
            row["aleatoric_raw"] = float(ale_raw)
        if mu_np is not None:
            row["evidence_mu"] = float(mu_np[i])
            row["prediction_error"] = abs(float(mu_np[i]) - float(rho[i]))
        elif "prediction_error" not in row:
            row["prediction_error"] = float("nan")
        rows.append(row)
    return rows


def audit_uncertainty_sanity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sanity metrics: spread, decoupling, total identity, τ-boundary ale lift."""
    if not rows:
        return {"ok": False, "reason": "empty"}

    epi = np.array([r["epistemic"] for r in rows], dtype=np.float64)
    ale = np.array([r["aleatoric"] for r in rows], dtype=np.float64)
    total = np.array([r["total"] for r in rows], dtype=np.float64)
    rho = np.array([r["rho"] for r in rows], dtype=np.float64)

    total_ok = all(r["total_matches_sum"] for r in rows)
    epi_std = float(np.std(epi))
    ale_std = float(np.std(ale))
    epi_finite = bool(np.all(np.isfinite(epi)))
    ale_finite = bool(np.all(np.isfinite(ale)))

    r_epi_ale = float("nan")
    if epi_std > 1e-12 and ale_std > 1e-12:
        r_epi_ale = float(np.corrcoef(epi, ale)[0, 1])

    tau_mask = np.abs(rho - TAU) <= 1.0
    ale_tau = float(np.mean(ale[tau_mask])) if tau_mask.any() else float("nan")
    ale_non = float(np.mean(ale[~tau_mask])) if (~tau_mask).any() else float("nan")

    alive_epi = epi_finite and epi_std >= NODE_EPI_STD_FLOOR
    alive_ale = ale_finite and ale_std >= NODE_ALE_STD_FLOOR
    ok = alive_epi and alive_ale and total_ok and epi_finite and ale_finite

    reasons: list[str] = []
    if not epi_finite:
        reasons.append("epistemic_non_finite")
    if not ale_finite:
        reasons.append("aleatoric_non_finite")
    if not alive_epi:
        reasons.append(f"epistemic_flat_std={epi_std:.2e}")
    if not alive_ale:
        reasons.append(f"aleatoric_flat_std={ale_std:.2e}")
    if not total_ok:
        reasons.append("total_neq_epi_plus_ale")

    return {
        "ok": ok,
        "reason": "ok" if ok else ";".join(reasons),
        "n_residues": len(rows),
        "epistemic_mean": float(np.mean(epi)),
        "epistemic_std": epi_std,
        "epistemic_p05": float(np.percentile(epi, 5)),
        "epistemic_p95": float(np.percentile(epi, 95)),
        "aleatoric_mean": float(np.mean(ale)),
        "aleatoric_std": ale_std,
        "aleatoric_p05": float(np.percentile(ale, 5)),
        "aleatoric_p95": float(np.percentile(ale, 95)),
        "total_std": float(np.std(total)),
        "r_epi_ale": r_epi_ale,
        "total_matches_epi_plus_ale": total_ok,
        "tau_boundary_n": int(tau_mask.sum()),
        "aleatoric_mean_tau_boundary": ale_tau,
        "aleatoric_mean_non_tau_boundary": ale_non,
        "aleatoric_tau_lift": (
            float(ale_tau - ale_non)
            if math.isfinite(ale_tau) and math.isfinite(ale_non)
            else float("nan")
        ),
        "telemetry_alive_epistemic": alive_epi,
        "telemetry_alive_aleatoric": alive_ale,
    }
