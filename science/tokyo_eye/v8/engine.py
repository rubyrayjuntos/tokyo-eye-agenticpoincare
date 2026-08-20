"""v8 curriculum training driver + Poincaré diagnostics (Sprint 4).

Isolated from v7/v66 train loops. Logs Gumbel temperature alongside radius
diagnostics every step.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from science.tokyo_eye.v8.heads import mechanism_margin_loss_v2, sdrp_cross_entropy
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8


class CurriculumRadiusController:
    """Open ``τ_ceiling`` from interior toward the boundary over epochs."""

    def __init__(
        self,
        tau_start: float = 0.70,
        tau_end: float = 0.995,
        total_epochs: int = 100,
    ) -> None:
        if not (0.0 < tau_start <= tau_end < 1.0):
            raise ValueError("require 0 < tau_start <= tau_end < 1")
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1")
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.total_epochs = int(total_epochs)

    def tau_ceiling(self, epoch: int) -> float:
        e = max(0, min(int(epoch), self.total_epochs))
        t = e / float(self.total_epochs)
        return self.tau_start + t * (self.tau_end - self.tau_start)


class GumbelTemperatureSchedule:
    """Cool Gumbel-Softmax temperature over epochs (hardness stays True).

    Sprint 9 default: exponential decay to ``tau_end`` by ``half_epochs``.
    Sprint 8 fallback: ``schedule='linear'``.
    """

    def __init__(
        self,
        tau_start: float = 1.0,
        tau_end: float = 0.3,
        total_epochs: int = 100,
        *,
        schedule: str = "exponential",
        alpha: float | None = None,
        half_epochs: int = 12,
    ) -> None:
        if tau_start <= 0 or tau_end <= 0:
            raise ValueError("temperatures must be > 0")
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1")
        sched = str(schedule).strip().lower()
        if sched not in ("exponential", "linear"):
            raise ValueError("schedule must be 'exponential' or 'linear'")
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.total_epochs = int(total_epochs)
        self.schedule = sched
        self.half_epochs = max(1, int(half_epochs))
        if alpha is None:
            # α so τ_start * exp(-α * half_epochs) = tau_end
            ratio = self.tau_start / self.tau_end
            self.alpha = float(math.log(ratio) / float(self.half_epochs))
        else:
            self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")

    def temperature(self, epoch: int) -> float:
        e = max(0, int(epoch))
        if self.schedule == "linear":
            e_c = min(e, self.total_epochs)
            t = e_c / float(self.total_epochs)
            return self.tau_start + t * (self.tau_end - self.tau_start)
        # exponential
        return max(self.tau_end, self.tau_start * math.exp(-self.alpha * e))


class PoincareDiagnosticsEngine:
    """Radius / saturation / radial-histogram entropy telemetry."""

    def __init__(
        self,
        *,
        boundary_radius: float = 0.90,
        core_radius: float = 0.30,
        n_bins: int = 20,
        eps: float = 1e-8,
    ) -> None:
        self.boundary_radius = float(boundary_radius)
        self.core_radius = float(core_radius)
        self.n_bins = int(n_bins)
        self.eps = float(eps)

    @torch.no_grad()
    def summarize(self, z: torch.Tensor) -> dict[str, float | bool]:
        r = torch.linalg.vector_norm(z.detach(), dim=-1)
        n = max(int(r.numel()), 1)
        mean_r = float(r.mean())
        max_r = float(r.max())
        min_r = float(r.min())
        boundary_sat = float((r > self.boundary_radius).float().mean())
        core_frac = float((r < self.core_radius).float().mean())
        # Radial histogram entropy
        hist = torch.histc(r.cpu(), bins=self.n_bins, min=0.0, max=1.0)
        p = hist / hist.sum().clamp_min(self.eps)
        p = p[p > 0]
        entropy = float((-(p * p.log())).sum()) if p.numel() else 0.0
        warnings = {
            "warn_core_collapse": core_frac > 0.85 and mean_r < self.core_radius,
            "warn_boundary_saturation": boundary_sat > 0.50,
            "warn_over_smooth_entropy": entropy < 0.5 and (max_r - min_r) < 0.05,
        }
        return {
            "mean_radius": mean_r,
            "max_radius": max_r,
            "min_radius": min_r,
            "radius_spread": max_r - min_r,
            "boundary_saturation": boundary_sat,
            "core_fraction": core_frac,
            "radial_entropy": entropy,
            **warnings,
        }


def train_v8_step(
    model: TokyoEyesHyperbolicV8,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    radius_controller: CurriculumRadiusController,
    gumbel_schedule: GumbelTemperatureSchedule,
    diagnostics: PoincareDiagnosticsEngine | None = None,
    max_grad_norm: float = 1.0,
    sdrp_coeff: float = 1.0,
    margin_coeff: float = 1.0,
    cv_coeff: float = 1.0,
    moe_quota_coeff: float = 5.0,
) -> dict[str, float]:
    """One curriculum train step: lift→attn→MoE losses + clip + telemetry."""
    model.train()
    tau_ceil = radius_controller.tau_ceiling(epoch)
    gumbel_tau = gumbel_schedule.temperature(epoch)
    model.set_moe_temperature(gumbel_tau)

    s = batch["s"]
    v = batch["v"]
    edge_index = batch["edge_index"]
    edge_type = batch["edge_type"]
    out = model(s, v, edge_index, edge_type, tau_ceiling=tau_ceil)

    loss_sdrp = sdrp_cross_entropy(out["sdrp_logits"], batch["sdrp_target"])
    loss_margin = mechanism_margin_loss_v2(
        out["mechanism_score"],
        batch["mechanism_pos"].to(out["mechanism_score"].dtype),
        batch["mechanism_neg"].to(out["mechanism_score"].dtype),
    )
    loss_cv_raw = out["moe_aux"]["cv_loss"]
    loss_quota_raw = out["moe_aux"].get("quota_loss")
    if loss_quota_raw is None:
        loss_quota_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_balance = (
        loss_cv_raw * float(cv_coeff)
        + loss_quota_raw * float(moe_quota_coeff)
    )
    loss = (
        float(sdrp_coeff) * loss_sdrp
        + float(margin_coeff) * loss_margin
        + loss_balance
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
    )
    optimizer.step()

    load = out["moe_aux"].get("load")
    if load is None:
        load = out["moe_aux"]["routing"].mean(dim=0)
    metrics: dict[str, float] = {
        "loss_total": float(loss.detach()),
        "loss_sdrp": float(loss_sdrp.detach()),
        "loss_margin": float(loss_margin.detach()),
        "loss_cv": float(loss_balance.detach()),
        "moe_quota_loss": float(loss_quota_raw.detach()),
        "tau_ceiling": float(tau_ceil),
        "gumbel_temperature": float(gumbel_tau),
        "grad_norm": grad_norm,  # pre-clip total norm (PyTorch convention)
        "max_grad_norm": float(max_grad_norm),
        "moe_load_min": float(load.min().detach()),
    }
    for i, v_load in enumerate(load.detach().tolist()):
        metrics[f"moe_load_e{i}"] = float(v_load)
    if diagnostics is not None:
        diag = diagnostics.summarize(out["z_hyp"])
        for k, val in diag.items():
            metrics[f"diag_{k}"] = float(val) if not isinstance(val, bool) else float(val)
    return metrics


__all__ = [
    "CurriculumRadiusController",
    "GumbelTemperatureSchedule",
    "PoincareDiagnosticsEngine",
    "train_v8_step",
]
