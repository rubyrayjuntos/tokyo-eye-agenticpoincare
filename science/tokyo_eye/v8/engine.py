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


class EpsilonGreedySchedule:
    """§2.6: decay train-time hard-override probability (mirrors Gumbel form).

    ``eps_end=0`` uses a tiny ``eps_end_eff`` only to define α, then clamps to 0.
    """

    def __init__(
        self,
        eps_start: float = 0.20,
        eps_end: float = 0.0,
        *,
        half_epochs: int = 12,
        alpha: float | None = None,
        eps_end_eff: float = 1e-3,
    ) -> None:
        if eps_start < 0.0 or eps_start > 1.0:
            raise ValueError("eps_start must be in [0, 1]")
        if eps_end < 0.0 or eps_end > 1.0:
            raise ValueError("eps_end must be in [0, 1]")
        if eps_end > eps_start:
            raise ValueError("eps_end must be <= eps_start")
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.half_epochs = max(1, int(half_epochs))
        self.eps_end_eff = float(eps_end_eff) if self.eps_end <= 0.0 else float(eps_end)
        if self.eps_end_eff <= 0.0 or self.eps_end_eff > self.eps_start:
            raise ValueError("eps_end_eff must be in (0, eps_start]")
        if alpha is None:
            self.alpha = float(
                math.log(self.eps_start / self.eps_end_eff) / float(self.half_epochs)
            )
        else:
            self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        self._floor_epoch: int | None = None

    def epsilon(self, epoch: int) -> float:
        e = max(0, int(epoch))
        if self.eps_start == 0.0:
            return 0.0
        raw = self.eps_start * math.exp(-self.alpha * e)
        val = max(self.eps_end, raw)
        if self.eps_end == 0.0 and raw < self.eps_end_eff:
            val = 0.0
        if val <= self.eps_end and self._floor_epoch is None:
            self._floor_epoch = e
        return float(val)

    @property
    def first_floor_epoch(self) -> int | None:
        """Epoch when ε first hit ``eps_end`` (populated after ``epsilon`` calls)."""
        return self._floor_epoch


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


class TransparencyEngine:
    """Freeze §9.2–§9.3: rim↔core flow and per-R message fractions."""

    def __init__(
        self,
        *,
        rim_radius: float = 0.90,
        core_radius: float = 0.30,
        eps: float = 1e-8,
    ) -> None:
        self.rim_radius = float(rim_radius)
        self.core_radius = float(core_radius)
        self.eps = float(eps)

    @torch.no_grad()
    def summarize(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        *,
        attn: torch.Tensor | None = None,
        num_relations: int = 6,
    ) -> dict[str, float]:
        r = torch.linalg.vector_norm(z.detach(), dim=-1)
        rim = r > self.rim_radius
        core = r < self.core_radius
        n_rel = int(num_relations)
        out: dict[str, float] = {
            "n_rim": float(rim.sum()),
            "n_core": float(core.sum()),
        }
        if edge_index.numel() == 0:
            for i in range(n_rel):
                out[f"msg_frac_r{i}"] = 0.0
            out["flow_rim_to_core"] = 0.0
            out["flow_core_to_rim"] = 0.0
            out["rim_r2_vs_r1"] = 0.0
            return out
        src = edge_index[0].long()
        dst = edge_index[1].long()
        et = edge_type.long()
        e = int(src.numel())
        w = attn.detach() if attn is not None and attn.numel() == e else torch.ones(
            e, device=z.device, dtype=z.dtype
        )
        wsum = float(w.sum().clamp_min(self.eps))
        for i in range(n_rel):
            m = et == i
            out[f"msg_frac_r{i}"] = float(w[m].sum() / wsum) if m.any() else 0.0
        rim_src = rim[src]
        core_dst = core[dst]
        core_src = core[src]
        rim_dst = rim[dst]
        out["flow_rim_to_core"] = float((w[rim_src & core_dst]).sum() / wsum)
        out["flow_core_to_rim"] = float((w[core_src & rim_dst]).sum() / wsum)
        r1 = (et == 1) & rim[src]
        r2 = (et == 2) & rim[src]
        rim_h = float((w[r1].sum() + w[r2].sum()).clamp_min(self.eps))
        out["rim_r2_vs_r1"] = float(w[r2].sum() / rim_h)
        return out


def train_v8_step(
    model: TokyoEyesHyperbolicV8,
    batch: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    radius_controller: CurriculumRadiusController,
    gumbel_schedule: GumbelTemperatureSchedule,
    diagnostics: PoincareDiagnosticsEngine | None = None,
    transparency: TransparencyEngine | None = None,
    max_grad_norm: float = 1.0,
    sdrp_coeff: float = 1.0,
    margin_coeff: float = 1.0,
    cv_coeff: float = 1.0,
    moe_quota_coeff: float = 5.0,
    switch_lb_coeff: float = 1.0,
    soft_quota_coeff: float = 5.0,
    eval_proxy_lb_coeff: float = 1.0,
    eval_proxy_quota_coeff: float = 140.0,
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
    chem = batch.get("gate_chem")
    if chem is None:
        chem = batch.get("chem")
    out = model(
        s,
        v,
        edge_index,
        edge_type,
        tau_ceiling=tau_ceil,
        chem=chem,
    )

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
    loss_switch_raw = out["moe_aux"].get("switch_lb_loss")
    if loss_switch_raw is None:
        loss_switch_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_soft_quota_raw = out["moe_aux"].get("soft_quota_loss")
    if loss_soft_quota_raw is None:
        loss_soft_quota_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_eval_proxy_raw = out["moe_aux"].get("eval_proxy_lb_loss")
    if loss_eval_proxy_raw is None:
        loss_eval_proxy_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_eval_proxy_quota_raw = out["moe_aux"].get("eval_proxy_quota_loss")
    if loss_eval_proxy_quota_raw is None:
        loss_eval_proxy_quota_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_majority_raw = out["moe_aux"].get("majority_hinge_loss")
    if loss_majority_raw is None:
        loss_majority_raw = torch.zeros(
            (), device=loss_cv_raw.device, dtype=loss_cv_raw.dtype
        )
    loss_balance = (
        loss_cv_raw * float(cv_coeff)
        + loss_quota_raw * float(moe_quota_coeff)
        + loss_switch_raw * float(switch_lb_coeff)
        + loss_soft_quota_raw * float(soft_quota_coeff)
        + loss_eval_proxy_raw * float(eval_proxy_lb_coeff)
        + loss_eval_proxy_quota_raw * float(eval_proxy_quota_coeff)
        + loss_majority_raw
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
        "moe_switch_lb_loss": float(loss_switch_raw.detach()),
        "moe_soft_quota_loss": float(loss_soft_quota_raw.detach()),
        "moe_eval_proxy_lb_loss": float(loss_eval_proxy_raw.detach()),
        "moe_eval_proxy_quota_loss": float(loss_eval_proxy_quota_raw.detach()),
        "tau_ceiling": float(tau_ceil),
        "gumbel_temperature": float(gumbel_tau),
        "grad_norm": grad_norm,  # pre-clip total norm (PyTorch convention)
        "max_grad_norm": float(max_grad_norm),
        "moe_majority_hinge_loss": float(loss_majority_raw.detach()),
        "moe_load_min": float(load.min().detach()),
    }
    for i, v_load in enumerate(load.detach().tolist()):
        metrics[f"moe_load_e{i}"] = float(v_load)
    soft_load = out["moe_aux"].get("soft_load")
    if soft_load is not None:
        for i, v_load in enumerate(soft_load.detach().tolist()):
            metrics[f"moe_soft_load_e{i}"] = float(v_load)
    if diagnostics is not None:
        for tag, tensor in (
            ("pre_moe_lift", out["z_lift"]),
            ("pre_moe_attn", out["z_attn"]),
            ("post_moe", out["z_hyp"]),
        ):
            diag = diagnostics.summarize(tensor)
            for k, val in diag.items():
                metrics[f"diag_{tag}_{k}"] = (
                    float(val) if not isinstance(val, bool) else float(val)
                )
        # Back-compat aliases on deployed (post-MoE) embedding.
        diag = diagnostics.summarize(out["z_hyp"])
        for k, val in diag.items():
            metrics[f"diag_{k}"] = float(val) if not isinstance(val, bool) else float(val)
    if transparency is not None:
        attn = None
        last = model.attn_layers[-1]
        attn = getattr(last, "last_attn", None)
        tele = transparency.summarize(
            out["z_hyp"], edge_index, edge_type, attn=attn
        )
        for k, val in tele.items():
            metrics[f"trans_{k}"] = float(val)
    return metrics


__all__ = [
    "CurriculumRadiusController",
    "GumbelTemperatureSchedule",
    "EpsilonGreedySchedule",
    "PoincareDiagnosticsEngine",
    "TransparencyEngine",
    "train_v8_step",
]
