"""V6 decoupled evidential uncertainty head (separate epi / ale trunks)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_trunk(total_input: int) -> nn.Sequential:
    reduced = total_input // 4
    return nn.Sequential(
        nn.Linear(total_input, total_input // 2),
        nn.SiLU(),
        nn.Linear(total_input // 2, reduced),
        nn.SiLU(),
    )


class DecoupledEvidentialHead(nn.Module):
    """Evidential head with independent epistemic and aleatoric MLP trunks."""

    def __init__(
        self,
        hidden_dim: int,
        extra_input_dim: int = 2,
        out_dim: int = 1,
        aleatoric_logvar_min: float = -12.0,
        aleatoric_logvar_max: float = 6.0,
        epistemic_temp_scaling: float = 1.0,
    ):
        super().__init__()
        total_input = hidden_dim + extra_input_dim
        self.epi_trunk = _make_trunk(total_input)
        self.ale_trunk = _make_trunk(total_input)
        reduced = total_input // 4
        self.logv_head = nn.Linear(reduced, out_dim)
        self.mu_head = nn.Linear(reduced, out_dim)
        self.loga_head = nn.Linear(reduced, out_dim)
        self.logb_head = nn.Linear(reduced, out_dim)
        self.aleatoric_logvar_min = float(aleatoric_logvar_min)
        self.aleatoric_logvar_max = float(aleatoric_logvar_max)
        self.epistemic_temp_scaling = float(epistemic_temp_scaling)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        h_epi = self.epi_trunk(x)
        h_ale = self.ale_trunk(x)
        nu = F.softplus(self.logv_head(h_epi)) + 1e-6
        mu = self.mu_head(h_ale)
        alpha = F.softplus(self.loga_head(h_ale)) + 1.0
        beta = F.softplus(self.logb_head(h_ale)) + 1e-6

        epistemic = (1.0 / nu) * self.epistemic_temp_scaling
        aleatoric_raw = beta / (alpha - 1.0 + 1e-6)
        aleatoric_log = torch.log(torch.clamp(aleatoric_raw, min=1e-12))
        aleatoric = torch.exp(
            torch.clamp(
                aleatoric_log,
                min=self.aleatoric_logvar_min,
                max=self.aleatoric_logvar_max,
            )
        )

        uncertainty = {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": epistemic + aleatoric,
        }
        evidence = {"mu": mu, "nu": nu, "alpha": alpha, "beta": beta}
        return uncertainty, evidence


def expand_coupled_uncertainty_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "uncertainty_head.",
) -> dict[str, torch.Tensor]:
    """Copy legacy shared-trunk weights into decoupled epi/ale trunks."""
    adapted = dict(state_dict)
    shared_key = f"{prefix}shared.0.weight"
    if shared_key not in adapted:
        return adapted
    for layer_idx in (0, 2):
        for param in ("weight", "bias"):
            src = f"{prefix}shared.{layer_idx}.{param}"
            if src not in adapted:
                continue
            for trunk in ("epi_trunk", "ale_trunk"):
                dst = f"{prefix}{trunk}.{layer_idx}.{param}"
                if dst not in adapted:
                    adapted[dst] = adapted[src].clone()
    return adapted


def uncertainty_head_is_decoupled(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("uncertainty_head.epi_trunk.") for k in state_dict)
