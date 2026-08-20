"""Frozen v3 GOSPConeMapper teacher for v6 shell-signal distillation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "science" / "dtie" / "v3" / "checkpoints" / "v2_bridge_epoch_014.pt"
_DEFAULT_BENCHMARK_PDB_DIR = _REPO_ROOT / "science" / "dtie" / "assets" / "benchmark_pdbs"


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    x = x.squeeze(-1) if x.dim() > 1 else x
    lo, hi = x.min(), x.max()
    if float(hi - lo) < 1e-6:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


class V2Teacher:
    """Loads frozen v3 scaffold (v2-bridge warmstart) and emits per-residue shell targets."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str = "cpu",
    ) -> None:
        from science.dtie.v3.gnn.model import GOSPConeMapper

        self.device = device
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"V2 teacher checkpoint not found: {checkpoint}")

        self.model: nn.Module = GOSPConeMapper()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.train(False)
        for p in self.model.parameters():
            p.requires_grad = False
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        logger.info("V2 teacher loaded from %s (%d params frozen)", checkpoint, len(state))

    @torch.no_grad()
    def infer(self, data: Any) -> dict[str, torch.Tensor]:
        """Run teacher forward; return normalized depth, epistemic, expert_id on CPU."""
        data = data.to(self.device)
        out = self.model(data)
        depth = _normalize01(out["cone_depth"])
        epistemic = out["uncertainty"]["epistemic"].squeeze(-1)
        expert_id = out["expert_weights"].argmax(dim=-1).float()
        return {
            "cone_depth_norm": depth.detach().cpu(),
            "epistemic": epistemic.detach().cpu(),
            "expert_id": expert_id.detach().cpu(),
        }

    def precompute(self, proteins: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        """Precompute teacher targets for all proteins (keyed by pdb_id)."""
        from experiments.training.v66.train_loop import attach_v6_features

        for prot in proteins:
            pdb_id = str(prot.get("pdb_id", "?")).upper()
            if pdb_id in self._cache:
                continue
            data = attach_v6_features(prot["data"])
            self._cache[pdb_id] = self.infer(data)
            logger.info(
                "V2 teacher targets for %s: depth_std=%.4f epi_mean=%.4f",
                pdb_id,
                float(self._cache[pdb_id]["cone_depth_norm"].std()),
                float(self._cache[pdb_id]["epistemic"].mean()),
            )
        return self._cache

    def targets_for(self, pdb_id: str) -> dict[str, torch.Tensor] | None:
        return self._cache.get(str(pdb_id).upper())


def resolve_default_v2_teacher_checkpoint() -> Path | None:
    env = os.environ.get("V2_TEACHER_CHECKPOINT")
    if env and Path(env).is_file():
        return Path(env)
    if _DEFAULT_CHECKPOINT.is_file():
        return _DEFAULT_CHECKPOINT
    return None


def resolve_default_benchmark_pdb_dir() -> Path | None:
    env = os.environ.get("BENCHMARK_PDB_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    if _DEFAULT_BENCHMARK_PDB_DIR.is_dir():
        return _DEFAULT_BENCHMARK_PDB_DIR
    return None
