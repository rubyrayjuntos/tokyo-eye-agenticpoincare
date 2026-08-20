"""Per-epoch T1a trunk rank (pre_mp / encoder_h) for cold z-norm retrains."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.diagnostics.trunk_hidden_occupancy import _collect, _svd_pack


def measure_t1a_trunk_ranks(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    device: str,
    *,
    max_proteins: int | None = None,
) -> dict[str, Any]:
    """Corpus-pooled SVD effective ranks for ``pre_mp`` and ``encoder_h``."""
    parts: list[dict[str, np.ndarray]] = []
    for prot in proteins[: max_proteins or len(proteins)]:
        bundle = _collect(model, prot, device)
        if bundle is None:
            continue
        if "pre_mp" not in bundle or "encoder_h" not in bundle:
            continue
        parts.append(bundle)
    if not parts:
        return {
            "n_proteins": 0,
            "n_residues": 0,
            "pre_mp_effective_rank": float("nan"),
            "encoder_h_effective_rank": float("nan"),
        }

    pre = np.concatenate([p["pre_mp"] for p in parts], axis=0)
    enc = np.concatenate([p["encoder_h"] for p in parts], axis=0)
    pre_s = _svd_pack(pre, name="pre_mp")
    enc_s = _svd_pack(enc, name="encoder_h")
    return {
        "n_proteins": len(parts),
        "n_residues": int(pre.shape[0]),
        "pre_mp_effective_rank": float(pre_s.get("effective_rank", float("nan"))),
        "encoder_h_effective_rank": float(enc_s.get("effective_rank", float("nan"))),
        "pre_mp_top1": float((pre_s.get("explained_var_topk") or {}).get("k1", float("nan"))),
        "encoder_h_top1": float((enc_s.get("explained_var_topk") or {}).get("k1", float("nan"))),
        "pre_mp_sigma2_sigma1": float(pre_s.get("sigma2_sigma1", float("nan"))),
        "encoder_h_sigma2_sigma1": float(enc_s.get("sigma2_sigma1", float("nan"))),
    }
