"""Isolated module initialization — seed stability independent of construction order.

Modules built after ``node_emb`` share a global torch RNG stream. Changing
``node_emb`` width (Linear 3→H vs 4→H) consumes a different number of draws and
silently reshuffles every subsequent init — including the MoE prototype bank —
even under the same ``torch.manual_seed(seed)``. See
``node_emb_width_rng_shift_check`` (2026-07-16).

Use :func:`derived_seed` + :func:`isolated_torch_seed` (or a private
``torch.Generator``) so gate/prototype init depends only on the training seed,
not on what was constructed earlier in ``__init__``.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Iterator

import torch


def derived_seed(namespace: str, seed: int) -> int:
    """Stable domain-separated seed from ``(namespace, seed)``.

    Keeps gate/prototype draws uncorrelated with whatever the global stream
    would have produced for the same numeric seed at the start of model init.
    """
    digest = hashlib.sha256(f"{namespace}:{int(seed)}".encode()).digest()
    # 63-bit positive int — safe for torch.Generator.manual_seed on all platforms.
    return int.from_bytes(digest[:8], "big") % (2**63)


@contextmanager
def isolated_torch_seed(seed: int) -> Iterator[None]:
    """Temporarily set torch CPU/CUDA RNG to ``seed``, then restore prior state."""
    cpu_state = torch.get_rng_state()
    cuda_states: list[torch.Tensor] | None = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
