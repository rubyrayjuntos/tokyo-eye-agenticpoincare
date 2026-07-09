"""G4 — stable holdout masks for v3 aleatoric shaping (var_penalty / hinge)."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Sequence

import numpy as np
import torch

G4_DEFAULT_HOLDOUT_FRACTION = 0.20
G4_DEFAULT_MASK_SEED = 42
# Default: whole-protein holdout — P8 on held-out structures, not within-protein interpolation.
G4_DEFAULT_HOLDOUT_MODE = "protein"
HoldoutMode = Literal["protein", "residue_stratified"]


def _stable_unit_float(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def corpus_protein_holdout_ids(
    pdb_ids: Sequence[str],
    *,
    holdout_fraction: float = G4_DEFAULT_HOLDOUT_FRACTION,
    seed: int = G4_DEFAULT_MASK_SEED,
) -> frozenset[str]:
    """Stable set of PDB IDs excluded from aleatoric shaping (all residues per protein)."""
    unique = sorted({str(p).upper() for p in pdb_ids if p})
    if len(unique) < 2:
        raise ValueError(f"need >=2 proteins for protein holdout, got {len(unique)}")
    holdout = {
        pdb
        for pdb in unique
        if _stable_unit_float(str(seed), "protein_holdout", pdb) < holdout_fraction
    }
    if not holdout:
        holdout = {unique[int(_stable_unit_float(str(seed), "fallback") * len(unique)) % len(unique)]}
    if len(holdout) >= len(unique):
        holdout = {unique[0]}
    return frozenset(holdout)


def aleatoric_shaping_holdout_masks(
    target_dehydron: np.ndarray,
    residue_ids: Sequence[str],
    *,
    pdb_id: str = "",
    holdout_fraction: float = G4_DEFAULT_HOLDOUT_FRACTION,
    seed: int = G4_DEFAULT_MASK_SEED,
    stratify_by_dehydron: bool = True,
    holdout_mode: HoldoutMode = G4_DEFAULT_HOLDOUT_MODE,
    corpus_holdout_proteins: frozenset[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, holdout_mask) boolean arrays [N], disjoint, covering all residues.

    ``protein`` mode: entire structure train or holdout (requires ``corpus_holdout_proteins``).
    ``residue_stratified`` mode: per-residue stable split stratified by target_dehydron.
    """
    n = int(target_dehydron.shape[0])
    if len(residue_ids) != n:
        raise ValueError(f"residue_ids length {len(residue_ids)} != n={n}")

    if holdout_mode == "protein":
        if corpus_holdout_proteins is None:
            raise ValueError("corpus_holdout_proteins required when holdout_mode='protein'")
        is_holdout = str(pdb_id).upper() in corpus_holdout_proteins
        holdout = np.full(n, is_holdout, dtype=bool)
        train = ~holdout
        return train, holdout

    holdout = np.zeros(n, dtype=bool)
    dehyd = target_dehydron.reshape(-1).astype(np.float64)
    strata = [0.0, 1.0] if stratify_by_dehydron else [None]
    for stratum in strata:
        if stratum is None:
            idxs = list(range(n))
        else:
            idxs = [i for i in range(n) if float(dehyd[i]) == stratum]
        for i in idxs:
            rid = str(residue_ids[i])
            u = _stable_unit_float(str(seed), pdb_id.upper(), rid, str(stratum))
            holdout[i] = u < holdout_fraction
    train = ~holdout
    if not holdout.any() or not train.any():
        raise ValueError(
            f"degenerate G4 masks for {pdb_id}: holdout={int(holdout.sum())} train={int(train.sum())}"
        )
    return train, holdout


def aleatoric_shaping_holdout_masks_torch(
    target_dehydron: torch.Tensor,
    residue_ids: Sequence[str],
    *,
    pdb_id: str = "",
    holdout_fraction: float = G4_DEFAULT_HOLDOUT_FRACTION,
    seed: int = G4_DEFAULT_MASK_SEED,
    device: torch.device | str | None = None,
    holdout_mode: HoldoutMode = G4_DEFAULT_HOLDOUT_MODE,
    corpus_holdout_proteins: frozenset[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch boolean masks on ``target_dehydron`` device."""
    dehyd_np = target_dehydron.detach().cpu().numpy().reshape(-1)
    train_np, hold_np = aleatoric_shaping_holdout_masks(
        dehyd_np,
        residue_ids,
        pdb_id=pdb_id,
        holdout_fraction=holdout_fraction,
        seed=seed,
        holdout_mode=holdout_mode,
        corpus_holdout_proteins=corpus_holdout_proteins,
    )
    dev = device if device is not None else target_dehydron.device
    return (
        torch.tensor(train_np, device=dev, dtype=torch.bool),
        torch.tensor(hold_np, device=dev, dtype=torch.bool),
    )


def tag_residue_rows_with_shaping_holdout(
    rows: list[dict[str, Any]],
    prot: dict[str, Any],
    *,
    holdout_fraction: float = G4_DEFAULT_HOLDOUT_FRACTION,
    seed: int = G4_DEFAULT_MASK_SEED,
    holdout_mode: HoldoutMode = G4_DEFAULT_HOLDOUT_MODE,
    corpus_holdout_proteins: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Annotate uncertainty audit rows with ``ale_shaping_holdout`` for G4 P8 eval."""
    target_dehydron = prot.get("target_dehydron")
    if target_dehydron is None:
        return rows
    dehyd = target_dehydron.detach().cpu().numpy().reshape(-1)
    residue_ids = list(prot.get("residue_ids") or [f"idx:{i}" for i in range(len(rows))])
    pdb_id = str(prot.get("pdb_id", "?"))
    _, holdout = aleatoric_shaping_holdout_masks(
        dehyd,
        residue_ids,
        pdb_id=pdb_id,
        holdout_fraction=holdout_fraction,
        seed=seed,
        holdout_mode=holdout_mode,
        corpus_holdout_proteins=corpus_holdout_proteins,
    )
    for i, row in enumerate(rows):
        row["ale_shaping_holdout"] = bool(holdout[i])
        row["holdout_granularity"] = holdout_mode
        row["structure_id"] = pdb_id
    return rows


def holdout_mask_metadata(
    rows: Sequence[dict[str, Any]],
    *,
    holdout_key: str = "ale_shaping_holdout",
) -> dict[str, Any]:
    """Summarize holdout granularity for G4 reports."""
    structures = sorted({str(r.get("structure_id", "?")) for r in rows})
    holdout_structures = sorted(
        {str(r.get("structure_id", "?")) for r in rows if r.get(holdout_key)}
    )
    granularity = (
        rows[0].get("holdout_granularity", G4_DEFAULT_HOLDOUT_MODE) if rows else G4_DEFAULT_HOLDOUT_MODE
    )
    thin_warning = None
    if granularity == "protein" and len(holdout_structures) == 1:
        thin_warning = (
            "n_holdout_proteins=1 at 20% of n=12 — holdout P8 and transfer-ratio verdict rest "
            "on a single structure; compare holdout_corpus_contrast and multi-seed eval before "
            "treating as cross-corpus generalization"
        )
    return {
        "holdout_granularity": granularity,
        "n_proteins": len(structures),
        "n_holdout_proteins": len(holdout_structures),
        "holdout_proteins": holdout_structures,
        "thin_holdout_warning": thin_warning,
        "granularity_caveat": (
            "protein holdout: P8 on whole held-out structures — cross-structure generalization"
            if granularity == "protein"
            else (
                "residue_stratified holdout: within-protein interpolation; adjacent residues "
                "may share context — weaker generalization claim than protein holdout"
            )
        ),
    }


def protein_biophysics_summary(prot: dict[str, Any]) -> dict[str, Any]:
    """Per-structure biophysics for holdout vs corpus contrast."""
    dehyd = prot["target_dehydron"].detach().cpu().numpy().reshape(-1)
    rho_t = prot.get("target_rho")
    if rho_t is not None:
        rho = rho_t.detach().cpu().numpy().reshape(-1)
    else:
        rho = prot["data"].x.detach().cpu().numpy()[:, 0]
    x = prot["data"].x.detach().cpu().numpy()
    ss = x[:, 2] if x.shape[1] > 2 else None
    summary: dict[str, Any] = {
        "pdb_id": str(prot.get("pdb_id", "?")).upper(),
        "fold_id": prot.get("fold_id"),
        "gene": prot.get("gene"),
        "n_residues": int(prot.get("n_residues", dehyd.size)),
        "dehydron_fraction": float(np.mean(dehyd)),
        "rho_mean": float(np.mean(rho)),
        "rho_std": float(np.std(rho)),
    }
    if ss is not None:
        summary["helix_fraction"] = float(np.mean(ss < 0.25))
        summary["sheet_fraction"] = float(np.mean((ss >= 0.25) & (ss < 0.75)))
        summary["coil_fraction"] = float(np.mean(ss >= 0.75))
    return summary


def holdout_corpus_contrast(
    proteins: Sequence[dict[str, Any]],
    holdout_proteins: frozenset[str],
    *,
    holdout_seed: int = G4_DEFAULT_MASK_SEED,
) -> dict[str, Any]:
    """Holdout protein(s) vs in-corpus mean — first check for idiosyncratic holdout."""
    summaries = {
        str(p.get("pdb_id", "?")).upper(): protein_biophysics_summary(p) for p in proteins
    }
    train_ids = [pid for pid in summaries if pid not in holdout_proteins]
    hold_ids = sorted(holdout_proteins & summaries.keys())
    if not train_ids or not hold_ids:
        return {"ok": False, "reason": "missing_train_or_holdout_summaries"}

    metrics = ["dehydron_fraction", "rho_mean", "rho_std", "n_residues"]
    for frac in ("helix_fraction", "sheet_fraction", "coil_fraction"):
        if all(summaries[pid].get(frac) is not None for pid in summaries):
            metrics.append(frac)

    corpus_means = {
        m: float(np.mean([summaries[pid][m] for pid in train_ids if summaries[pid].get(m) is not None]))
        for m in metrics
    }
    corpus_stds = {
        m: float(np.std([summaries[pid][m] for pid in train_ids if summaries[pid].get(m) is not None]))
        for m in metrics
    }

    holdout_details: list[dict[str, Any]] = []
    for pid in hold_ids:
        s = summaries[pid]
        deltas: dict[str, Any] = {}
        z_scores: dict[str, float] = {}
        for m in metrics:
            val = s.get(m)
            if val is None or m not in corpus_means:
                continue
            mean = corpus_means[m]
            std = corpus_stds[m]
            deltas[f"{m}_delta_vs_corpus_mean"] = float(val - mean)
            z_scores[f"{m}_z_vs_corpus"] = (
                float((val - mean) / std) if std > 1e-12 else float("nan")
            )
        holdout_details.append(
            {
                "pdb_id": pid,
                "fold_id": s.get("fold_id"),
                "gene": s.get("gene"),
                "n_residues": s.get("n_residues"),
                "corpus_mean": {m: corpus_means[m] for m in metrics},
                "holdout_values": {m: s.get(m) for m in metrics},
                "delta_vs_corpus": deltas,
                "z_vs_corpus": z_scores,
            }
        )

    return {
        "holdout_seed": holdout_seed,
        "n_corpus_proteins": len(train_ids),
        "n_holdout_proteins": len(hold_ids),
        "holdout_proteins": hold_ids,
        "corpus_protein_ids": sorted(train_ids),
        "corpus_means": corpus_means,
        "corpus_stds": corpus_stds,
        "holdout_details": holdout_details,
        "interpretation": (
            "Inspect z_vs_corpus: |z|>1.5 on dehydron_fraction or rho_mean suggests holdout "
            "protein is atypical vs training set — surprising G4 verdict may reflect structure "
            "idiosyncrasy, not shaping recipe"
        ),
    }
