"""Corpus manifest loading for v6 GNN training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO_ROOT / "manifests" / "v6_corpus_120.json"


def load_corpus_manifest(manifest_path: Path | None = None) -> dict[str, Any]:
    """Load corpus JSON manifest."""
    path = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {path}")
    with path.open() as handle:
        data = json.load(handle)
    if "proteins" not in data:
        raise ValueError(f"Manifest {path} missing 'proteins' list")
    return data


def iter_corpus_entries(
    manifest_path: Path | None = None,
    *,
    max_proteins: int | None = None,
    stage0_only: bool = False,
) -> list[dict[str, Any]]:
    """Return protein entries from manifest, optionally capped."""
    data = load_corpus_manifest(manifest_path)
    entries: list[dict[str, Any]] = []
    for entry in data["proteins"]:
        if not entry.get("enabled", True):
            continue
        if stage0_only and not entry.get("stage0", False):
            continue
        entries.append(entry)
    if max_proteins is not None:
        entries = entries[:max_proteins]
    return entries


def load_training_proteins(
    pdb_dir: Path,
    manifest_path: Path | None = None,
    *,
    max_proteins: int | None = None,
    max_residues: int = 800,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """
    Load protein graphs from corpus manifest.

    Skips structures with more than ``max_residues`` (GPU memory guard).
    Caches parsed graphs under ``pdb_dir/corpus_cache/`` to avoid re-parsing on restart.

    Returns (loaded_proteins, failed_count).
    """
    import hashlib

    from experiments.training.v6._data import load_protein_graph

    manifest = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
    cache_parent = Path(pdb_dir)
    cache_dir = cache_parent / "corpus_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path("/tmp/dtie_pdb_cache/corpus_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using writable corpus cache at %s (pdb_dir not writable)", cache_dir)
    cache_key = hashlib.sha256(
        f"{manifest.resolve()}|{max_proteins}|{max_residues}|bf_v1".encode()
    ).hexdigest()[:16]
    cache_path = cache_dir / f"graphs_{cache_key}.pt"

    if use_cache and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        proteins = payload.get("proteins", [])
        failed = int(payload.get("failed", 0))
        logger.info(
            "Loaded %d proteins from cache %s (%d prior failures, max_residues=%d)",
            len(proteins),
            cache_path.name,
            failed,
            max_residues,
        )
        return proteins, failed

    entries = iter_corpus_entries(manifest_path, max_proteins=max_proteins)
    proteins: list[dict[str, Any]] = []
    failed = 0

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        prot = load_protein_graph(pdb_id, chain, pdb_dir)
        if prot is None:
            failed += 1
            logger.warning("Failed to load %s chain %s", pdb_id, chain)
            continue
        if prot["n_residues"] > max_residues:
            failed += 1
            logger.warning(
                "Skipping %s chain %s: %d residues > max_residues=%d",
                pdb_id,
                chain,
                prot["n_residues"],
                max_residues,
            )
            continue
        prot["fold_class"] = entry.get("fold_class", "unknown")
        prot["gene"] = entry.get("gene", "")
        from science.training.mlflow_governance import resolve_protein_fold_id

        prot["fold_id"] = resolve_protein_fold_id(entry)
        proteins.append(prot)
        logger.info(
            "  %s (%s): %d residues [%s]",
            pdb_id,
            entry.get("gene", ""),
            prot["n_residues"],
            entry.get("fold_class", ""),
        )

    if use_cache and proteins:
        torch.save({"proteins": proteins, "failed": failed, "max_residues": max_residues}, cache_path)
        logger.info("Wrote corpus cache %s", cache_path)

    return proteins, failed
