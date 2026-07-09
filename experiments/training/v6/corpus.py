"""Corpus manifest loading for v6 GNN training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from science.training.corpus_governance import (
    STAGE_A_MAX_RESIDUES,
    is_locked_stage_a_manifest_path,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO_ROOT / "manifests" / "v6_corpus_120.json"


class LockedCorpusLoadError(RuntimeError):
    """Locked Stage A structure failed to load — manifest/loader split-brain."""


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


def _barcode_cache_suffix(
    *,
    use_dehydron_barcode: bool = False,
    use_binned_dehydron: bool = False,
) -> str:
    """Cache-key tag when barcode sidecars widen ``data.x`` (Task 5 / Task 6)."""
    import os

    if not use_dehydron_barcode:
        use_dehydron_barcode = os.environ.get("USE_DEHYDRON_BARCODE", "").lower() in (
            "1",
            "true",
            "yes",
        )
    if not use_binned_dehydron:
        use_binned_dehydron = os.environ.get("USE_BINNED_DEHYDRON", "").lower() in (
            "1",
            "true",
            "yes",
        )
    if not use_dehydron_barcode:
        return ""
    if use_binned_dehydron:
        return "|dbh_v1_binned"
    return "|dbh_v1"


def load_training_proteins(
    pdb_dir: Path,
    manifest_path: Path | None = None,
    *,
    max_proteins: int | None = None,
    max_residues: int | None = None,
    use_cache: bool = True,
    use_dehydron_barcode: bool = False,
    use_binned_dehydron: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """
    Load protein graphs from corpus manifest.

    Structures larger than ``max_residues`` are skipped for open corpora; for the
    locked Stage A manifest any load failure or residue-cap exclusion is a hard error.

    Caches parsed graphs under ``pdb_dir/corpus_cache/`` to avoid re-parsing on restart.

    Returns (loaded_proteins, failed_count).
    """
    import hashlib

    from experiments.training.v6._data import load_protein_graph

    manifest = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
    locked_manifest = is_locked_stage_a_manifest_path(manifest)
    if max_residues is None:
        max_residues = STAGE_A_MAX_RESIDUES if locked_manifest else 800
    cache_parent = Path(pdb_dir)
    cache_dir = cache_parent / "corpus_cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path("/tmp/dtie_pdb_cache/corpus_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using writable corpus cache at %s (pdb_dir not writable)", cache_dir)
    manifest_data = load_corpus_manifest(manifest_path)
    residue_stage = int(manifest_data.get("residue_stage", 0) or 0)
    barcode_tag = _barcode_cache_suffix(
        use_dehydron_barcode=use_dehydron_barcode,
        use_binned_dehydron=use_binned_dehydron,
    )
    cache_key = hashlib.sha256(
        f"{manifest.resolve()}|{max_proteins}|{max_residues}|bf_v1|rs{residue_stage}{barcode_tag}".encode()
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
        logger.info("Loading %s:%s from governed DB...", pdb_id, chain)
        prot = load_protein_graph(pdb_id, chain, pdb_dir)
        if prot is None or int(prot.get("n_residues", 0)) <= 0:
            failed += 1
            msg = f"Failed to load {pdb_id} chain {chain}"
            if locked_manifest:
                raise LockedCorpusLoadError(
                    f"{msg} — locked Stage A corpus load violation; fix manifest chain "
                    f"or PDB availability, not a skippable warning."
                )
            logger.warning(msg)
            continue
        if prot["n_residues"] > max_residues:
            failed += 1
            msg = (
                f"{pdb_id} chain {chain}: {prot['n_residues']} residues > "
                f"max_residues={max_residues}"
            )
            if locked_manifest:
                raise LockedCorpusLoadError(
                    f"{msg} — locked Stage A structure excluded by residue cap; "
                    f"raise STAGE_A_MAX_RESIDUES ({STAGE_A_MAX_RESIDUES}) or demote."
                )
            logger.warning("Skipping %s", msg)
            continue
        prot["fold_class"] = entry.get("fold_class", "unknown")
        prot["gene"] = entry.get("gene", "")
        from science.training.mlflow_governance import resolve_protein_fold_id

        prot["fold_id"] = resolve_protein_fold_id(entry)
        if residue_stage >= 2:
            from experiments.training.v6.pipeline_labels import attach_pipeline_supervision

            attach_pipeline_supervision(prot, pdb_dir)
        elif residue_stage >= 1:
            from experiments.training.v6.residue_labels import attach_residue_supervision

            attach_residue_supervision(prot, pdb_dir)
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

    if locked_manifest and failed:
        raise LockedCorpusLoadError(
            f"{failed} locked Stage A structure(s) failed to load — corpus is not trainable."
        )

    return proteins, failed
