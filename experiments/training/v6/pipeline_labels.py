"""Pipeline-derived residue labels (cryptic pockets + source leaks) for ResidueStage2."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SIDECAR_DIR = _REPO_ROOT / "manifests" / "pipeline_labels"
_LEAK_POSITIVE_QUANTILE = 0.75


def resolve_structure_id(pdb_id: str) -> str:
    """Governed structure_id for RCSB-sourced PDB entries (lowercase pdb code)."""
    return str(pdb_id).strip().lower()


def training_key_from_governed_residue_id(residue_id: str) -> str | None:
    """Map ``4obe:A:31`` → training graph key ``A:31:``."""
    parts = str(residue_id).strip().split(":")
    if len(parts) < 3:
        return None
    chain = parts[1].upper()
    try:
        seq = int(parts[2])
    except ValueError:
        return None
    return f"{chain}:{seq}:"


def training_key_matches_chain(training_key: str, chain_id: str) -> bool:
    return training_key.upper().startswith(f"{chain_id.upper()}:")


@dataclass
class PipelineFacts:
    structure_id: str
    cryptic_residue_ids: list[str] = field(default_factory=list)
    leak_scores: dict[str, float] = field(default_factory=dict)
    source: str = "none"

    @property
    def has_cryptic(self) -> bool:
        return bool(self.cryptic_residue_ids)

    @property
    def has_leaks(self) -> bool:
        return bool(self.leak_scores)


def _load_from_db(structure_id: str) -> PipelineFacts | None:
    db_url = os.environ.get("DATABASE_URL") or os.environ.get("TEST_DATABASE_URL")
    if not db_url:
        return None
    try:
        import psycopg
    except ImportError:
        logger.debug("psycopg unavailable — skipping DB pipeline label load")
        return None

    cryptic_ids: list[str] = []
    leak_scores: dict[str, float] = {}
    try:
        with psycopg.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT residue_ids
                FROM fact_cryptic_site
                WHERE structure_id = %s
                """,
                (structure_id,),
            )
            for (residue_ids_json,) in cur.fetchall():
                if isinstance(residue_ids_json, str):
                    residue_ids_json = json.loads(residue_ids_json)
                if isinstance(residue_ids_json, list):
                    cryptic_ids.extend(str(rid) for rid in residue_ids_json)

            cur.execute(
                """
                SELECT residue_id, leak_score
                FROM fact_source_leak
                WHERE structure_id = %s
                """,
                (structure_id,),
            )
            for residue_id, leak_score in cur.fetchall():
                if residue_id is None:
                    continue
                leak_scores[str(residue_id)] = float(leak_score or 0.0)
    except Exception as exc:
        logger.warning("Pipeline label DB load failed for %s: %s", structure_id, exc)
        return None

    if not cryptic_ids and not leak_scores:
        return None
    return PipelineFacts(
        structure_id=structure_id,
        cryptic_residue_ids=sorted(set(cryptic_ids)),
        leak_scores=leak_scores,
        source="db",
    )


def _load_from_sidecar(structure_id: str, sidecar_dir: Path | None = None) -> PipelineFacts | None:
    root = sidecar_dir or _DEFAULT_SIDECAR_DIR
    path = root / f"{structure_id}.json"
    if not path.is_file():
        return None
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Invalid pipeline label sidecar %s: %s", path, exc)
        return None

    cryptic_ids = [str(rid) for rid in payload.get("cryptic_residue_ids", [])]
    leak_scores: dict[str, float] = {}
    for row in payload.get("source_leaks", []):
        if isinstance(row, dict) and row.get("residue_id"):
            leak_scores[str(row["residue_id"])] = float(row.get("leak_score") or 0.0)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            leak_scores[str(row[0])] = float(row[1] or 0.0)
    if not cryptic_ids and not leak_scores:
        return None
    return PipelineFacts(
        structure_id=structure_id,
        cryptic_residue_ids=cryptic_ids,
        leak_scores=leak_scores,
        source=f"sidecar:{path.name}",
    )


def load_pipeline_facts(
    pdb_id: str,
    *,
    sidecar_dir: Path | None = None,
) -> PipelineFacts | None:
    """Load governed pipeline facts from DB, then JSON sidecar fallback."""
    structure_id = resolve_structure_id(pdb_id)
    facts = _load_from_db(structure_id)
    if facts is not None:
        return facts
    return _load_from_sidecar(structure_id, sidecar_dir)


def _leak_targets_for_chain(
    residue_ids: list[str],
    chain_id: str,
    leak_scores: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Binary leak targets + mask for residues on ``chain_id`` with pipeline facts."""
    n = len(residue_ids)
    target = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    chain_scores: list[tuple[int, float]] = []

    id_to_idx = {rid.upper(): i for i, rid in enumerate(residue_ids)}
    for governed_id, score in leak_scores.items():
        key = training_key_from_governed_residue_id(governed_id)
        if key is None or not training_key_matches_chain(key, chain_id):
            continue
        idx = id_to_idx.get(key.upper())
        if idx is None:
            continue
        chain_scores.append((idx, score))

    if not chain_scores:
        return target, mask

    scores_only = np.array([s for _, s in chain_scores], dtype=np.float64)
    threshold = float(np.quantile(scores_only, _LEAK_POSITIVE_QUANTILE))
    mask[:] = True
    for idx, score in chain_scores:
        target[idx] = 1.0 if score >= threshold else 0.0

    return target, mask


def _cryptic_targets_for_chain(
    residue_ids: list[str],
    chain_id: str,
    cryptic_residue_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    n = len(residue_ids)
    target = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    matched: set[str] = set()

    id_to_idx = {rid.upper(): i for i, rid in enumerate(residue_ids)}
    for governed_id in cryptic_residue_ids:
        key = training_key_from_governed_residue_id(governed_id)
        if key is None or not training_key_matches_chain(key, chain_id):
            continue
        idx = id_to_idx.get(key.upper())
        if idx is None:
            continue
        target[idx] = 1.0
        matched.add(key.upper())

    if matched:
        mask[:] = True
    return target, mask, matched


def apply_pipeline_labels(prot: dict[str, Any], facts: PipelineFacts) -> None:
    """Overlay pipeline cryptic / source-leak supervision on an loaded protein dict."""
    chain = str(prot.get("chain", "A"))
    residue_ids = prot["residue_ids"]

    meta = dict(prot.get("label_meta") or {})
    meta["pipeline_source"] = facts.source

    if facts.has_cryptic:
        pocket, pocket_mask, matched = _cryptic_targets_for_chain(
            residue_ids,
            chain,
            facts.cryptic_residue_ids,
        )
        if matched:
            prot["target_pocket"] = torch.tensor(pocket, dtype=torch.float32).unsqueeze(1)
            prot["pocket_label_mask"] = torch.tensor(pocket_mask, dtype=torch.bool)
            meta["pocket_source"] = "pipeline_cryptic_site"
            meta["pocket_pos_rate"] = float(pocket.mean())
            logger.info(
                "%s chain %s: pipeline cryptic pocket labels (%d/%d pos)",
                prot.get("pdb_id", "?"),
                chain,
                int(pocket.sum()),
                len(residue_ids),
            )

    if facts.has_leaks:
        leak, leak_mask = _leak_targets_for_chain(residue_ids, chain, facts.leak_scores)
        if leak_mask.any():
            prot["target_leak"] = torch.tensor(leak, dtype=torch.float32).unsqueeze(1)
            prot["leak_label_mask"] = torch.tensor(leak_mask, dtype=torch.bool)
            meta["leak_source"] = "pipeline_source_leak"
            meta["leak_pos_rate"] = float(leak[leak_mask].mean()) if leak_mask.any() else 0.0
            logger.info(
                "%s chain %s: pipeline source-leak labels (%d/%d pos, masked=%d)",
                prot.get("pdb_id", "?"),
                chain,
                int(leak.sum()),
                len(residue_ids),
                int(leak_mask.sum()),
            )

    prot["label_meta"] = meta


def attach_pipeline_supervision(prot: dict[str, Any], pdb_dir: Path) -> None:
    """
    ResidueStage2: RCSB proxy labels (interface + pocket fallback) plus pipeline overlay.
    """
    from experiments.training.v6.residue_labels import attach_residue_supervision

    attach_residue_supervision(prot, pdb_dir)
    pdb_id = str(prot.get("pdb_id", "")).upper()
    facts = load_pipeline_facts(pdb_id)
    if facts is None:
        logger.info("%s: no pipeline facts — RCSB proxy labels only", pdb_id)
        return
    apply_pipeline_labels(prot, facts)
