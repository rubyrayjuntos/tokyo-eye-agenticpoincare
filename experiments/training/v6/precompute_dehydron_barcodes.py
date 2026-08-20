#!/usr/bin/env python3
"""Precompute versioned dehydron barcode sidecars for corpus training graphs.

Writes one ``.pt`` per manifest entry under
``{out_dir}/{PDB_ID}_{chain}_{BARCODE_FEATURE_VERSION}.pt`` with per-residue
tensors (``scalars``, ``missing``, optional ``binned``) and ``metadata``.

Corpus z-score μ/σ is computed once over all valid (non-missing) rows and
applied before save. Stats are written to ``{out_dir}/corpus_zscore_stats.json``.

Example (host):
  PYTHONPATH=. python -m experiments.training.v6.precompute_dehydron_barcodes \\
    --manifest manifests/v6_corpus_stage_a_small_v1.json \\
    --pdb-dir pdb_cache \\
    --out-dir checkpoints/v65/dehydron_barcode_v1 \\
    --max-proteins 1 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v6 import _data
from experiments.training.v6.corpus import iter_corpus_entries
from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    SCALAR_NAMES,
    apply_corpus_zscore,
    barcode_sidecar_filename,
    compute_corpus_zscore_stats,
    featurize_chain_dehydron_barcode,
)
from science.dtie.common.residue_features import FeatureMode, build_from_pdb_chain

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO_ROOT / "manifests" / "v6_corpus_stage_a_small_v1.json"


def _sidecar_path(out_dir: Path, pdb_id: str, chain: str) -> Path:
    return out_dir / barcode_sidecar_filename(pdb_id, chain)


def _resolve_pdb_path(pdb_id: str, pdb_dir: Path) -> Path:
    return _data._download_pdb(pdb_id.upper(), pdb_dir)


def stats_missing_for_training_rows(
    payload: dict[str, Any],
    training_residue_indices: list[int],
) -> np.ndarray:
    """Mask sidecar-only PDB rows out of corpus normalization statistics."""
    residue_indices = np.asarray(payload["residue_indices"], dtype=np.int32).reshape(-1)
    missing = np.asarray(payload["missing"], dtype=np.float32).copy()
    if missing.shape != (residue_indices.shape[0], 1):
        raise ValueError(
            "missing/residue_indices shape mismatch: "
            f"{missing.shape} vs {residue_indices.shape}"
        )
    training_rows = np.isin(
        residue_indices,
        np.asarray(training_residue_indices, dtype=np.int32),
    )
    missing[~training_rows, 0] = 1.0
    return missing


def _save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "scalars": torch.as_tensor(payload["scalars"], dtype=torch.float32),
        "missing": torch.as_tensor(payload["missing"], dtype=torch.float32),
        "residue_indices": torch.as_tensor(payload["residue_indices"], dtype=torch.int32),
        "metadata": payload["metadata"],
    }
    if payload.get("binned") is not None:
        save_obj["binned"] = torch.as_tensor(payload["binned"], dtype=torch.float32)
    if payload.get("edge_pairs") is not None:
        save_obj["edge_pairs"] = torch.as_tensor(payload["edge_pairs"], dtype=torch.int32)
    if payload.get("edge_scalars") is not None:
        save_obj["edge_scalars"] = torch.as_tensor(payload["edge_scalars"], dtype=torch.float32)
    torch.save(save_obj, path)


def _load_locked_zscore_stats(path: Path) -> dict[str, Any]:
    stats = json.loads(Path(path).read_text())
    if stats.get("version") != BARCODE_FEATURE_VERSION:
        raise ValueError(
            "z-score stats version mismatch: "
            f"{stats.get('version')!r} != {BARCODE_FEATURE_VERSION!r}"
        )
    if stats.get("scalar_names") != list(SCALAR_NAMES):
        raise ValueError(
            "z-score stats scalar_names mismatch: "
            f"{stats.get('scalar_names')!r} != {list(SCALAR_NAMES)!r}"
        )
    return stats


def precompute_single_pdb(
    *,
    pdb_id: str,
    chain: str,
    pdb_dir: Path,
    out_dir: Path,
    zscore_stats_path: Path,
    use_binned: bool,
) -> Path:
    """Write one OOD sidecar using locked training-corpus normalization."""
    stats_path = Path(zscore_stats_path)
    stats = _load_locked_zscore_stats(stats_path)
    pdb_id = str(pdb_id).upper()
    chain = str(chain)
    pdb_path = _resolve_pdb_path(pdb_id, Path(pdb_dir))
    payload = featurize_chain_dehydron_barcode(
        pdb_path,
        chain,
        use_binned=use_binned,
    )
    training_features = build_from_pdb_chain(
        pdb_path,
        chain,
        mode=FeatureMode.MASTER,
    )
    training_residue_indices = [
        int(feature.residue_index) for feature in training_features
    ]
    stats_missing_for_training_rows(payload, training_residue_indices)
    payload["scalars"] = apply_corpus_zscore(
        payload["scalars"],
        payload["missing"],
        stats,
    )
    metadata = dict(payload["metadata"])
    metadata.update(
        {
            "pdb_id": pdb_id,
            "chain": chain,
            "pdb_path": str(pdb_path),
            "sidecar_version": BARCODE_FEATURE_VERSION,
            "corpus_zscore": {
                "path": str(stats_path),
                "scope": stats.get("scope", "master_training_graph_rows"),
                "n_valid_rows": int(stats["n_valid_rows"]),
                "mean": stats["mean"],
                "std": stats["std"],
            },
        }
    )
    payload["metadata"] = metadata
    sidecar = _sidecar_path(Path(out_dir), pdb_id, chain)
    _save_payload(sidecar, payload)
    logger.info(
        "Wrote OOD sidecar %s using locked stats %s",
        sidecar,
        stats_path,
    )
    return sidecar


def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    entries = iter_corpus_entries(manifest, max_proteins=args.max_proteins)

    if not entries:
        logger.error("No enabled proteins in manifest %s", manifest)
        return 1

    logger.info(
        "Dehydron barcode precompute: %d structures from %s -> %s (binned=%s, version=%s)",
        len(entries),
        manifest,
        out_dir,
        args.binned,
        BARCODE_FEATURE_VERSION,
    )

    staged: list[dict[str, Any]] = []
    failures: list[str] = []

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        try:
            pdb_path = _resolve_pdb_path(pdb_id, pdb_dir)
            payload = featurize_chain_dehydron_barcode(
                pdb_path,
                chain,
                use_binned=args.binned,
            )
            training_features = build_from_pdb_chain(
                pdb_path,
                chain,
                mode=FeatureMode.MASTER,
            )
            training_residue_indices = [
                int(feature.residue_index) for feature in training_features
            ]
            stats_missing = stats_missing_for_training_rows(
                payload,
                training_residue_indices,
            )
            metadata = payload["metadata"]
            if not isinstance(metadata, dict):
                raise TypeError(f"metadata must be dict, got {type(metadata)!r}")
            metadata = {
                **metadata,
                "pdb_id": pdb_id,
                "chain": chain,
                "pdb_path": str(pdb_path),
                "sidecar_version": BARCODE_FEATURE_VERSION,
            }
            payload["metadata"] = metadata
            staged.append(
                {
                    "pdb_id": pdb_id,
                    "chain": chain,
                    "sidecar": _sidecar_path(out_dir, pdb_id, chain),
                    "payload": payload,
                    "stats_missing": stats_missing,
                }
            )
            logger.info(
                "featurized %s:%s — residues=%d midpoints=%d bars=%d",
                pdb_id,
                chain,
                int(payload["scalars"].shape[0]),  # type: ignore[attr-defined]
                int(metadata.get("n_midpoints", 0)),
                int(metadata.get("n_bars", 0)),
            )
        except Exception as exc:
            msg = f"{pdb_id}:{chain}: {exc}"
            failures.append(msg)
            logger.error("FAIL %s", msg)
            if args.fail_fast:
                break

    if not staged:
        logger.error("All %d structure(s) failed before z-score", len(entries))
        return 1

    zscore_stats = compute_corpus_zscore_stats(
        [item["payload"]["scalars"] for item in staged],
        [item["stats_missing"] for item in staged],
    )
    zscore_stats["manifest"] = str(manifest)
    zscore_stats["scope"] = "master_training_graph_rows"
    zscore_path = out_dir / "corpus_zscore_stats.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    zscore_path.write_text(json.dumps(zscore_stats, indent=2))
    logger.info(
        "Corpus z-score stats: n_valid_rows=%d -> %s",
        zscore_stats["n_valid_rows"],
        zscore_path,
    )

    results: list[dict] = []
    for item in staged:
        payload = item["payload"]
        payload["scalars"] = apply_corpus_zscore(
            payload["scalars"],
            payload["missing"],
            zscore_stats,
        )
        metadata = dict(payload["metadata"])
        metadata["corpus_zscore"] = {
            "path": str(zscore_path),
            "scope": zscore_stats["scope"],
            "n_valid_rows": zscore_stats["n_valid_rows"],
            "mean": zscore_stats["mean"],
            "std": zscore_stats["std"],
        }
        payload["metadata"] = metadata
        _save_payload(item["sidecar"], payload)
        results.append(
            {
                "pdb_id": item["pdb_id"],
                "chain": item["chain"],
                "sidecar": str(item["sidecar"]),
                "n_residues": int(payload["scalars"].shape[0]),
                "n_midpoints": int(metadata.get("n_midpoints", 0)),
                "n_bars": int(metadata.get("n_bars", 0)),
            }
        )
        logger.info("wrote %s", item["sidecar"].name)

    report = {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "binned": args.binned,
        "barcode_feature_version": BARCODE_FEATURE_VERSION,
        "corpus_zscore_stats": str(zscore_path),
        "attempted": len(entries),
        "succeeded": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    print(
        json.dumps(
            {
                "succeeded": len(results),
                "failed": len(failures),
                "corpus_zscore_stats": str(zscore_path),
                "version": BARCODE_FEATURE_VERSION,
            },
            indent=2,
        )
    )
    if failures:
        for failure in failures:
            print(f"  ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Precompute {BARCODE_FEATURE_VERSION} sidecars with corpus z-score"
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Corpus manifest JSON (default: stage_a_small_v1)",
    )
    parser.add_argument(
        "--pdb-id",
        default="",
        help="Precompute one OOD PDB instead of a manifest",
    )
    parser.add_argument("--chain", default="A", help="Chain for --pdb-id")
    parser.add_argument(
        "--zscore-stats",
        type=Path,
        default=None,
        help="Locked training-corpus stats (required with --pdb-id)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=_REPO_ROOT / "pdb_cache",
        help="Directory for PDB files (download via RCSB if missing)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for sidecar .pt files",
    )
    parser.add_argument(
        "--binned",
        action="store_true",
        help="Include L1-normalized H1 persistence histogram (40 bins)",
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--report", type=str, default="", help="Write JSON summary to path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.pdb_id:
        if args.zscore_stats is None:
            parser.error("--zscore-stats is required with --pdb-id")
        precompute_single_pdb(
            pdb_id=args.pdb_id,
            chain=args.chain,
            pdb_dir=args.pdb_dir,
            out_dir=args.out_dir,
            zscore_stats_path=args.zscore_stats,
            use_binned=args.binned,
        )
        raise SystemExit(0)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
