#!/usr/bin/env python3
"""Precompute dehydron_barcode_v1 sidecars for corpus training graphs.

Writes one ``.pt`` per manifest entry under ``{out_dir}/{PDB_ID}_{chain}_dehydron_barcode_v1.pt``
with per-residue tensors (``scalars``, ``missing``, optional ``binned``) and ``metadata``.

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

import torch

from experiments.training.v6 import _data
from experiments.training.v6.corpus import iter_corpus_entries
from science.dtie.common.dehydron_barcode_features import (
    BARCODE_FEATURE_VERSION,
    featurize_chain_dehydron_barcode,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO_ROOT / "manifests" / "v6_corpus_stage_a_small_v1.json"


def _sidecar_path(out_dir: Path, pdb_id: str, chain: str) -> Path:
    return out_dir / f"{pdb_id.upper()}_{chain}_dehydron_barcode_v1.pt"


def _resolve_pdb_path(pdb_id: str, pdb_dir: Path) -> Path:
    return _data._download_pdb(pdb_id.upper(), pdb_dir)


def _save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "scalars": torch.as_tensor(payload["scalars"], dtype=torch.float32),
        "missing": torch.as_tensor(payload["missing"], dtype=torch.float32),
        "metadata": payload["metadata"],
    }
    if payload.get("binned") is not None:
        save_obj["binned"] = torch.as_tensor(payload["binned"], dtype=torch.float32)
    torch.save(save_obj, path)


def run(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest)
    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    entries = iter_corpus_entries(manifest, max_proteins=args.max_proteins)

    if not entries:
        logger.error("No enabled proteins in manifest %s", manifest)
        return 1

    logger.info(
        "Dehydron barcode precompute: %d structures from %s -> %s (binned=%s)",
        len(entries),
        manifest,
        out_dir,
        args.binned,
    )

    results: list[dict] = []
    failures: list[str] = []

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry.get("chain", "A"))
        sidecar = _sidecar_path(out_dir, pdb_id, chain)
        try:
            pdb_path = _resolve_pdb_path(pdb_id, pdb_dir)
            payload = featurize_chain_dehydron_barcode(
                pdb_path,
                chain,
                use_binned=args.binned,
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
            _save_payload(sidecar, payload)

            n_midpoints = int(metadata.get("n_midpoints", 0))
            n_bars = int(metadata.get("n_bars", 0))
            n_residues = int(payload["scalars"].shape[0])  # type: ignore[attr-defined]
            summary = {
                "pdb_id": pdb_id,
                "chain": chain,
                "sidecar": str(sidecar),
                "n_residues": n_residues,
                "n_midpoints": n_midpoints,
                "n_bars": n_bars,
            }
            results.append(summary)
            logger.info(
                "OK %s:%s — %d residues, n_midpoints=%d, n_bars=%d -> %s",
                pdb_id,
                chain,
                n_residues,
                n_midpoints,
                n_bars,
                sidecar.name,
            )
        except Exception as exc:
            msg = f"{pdb_id}:{chain}: {exc}"
            failures.append(msg)
            logger.error("FAIL %s", msg)
            if args.fail_fast:
                break

    report = {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "binned": args.binned,
        "attempted": len(entries),
        "succeeded": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    print(json.dumps({"succeeded": len(results), "failed": len(failures)}, indent=2))
    if not results:
        logger.error("All %d structure(s) failed", len(entries))
        return 1
    if failures:
        for failure in failures:
            print(f"  ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute dehydron_barcode_v1 sidecars for corpus structures"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Corpus manifest JSON (default: stage_a_small_v1)",
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
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
