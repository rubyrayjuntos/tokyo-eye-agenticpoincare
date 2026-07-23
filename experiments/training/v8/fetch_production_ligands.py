#!/usr/bin/env python3
"""Fetch PDBbind 2020 refined ligands from Hugging Face and flat-stage them.

Source: ``photonmz/pdbbindpp-2020`` (``pbpp-2020.zip`` with per-PDB folders
containing ``ligand.mol2`` / ``ligand.sdf``).

Stages into ``data/pdbbind/ligands/{pdb_id}.mol2`` (Option C; SDF fallback).
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def _target_pdbs(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text())
    out: set[str] = set()
    for split in ("train", "val", "core_test"):
        for item in manifest.get(split, []):
            out.add(str(item["pdb_id"]).lower())
    return out


def download_and_stage_ligands(
    *,
    manifest_path: Path,
    dst_dir: Path,
    repo_id: str = "photonmz/pdbbindpp-2020",
    zip_name: str = "pbpp-2020.zip",
) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    targets = _target_pdbs(manifest_path)
    print(f"[fetch] leak-wall targets: {len(targets)}")

    print(f"[fetch] downloading {repo_id}/{zip_name} …")
    zip_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=zip_name,
        )
    )
    print(f"[fetch] local zip: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")

    n_mol2 = n_sdf = n_miss = 0
    missing: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        # Support both ``1a30/ligand.mol2`` and ``refined-set/1a30/ligand.mol2``
        for pdb_id in sorted(targets):
            mol2_cands = [
                f"pbpp-2020/{pdb_id}/{pdb_id}_ligand.mol2",
                f"{pdb_id}/{pdb_id}_ligand.mol2",
                f"{pdb_id}/ligand.mol2",
                f"refined-set/{pdb_id}/ligand.mol2",
                f"refined_set/{pdb_id}/ligand.mol2",
            ]
            sdf_cands = [
                f"pbpp-2020/{pdb_id}/{pdb_id}_ligand.sdf",
                f"{pdb_id}/{pdb_id}_ligand.sdf",
                f"{pdb_id}/ligand.sdf",
                f"refined-set/{pdb_id}/ligand.sdf",
                f"refined_set/{pdb_id}/ligand.sdf",
            ]
            staged = False
            for cand in mol2_cands:
                if cand in names:
                    out = dst_dir / f"{pdb_id}.mol2"
                    out.write_bytes(zf.read(cand))
                    n_mol2 += 1
                    staged = True
                    break
            if not staged:
                for cand in sdf_cands:
                    if cand in names:
                        out = dst_dir / f"{pdb_id}.sdf"
                        out.write_bytes(zf.read(cand))
                        n_sdf += 1
                        staged = True
                        break
            if not staged:
                n_miss += 1
                missing.append(pdb_id)

    report = {
        "n_targets": len(targets),
        "n_mol2": n_mol2,
        "n_sdf": n_sdf,
        "n_missing": n_miss,
        "coverage": (n_mol2 + n_sdf) / max(len(targets), 1),
        "dst": str(dst_dir),
        "zip": str(zip_path),
        "missing_examples": missing[:20],
    }
    (dst_dir / "_fetch_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch HF PDBbind ligands → flat stage")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/v8_pdbbind_refined_cluster30_v1.json"),
    )
    p.add_argument("--dst", type=Path, default=Path("data/pdbbind/ligands"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_and_stage_ligands(manifest_path=args.manifest, dst_dir=args.dst)
