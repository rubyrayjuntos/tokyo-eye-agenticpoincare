"""Export NGL structure + Poincaré disc HTML for every protein in a training corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from experiments.training.v6._data import _download_pdb
from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import attach_v6_features
from science.dtie.v6.visualization.interactive_viewer import (
    build_residue_channel_lookup,
    nodes_from_training_inference,
    write_offline_annotated_pdb,
    write_structure_viewers,
)
from shared.gnn_viewer_paths import viewer_output_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("export_corpus_viewers")


def export_corpus_viewers(
    *,
    checkpoint: Path,
    corpus: Path,
    pdb_dir: Path,
    device: str = "cpu",
    output_root: Path | None = None,
    max_proteins: int | None = None,
) -> list[dict[str, str]]:
    """Run inference and write both HTML viewers per structure."""
    model = load_v6_model(checkpoint, device)
    proteins, failed = load_training_proteins(
        pdb_dir,
        corpus,
        max_proteins=max_proteins,
        use_cache=True,
    )
    if failed:
        logger.warning("%d corpus entries failed to load", failed)
    if not proteins:
        raise RuntimeError(f"No proteins loaded from {corpus}")

    root = output_root or viewer_output_dir()
    model_version = "GOSPConeMapper-v6"
    checkpoint_path = str(checkpoint)
    written: list[dict[str, str]] = []

    with torch.no_grad():
        for prot in proteins:
            pdb_id = str(prot["pdb_id"]).upper()
            chain = str(prot.get("chain", "A"))
            sid = pdb_id.lower()
            out_dir = root / sid
            pdb_path = out_dir / f"{sid}_gosp_native.pdb"

            output = model(attach_v6_features(prot["data"].to(device)))
            nodes = nodes_from_training_inference(prot, output)
            node_lookup = build_residue_channel_lookup(nodes)

            chain_pdb = _download_pdb(pdb_id, pdb_dir)
            n_atoms = write_offline_annotated_pdb(
                chain_pdb,
                node_lookup,
                pdb_path,
                chain=chain,
            )
            if n_atoms == 0:
                logger.warning("Skipping %s — no atoms annotated", pdb_id)
                continue

            pdb_text = pdb_path.read_text(encoding="utf-8")
            curvature = None
            if hasattr(model, "curvature"):
                curvature = float(model.curvature.detach().cpu().item())

            paths = write_structure_viewers(
                structure_id=sid,
                pdb_text=pdb_text,
                nodes=nodes,
                model_version=model_version,
                out_dir=out_dir,
                checkpoint_path=checkpoint_path,
                curvature=curvature,
            )
            logger.info(
                "%s: wrote structure=%s disc=%s (%d residues, %d atoms)",
                pdb_id,
                paths["structure_html"],
                paths["disc_html"],
                prot["n_residues"],
                n_atoms,
            )
            written.append({"pdb_id": pdb_id, **paths})

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("manifests/v6_corpus_stage_a_small_v1.json"),
    )
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-proteins", type=int, default=None)
    args = parser.parse_args()

    written = export_corpus_viewers(
        checkpoint=args.checkpoint,
        corpus=args.corpus,
        pdb_dir=args.pdb_dir,
        device=args.device,
        output_root=args.output_root,
        max_proteins=args.max_proteins,
    )
    print(f"Exported {len(written)} structure viewer pairs under {args.output_root or viewer_output_dir()}")


if __name__ == "__main__":
    main()
