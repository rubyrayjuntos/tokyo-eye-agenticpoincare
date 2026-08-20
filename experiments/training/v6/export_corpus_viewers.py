"""Export NGL structure + Poincaré disc HTML for every protein in a training corpus."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from experiments.training.v6._data import _download_pdb
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.v5.gnn.model import precompute_clustering
from science.dtie.v6.visualization.interactive_viewer import (
    build_residue_channel_lookup,
    nodes_from_training_inference,
    write_offline_annotated_pdb,
    write_structure_viewers,
)
from science.dtie.v6.visualization.shell_signal_gate import (
    evaluate_shell_signal_ssot,
    write_shell_gate_report,
)
from science.training.gnn_lineage import load_model_from_checkpoint
from shared.gnn_viewer_paths import interactive_viewer_enabled, viewer_output_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("export_corpus_viewers")


def export_corpus_viewers(
    *,
    checkpoint: Path | None = None,
    model: torch.nn.Module | None = None,
    proteins: list[dict[str, Any]] | None = None,
    corpus: Path | None = None,
    pdb_dir: Path | None = None,
    device: str = "cpu",
    output_root: Path | None = None,
    mirror_root: Path | None = None,
    max_proteins: int | None = None,
    structural_disc_frozen: bool = False,
    checkpoint_label: str | None = None,
) -> list[dict[str, str]]:
    """Run inference and write structure + disc + split HTML per structure.

    When ``model`` is provided, ``checkpoint`` is optional (used only for metadata paths).
  When ``proteins`` is omitted, loads from ``corpus`` + ``pdb_dir``.
    """
    if model is None:
        if checkpoint is None:
            raise ValueError("Provide checkpoint or model")
        model = load_model_from_checkpoint(checkpoint, device)
    if proteins is None:
        if corpus is None or pdb_dir is None:
            raise ValueError("Provide proteins or corpus+pdb_dir")
        proteins, failed = load_training_proteins(
            pdb_dir,
            corpus,
            max_proteins=max_proteins,
            use_cache=True,
        )
        if failed:
            logger.warning("%d corpus entries failed to load", failed)
    if not proteins:
        raise RuntimeError("No proteins to export viewers for")

    run_root = output_root or viewer_output_dir()
    mirror = mirror_root if mirror_root is not None else viewer_output_dir()
    ckpt_meta = checkpoint_label or (str(checkpoint) if checkpoint else "in_memory")
    model_version = "GOSPConeMapper-v6"
    disc_layout_label = (
        "structural SSOT (ρ, τ, Cα)" if structural_disc_frozen else "gnn_learned"
    )
    written: list[dict[str, str]] = []
    gate_summary: list[dict[str, Any]] = []

    model.eval()
    pdb_cache_dir = pdb_dir or Path("/tmp/dtie_pdb_cache")
    with torch.no_grad():
        for prot in proteins:
            pdb_id = str(prot["pdb_id"]).upper()
            chain = str(prot.get("chain", "A"))
            sid = pdb_id.lower()

            # Always use prepare_training_batch so role-edges / rim fanout /
            # geometric angular prior match the training forward (not bare features).
            data = prepare_training_batch(
                model,
                prot,
                device,
                structural_disc_frozen=structural_disc_frozen,
            )
            in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
            if data.x.size(-1) > in_f:
                data.x = data.x[:, :in_f].contiguous()
            data = data.to(device)
            data = precompute_clustering(data)
            output = model(data)
            nodes = nodes_from_training_inference(prot, output, data=data)
            node_lookup = build_residue_channel_lookup(nodes)

            chain_pdb = _download_pdb(pdb_id, pdb_cache_dir)
            curvature = None
            if hasattr(model, "curvature"):
                curvature = float(model.curvature.detach().cpu().item())

            verdict = evaluate_shell_signal_ssot(
                nodes,
                structure_id=sid,
                structural_disc_frozen=structural_disc_frozen,
            )

            primary_dir = run_root / sid
            primary_dir.mkdir(parents=True, exist_ok=True)
            pdb_path = primary_dir / f"{sid}_gosp_native.pdb"
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

            for root in {run_root, mirror}:
                out_dir = root / sid
                out_dir.mkdir(parents=True, exist_ok=True)
                if root != run_root:
                    (out_dir / f"{sid}_gosp_native.pdb").write_text(
                        pdb_text, encoding="utf-8"
                    )
                paths = write_structure_viewers(
                    structure_id=sid,
                    pdb_text=pdb_text,
                    nodes=nodes,
                    model_version=model_version,
                    out_dir=out_dir,
                    checkpoint_path=ckpt_meta,
                    curvature=curvature,
                    disc_layout_label=disc_layout_label,
                )
                write_shell_gate_report(
                    verdict,
                    out_dir / f"{sid}_shell_signal_gate.json",
                )

            logger.info(
                "%s: viewers → %s (gate %s)",
                pdb_id,
                paths.get("split_html", paths["structure_html"]),
                "PASS" if verdict.passed else "FAIL",
            )
            gate_summary.append(verdict.to_dict())
            written.append({"pdb_id": pdb_id, **paths})

    manifest_path = run_root / "viewer_manifest.json"
    manifest_path.write_text(
        json.dumps({"structures": written, "shell_gates": gate_summary}, indent=2),
        encoding="utf-8",
    )
    return written


def export_training_run_viewers(
    model: torch.nn.Module,
    proteins: list[dict[str, Any]],
    *,
    output_dir: Path,
    pdb_dir: Path,
    device: str,
    structural_disc_frozen: bool = False,
    checkpoint_label: str | None = None,
) -> list[dict[str, str]]:
    """Write full interactive viewer set under ``{output_dir}/viewers`` after training."""
    if not interactive_viewer_enabled():
        logger.info("GNN_INTERACTIVE_HTML disabled — skipping viewer export")
        return []
    run_viewers = output_dir / "viewers"
    return export_corpus_viewers(
        model=model,
        proteins=proteins,
        pdb_dir=pdb_dir,
        device=device,
        output_root=run_viewers,
        mirror_root=viewer_output_dir(),
        structural_disc_frozen=structural_disc_frozen,
        checkpoint_label=checkpoint_label,
    )


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
    parser.add_argument(
        "--structural-disc-frozen",
        action="store_true",
        help="Attach structural SSOT disc before forward (slim preset runs)",
    )
    args = parser.parse_args()

    written = export_corpus_viewers(
        checkpoint=args.checkpoint,
        corpus=args.corpus,
        pdb_dir=args.pdb_dir,
        device=args.device,
        output_root=args.output_root,
        max_proteins=args.max_proteins,
        structural_disc_frozen=args.structural_disc_frozen,
    )
    root = args.output_root or viewer_output_dir()
    print(f"Exported {len(written)} viewer sets under {root}")


if __name__ == "__main__":
    main()
