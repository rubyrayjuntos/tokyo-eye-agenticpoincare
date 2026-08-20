"""Regenerate NGL + Poincaré viewers for the 4 controlled width-ablation runs.

Refreshes each run's ``viewers/`` dir with the fixed viewer HTML (NGL color-scheme
rebuild on metric change + physics Investigation default + matched disc↔NGL colormap).

DB-free: features are rebuilt from cached PDBs via the MASTER-parity PDB path
(``TRAINING_LOAD_FROM_PDB=1``). Input mode is set per run so ``data.x`` width matches
``node_emb``: 3d runs → topology_three_vector, 4d runs → legacy_four_vector.

IMPORTANT: prefer mature end-of-run checkpoints (``phase_12.pt`` / ``v66_phase*_12prot.pt``)
over ``v66_best_disc.pt``. The disc-occupancy saver can freeze an early
origin-collapsed epoch (high σ₂/σ₁ with ``disc_r_std≈0``) — regenerating from that
file makes every Poincaré look like a single center dot even when the mature run
is healthy. Caught 2026-07-17 on controlled 3d seed2 (best_disc=ep7 collapse;
phase_12=ep30 ``disc_r_std≈0.15``).

Run: python -m experiments.diagnostics.regen_controlled_ablation_viewers
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("regen_controlled_viewers")

RUN_ROOT = Path("checkpoints/v66/runs")
MANIFEST = Path("manifests/v6_corpus_stage_a_small_v1.json")
# Prefer the science-container cache mount; fall back to repo pdb_cache.
PDB_DIR = (
    Path("/tmp/dtie_pdb_cache")
    if Path("/tmp/dtie_pdb_cache").is_dir()
    else Path("pdb_cache")
)

RUNS = [
    ("fix1_s4_stack_initseed_controlled_3d_seed1_v1", "topology_three_vector"),
    ("fix1_s4_stack_initseed_controlled_3d_seed2_v1", "topology_three_vector"),
    ("fix1_s4_stack_initseed_controlled_4d_seed1_v1", "legacy_four_vector"),
    ("fix1_s4_stack_initseed_controlled_4d_seed2_v1", "legacy_four_vector"),
]

# Prefer mature phase / full-run artifacts. ``*_best_disc.pt`` is last-resort only.
_CHECKPOINT_CANDIDATES = (
    "phase_12.pt",
    "v66_phase12_12prot.pt",
    "phase_2.pt",
    "v66_phase2_12prot.pt",
    "v66_best.pt",
    "v66_best_disc.pt",  # last — may be an early origin-collapse snapshot
)


def _pick_checkpoint(run_dir: Path) -> Path | None:
    for name in _CHECKPOINT_CANDIDATES:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _enrich_checkpoint_metadata(run_dir: Path, checkpoint: Path, tmp_dir: Path) -> Path:
    """Bare ``phase_*.pt`` often omit architecture/training_config.

    Borrow metadata from ``v66_best_disc.pt`` / latest epoch snapshot so
    ``load_model_from_checkpoint`` rebuilds feeler flags (rim fanout, geom
    prior, T1a z-norm) that the mature weights expect.
    """
    import torch

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        return checkpoint
    if raw.get("architecture") and raw.get("training_config"):
        return checkpoint

    arch = raw.get("architecture")
    tc = raw.get("training_config")
    meta_src = run_dir / "v66_best_disc.pt"
    if meta_src.is_file():
        meta = torch.load(meta_src, map_location="cpu", weights_only=False)
        arch = arch or meta.get("architecture")
        tc = tc or meta.get("training_config")
    # Prefer latest epoch snapshot for training_config (has input_feature_zscore etc.)
    epochs = sorted((run_dir / "epochs").glob("epoch_*.pt")) if (run_dir / "epochs").is_dir() else []
    if epochs:
        ep = torch.load(epochs[-1], map_location="cpu", weights_only=False)
        if isinstance(ep, dict) and ep.get("training_config"):
            tc = ep["training_config"]

    if not arch and not tc:
        return checkpoint

    enriched = {
        "model_state_dict": raw.get("model_state_dict", raw),
        "architecture": arch,
        "training_config": tc,
        "global_epoch": raw.get("global_epoch"),
        "phase": raw.get("phase"),
        "phase_name": raw.get("phase_name"),
    }
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{run_dir.name}__{checkpoint.stem}__enriched.pt"
    torch.save(enriched, out)
    logger.info(
        "Enriched %s with architecture/training_config → %s",
        checkpoint.name,
        out.name,
    )
    return out


def _chain_map() -> dict[str, str]:
    manifest = json.loads(MANIFEST.read_text())
    out: dict[str, str] = {}
    for p in manifest.get("proteins", []):
        pid = str(p.get("pdb_id") or p.get("id") or "").upper()
        if pid:
            out[pid] = str(p.get("chain", "A"))
    return out


def _run_structures(run_id: str) -> list[str]:
    vm = RUN_ROOT / run_id / "viewers" / "viewer_manifest.json"
    data = json.loads(vm.read_text())
    return [str(s["pdb_id"]).upper() for s in data.get("structures", [])]


def main() -> None:
    os.environ["TRAINING_LOAD_FROM_PDB"] = "1"
    chain_map = _chain_map()
    tmp_dir = Path("/tmp/tokyoeye_viewer_ckpt_enrich")

    for run_id, input_mode in RUNS:
        os.environ["GNN_INPUT_MODE"] = input_mode
        # Import inside loop so gnn_feature_set_id() re-reads GNN_INPUT_MODE.
        from experiments.training.v6._data import load_protein_graph
        from experiments.training.v6.export_corpus_viewers import export_corpus_viewers

        run_dir = RUN_ROOT / run_id
        checkpoint = _pick_checkpoint(run_dir)
        viewers_dir = run_dir / "viewers"
        if checkpoint is None:
            logger.error("No usable checkpoint under %s — skipping", run_dir)
            continue
        if checkpoint.name == "v66_best_disc.pt":
            logger.warning(
                "%s: falling back to v66_best_disc.pt (may be early origin-collapse)",
                run_id,
            )
        load_ckpt = _enrich_checkpoint_metadata(run_dir, checkpoint, tmp_dir)

        pdb_ids = _run_structures(run_id)
        logger.info(
            "=== %s (mode=%s, ckpt=%s → load=%s) — %d structures ===",
            run_id,
            input_mode,
            checkpoint.name,
            load_ckpt.name,
            len(pdb_ids),
        )

        proteins = []
        for pid in pdb_ids:
            chain = chain_map.get(pid, "A")
            prot = load_protein_graph(pid, chain, PDB_DIR)
            if prot is None:
                logger.warning("%s: failed to build graph — skipping", pid)
                continue
            proteins.append(prot)

        if not proteins:
            logger.error("%s: no proteins built — skipping run", run_id)
            continue

        written = export_corpus_viewers(
            checkpoint=load_ckpt,
            proteins=proteins,
            pdb_dir=PDB_DIR,
            device="cpu",
            output_root=viewers_dir,
            mirror_root=viewers_dir,  # collapse mirror → no shared-dir clobber
            structural_disc_frozen=False,
            checkpoint_label=str(checkpoint),  # footer shows the mature file name
        )
        logger.info("%s: regenerated %d viewer sets → %s", run_id, len(written), viewers_dir)


if __name__ == "__main__":
    main()
