"""P4 biology grade pack: sealed vs expand champion on 4OBE (post-lift probes).

Logs a JSON report under the diagnostics dir. Does not claim disc-alone Pass.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.training.v66.healthy_fix1 import HEALTHY_FIX1_CKPT
from science.training.gnn_lineage import load_model_from_checkpoint


def _resolve_champion_ckpt(champion_dir: Path) -> Path | None:
    """Prefer fat checkpoints with training_config — never bare phase_12.pt."""

    def _has_training_config(path: Path) -> bool:
        try:
            import torch

            raw = torch.load(path, map_location="cpu", weights_only=False)
            return isinstance(raw, dict) and isinstance(raw.get("training_config"), dict)
        except Exception:
            return False

    # Prefer held continue epoch: last snap with probe_r_proj_depth ≥ 0.95 when
    # metrics exist; else last fat epoch_*.pt (never bare phase_12).
    epochs = champion_dir / "epochs"
    metrics_path = champion_dir / "metrics.json"
    if epochs.is_dir() and metrics_path.is_file():
        try:
            rows = json.loads(metrics_path.read_text())
            held_ge: int | None = None
            for row in reversed(rows):
                h = row.get("health") or {}
                pr = h.get("probe_r_proj_depth")
                if pr is not None and float(pr) >= 0.95:
                    held_ge = int(row.get("global_epoch") or 0)
                    break
            if held_ge:
                snap = epochs / f"epoch_{held_ge:03d}.pt"
                if snap.is_file() and _has_training_config(snap):
                    return snap
        except Exception:
            pass
    if epochs.is_dir():
        snaps = sorted(epochs.glob("epoch_*.pt"))
        for snap in reversed(snaps):
            if _has_training_config(snap):
                return snap

    for pattern in ("v66_phase12_*.pt", "v66_best_disc.pt", "v66_best.pt", "v66_healthy_sealed.pt"):
        matches = sorted(champion_dir.glob(pattern))
        for p in reversed(matches):
            if p.is_file() and _has_training_config(p):
                return p

    phase12 = champion_dir / "phase_12.pt"
    if phase12.is_file() and _has_training_config(phase12):
        return phase12
    return None


def _forward_disc_stats(model: torch.nn.Module, prot: dict[str, Any], device: str) -> dict[str, float]:
    from experiments.training.v66.train_loop import prepare_training_batch

    data = prot["data"]
    if data.x.size(-1) > 3:
        if getattr(data, "sasa", None) is None:
            data.sasa = data.x[:, 3].detach().clone()
        data.x = data.x[:, :3].contiguous()
        prot = {**prot, "data": data}
    batch = prepare_training_batch(model, prot, device)
    with torch.no_grad():
        out = model(batch)
    proj = out.get("hyp_projections_2d")
    if proj is None:
        raise RuntimeError("missing hyp_projections_2d in model output")
    r = proj.norm(dim=-1)
    depth = out.get("cone_depth")
    stats = {
        "disc_r_mean": float(r.mean()),
        "disc_r_max": float(r.max()),
        "n": int(r.numel()),
    }
    if depth is not None and torch.is_tensor(depth):
        stats["cone_depth_mean"] = float(depth.float().mean())
        stats["cone_depth_std"] = float(depth.float().std())
        # Spearman-ish: Pearson on ranks omitted; use linear corr |p| vs depth
        r_c = r - r.mean()
        d_c = depth.float() - depth.float().mean()
        denom = float(r_c.norm() * d_c.norm()) + 1e-8
        stats["corr_r_depth"] = float((r_c * d_c).sum() / denom)
    return stats


def grade_pack(
    *,
    sealed: Path,
    champion_dir: Path,
    output_dir: Path,
    device: str,
    pdb_id: str = "4OBE",
) -> dict[str, Any]:
    from experiments.training.v66._data import load_protein_graph_from_pdb_legacy

    output_dir.mkdir(parents=True, exist_ok=True)
    champ_ckpt = _resolve_champion_ckpt(champion_dir)
    pdb_dir = Path("/tmp/dtie_pdb_cache")
    if not pdb_dir.is_dir():
        pdb_dir = Path("pdb_cache")

    prot = load_protein_graph_from_pdb_legacy(pdb_id, "A", pdb_dir)
    if prot is None:
        raise FileNotFoundError(f"could not load {pdb_id} from {pdb_dir}")

    sealed_model = load_model_from_checkpoint(sealed, device)
    sealed_model.eval()
    sealed_stats = _forward_disc_stats(sealed_model, prot, device)

    report: dict[str, Any] = {
        "gate": "FIX1_BIOLOGY_PACK_P4",
        "pdb_id": pdb_id,
        "sealed_checkpoint": str(sealed),
        "champion_dir": str(champion_dir),
        "champion_checkpoint": str(champ_ckpt) if champ_ckpt else None,
        "sealed_stats": sealed_stats,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Post-lift / disc occupancy smoke only — not disc-alone Pass.",
            "RAF1 ≠ PPI typed edges.",
            "Sealed remains immutable baseline.",
        ],
    }

    if champ_ckpt is not None and champ_ckpt.is_file():
        champ_model = load_model_from_checkpoint(champ_ckpt, device)
        champ_model.eval()
        # Reload prot clone for clean forward
        prot2 = load_protein_graph_from_pdb_legacy(pdb_id, "A", pdb_dir)
        champ_stats = _forward_disc_stats(champ_model, prot2, device)
        report["champion_stats"] = champ_stats
        report["delta_disc_r_mean"] = champ_stats["disc_r_mean"] - sealed_stats["disc_r_mean"]
        # Soft hold: champion disc_r within 0.08 of sealed and still mature
        report["passed"] = (
            champ_stats["disc_r_mean"] >= 0.20
            and abs(report["delta_disc_r_mean"]) <= 0.08
        )
    else:
        report["champion_stats"] = None
        report["passed"] = False
        report["error"] = "champion checkpoint missing — run P1–P3 first"

    # Optional: attach resilience stamps if present
    for name in ("resilience_gate.json",):
        p = champion_dir / name
        if p.is_file():
            report["champion_resilience"] = json.loads(p.read_text())

    out = output_dir / "biology_pack_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    # Promote tag file when Pass
    if report.get("passed"):
        promo = output_dir / "promoted_trunk.json"
        promo.write_text(
            json.dumps(
                {
                    "promoted_trunk": True,
                    "champion_checkpoint": report["champion_checkpoint"],
                    "sealed_baseline": str(sealed),
                    "graded_at": report["graded_at"],
                },
                indent=2,
            )
            + "\n"
        )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sealed-checkpoint", type=Path, default=HEALTHY_FIX1_CKPT)
    p.add_argument("--champion-dir", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/v66/diagnostics/fix1_expand_biology"),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pdb-id", default="4OBE")
    args = p.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    report = grade_pack(
        sealed=args.sealed_checkpoint,
        champion_dir=args.champion_dir,
        output_dir=args.output_dir,
        device=args.device,
        pdb_id=args.pdb_id,
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report.get("passed") else 1)


if __name__ == "__main__":
    main()
