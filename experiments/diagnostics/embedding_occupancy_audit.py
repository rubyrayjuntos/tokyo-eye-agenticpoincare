"""Embedding occupancy audit — reproducible disc/ball rank (centered SVD).

Primary occupancy signal for disc_2d: centered σ₂/σ₁ and effective_rank.
Origin-referenced angular span is diagnostic-only — it inflates when a
rank-1 streak passes near the disc origin (origin-straddle artifact).

Usage:
  python -m experiments.diagnostics.embedding_occupancy_audit \\
      --checkpoint checkpoints/v6/tokyo_eyes_v6.pt \\
      --structures 11QE:A,4OBE:A,1IVO:A

  python -m experiments.diagnostics.embedding_occupancy_audit \\
      --compare-checkpoints \\
        checkpoints/v6/runs/shell_p2_hypmix2/v6_best.pt \\
        checkpoints/v6/runs/shell_p2_hypmix_final/v6_best.pt \\
        checkpoints/v6/tokyo_eyes_v6.pt \\
      --structures 11QE:A

  python -m experiments.diagnostics.embedding_occupancy_audit \\
      --checkpoint checkpoints/v6/runs/shell_p2_hypmix2/v6_best.pt \\
      --structures 11QE:A --export-scatter /tmp/11qe_hypmix2_disc.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v6._data import load_protein_graph
from experiments.training.v6.assess_checkpoint import load_v6_model
from experiments.training.v6.train_loop import attach_v6_features
from science.dtie.v5.gnn.runner import V5GNNRunner


def _arch_dict(raw: dict[str, Any]) -> dict[str, Any]:
    arch = raw.get("architecture") or {}
    if isinstance(arch, dict):
        return arch
    if isinstance(arch, str):
        return {"name": arch}
    return {}


def _training_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    tc = raw.get("training_config") or {}
    return tc if isinstance(tc, dict) else {}


def _is_v6_checkpoint(raw: dict[str, Any]) -> bool:
    state = raw.get("model_state_dict", raw)
    if not isinstance(state, dict):
        return False
    keys = list(state.keys())
    return any(k.startswith("gate.") for k in keys) or any(
        k.startswith("experts.") for k in keys
    )


def load_audit_model(
    checkpoint_path: Path,
    device: str = "cpu",
    *,
    legacy_disc_projection: bool | None = None,
) -> tuple[torch.nn.Module, str]:
    """Load v5 or v6 checkpoint for occupancy audit."""
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if _is_v6_checkpoint(raw):
        return (
            load_v6_model(
                checkpoint_path,
                device,
                legacy_disc_projection=legacy_disc_projection,
            ),
            "v6",
        )
    runner = V5GNNRunner(checkpoint_path=str(checkpoint_path), device=device)
    runner._load_model_sync()
    return runner._model, "v5"  # type: ignore[return-value]


def _forward_audit(
    model: torch.nn.Module,
    version: str,
    prot: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    data = prot["data"].to(device)
    if version == "v6":
        data = attach_v6_features(data)
        with torch.no_grad():
            return model(data)
    from science.dtie.v5.gnn.model import precompute_clustering

    data = precompute_clustering(data)
    with torch.no_grad():
        return model(data)

@dataclass
class LayerOccupancy:
    layer: str
    n: int
    sigma: list[float]
    sigma_ratio: list[float]
    effective_rank: float
    line_thickness_rms: float | None
    line_thickness_max: float | None
    n_within_5pct_rmax: int | None
    origin_angular_span_p5_p95_deg: float | None
    centroid_angular_span_p5_p95_deg: float | None
    disc_r_min: float | None
    disc_r_p50: float | None
    disc_r_max: float | None
    xy_x_min: float | None
    xy_x_max: float | None
    xy_y_min: float | None
    xy_y_max: float | None


@dataclass
class PrototypeOccupancy:
    n_proto: int
    dim: int
    sigma: list[float]
    sigma_ratio: list[float]
    effective_rank: float


@dataclass
class StructureAudit:
    structure_id: str
    chain: str
    n_residues: int
    layers: list[LayerOccupancy]


@dataclass
class CheckpointAudit:
    checkpoint: str
    global_epoch: int | None
    phase_name: str | None
    score: float | None
    learned_curvature: float | None
    gate_disc_scale: float | None
    deep_hyperbolic_gate: bool | None
    prototype: PrototypeOccupancy | None
    structures: list[StructureAudit]


def _effective_rank(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or s[0] < 1e-15:
        return 0.0
    return float(np.sum(s * s) / (s[0] * s[0]))


def _angular_deg(xy: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))


def _line_thickness(xy: np.ndarray) -> tuple[float, float]:
    """RMS and max perpendicular distance to the PC1 line (centered SVD)."""
    X = xy - xy.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]
    proj = (X @ pc1)[:, None] * pc1[None, :]
    resid = X - proj
    thickness = np.linalg.norm(resid, axis=1)
    return float(np.sqrt((thickness**2).mean())), float(thickness.max())


def _layer_occupancy(layer: str, pts: np.ndarray) -> LayerOccupancy:
    X = pts - pts.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    sn = (s / (s[0] + 1e-12)).tolist()
    origin_span = centroid_span = None
    thickness_rms = thickness_max = None
    n_near_origin = None
    r_min = r_p50 = r_max = None
    x_min = x_max = y_min = y_max = None
    if pts.shape[1] == 2 and len(pts):
        ang_origin = _angular_deg(pts)
        origin_span = float(np.percentile(ang_origin, 95) - np.percentile(ang_origin, 5))
        ang_centroid = _angular_deg(X)
        centroid_span = float(np.percentile(ang_centroid, 95) - np.percentile(ang_centroid, 5))
        thickness_rms, thickness_max = _line_thickness(pts)
        r = np.linalg.norm(pts, axis=1)
        rmax = float(r.max()) if len(r) else 1.0
        n_near_origin = int(np.sum(r < 0.05 * rmax))
        r_min, r_p50, r_max = float(r.min()), float(np.median(r)), float(r.max())
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    return LayerOccupancy(
        layer=layer,
        n=int(len(pts)),
        sigma=[float(x) for x in s[:4]],
        sigma_ratio=[float(x) for x in sn[:4]],
        effective_rank=_effective_rank(s),
        line_thickness_rms=thickness_rms,
        line_thickness_max=thickness_max,
        n_within_5pct_rmax=n_near_origin,
        origin_angular_span_p5_p95_deg=origin_span,
        centroid_angular_span_p5_p95_deg=centroid_span,
        disc_r_min=r_min,
        disc_r_p50=r_p50,
        disc_r_max=r_max,
        xy_x_min=x_min,
        xy_x_max=x_max,
        xy_y_min=y_min,
        xy_y_max=y_max,
    )


def _prototype_occupancy(model: torch.nn.Module) -> PrototypeOccupancy | None:
    gate = getattr(model, "gate", None)
    if gate is None or not hasattr(gate, "prototype_bank"):
        return None
    with torch.no_grad():
        proto = gate.prototype_bank(model.curvature).detach().cpu().numpy()
    X = proto - proto.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    sn = (s / (s[0] + 1e-12)).tolist()
    return PrototypeOccupancy(
        n_proto=int(proto.shape[0]),
        dim=int(proto.shape[1]),
        sigma=[float(x) for x in s[:4]],
        sigma_ratio=[float(x) for x in sn[:4]],
        effective_rank=_effective_rank(s),
    )


def _parse_structures(raw: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in raw:
        if ":" in item:
            pdb, chain = item.split(":", 1)
        else:
            pdb, chain = item, "A"
        out.append((pdb.upper(), chain))
    return out


def audit_checkpoint(
    checkpoint_path: Path,
    structures: list[tuple[str, str]],
    pdb_dir: Path,
    device: str = "cpu",
    *,
    legacy_disc_projection: bool | None = None,
) -> CheckpointAudit:
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tc = _training_config_dict(raw)
    arch = _arch_dict(raw)
    model, version = load_audit_model(
        checkpoint_path,
        device,
        legacy_disc_projection=legacy_disc_projection,
    )
    learned_c = float(getattr(model, "curvature").detach().cpu().item())
    proto = _prototype_occupancy(model) if version == "v6" else None

    structure_audits: list[StructureAudit] = []
    for pdb_id, chain in structures:
        prot = load_protein_graph(pdb_id, chain, pdb_dir)
        if prot is None:
            raise RuntimeError(f"Failed to load {pdb_id}:{chain}")
        out = _forward_audit(model, version, prot, device)
        layers = [
            _layer_occupancy("disc_2d", out["hyp_projections_2d"].detach().cpu().numpy()),
            _layer_occupancy("ball_3d", out["hyp_projections_3d"].detach().cpu().numpy()),
            _layer_occupancy("x_routed_hyp", out["x_routed_hyp"].detach().cpu().numpy()),
        ]
        structure_audits.append(
            StructureAudit(
                structure_id=pdb_id.lower(),
                chain=chain,
                n_residues=int(prot["n_residues"]),
                layers=layers,
            )
        )

    gate_scale = arch.get("gate_disc_scale", tc.get("gate_disc_scale"))
    return CheckpointAudit(
        checkpoint=str(checkpoint_path.resolve()),
        global_epoch=raw.get("global_epoch"),
        phase_name=raw.get("phase_name"),
        score=float(raw["score"]) if raw.get("score") is not None else None,
        learned_curvature=learned_c,
        gate_disc_scale=float(gate_scale) if gate_scale is not None else None,
        deep_hyperbolic_gate=bool(arch.get("deep_hyperbolic_gate", tc.get("deep_hyperbolic_gate", False))),
        prototype=proto,
        structures=structure_audits,
    )


def export_disc_scatter_from_model(
    model: torch.nn.Module,
    structure: tuple[str, str],
    pdb_dir: Path,
    output: Path,
    device: str = "cpu",
    *,
    title_suffix: str = "",
    legacy_disc_projection: bool | None = None,
) -> None:
    """Render disc_2d scatter from a loaded model (training-time visual check)."""
    import matplotlib.pyplot as plt

    _ = legacy_disc_projection
    pdb_id, chain = structure
    prot = load_protein_graph(pdb_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"Failed to load {pdb_id}:{chain}")
    xy = _forward_audit(model, "v6", prot, device)["hyp_projections_2d"].cpu().numpy()
    c = float(model.curvature.detach().cpu())
    r_ball = 1.0 / math.sqrt(c) if c > 0 else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), r_ball, fill=False, color="#666", linestyle="--", linewidth=0.8)
    ax.add_patch(circle)
    ax.scatter(xy[:, 0], xy[:, 1], s=12, c="#eca53a", alpha=0.75, edgecolors="none")
    ax.set_aspect("equal")
    title = f"{pdb_id}:{chain} disc_2d"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_disc_scatter(
    checkpoint_path: Path,
    structure: tuple[str, str],
    pdb_dir: Path,
    output: Path,
    device: str = "cpu",
    *,
    legacy_disc_projection: bool | None = None,
) -> None:
    model, _version = load_audit_model(
        checkpoint_path,
        device,
        legacy_disc_projection=legacy_disc_projection,
    )
    pdb_id, chain = structure
    title = f"{checkpoint_path.parent.name}"
    export_disc_scatter_from_model(
        model,
        (pdb_id, chain),
        pdb_dir,
        output,
        device=device,
        title_suffix=title,
    )
    print(f"Wrote scatter → {output}")


def _print_checkpoint(audit: CheckpointAudit) -> None:
    print("=" * 78)
    print(f"CHECKPOINT: {audit.checkpoint}")
    print(
        f"  global_epoch={audit.global_epoch}  phase={audit.phase_name!r}  "
        f"score={audit.score}  curvature={audit.learned_curvature}"
    )
    print(
        f"  gate_disc_scale={audit.gate_disc_scale}  "
        f"deep_hyperbolic_gate={audit.deep_hyperbolic_gate}"
    )
    if audit.prototype:
        p = audit.prototype
        print(
            f"  PROTOTYPE_BANK: n={p.n_proto} dim={p.dim} eff_rank={p.effective_rank:.4f} "
            f"sigma_ratio={p.sigma_ratio[:3]}"
        )
    for st in audit.structures:
        print("-" * 78)
        print(f"STRUCTURE {st.structure_id}:{st.chain}  n_residues={st.n_residues}")
        for layer in st.layers:
            print(f"  [{layer.layer}]")
            print(f"    n={layer.n}  eff_rank={layer.effective_rank:.4f}")
            print(f"    sigma={layer.sigma[:3]}")
            print(f"    sigma_ratio={layer.sigma_ratio[:3]}")
            if layer.origin_angular_span_p5_p95_deg is not None:
                print(
                    f"  PRIMARY (centered SVD): sigma2/sigma1={layer.sigma_ratio[1]:.4f}  "
                    f"line_thickness_rms={layer.line_thickness_rms:.5f}"
                )
                print(
                    f"    |disc| min/p50/max={layer.disc_r_min:.4f}/{layer.disc_r_p50:.4f}/"
                    f"{layer.disc_r_max:.4f}  n_within_5pct_rmax={layer.n_within_5pct_rmax}"
                )
                print(
                    f"    x=[{layer.xy_x_min:.4f}, {layer.xy_x_max:.4f}]  "
                    f"y=[{layer.xy_y_min:.4f}, {layer.xy_y_max:.4f}]"
                )
                print(
                    f"  DIAGNOSTIC (angular — misleading for rank-1 streaks): "
                    f"origin_span={layer.origin_angular_span_p5_p95_deg:.1f}°  "
                    f"centroid_span={layer.centroid_angular_span_p5_p95_deg:.1f}°  "
                    f"(origin-span inflates when streak crosses near origin)"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding occupancy audit (full tables)")
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint .pt path")
    parser.add_argument(
        "--compare-checkpoints",
        nargs="+",
        default=None,
        help="Compare multiple checkpoints on the same structures",
    )
    parser.add_argument(
        "--structures",
        default="11QE:A,4OBE:A,1IVO:A,4MNE:A",
        help="Comma-separated PDB:chain list",
    )
    parser.add_argument("--pdb-dir", default="/tmp/dtie_pdb_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None, help="Optional JSON dump of full audit")
    parser.add_argument(
        "--export-scatter",
        default=None,
        help="Write disc_2d scatter PNG (requires --checkpoint and single structure)",
    )
    parser.add_argument(
        "--legacy-disc-projection",
        action="store_true",
        help="Force legacy post-routing disc projection at inference",
    )
    parser.add_argument(
        "--no-legacy-disc-projection",
        action="store_true",
        help="Force new pre-routing disc projection at inference",
    )
    args = parser.parse_args()

    legacy_override: bool | None = None
    if args.legacy_disc_projection:
        legacy_override = True
    if args.no_legacy_disc_projection:
        legacy_override = False

    structures = _parse_structures(args.structures.split(","))
    pdb_dir = Path(args.pdb_dir)

    checkpoints: list[Path] = []
    if args.compare_checkpoints:
        checkpoints = [Path(p) for p in args.compare_checkpoints]
    elif args.checkpoint:
        checkpoints = [Path(args.checkpoint)]
    else:
        parser.error("Provide --checkpoint or --compare-checkpoints")

    audits: list[CheckpointAudit] = []
    for ckpt in checkpoints:
        if not ckpt.is_file():
            raise SystemExit(f"Missing checkpoint: {ckpt}")
        audit = audit_checkpoint(
            ckpt,
            structures,
            pdb_dir,
            device=args.device,
            legacy_disc_projection=legacy_override,
        )
        audits.append(audit)
        _print_checkpoint(audit)

    if args.export_scatter:
        if len(checkpoints) != 1 or len(structures) != 1:
            parser.error("--export-scatter requires one --checkpoint and one structure")
        export_disc_scatter(
            checkpoints[0],
            structures[0],
            pdb_dir,
            Path(args.export_scatter),
            device=args.device,
            legacy_disc_projection=legacy_override,
        )

    if args.json_out:
        payload = [asdict(a) for a in audits]
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON → {args.json_out}")

    if len(audits) > 1:
        print("=" * 78)
        print("DISC_2D sigma2/sigma1 COMPARISON (primary occupancy — centered SVD)")
        header = f"{'structure':<12}" + "".join(f"{Path(a.checkpoint).parent.name:>22}" for a in audits)
        print(header)
        for st_idx, (pdb, chain) in enumerate(structures):
            sid = f"{pdb.lower()}:{chain}"
            row = f"{sid:<12}"
            for audit in audits:
                layer = audit.structures[st_idx].layers[0]
                ratio = layer.sigma_ratio[1] if len(layer.sigma_ratio) > 1 else 0.0
                row += f"{ratio:>22.4f}"
            print(row)
        print()
        print("origin_angular_span (DIAGNOSTIC ONLY — origin-straddle artifact):")
        print(header)
        for st_idx, (pdb, chain) in enumerate(structures):
            sid = f"{pdb.lower()}:{chain}"
            row = f"{sid:<12}"
            for audit in audits:
                layer = audit.structures[st_idx].layers[0]
                span = layer.origin_angular_span_p5_p95_deg
                row += f"{span:>22.1f}" if span is not None else f"{'n/a':>22}"
            print(row)


if __name__ == "__main__":
    main()
