"""TokyoEye-v8 Poincaré disc viewer export (Sprint 6 Milestone 2).

Isolated from v66 viewer writers — self-contained HTML under
``data/local_objects/gnn_viewer/v8/<pdb_id>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from science.tokyo_eye.v8.equiformer_frontend import (
    DEFAULT_WEIGHT_MAP,
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    load_weight_map,
)
from science.tokyo_eye.v8.loader import DEFAULT_CHAIN, DEFAULT_PDB_DIR, load_structure_batch
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8

DEFAULT_OUT_ROOT = Path("data/local_objects/gnn_viewer/v8")


def _as_numpy(z: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(z):
        return z.detach().cpu().numpy().astype(np.float64)
    return np.asarray(z, dtype=np.float64)


def n_unique_rows(z: torch.Tensor | np.ndarray, *, decimals: int = 4) -> int:
    z_np = _as_numpy(z)
    return len({tuple(np.round(row, decimals)) for row in z_np})


def _pca_explained(z: torch.Tensor | np.ndarray) -> tuple[float, float]:
    """Return (pc1_frac, pc2_frac) of centered variance."""
    z_np = _as_numpy(z)
    centered = z_np - z_np.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    ev = s * s
    total = float(ev.sum()) + 1e-12
    pc1 = float(ev[0] / total) if len(ev) else 1.0
    pc2 = float(ev[1] / total) if len(ev) > 1 else 0.0
    return pc1, pc2


def select_layout_source(
    *,
    z_hyp: torch.Tensor,
    z_attn: torch.Tensor | None = None,
    z_lift: torch.Tensor | None = None,
    h_euc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Pick disc layout by 2D content, preferring hyperbolic tensors.

    A 1D curve can still have N unique XY points (flat band in HTML). Prefer
    ``z_hyp`` / ``z_lift`` when PC2 carries real mass; fall back to ``h_euc``
    only if hyp boards are quasi-1D.
    """
    hyp: list[tuple[str, torch.Tensor]] = [("z_hyp", z_hyp)]
    if z_lift is not None:
        hyp.append(("z_lift", z_lift))
    if z_attn is not None:
        hyp.append(("z_attn", z_attn))

    def _score(tensor: torch.Tensor) -> tuple[float, float, float]:
        pc1, pc2 = _pca_explained(tensor)
        xy = ball_to_disc(tensor)
        u = len({(round(float(a), 3), round(float(b), 3)) for a, b in xy})
        score = float(u) * (pc2 + 0.02) * (1.0 - min(pc1, 0.99))
        return score, pc1, pc2

    # Prefer a hyperbolic board with meaningful PC2 (>=10% variance).
    hyp_ok: list[tuple[float, str, torch.Tensor, float, float]] = []
    for name, tensor in hyp:
        score, pc1, pc2 = _score(tensor)
        if pc2 >= 0.10:
            hyp_ok.append((score, name, tensor, pc1, pc2))
    if hyp_ok:
        hyp_ok.sort(key=lambda t: t[0], reverse=True)
        _, name, tensor, pc1, pc2 = hyp_ok[0]
        return tensor, f"{name}(pc1={pc1:.2f},pc2={pc2:.2f})"

    # Otherwise pick best among all candidates (may use h_euc as last resort).
    candidates = list(hyp)
    if h_euc is not None:
        candidates.append(("h_euc", h_euc))
    best_name, best_z, best_score, best_pc = "z_hyp", z_hyp, -1.0, (1.0, 0.0)
    for name, tensor in candidates:
        score, pc1, pc2 = _score(tensor)
        if score > best_score:
            best_name, best_z, best_score, best_pc = name, tensor, score, (pc1, pc2)
    return best_z, f"{best_name}(pc1={best_pc[0]:.2f},pc2={best_pc[1]:.2f})"


def ball_to_disc(z: torch.Tensor | np.ndarray) -> np.ndarray:
    """Map high-dim states to a 2D Poincaré disc for viewing.

    PCA on centered vectors, **axis-whiten** so a dominant PC1 does not paint
    a flat horizontal band, then scale into the open disc and stereograph.
    """
    z_np = _as_numpy(z)
    if z_np.ndim != 2 or z_np.shape[1] < 2:
        raise ValueError(f"z must be [N, D>=2], got {z_np.shape}")
    centered = z_np - z_np.mean(axis=0, keepdims=True)
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        plane = centered @ vt[:2].T
        # Whiten PC axes (avoid ribbon collapse in the HTML canvas)
        scale = np.maximum(s[:2], 1e-6)
        plane = plane / scale
    except np.linalg.LinAlgError:
        plane = centered[:, :2]
    pr = np.linalg.norm(plane, axis=1)
    max_r = float(pr.max()) if pr.size else 1.0
    if max_r < 1e-8:
        n = z_np.shape[0]
        ang = 2.0 * np.pi * np.arange(n) / max(n, 1)
        plane = np.stack([0.35 * np.cos(ang), 0.35 * np.sin(ang)], axis=1)
    else:
        plane = plane * (0.85 / max_r)
    norm_sq = np.sum(plane * plane, axis=1, keepdims=True)
    xy = (2.0 * plane) / (1.0 + norm_sq)
    r = np.linalg.norm(xy, axis=1, keepdims=True)
    xy = xy * np.minimum(1.0, 0.999 / np.maximum(r, 1e-12))
    return xy.astype(np.float32)


def _write_disc_html(
    *,
    structure_id: str,
    points: list[dict[str, Any]],
    output_path: Path,
    checkpoint_path: str | None,
    curvature: float,
) -> None:
    sid = structure_id.strip().lower()
    title = f"{sid.upper()} — TokyoEye-v8 Poincaré Disc"
    ckpt_note = (
        f" | ckpt: {Path(checkpoint_path).name}" if checkpoint_path else ""
    )
    points_json = json.dumps(points)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ margin:0; background:#111; color:#eee; font-family:Georgia,serif; }}
#toolbar {{ padding:10px 14px; background:#1a1a1a; font-size:12px; display:flex; gap:14px; flex-wrap:wrap; align-items:center; }}
#canvas {{ display:block; width:100vw; height:calc(100vh - 56px); }}
#tip {{ position:fixed; pointer-events:none; background:rgba(0,0,0,.9); border:1px solid #444; padding:6px 8px; font-size:11px; display:none; }}
</style></head><body>
<div id="toolbar">
  <b>{sid.upper()}</b> | tokyo-eye-v8{ckpt_note} | κ={curvature:.3f}
  <span id="layoutTag"></span>
  <label>Color
    <select id="metric">
      <option value="radius">Physical |z| radius</option>
      <option value="dehydron" selected>Dehydron flag</option>
      <option value="risk">Epistemic risk β/α</option>
      <option value="mech">Mechanism score σ</option>
      <option value="expert">MoE expert</option>
    </select>
  </label>
  <span id="stats"></span>
</div>
<canvas id="canvas"></canvas><div id="tip"></div>
<script>
const POINTS = {points_json};
const LAYOUT = POINTS.length ? (POINTS[0].layout || 'z_hyp') : 'z_hyp';
document.getElementById('layoutTag').textContent = ' | layout=' + LAYOUT;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tip = document.getElementById('tip');
const metricSel = document.getElementById('metric');
let hover = -1;
function resize() {{ canvas.width = innerWidth; canvas.height = innerHeight-56; draw(); }}
function color(t) {{
  const r = Math.round(40 + 200*t), g = Math.round(80 + 40*(1-t)), b = Math.round(200 - 140*t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function val(p,k) {{
  if (k==='dehydron') return p.dehydron;
  if (k==='risk') return p.risk;
  if (k==='mech') return p.mech;
  if (k==='expert') return (p.expert || 0) / 3.0;
  return p.radius;
}}
function draw() {{
  const w=canvas.width, h=canvas.height, cx=w/2, cy=h/2, R=Math.min(w,h)*0.42;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='#333'; ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2); ctx.stroke();
  const key=metricSel.value;
  let mn=1e9, mx=-1e9;
  for (const p of POINTS) {{ const v=val(p,key); mn=Math.min(mn,v); mx=Math.max(mx,v); }}
  const span=Math.max(1e-6, mx-mn);
  POINTS.forEach((p,i) => {{
    const t=(val(p,key)-mn)/span;
    const x=cx + p.x*R, y=cy - p.y*R;
    ctx.fillStyle=color(t);
    ctx.beginPath(); ctx.arc(x,y, i===hover?5:3, 0, Math.PI*2); ctx.fill();
  }});
  document.getElementById('stats').textContent = `N=${{POINTS.length}} metric=${{key}} uniqueXY=${{new Set(POINTS.map(p=>p.x.toFixed(3)+','+p.y.toFixed(3))).size}}`;
}}
metricSel.onchange=draw;
canvas.onmousemove = (e) => {{
  const rect=canvas.getBoundingClientRect();
  const mx=e.clientX-rect.left, my=e.clientY-rect.top;
  const w=canvas.width, h=canvas.height, cx=w/2, cy=h/2, R=Math.min(w,h)*0.42;
  let best=-1, bd=9;
  POINTS.forEach((p,i) => {{
    const x=cx+p.x*R, y=cy-p.y*R;
    const d=Math.hypot(x-mx,y-my);
    if (d<bd) {{ bd=d; best=i; }}
  }});
  hover=best;
  if (best>=0) {{
    const p=POINTS[best];
    tip.style.display='block'; tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';
    tip.textContent = `#${{p.i}} dehydron=${{p.dehydron}} r=${{p.radius.toFixed(3)}} risk=${{p.risk.toFixed(3)}} expert=${{p.expert}}`;
  }} else tip.style.display='none';
  draw();
}};
addEventListener('resize', resize); resize();
</script></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def _eqf_split_nodes(
    *,
    z_hyp: torch.Tensor,
    xy: np.ndarray,
    dehydron_labels: torch.Tensor,
    evidence: torch.Tensor,
    residue_records: list[Any],
    expert_id: torch.Tensor | None,
    curvature: float,
) -> list[Any]:
    """Bridge EQF tensors onto the existing 3D+disc split writer.

    Disc XY is ``ball_to_disc(z_hyp)``. Occupancy/cone_depth is hyperbolic
    radius from the origin (not a cone_depth lift of the July store).
    ρ/τ channels are the Mode-A dehydron labels so physics coloring matches
    the train target, not evidential investigation.
    """
    from science.dtie.common.interfaces import GNNNodeOutput
    from science.tokyo_eye.v8.attention import poincare_dist

    n = int(z_hyp.shape[0])
    if len(residue_records) != n:
        raise ValueError(
            f"residue record count {len(residue_records)} != z_hyp N={n}"
        )
    origin = torch.zeros_like(z_hyp)
    hyp_r = poincare_dist(z_hyp, origin, c=curvature).detach()
    depth_max = hyp_r.max().clamp_min(1e-8)
    cone_depth = (hyp_r / depth_max) * 8.0
    cone_width = torch.exp(-cone_depth)
    nu = evidence[:, 1].clamp_min(1e-4)
    alpha = evidence[:, 2].clamp_min(1.0001)
    beta = evidence[:, 3].clamp_min(1e-4)
    aleatoric = beta / (alpha - 1.0)
    epistemic = beta / (nu * (alpha - 1.0))
    dehyd = dehydron_labels.detach().cpu().numpy().astype(np.float32)
    experts = (
        expert_id.detach().cpu().numpy().astype(int)
        if expert_id is not None
        else np.zeros(n, dtype=int)
    )
    z_np = _as_numpy(z_hyp)
    nodes = []
    for i, rec in enumerate(residue_records):
        rho = float(dehyd[i])
        onehot = np.zeros(4, dtype=np.float32)
        onehot[int(experts[i]) % 4] = 1.0
        nodes.append(
            GNNNodeOutput(
                residue_index=int(rec.residue_index),
                chain_label=str(rec.chain_label),
                input_features=np.asarray([rho, rho, 0.0, 0.0], dtype=np.float32),
                projections=z_np[i].astype(np.float32),
                cone_depth=float(cone_depth[i]),
                cone_width=float(cone_width[i]),
                epistemic_uncertainty=float(epistemic[i]),
                aleatoric_uncertainty=float(aleatoric[i]),
                total_uncertainty=float(aleatoric[i] + epistemic[i]),
                x_hyp=z_np[i],
                hyp_projections=np.asarray([xy[i, 0], xy[i, 1]], dtype=np.float64),
                expert_weights=onehot,
            )
        )
    return nodes


def export_v8_structure_viewers(
    *,
    pdb_id: str,
    z_hyp: torch.Tensor,
    dehydron_labels: torch.Tensor,
    mechanism_score: torch.Tensor,
    evidence: torch.Tensor,
    checkpoint_path: str | None = None,
    curvature: float = 1.0,
    out_root: Path | str = DEFAULT_OUT_ROOT,
    z_attn: torch.Tensor | None = None,
    z_lift: torch.Tensor | None = None,
    h_euc: torch.Tensor | None = None,
    expert_id: torch.Tensor | None = None,
    pdb_text: str | None = None,
    residue_records: list[Any] | None = None,
) -> dict[str, str]:
    """Write disc HTML + JSON sidecar, and the 3D+disc split when PDB is given."""
    pdb_id = pdb_id.strip().upper()
    out_dir = Path(out_root) / pdb_id.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_z, layout_name = select_layout_source(
        z_hyp=z_hyp, z_attn=z_attn, z_lift=z_lift, h_euc=h_euc
    )
    xy = ball_to_disc(layout_z)
    radius = torch.linalg.vector_norm(z_hyp, dim=-1).cpu().numpy()
    risk = (evidence[:, 3] / evidence[:, 2].clamp_min(1e-4)).cpu().numpy()
    mech = torch.sigmoid(mechanism_score).cpu().numpy()
    dehyd = dehydron_labels.cpu().numpy().astype(float)
    if expert_id is None:
        experts = np.zeros(xy.shape[0], dtype=int)
    else:
        experts = expert_id.detach().cpu().numpy().astype(int)
    points = []
    for i in range(xy.shape[0]):
        points.append(
            {
                "i": int(i),
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "radius": float(radius[i]),
                "dehydron": float(dehyd[i]),
                "risk": float(risk[i]),
                "mech": float(mech[i]),
                "expert": int(experts[i]),
                "layout": layout_name,
            }
        )
    disc_path = out_dir / f"{pdb_id.lower()}_poincare_disc.html"
    _write_disc_html(
        structure_id=pdb_id,
        points=points,
        output_path=disc_path,
        checkpoint_path=checkpoint_path,
        curvature=curvature,
    )
    manifest = {
        "structure_id": pdb_id,
        "n_nodes": len(points),
        "dehydron_frac": float(dehyd.mean()) if len(dehyd) else 0.0,
        "checkpoint": checkpoint_path,
        "disc_html": str(disc_path),
        "curvature": curvature,
        "layout": layout_name,
        "unique_layout_rows": n_unique_rows(layout_z),
        "unique_z_hyp_rows": n_unique_rows(z_hyp),
    }
    man_path = out_dir / "viewer_manifest.json"
    paths = {"disc_html": str(disc_path), "manifest": str(man_path)}
    if pdb_text and residue_records:
        from science.dtie.v6.visualization.interactive_viewer import (
            write_structure_viewers,
        )

        split_nodes = _eqf_split_nodes(
            z_hyp=z_hyp,
            xy=xy,
            dehydron_labels=dehydron_labels,
            evidence=evidence,
            residue_records=residue_records,
            expert_id=expert_id,
            curvature=curvature,
        )
        split_paths = write_structure_viewers(
            structure_id=pdb_id,
            pdb_text=pdb_text,
            nodes=split_nodes,
            model_version="TokyoEye-EQF",
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            curvature=curvature,
            disc_layout_label=f"z_hyp→disc {layout_name} (not cone_depth lift)",
        )
        paths.update(split_paths)
        manifest["split_html"] = split_paths.get("split_html")
        manifest["structure_html"] = split_paths.get("structure_html")
    man_path.write_text(json.dumps(manifest, indent=2))
    return paths


def main() -> None:
    from science.tokyo_eye.v8.biophysics import parse_residue_records_from_pdb_chain
    from science.tokyo_eye.v8.loader import ensure_pdb_cached, parse_enabled_manifest
    from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

    p = argparse.ArgumentParser(description="Export TokyoEye-v8 Poincaré viewers")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pdb", type=str, default="4OBE")
    p.add_argument("--chain", type=str, default=DEFAULT_CHAIN)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Mode B/C proteins[] manifest (exports all enabled entries)",
    )
    p.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument(
        "--dehydron-wrap-max",
        type=int,
        default=None,
        help="Override DEHYDRON_WRAP_MAX (Mode C retune used τ=1)",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Stub trunks only (default: live SE(3)-lite matching train)",
    )
    args = p.parse_args()

    if args.dehydron_wrap_max is not None:
        set_dehydron_wrap_max(int(args.dehydron_wrap_max))

    if args.manifest is not None and Path(args.manifest).is_file():
        entries = parse_enabled_manifest(args.manifest)
    else:
        entries = [{"pdb_id": args.pdb.strip().upper(), "chain": args.chain.strip()}]

    cfg = load_weight_map(args.weight_map)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[v8-export] WARN: CUDA unavailable — falling back to cpu")
        args.device = "cpu"
    device = torch.device(args.device)
    live = not bool(args.freeze_backbone)
    frontend = StubEquiformerFrontend(
        in_dim=3,
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        num_backbone_blocks=int(cfg.get("num_backbone_blocks", 7)),
        live_backbone=live,
    )
    spine = TokyoEyesHyperbolicV8(
        scalar_dim=int(cfg["scalar_dim"]),
        vector_dim=int(cfg["vector_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_attn_layers=2,
        num_sdrp_classes=5,
        c=float(cfg.get("curvature_c", 1.0)),
    )
    system = TokyoEyeV8WithFrontend(frontend, spine).to(device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    system.load_state_dict(state, strict=False)
    system.eval()
    tau_ceil = float(cfg.get("tau_end", 0.995))
    curvature = float(cfg.get("curvature_c", 1.0))

    exported: list[dict[str, Any]] = []
    for entry in entries:
        pdb_id = entry["pdb_id"]
        chain = entry["chain"]
        batch = load_structure_batch(
            pdb_id,
            chain,
            pdb_dir=args.pdb_dir,
            device=device,
            graph_cache_dir=Path(args.pdb_dir) / "v8_graph_cache",
        )
        with torch.no_grad():
            out = system(
                batch["x"],
                batch["edge_index"],
                batch["edge_type"],
                tau_ceiling=tau_ceil,
            )
        pdb_path = ensure_pdb_cached(pdb_id, args.pdb_dir)
        pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
        records = [
            rec
            for rec in parse_residue_records_from_pdb_chain(pdb_path, chain)
            if rec.get_atom("CA") is not None
        ]
        paths = export_v8_structure_viewers(
            pdb_id=pdb_id,
            z_hyp=out["z_hyp"],
            dehydron_labels=batch["dehydron_labels"],
            mechanism_score=out["mechanism_score"],
            evidence=out["evidence"],
            checkpoint_path=str(args.ckpt),
            curvature=curvature,
            out_root=args.out_root,
            z_attn=out.get("z_attn"),
            z_lift=out.get("z_lift"),
            h_euc=out.get("h_euc"),
            expert_id=out["moe_aux"]["routing"].argmax(dim=-1),
            pdb_text=pdb_text,
            residue_records=records,
        )
        print(
            f"[v8-export] {pdb_id}:{chain} N={batch['num_nodes']} "
            f"dehydron_frac={float(batch['dehydron_frac']):.3f} → {paths['disc_html']}"
        )
        exported.append({"pdb_id": pdb_id, "chain": chain, **paths})

    print(
        json.dumps(
            {
                "ok": True,
                "n_structures": len(exported),
                "backbone_mode": "live_se3_lite" if live else "frozen_stub",
                "dehydron_wrap_max": args.dehydron_wrap_max,
                "exports": exported,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
