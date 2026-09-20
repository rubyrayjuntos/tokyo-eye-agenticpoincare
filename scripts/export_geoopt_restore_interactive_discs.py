"""Interactive Poincaré discs for tokyo_eye_equ_geoopt_restore (Equiformer-pool spine).

Uses the sealed best ckpt + EquiformerPoolFrontend (SE(3)-lite forbidden).
Writes standalone HTML under data/local_objects/gnn_viewer/equ_geoopt_restore/.
Also emits the existing disc+split artifacts via export_v8_structure_viewers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v8.equ_geoopt_restore import (
    DEFAULT_FRONTEND_CKPT,
    DEFAULT_MANIFEST,
    PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE,
    RV_TAU_END,
    assert_frontend_bank,
    load_boot_split,
)
from experiments.training.v8.export_viewers import (
    ball_to_disc,
    export_v8_structure_viewers,
)
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import (
    _load_train_batch,
    _resolve_entry,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.biophysics import parse_residue_records_from_pdb_chain
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.loader import ensure_pdb_cached
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

BEST = Path("checkpoints/tokyoeye/runs/eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt")
OUT_ROOT = Path("data/local_objects/gnn_viewer/equ_geoopt_restore")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")


def _write_enhanced_disc_html(
    *,
    structure_id: str,
    chain: str,
    role: str,
    points: list[dict[str, Any]],
    output_path: Path,
    checkpoint_path: str,
    curvature: float,
    mean_ball_r: float,
) -> None:
    sid = structure_id.strip().upper()
    title = f"{sid}:{chain} — geoopt_restore Poincaré disc"
    points_json = json.dumps(points)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ margin:0; background:#111; color:#eee; font-family:Georgia,serif; }}
#toolbar {{ padding:10px 14px; background:#1a1a1a; font-size:12px; display:flex; gap:14px; flex-wrap:wrap; align-items:center; }}
#canvas {{ display:block; width:100vw; height:calc(100vh - 56px); }}
#tip {{ position:fixed; pointer-events:none; background:rgba(0,0,0,.92); border:1px solid #555; padding:8px 10px; font-size:12px; display:none; max-width:280px; line-height:1.35; }}
code {{ color:#9cf; }}
</style></head><body>
<div id="toolbar">
  <b>{sid}:{chain}</b> <span style="opacity:.7">[{role}]</span>
  | geoopt_restore | κ={curvature:.3f} | mean ‖z‖₂={mean_ball_r:.3f}
  | ckpt: {Path(checkpoint_path).name}
  <label>Color
    <select id="metric">
      <option value="dehydron" selected>Dehydron</option>
      <option value="radius">‖z‖₂ ball radius</option>
      <option value="mech">Mechanism σ</option>
      <option value="risk">Epistemic risk β/α</option>
      <option value="expert">MoE expert E0–E3</option>
    </select>
  </label>
  <span id="stats"></span>
  <span style="opacity:.55">unit disc = Poincaré boundary; sat gate is on ‖z‖₂ (&lt;0.50), not disc radius</span>
</div>
<canvas id="canvas"></canvas><div id="tip"></div>
<script>
const POINTS = {points_json};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tip = document.getElementById('tip');
const metricSel = document.getElementById('metric');
const stats = document.getElementById('stats');

function metricVal(p, k) {{
  if (k==='dehydron') return p.dehydron;
  if (k==='radius') return p.radius;
  if (k==='mech') return p.mech;
  if (k==='risk') return p.risk;
  if (k==='expert') return (p.expert || 0) / 3.0;
  return 0;
}}
function colorize(t) {{
  // RdYlBu-ish
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, 2*t));
  const b = Math.round(255 * Math.min(1, 2*(1-t)));
  const g = Math.round(255 * (1 - Math.abs(t-0.5)*2));
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function expertColor(e) {{
  const pal = ['#4e79a7','#f28e2b','#e15759','#76b7b2'];
  return pal[(e|0) % 4];
}}

let W=0, H=0, scale=1, cx=0, cy=0;
function resize() {{
  W = canvas.width = window.innerWidth;
  H = canvas.height = window.innerHeight - 56;
  scale = 0.42 * Math.min(W, H);
  cx = W/2; cy = H/2 + 8;
  draw();
}}
function toScreen(x,y) {{ return [cx + x*scale, cy - y*scale]; }}
function drawCircle(r, stroke, dash) {{
  ctx.beginPath();
  ctx.setLineDash(dash || []);
  ctx.arc(cx, cy, r*scale, 0, Math.PI*2);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
}}
function draw() {{
  ctx.fillStyle = '#111';
  ctx.fillRect(0,0,W,H);
  drawCircle(1.0, '#888');
  drawCircle(0.5, '#444', [4,4]);
  const k = metricSel.value;
  let vals = POINTS.map(p => metricVal(p,k));
  let vmin = Math.min(...vals), vmax = Math.max(...vals);
  if (!(vmax > vmin)) {{ vmin = 0; vmax = 1; }}
  if (k==='dehydron' || k==='mech') {{ vmin=0; vmax=1; }}
  if (k==='radius') {{ vmin=0; vmax=Math.max(0.5, vmax); }}
  if (k==='expert') {{ vmin=0; vmax=1; }}
  for (const p of POINTS) {{
    const [sx,sy] = toScreen(p.x, p.y);
    let t = (metricVal(p,k) - vmin) / (vmax - vmin + 1e-12);
    ctx.beginPath();
    ctx.arc(sx, sy, 3.2, 0, Math.PI*2);
    ctx.fillStyle = (k==='expert') ? expertColor(p.expert) : colorize(t);
    ctx.fill();
  }}
  const meanR = POINTS.reduce((a,p)=>a+p.radius,0)/POINTS.length;
  stats.textContent = `N=${{POINTS.length}} | mean ‖z‖₂=${{meanR.toFixed(3)}} | color=${{k}}`;
}}
metricSel.addEventListener('change', draw);
window.addEventListener('resize', resize);

function hit(mx, my) {{
  let best=null, bestD=9;
  for (const p of POINTS) {{
    const [sx,sy] = toScreen(p.x, p.y);
    const d = Math.hypot(sx-mx, sy-my);
    if (d < bestD) {{ bestD=d; best=p; }}
  }}
  return best;
}}
canvas.addEventListener('mousemove', (ev) => {{
  const rect = canvas.getBoundingClientRect();
  const p = hit(ev.clientX - rect.left, ev.clientY - rect.top);
  if (!p) {{ tip.style.display='none'; return; }}
  tip.style.display='block';
  tip.style.left = (ev.clientX + 12) + 'px';
  tip.style.top = (ev.clientY + 12) + 'px';
  tip.innerHTML =
    `<b>${{p.chain}}${{p.resi}} ${{p.aa}}</b><br>` +
    `i=${{p.i}} | disc=(${{p.x.toFixed(3)}}, ${{p.y.toFixed(3)}})<br>` +
    `‖z‖₂=${{p.radius.toFixed(4)}} | dehydron=${{p.dehydron.toFixed(3)}}<br>` +
    `MoE <code>E${{p.expert}}</code> | mech σ=${{p.mech.toFixed(3)}} | risk=${{p.risk.toFixed(3)}}`;
}});
canvas.addEventListener('mouseleave', () => tip.style.display='none');
resize();
</script></body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    set_dehydron_wrap_max(1)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digest = assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    system, load_info = build_equiformer_pool_system(
        cfg,
        equiformer_ckpt=DEFAULT_FRONTEND_CKPT,
        device=device,
        freeze_backbone=True,
        max_neighbors=50,
    )
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    blob = torch.load(BEST, map_location=device, weights_only=False)
    system.load_state_dict(blob["model"], strict=False)
    system.eval()
    curvature = (
        float(system.spine.c.detach().cpu())
        if hasattr(system.spine.c, "detach")
        else float(system.spine.c)
    )
    print("loaded", BEST, "epoch", blob.get("epoch"), "mode", load_info.get("mode"))

    split = load_boot_split(DEFAULT_MANIFEST)
    entries = [("train", e) for e in split["train"]] + [("probe", e) for e in split["probe"]]
    index_rows: list[dict[str, Any]] = []

    for role, raw in entries:
        entry = _resolve_entry(raw, PDB_DIR)
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry["chain"])
        batch = _load_train_batch(
            entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE
        )
        with torch.no_grad():
            out = system(
                batch["x"],
                batch["edge_index"],
                batch["edge_type"],
                tau_ceiling=RV_TAU_END,
            )
        z_hyp = out["z_hyp"]
        expert_id = out["moe_aux"]["routing"].argmax(dim=-1)
        pdb_path = ensure_pdb_cached(pdb_id, PDB_DIR)
        pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
        records = [
            rec
            for rec in parse_residue_records_from_pdb_chain(pdb_path, chain)
            if rec.get_atom("CA") is not None
        ]
        if len(records) != int(z_hyp.shape[0]):
            # fall back to index-aligned stub labels
            print(
                f"WARN {pdb_id}:{chain} records={len(records)} != N={int(z_hyp.shape[0])} — pad/trim"
            )
            if len(records) > int(z_hyp.shape[0]):
                records = records[: int(z_hyp.shape[0])]
            else:
                while len(records) < int(z_hyp.shape[0]):
                    records.append(records[-1] if records else None)

        xy = ball_to_disc(z_hyp)
        radius = torch.linalg.vector_norm(z_hyp, dim=-1).cpu().numpy()
        dehyd = batch["dehydron_labels"].detach().cpu().numpy().astype(float)
        mech = torch.sigmoid(out["mechanism_score"]).detach().cpu().numpy()
        evid = out["evidence"].detach().cpu()
        risk = (evid[:, 3] / evid[:, 2].clamp_min(1e-4)).numpy()
        experts = expert_id.detach().cpu().numpy().astype(int)
        points = []
        for i in range(xy.shape[0]):
            rec = records[i] if i < len(records) and records[i] is not None else None
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
                    "resi": int(getattr(rec, "residue_index", i + 1)),
                    "aa": str(getattr(rec, "residue_name", "UNK")),
                    "chain": str(getattr(rec, "chain_label", chain)),
                    "layout": "z_hyp",
                }
            )

        safe_chain = chain.replace("/", "_")
        disc_name = f"{pdb_id.lower()}_{safe_chain}_poincare_disc.html"
        disc_path = OUT_ROOT / disc_name
        _write_enhanced_disc_html(
            structure_id=pdb_id,
            chain=chain,
            role=role,
            points=points,
            output_path=disc_path,
            checkpoint_path=str(BEST),
            curvature=curvature,
            mean_ball_r=float(radius.mean()),
        )

        # Also write canonical v8 disc+split under subfolder
        paths = export_v8_structure_viewers(
            pdb_id=f"{pdb_id}_{safe_chain}",
            z_hyp=z_hyp,
            dehydron_labels=batch["dehydron_labels"],
            mechanism_score=out["mechanism_score"],
            evidence=out["evidence"],
            checkpoint_path=str(BEST),
            curvature=curvature,
            out_root=OUT_ROOT,
            z_attn=out.get("z_attn"),
            z_lift=out.get("z_lift"),
            h_euc=out.get("h_euc"),
            expert_id=expert_id,
            pdb_text=pdb_text,
            residue_records=records if all(r is not None for r in records) else None,
        )
        row = {
            "role": role,
            "pdb_id": pdb_id,
            "chain": chain,
            "n": int(z_hyp.shape[0]),
            "mean_ball_r": float(radius.mean()),
            "dehydron_frac": float(dehyd.mean()),
            "interactive_html": str(disc_path),
            **paths,
        }
        index_rows.append(row)
        print(
            f"[disc] {pdb_id}:{chain} [{role}] N={row['n']} "
            f"r̄={row['mean_ball_r']:.3f} → {disc_path.name}"
        )

    # Index page
    links = "\n".join(
        f'<li><a href="{Path(r["interactive_html"]).name}">'
        f'{r["pdb_id"]}:{r["chain"]}</a> '
        f'[{r["role"]}] N={r["n"]} ‖z‖₂̄={r["mean_ball_r"]:.3f} '
        f'dh={r["dehydron_frac"]:.2f}</li>'
        for r in index_rows
    )
    index_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>geoopt_restore Poincaré discs</title>
<style>
body {{ background:#111; color:#eee; font-family:Georgia,serif; padding:24px; }}
a {{ color:#9cf; }} li {{ margin:8px 0; }}
</style></head><body>
<h1>tokyo_eye_equ_geoopt_restore — interactive Poincaré discs</h1>
<p>Best epoch {blob.get("epoch")} | frontend {digest[:16]}… | Equiformer-pool + geoopt lift</p>
<p>Hover: residue id, AA, ‖z‖₂, dehydron, MoE expert. Color dropdown: dehydron / radius / mech / risk / expert.</p>
<ul>{links}</ul>
</body></html>
"""
    (OUT_ROOT / "index.html").write_text(index_html, encoding="utf-8")
    (OUT_ROOT / "interactive_discs_manifest.json").write_text(
        json.dumps(
            {
                "gate_id": "tokyo_eye_equ_geoopt_restore",
                "ckpt": str(BEST),
                "best_epoch": blob.get("epoch"),
                "frontend_sha256": digest,
                "frontend_mode": "equiformer_v3_pool",
                "proteins": index_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("INDEX", OUT_ROOT / "index.html")
    print("DONE", len(index_rows), "discs")


if __name__ == "__main__":
    main()
