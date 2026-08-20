"""Overlay per-expert health metrics (4 experts, one chart per metric family).

MLflow's default UI plots one metric name per chart, so ``expert_0_depth_mean``
and ``expert_1_depth_mean`` never share axes. This script reads ``metrics.json``
and writes a self-contained HTML page with colored expert lines.

Example::

    python -m experiments.diagnostics.plot_expert_metrics_overlay \\
      --metrics checkpoints/v7/runs/tokyo_eye_v7_bprime_health_continue_v1/metrics.json \\
      --out checkpoints/v7/diagnostics/expert_overlay_continue_v1.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_BASES = (
    "depth_mean",
    "disc_r_mean",
    "disc_r_std",
    "r_depth_tau",
    "rho_mean",
    "tau_mean",
    "ss_helix_frac",
    "ss_coil_frac",
    "ss_sheet_frac",
)

# Distinct, colorblind-friendlier palette (not MLflow default purple soup).
EXPERT_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00")


def _finite(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def load_expert_series(
    metrics_path: Path,
    *,
    bases: tuple[str, ...] = DEFAULT_BASES,
    num_experts: int = 4,
) -> dict[str, Any]:
    rows = json.loads(metrics_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected list metrics.json, got {type(rows)}")
    xs: list[int] = []
    series: dict[str, dict[int, list[float | None]]] = {
        b: {e: [] for e in range(num_experts)} for b in bases
    }
    for row in rows:
        ge = int(row.get("global_epoch") or 0)
        xs.append(ge)
        health = row.get("health") or {}
        for b in bases:
            for e in range(num_experts):
                series[b][e].append(_finite(health.get(f"expert_{e}_{b}")))
    return {
        "metrics_path": str(metrics_path),
        "global_epochs": xs,
        "bases": list(bases),
        "num_experts": num_experts,
        "series": {
            b: {str(e): vals for e, vals in series[b].items()} for b in bases
        },
    }


def render_html(payload: dict[str, Any], *, title: str) -> str:
    """Minimal Chart.js page — no build step."""
    data_json = json.dumps(payload)
    colors = json.dumps(list(EXPERT_COLORS))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 1.5rem; background: #fafafa; color: #111; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 0.25rem; }}
    .meta {{ color: #555; font-size: 0.9rem; margin-bottom: 1.25rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem; }}
    .card h2 {{ font-size: 0.95rem; margin: 0 0 0.5rem; font-weight: 600; }}
    canvas {{ width: 100% !important; height: 220px !important; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Source: <code id="src"></code> · experts colored e0–e3 · NaNs omitted as gaps</p>
  <div class="grid" id="grid"></div>
  <script>
    const payload = {data_json};
    const colors = {colors};
    document.getElementById('src').textContent = payload.metrics_path;
    const xs = payload.global_epochs;
    const grid = document.getElementById('grid');
    for (const base of payload.bases) {{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = '<h2>expert_*_' + base + '</h2><canvas></canvas>';
      grid.appendChild(card);
      const ctx = card.querySelector('canvas').getContext('2d');
      const datasets = [];
      for (let e = 0; e < payload.num_experts; e++) {{
        datasets.push({{
          label: 'expert_' + e,
          data: payload.series[base][String(e)],
          borderColor: colors[e % colors.length],
          backgroundColor: colors[e % colors.length],
          tension: 0.15,
          spanGaps: false,
          pointRadius: 2,
          borderWidth: 2,
        }});
      }}
      new Chart(ctx, {{
        type: 'line',
        data: {{ labels: xs, datasets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {{ mode: 'index', intersect: false }},
          plugins: {{ legend: {{ position: 'bottom' }} }},
          scales: {{
            x: {{ title: {{ display: true, text: 'global_epoch' }} }},
            y: {{ title: {{ display: true, text: base }} }},
          }},
        }},
      }});
    }}
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Per-expert metric overlay (4 experts)")
    p.add_argument("--num-experts", type=int, default=4)
    args = p.parse_args(argv)
    payload = load_expert_series(args.metrics, num_experts=args.num_experts)
    html = render_html(payload, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(payload['global_epochs'])} epochs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
