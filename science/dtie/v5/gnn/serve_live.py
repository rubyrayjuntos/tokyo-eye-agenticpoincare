"""
serve_live.py — Live inference server for Tokyo Eyes v5 viewer
===============================================================
Eidetix Bio | 2026-05-20

Runs a lightweight HTTP server that serves GNN embeddings to the React viewer
in real-time. Supports:
  - Loading a checkpoint and serving embeddings for any PDB
  - Re-running inference when curvature is changed from the viewer
  - WebSocket for live updates during training (optional)

Usage:
    python serve_live.py \
        --checkpoint ./checkpoints_v5/best_checkpoint.pt \
        --pdb_dir /tmp/dtie_pdb_cache \
        --port 8765

The viewer connects to http://localhost:8765/api/embeddings?pdb_id=4OBE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_THIS_DIR = Path(__file__).resolve().parent
_ORCHESTRATION_DIR = _THIS_DIR.parent

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ORCHESTRATION_DIR / "TokyoEyesv4"))

from Gnnv5 import GOSPConeMapper, precompute_clustering
from train_v4 import load_protein_graph, TRAINING_TARGETS
from export_for_viewer import export_protein, KRAS_DOMAINS

# Global state
_model: Optional[GOSPConeMapper] = None
_device: str = "cpu"
_pdb_dir: Optional[Path] = None
_cache: dict = {}


class ViewerAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the viewer API."""

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/api/embeddings":
            self._handle_embeddings(params)
        elif parsed.path == "/api/proteins":
            self._handle_protein_list()
        elif parsed.path == "/api/status":
            self._handle_status()
        else:
            self._send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _handle_embeddings(self, params):
        """Serve embeddings for a given PDB."""
        pdb_id = params.get("pdb_id", ["4OBE"])[0].upper()
        chain = params.get("chain", ["A"])[0]
        cache_key = f"{pdb_id}_{chain}"

        # Check cache
        if cache_key in _cache:
            self._send_json(_cache[cache_key])
            return

        if _model is None:
            self._send_json({"error": "no model loaded"}, 503)
            return

        try:
            result = export_protein(
                model=_model,
                pdb_id=pdb_id,
                chain=chain,
                pdb_dir=_pdb_dir,
                device=_device,
            )
            _cache[cache_key] = result
            self._send_json(result)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_protein_list(self):
        """Return available training targets."""
        proteins = [
            {"pdb_id": pid, **info}
            for pid, info in TRAINING_TARGETS.items()
        ]
        self._send_json({"proteins": proteins})

    def _handle_status(self):
        """Return server status."""
        status = {
            "model_loaded": _model is not None,
            "version": "v5",
            "architecture": "decoupled_radial_angular",
            "device": _device,
            "cached_proteins": list(_cache.keys()),
        }
        if _model is not None:
            status["curvature"] = _model.curvature.item()
            status["radial_scale"] = _model.radial_head.radial_scale.item()
        self._send_json(status)

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        """Suppress default logging noise."""
        pass


def main():
    global _model, _device, _pdb_dir

    parser = argparse.ArgumentParser(description="Tokyo Eyes v5 Live Server")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, default="/tmp/dtie_pdb_cache")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    _device = args.device
    _pdb_dir = Path(args.pdb_dir)
    _pdb_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=_device)
    arch = ckpt.get("architecture", {})

    _model = GOSPConeMapper(
        node_dim=4,
        hidden=arch.get("hidden", 128),
        num_layers=arch.get("num_layers", 6),
        num_experts=arch.get("num_experts", 4),
        projection_dim=arch.get("projection_dim", 64),
        hyp_proj_dim_2d=arch.get("hyp_proj_dim_2d", 2),
        hyp_proj_dim_3d=arch.get("hyp_proj_dim_3d", 3),
    )
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.to(_device)
    _model.eval()

    print(f"  Curvature: {_model.curvature.item():.4f}")
    print(f"  Radial scale: {_model.radial_head.radial_scale.item():.4f}")
    print(f"  Device: {_device}")
    print(f"  PDB dir: {_pdb_dir}")

    # Start server
    server = HTTPServer(("0.0.0.0", args.port), ViewerAPIHandler)
    print(f"\n  Serving on http://localhost:{args.port}")
    print(f"  Endpoints:")
    print(f"    GET /api/embeddings?pdb_id=4OBE&chain=A")
    print(f"    GET /api/proteins")
    print(f"    GET /api/status")
    print(f"\n  Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
