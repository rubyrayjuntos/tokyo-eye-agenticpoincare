from pathlib import Path
import json
import torch

from science.tokyo_eye.v8.equiformer_frontend import (
    DEFAULT_WEIGHT_MAP,
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    load_weight_map,
)
from science.tokyo_eye.v8.loader import DEFAULT_CHAIN, ensure_pdb_cached, load_structure_batch
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from science.tokyo_eye.v8.biophysics import parse_residue_records_from_pdb_chain
from experiments.training.v8.export_viewers import export_v8_structure_viewers

CKPT = Path("checkpoints/tokyoeye/runs/eqf_equ_lift_radius_20260916/tokyoeye_best.pt")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
OUT_ROOT = Path("data/local_objects/gnn_viewer/equ_lift_radius")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PDBS = ["4OBE", "1UBQ", "1TIM", "1HNG"]

cfg = load_weight_map(DEFAULT_WEIGHT_MAP)
frontend = StubEquiformerFrontend(
    in_dim=3,
    scalar_dim=int(cfg["scalar_dim"]),
    vector_dim=int(cfg["vector_dim"]),
    num_backbone_blocks=int(cfg.get("num_backbone_blocks", 7)),
    live_backbone=True,
)
spine = TokyoEyesHyperbolicV8(
    scalar_dim=int(cfg["scalar_dim"]),
    vector_dim=int(cfg["vector_dim"]),
    hidden_dim=int(cfg["hidden_dim"]),
    num_attn_layers=2,
    num_sdrp_classes=5,
    c=float(cfg.get("curvature_c", 1.0)),
)
system = TokyoEyeV8WithFrontend(frontend, spine).to(DEVICE)
blob = torch.load(CKPT, map_location=DEVICE, weights_only=False)
state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
missing, unexpected = system.load_state_dict(state, strict=False)
print(f"[export] device={DEVICE} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
system.spine.tau_clamp_mode = "final_only"
system.eval()
tau_ceil = float(cfg.get("tau_end", 0.995))
curvature = float(cfg.get("curvature_c", 1.0))
OUT_ROOT.mkdir(parents=True, exist_ok=True)

exported = []
for pdb_id in PDBS:
    chain = DEFAULT_CHAIN
    batch = load_structure_batch(
        pdb_id, chain, pdb_dir=PDB_DIR, device=DEVICE,
        graph_cache_dir=PDB_DIR / "v8_graph_cache",
    )
    with torch.no_grad():
        out = system(batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=tau_ceil)
    z = out["z_hyp"]
    r = z.norm(dim=-1)
    print(
        f"[export] {pdb_id} N={batch['num_nodes']} "
        f"r_mean={float(r.mean()):.4f} r_std={float(r.std()):.4f} r_max={float(r.max()):.4f}",
        flush=True,
    )
    pdb_path = ensure_pdb_cached(pdb_id, PDB_DIR)
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
        checkpoint_path=str(CKPT),
        curvature=curvature,
        out_root=OUT_ROOT,
        z_attn=out.get("z_attn"),
        z_lift=out.get("z_lift"),
        h_euc=out.get("h_euc"),
        expert_id=out["moe_aux"]["routing"].argmax(dim=-1),
        pdb_text=pdb_text,
        residue_records=records,
    )
    print(json.dumps({"pdb_id": pdb_id, **paths}, indent=2), flush=True)
    exported.append({"pdb_id": pdb_id, **paths})

(OUT_ROOT / "export_manifest.json").write_text(
    json.dumps(
        {
            "ckpt": str(CKPT),
            "gate": "tokyo_eye_equ_lift_radius",
            "mlflow_run_id": "09f422b04335450d9cbc55ad2f675181",
            "tau_clamp_mode": "final_only",
            "exports": exported,
        },
        indent=2,
    )
    + "\n"
)
print("[export] DONE", OUT_ROOT, flush=True)
