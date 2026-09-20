"""Compare MoE expert loads: train(Gumbel STE) vs eval(argmax) on geoopt_restore best."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from experiments.training.v8.equ_geoopt_restore import (
    DEFAULT_FRONTEND_CKPT, DEFAULT_MANIFEST, PINNED_FRONTEND_SHA256,
    RV_TAU_CLAMP_MODE, RV_TAU_END, assert_frontend_bank, load_boot_split,
)
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import _load_train_batch, _resolve_entry
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

BEST = Path("checkpoints/tokyoeye/runs/eqf_equ_geoopt_restore_20260916/tokyoeye_best.pt")
OUT = Path("data/local_objects/frontend_ablation/moe_eval_vs_train_loads.json")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")

def loads_from_routing(routing: torch.Tensor) -> list[float]:
    return routing.mean(dim=0).detach().float().cpu().tolist()

def main():
    set_dehydron_wrap_max(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    system, _ = build_equiformer_pool_system(
        cfg, equiformer_ckpt=DEFAULT_FRONTEND_CKPT, device=device,
        freeze_backbone=True, max_neighbors=50,
    )
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    blob = torch.load(BEST, map_location=device, weights_only=False)
    system.load_state_dict(blob["model"], strict=False)
    # set gumbel tau like late train
    system.set_moe_temperature(0.5)

    split = load_boot_split(DEFAULT_MANIFEST)
    entries = [_resolve_entry(e, PDB_DIR) for e in (split["train"] + split["probe"])]

    eval_loads = []
    train_loads = []  # single Gumbel sample each
    train_loads_mean = []  # average over 8 Gumbel draws
    per = []

    for entry in entries:
        pdb_id = str(entry["pdb_id"]).upper()
        chain = str(entry["chain"])
        batch = _load_train_batch(entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE)
        # eval
        system.eval()
        with torch.no_grad():
            out_e = system(batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=RV_TAU_END)
            le = loads_from_routing(out_e["moe_aux"]["routing"])
            hard = out_e["moe_aux"]["routing"].argmax(-1).cpu().numpy()
            r = torch.linalg.vector_norm(out_e["z_hyp"], dim=-1).cpu().numpy()
        # train mode — one draw
        system.train()
        with torch.no_grad():  # no grad but Gumbel still stochastic in train
            out_t = system(batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=RV_TAU_END)
            lt = loads_from_routing(out_t["moe_aux"]["routing"])
            draws = []
            for _ in range(8):
                o = system(batch["x"], batch["edge_index"], batch["edge_type"], tau_ceiling=RV_TAU_END)
                draws.append(loads_from_routing(o["moe_aux"]["routing"]))
            lt_mean = np.mean(draws, axis=0).tolist()

        eval_loads.append(le)
        train_loads.append(lt)
        train_loads_mean.append(lt_mean)
        per.append({
            "pdb": f"{pdb_id}:{chain}",
            "eval_load": le,
            "train_one_draw": lt,
            "train_mean_8": lt_mean,
            "eval_experts_used": int(len(np.unique(hard))),
            "eval_r_by_expert": {
                f"E{i}": float(r[hard == i].mean()) if (hard == i).any() else None
                for i in range(4)
            },
        })
        print(f"{pdb_id}:{chain} eval={['%.3f'%x for x in le]} train8={['%.3f'%x for x in lt_mean]}")

    ev = np.mean(eval_loads, axis=0)
    tr = np.mean(train_loads_mean, axis=0)
    report = {
        "ckpt": str(BEST),
        "epoch": blob.get("epoch"),
        "mean_eval_load": ev.tolist(),
        "mean_train_gumbel_8_load": tr.tolist(),
        "eval_experts_with_load_gt_0.05": int((ev > 0.05).sum()),
        "train_experts_with_load_gt_0.05": int((tr > 0.05).sum()),
        "finding": (
            "If eval collapses to 2 experts while train Gumbel looks balanced, "
            "MLflow moe_load_* logged during train was a false healthy signal."
        ),
        "per_structure": per,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("MEAN eval", ev, "train8", tr)
    print("WROTE", OUT)

if __name__ == "__main__":
    main()
