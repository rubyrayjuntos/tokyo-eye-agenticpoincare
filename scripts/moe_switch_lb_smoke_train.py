"""Cheap smoke: few-epoch train with Switch soft LB; report eval-argmax loads each epoch.

Not a sealed card. Goal: see if corrected LB moves eval utilization before full cold run.
"""
from __future__ import annotations

import json
from pathlib import Path

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
from experiments.training.v8.run_tokyo_eye_equ_geoopt_restore import (
    _load_train_batch,
    _resolve_entry,
)
from experiments.training.v8.run_v8_experiment import build_equiformer_pool_system
from science.tokyo_eye.v8.engine import (
    CurriculumRadiusController,
    GumbelTemperatureSchedule,
    train_v8_step,
)
from science.tokyo_eye.v8.equiformer_frontend import DEFAULT_WEIGHT_MAP, load_weight_map
from science.tokyo_eye.v8.moe_eval_gates import eval_moe_utilization
from science.tokyo_eye.v8.r0_r5_graph import set_dehydron_wrap_max

OUT = Path("data/local_objects/frontend_ablation/moe_eval_proxy_lb_smoke.json")
PDB_DIR = Path("/tmp/dtie_pdb_cache")
GRAPH_CACHE = Path("data/graph_cache")
EPOCHS = 2
LR = 1e-3


def eval_loads(system, entries, device, tau):
    system.eval()
    routes = []
    with torch.no_grad():
        for entry in entries:
            batch = _load_train_batch(
                entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE
            )
            out = system(
                batch["x"],
                batch["edge_index"],
                batch["edge_type"],
                tau_ceiling=tau,
            )
            routes.append(out["moe_aux"]["routing"])
    routing = torch.cat(routes, dim=0)
    return eval_moe_utilization(routing), [float(x) for x in routing.mean(0).tolist()]


def main():
    set_dehydron_wrap_max(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert_frontend_bank(DEFAULT_FRONTEND_CKPT, expected=PINNED_FRONTEND_SHA256)
    cfg = dict(load_weight_map(DEFAULT_WEIGHT_MAP))
    cfg.setdefault("cv_coeff", 10.0)
    cfg.setdefault("moe_quota_coeff", 5.0)
    cfg.setdefault("switch_lb_coeff", 1.0)
    cfg.setdefault("soft_quota_coeff", 5.0)
    cfg.setdefault("eval_proxy_lb_coeff", 1.0)
    cfg.setdefault("eval_proxy_quota_coeff", 5.0)
    cfg.setdefault("gumbel_tau_start", 1.0)
    cfg.setdefault("gumbel_tau_end", 0.5)

    # Cold spine + frozen MPtrj frontend (no warm from collapsed θ)
    system, load_info = build_equiformer_pool_system(
        cfg,
        equiformer_ckpt=DEFAULT_FRONTEND_CKPT,
        device=device,
        freeze_backbone=True,
        max_neighbors=50,
    )
    system.spine.tau_clamp_mode = RV_TAU_CLAMP_MODE
    print("frontend", load_info.get("mode"), "cold spine (no collapsed warm-start)")

    split = load_boot_split(DEFAULT_MANIFEST)
    train_entries = [_resolve_entry(e, PDB_DIR) for e in split["train"]]
    probe_entries = [_resolve_entry(e, PDB_DIR) for e in split["probe"]]
    all_entries = train_entries + probe_entries

    # trainable = spine only
    params = [p for p in system.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    radius = CurriculumRadiusController(
        tau_start=0.70, tau_end=float(RV_TAU_END), total_epochs=EPOCHS
    )
    gumbel = GumbelTemperatureSchedule(
        float(cfg["gumbel_tau_start"]),
        float(cfg["gumbel_tau_end"]),
        EPOCHS,
        schedule="exponential",
        half_epochs=max(1, EPOCHS),
    )

    # epoch -1: init eval
    history = []
    gate0, load0 = eval_loads(system, all_entries, device, RV_TAU_END)
    history.append({"epoch": -1, "eval_load": load0, "eval_gates": gate0})
    print(f"ep-1 init eval_load={['%.3f'%x for x in load0]} pass={gate0['all_pass']}")

    for epoch in range(EPOCHS):
        system.train()
        # build s,v batches via full system path used in train_v8_step needs s,v in batch
        # Use same pattern as geoopt runner: forward through TokyoEyeV8WithFrontend wrapper
        # train_v8_step expects batch["s"], batch["v"] — get from frontend
        step_metrics = []
        for entry in train_entries:
            batch_x = _load_train_batch(
                entry, device=device, pdb_dir=PDB_DIR, graph_cache_dir=GRAPH_CACHE
            )
            with torch.no_grad():
                s, v = system.frontend(
                    batch_x["x"],
                    edge_index=batch_x["edge_index"],
                    edge_type=batch_x["edge_type"],
                )
            # mechanism / sdrp targets
            n = int(batch_x["num_nodes"])
            batch = {
                "s": s.detach().requires_grad_(False),  # frontend frozen
                "v": v.detach().requires_grad_(False),
                "edge_index": batch_x["edge_index"],
                "edge_type": batch_x["edge_type"],
                "sdrp_target": batch_x["sdrp_target"],
                "mechanism_pos": batch_x["dehydron_labels"].float(),
                "mechanism_neg": 1.0 - batch_x["dehydron_labels"].float(),
            }
            # s,v need grad through spine only — detach is fine; spine gets leaf inputs
            batch["s"] = s.detach()
            batch["v"] = v.detach()
            m = train_v8_step(
                system.spine,
                batch,
                opt,
                epoch=epoch,
                radius_controller=radius,
                gumbel_schedule=gumbel,
                cv_coeff=float(cfg["cv_coeff"]),
                moe_quota_coeff=float(cfg["moe_quota_coeff"]),
                switch_lb_coeff=float(cfg["switch_lb_coeff"]),
                soft_quota_coeff=float(cfg["soft_quota_coeff"]),
                eval_proxy_lb_coeff=float(cfg["eval_proxy_lb_coeff"]),
                eval_proxy_quota_coeff=float(cfg["eval_proxy_quota_coeff"]),
            )
            step_metrics.append(m)

        gate, load = eval_loads(system, all_entries, device, float(radius.tau_ceiling(epoch)))
        mean_loss = sum(m["loss_total"] for m in step_metrics) / max(len(step_metrics), 1)
        mean_sw = sum(m.get("moe_switch_lb_loss", 0.0) for m in step_metrics) / max(
            len(step_metrics), 1
        )
        mean_px = sum(m.get("moe_eval_proxy_lb_loss", 0.0) for m in step_metrics) / max(
            len(step_metrics), 1
        )
        row = {
            "epoch": epoch,
            "mean_loss": mean_loss,
            "mean_switch_lb": mean_sw,
            "mean_eval_proxy_lb": mean_px,
            "eval_load": load,
            "eval_gates": gate,
            "train_moe_load_last_step": [
                step_metrics[-1].get(f"moe_load_e{i}") for i in range(4)
            ],
        }
        history.append(row)
        print(
            f"ep{epoch} loss={mean_loss:.4f} switch_lb={mean_sw:.4f} proxy_lb={mean_px:.4f} "
            f"eval_load={['%.3f'%x for x in load]} pass={gate['all_pass']} "
            f"(train_gumbel_last={['%.3f'%float(x or 0) for x in row['train_moe_load_last_step']]})"
        )

    report = {
        "epochs": EPOCHS,
        "switch_lb_coeff": cfg["switch_lb_coeff"],
        "soft_quota_coeff": cfg["soft_quota_coeff"],
        "warm_start": None,
        "lb_on": "F.softmax(raw_logits) pre-Gumbel",
        "history": history,
        "final_eval_pass": history[-1]["eval_gates"]["all_pass"],
        "final_eval_load": history[-1]["eval_load"],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
