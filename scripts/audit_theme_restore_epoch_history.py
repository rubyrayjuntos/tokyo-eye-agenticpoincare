"""Per-epoch probe metrics for tokyo_eye_equ_theme_restore — cheaper checks before spread_hold."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mlflow.tracking import MlflowClient

RID = "0fc2986e0f49476694dfff3a4a955af5"
CKPT = Path("checkpoints/tokyoeye/pretrained/hf/checkpoint/mptrj_gradient.pt")
PIN = "59c6c23573a3b05b347662f473209d1bb1ccb5b85f6624a8f87072e7c397addf"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    # --- frontend hash ---
    print("=== frontend_sha256 ===")
    print("pin", PIN)
    if CKPT.is_file():
        digest = sha256_file(CKPT)
        print("file", CKPT, "bytes", CKPT.stat().st_size)
        print("sha256", digest)
        print("matches_pin", digest == PIN)
    else:
        print("MISSING", CKPT)

    # pins / docs note
    pins = json.loads(Path("data/gates/tokyo_eye_equ_theme_restore_pins.json").read_text())
    print("pins.frontend_ckpt", pins.get("frontend_ckpt"))
    print("pins.frontend_mode", pins.get("frontend_mode"))

    client = MlflowClient("http://mlflow:5000")
    run = client.get_run(RID)
    print("\n=== run ===")
    print("run_id", run.info.run_id, "status", run.info.status)

    # history for key metrics
    keys = [
        "probe_mean_spread",
        "probe_mean_sat",
        "probe_mean_h_norm",
        "probe_mean_theme_auprc",
        "probe_n_themes_pass_auprc",
        "seal_mean_spread",
        "seal_mean_theme_auprc",
        "gate_radius_spread_gate_value",
        "gate_mean_theme_auprc_gate_value",
        "theme_auprc_ig_like",
        "theme_auprc_lysozyme_like",
        "theme_auprc_ubiquitin_grasp",
        "theme_auprc_tim_barrel",
        "theme_auprc_globin",
        "theme_auprc_ploop_ntpase",
        "diag_radius_spread",  # train-step, not sealed
    ]
    print("\n=== metric history (step, value) ===")
    tables: dict[str, list[tuple[int, float]]] = {}
    for k in keys:
        hist = client.get_metric_history(RID, k)
        rows = [(int(m.step), float(m.value)) for m in hist]
        tables[k] = rows
        if rows:
            print(f"\n{k} n={len(rows)}")
            for step, val in rows:
                print(f"  step={step} value={val}")
        else:
            print(f"\n{k} n=0")

    # Align by step for joint view
    spread = {s: v for s, v in tables.get("probe_mean_spread", [])}
    auprc = {s: v for s, v in tables.get("probe_mean_theme_auprc", [])}
    npass = {s: v for s, v in tables.get("probe_n_themes_pass_auprc", [])}
    sat = {s: v for s, v in tables.get("probe_mean_sat", [])}
    steps = sorted(set(spread) | set(auprc))
    print("\n=== joint probe steps (spread>0.15 AND mean_auprc>=0.6?) ===")
    print("step | spread | sat | mean_auprc | n_themes>=0.55 | both_gates?")
    both_hits = []
    for st in steps:
        sp = spread.get(st)
        au = auprc.get(st)
        np_ = npass.get(st)
        sa = sat.get(st)
        if sp is None or au is None:
            both = None
        else:
            both = (sp > 0.15) and (au >= 0.6)
            if both:
                both_hits.append(st)
        print(
            f"{st:4d} | "
            f"{'NA' if sp is None else f'{sp:.6f}'} | "
            f"{'NA' if sa is None else f'{sa:.6f}'} | "
            f"{'NA' if au is None else f'{au:.6f}'} | "
            f"{'NA' if np_ is None else f'{np_:.0f}'} | "
            f"{both}"
        )

    print("\n=== cheaper-check verdict ===")
    print("epochs_with_spread_gt_0p15_AND_mean_auprc_ge_0p6:", both_hits if both_hits else "NONE")
    if spread and auprc:
        # crude co-movement: early vs late
        s_steps = sorted(set(spread) & set(auprc))
        if len(s_steps) >= 2:
            s0, s1 = s_steps[0], s_steps[-1]
            print(
                f"first_joint_step={s0} spread={spread[s0]:.6f} auprc={auprc[s0]:.6f}"
            )
            print(
                f"last_joint_step={s1} spread={spread[s1]:.6f} auprc={auprc[s1]:.6f}"
            )
            print(
                f"delta_spread={spread[s1]-spread[s0]:+.6f} delta_auprc={auprc[s1]-auprc[s0]:+.6f}"
            )

    # Also list all metric keys for completeness
    print("\n=== all metric keys on run ===")
    for k in sorted(run.data.metrics):
        print(k, run.data.metrics[k])


if __name__ == "__main__":
    main()
