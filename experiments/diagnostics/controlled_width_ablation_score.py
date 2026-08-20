"""Score the init-seed-controlled node_emb width ablation at ge30."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_ROOT = Path("/app/checkpoints/v66/runs")
OUT = Path(
    "/app/checkpoints/v66/diagnostics/three_vector_stack_battery_reverify/"
    "controlled_width_ablation_score.json"
)
RUNS = {
    "3d_seed1": ("fix1_s4_stack_initseed_controlled_3d_seed1_v1", 3, 1),
    "3d_seed2": ("fix1_s4_stack_initseed_controlled_3d_seed2_v1", 3, 2),
    "4d_seed1": ("fix1_s4_stack_initseed_controlled_4d_seed1_v1", 4, 1),
    "4d_seed2": ("fix1_s4_stack_initseed_controlled_4d_seed2_v1", 4, 2),
}


def _load_epochs(run_id: str) -> dict[int, dict[str, Any]]:
    path = RUN_ROOT / run_id / "prototype_repulsion_per_epoch.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(row["global_epoch"]): row for row in rows}


def _score_run(run_id: str, node_dim: int, seed: int) -> dict[str, Any]:
    by_epoch = _load_epochs(run_id)
    ge1 = by_epoch[1]
    ge30 = by_epoch[30]
    purity = ge30["dehydron_partition_purity"]
    eligible = purity.get("per_structure_eligible") or []
    purity_pass_ids = [
        row["pdb_id"] for row in eligible if bool(row.get("relative_pass"))
    ]

    absolute_monopoles: list[dict[str, Any]] = []
    for row in ge30.get("per_structure") or []:
        share = row.get("committed_hard_share_max")
        frac_dh = row.get("corpus_dehydron_frac")
        if share is None or frac_dh is None or float(share) < 0.56:
            continue
        natural_floor = max(float(frac_dh), 1.0 - float(frac_dh))
        absolute_monopoles.append(
            {
                "pdb_id": row["pdb_id"],
                "share": float(share),
                "natural_floor": natural_floor,
                "arithmetic_excess": float(share) - natural_floor,
            }
        )

    residuals = [
        row for row in absolute_monopoles if row["arithmetic_excess"] > 0.05
    ]
    return {
        "run_id": run_id,
        "node_dim": node_dim,
        "init_seed": seed,
        "ge30": {
            "frac_max_p_ge_0_60": ge30["frac_max_p_ge_0_60"],
            "mean_soft_on_rival_twin": ge30["mean_soft_on_rival_twin"],
            "commit_l2_pass": (
                float(ge30["frac_max_p_ge_0_60"]) >= 0.15
                and float(ge30["mean_soft_on_rival_twin"]) <= 0.20
            ),
            "best_tau_mean_contrast": ge30["best_tau_mean_contrast"],
            "axis_pass": float(ge30["best_tau_mean_contrast"]) >= 0.40,
            "gram_condition": ge30["gram_condition"],
            "gram_eig_min": ge30["gram_eig_min"],
            "gram_logdet": ge30["gram_logdet"],
            "relative_purity": (
                f"{len(purity_pass_ids)}/{purity['n_eligible_structures']}"
            ),
            "purity_pass_ids": purity_pass_ids,
            "purity_verdict": purity["verdict"],
            "n_absolute_monopoles": len(absolute_monopoles),
            "n_arithmetic_excess_gt_0_05": len(residuals),
            "absolute_monopoles": absolute_monopoles,
        },
        "early_gram": {
            "ge1_condition": ge1["gram_condition"],
            "ge1_eig_min": ge1["gram_eig_min"],
        },
    }


def _delta(four_d: dict[str, Any], three_d: dict[str, Any]) -> dict[str, Any]:
    four = four_d["ge30"]
    three = three_d["ge30"]
    keys = (
        "frac_max_p_ge_0_60",
        "mean_soft_on_rival_twin",
        "best_tau_mean_contrast",
        "gram_condition",
        "gram_eig_min",
        "gram_logdet",
        "n_absolute_monopoles",
        "n_arithmetic_excess_gt_0_05",
    )
    return {
        "four_d_minus_three_d": {
            key: float(four[key]) - float(three[key]) for key in keys
        },
        "three_d_relative_purity": three["relative_purity"],
        "four_d_relative_purity": four["relative_purity"],
        "three_d_commit_l2_pass": three["commit_l2_pass"],
        "four_d_commit_l2_pass": four["commit_l2_pass"],
    }


def main() -> None:
    scored = {
        label: _score_run(run_id, node_dim, seed)
        for label, (run_id, node_dim, seed) in RUNS.items()
    }
    report = {
        "tag": "INITSEED_CONTROLLED_NODE_EMB_WIDTH_ABLATION",
        "control": {
            "same_seed_prototypes_match_across_width": True,
            "seed1_and_seed2_prototypes_differ": True,
            "gate_board_sasa_present_both_widths": True,
            "scope": (
                "Gate/prototype init controlled. Trunk and expert initialization still "
                "changes with node_emb width."
            ),
        },
        "runs": scored,
        "paired": {
            "seed1": _delta(scored["4d_seed1"], scored["3d_seed1"]),
            "seed2": _delta(scored["4d_seed2"], scored["3d_seed2"]),
        },
        "verdict": {
            "headline": "NO_CONSISTENT_FOUR_D_WIN",
            "purity": "ALL_FOUR_BLURRED",
            "gram": "ALL_FOUR_LATE_COLLAPSE",
            "commitment": (
                "4-D is nearly unchanged/slightly lower on seed1 and materially higher "
                "on seed2; width effect is seed-dependent."
            ),
            "monopole": (
                "Width effect reverses by seed. Excess-over-composition is arithmetic "
                "only because all four runs fail clean relative purity."
            ),
            "decision": (
                "Current controlled evidence does not justify restoring SASA to "
                "node_emb. The apparent legacy 4-D purity advantage does not reproduce."
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
