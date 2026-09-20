"""C1 topology curriculum — split, freeze, hygiene (no GPU, no 150-epoch train)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from experiments.training.v8.b0_topology_observation import HELD_OUT_PDBS
from experiments.training.v8.c1_topology_curriculum import (
    C1_BEST_MOE_MIN,
    C1_BOUNDARY_RADIUS,
    C1_EPOCHS,
    C1_GUMBEL_HALF_EPOCHS,
    C1_TAU_END,
    C1_WRAP_MAX,
    TRAIN_THEME_COUNTS,
    build_c1_stamp,
    c1_hygiene_pass,
    epoch_is_best_eligible,
    freeze_entire_frontend,
    load_c1_split,
    row_moe_load_min,
    spine_param_group,
    theme_counts_viable,
)
from science.tokyo_eye.v8.engine import GumbelTemperatureSchedule
from science.tokyo_eye.v8.equiformer_frontend import build_param_groups

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "v8_c1_topology_curriculum_v1.json"

HOLDOUT_KEYS = {
    ("1HNG", "A"),
    ("1ALC", "A"),
    ("1A5R", "A"),
    ("1NAL", "1"),
    ("2HHB", "B"),
    ("1GKY", "A"),
}
FORBIDDEN_PDBS = {"1UD7", "1TIT", "1QHG", "1EIH", "1A0C", "1F88", "1BG1", "2Z6H", "1IVO", "2SHP"}


def test_split_25_train_6_holdout_disjoint_theme_counts() -> None:
    split = load_c1_split(MANIFEST)
    train, holdout = split["train"], split["holdout"]
    assert len(train) == 25
    assert len(holdout) == 6
    train_keys = {(p["pdb_id"], p["chain"]) for p in train}
    hold_keys = {(p["pdb_id"], p["chain"]) for p in holdout}
    assert train_keys.isdisjoint(hold_keys)
    assert hold_keys == HOLDOUT_KEYS
    assert {p["pdb_id"] for p in train + holdout}.isdisjoint(HELD_OUT_PDBS)
    assert {p["pdb_id"] for p in train + holdout}.isdisjoint(FORBIDDEN_PDBS)
    for theme, n in TRAIN_THEME_COUNTS.items():
        assert sum(1 for p in train if p["theme"] == theme) == n, theme
    kras = next(p for p in train if p["pdb_id"] == "4OBE")
    assert kras["role"] == "train"
    assert kras["theme"] == "ploop_ntpase"
    nmr = next(p for p in holdout if p["pdb_id"] == "1A5R")
    assert nmr["flags"]["nmr_model1"] is True
    nal = next(p for p in holdout if p["pdb_id"] == "1NAL")
    assert nal["chain"] == "1"
    assert nal["flags"]["first_ca_polymer"] is True
    ghl = next(p for p in train if p["pdb_id"] == "1GHL")
    assert ghl["flags"]["first_ca_polymer"] is True
    new_train = {
        "1WIT",
        "2RHE",
        "1LZ1",
        "153L",
        "1NDD",
        "1WM3",
        "1YPI",
        "1BTM",
        "1MXS",
        "1HHO",
        "1ECA",
        "1TEV",
        "1WE2",
    }
    assert new_train <= {p["pdb_id"] for p in train}


def test_freeze_entire_frontend_zeros_bank_and_adapters() -> None:
    class TinyFrontend(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.adapter = nn.Linear(2, 2)

    class TinySpine(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2)

    class TinySystem(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frontend = TinyFrontend()
            self.spine = TinySpine()

    system = TinySystem()
    n_frozen = freeze_entire_frontend(system)
    assert n_frozen >= 4
    assert system.frontend.backbone.weight.requires_grad is False
    assert system.frontend.adapter.weight.requires_grad is False
    assert system.spine.proj.weight.requires_grad is True
    groups = spine_param_group(system, lr=3e-4)
    opt = torch.optim.Adam(groups)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(system.spine.proj.weight) in opt_ids
    assert id(system.frontend.adapter.weight) not in opt_ids
    assert id(system.frontend.backbone.weight) not in opt_ids

    other = TinySystem()
    weak = build_param_groups(
        other.frontend, other.spine, lr_backbone=1e-5, lr_hyperbolic=3e-4, freeze_backbone=True
    )
    weak_ids = {id(p) for g in weak for p in g["params"]}
    assert id(other.frontend.adapter.weight) in weak_ids
    assert id(other.frontend.backbone.weight) not in weak_ids


def _holdout_row(*, loaded: bool = True, sat: float = 0.2, moe_min: float = 0.05) -> dict:
    return {
        "loaded": loaded,
        "z_hyp_finite": loaded,
        "curvature_finite": loaded,
        "boundary_saturation": sat,
        "moe_load_min": moe_min,
    }


def test_c1_hygiene_pass_and_fail_cases() -> None:
    holdouts = [_holdout_row() for _ in range(6)]
    kras = {
        "loaded": True,
        "z_hyp_finite": True,
        "curvature_finite": True,
        "pdb_id": "4OBE",
    }
    ok = c1_hygiene_pass(holdouts, kras)
    assert ok["hygiene_pass"] is True
    assert ok["biology_pass"] is False
    assert ok["n_holdout_loaded"] == 6

    fail_count = c1_hygiene_pass(holdouts[:5], kras)
    assert fail_count["hygiene_pass"] is False

    hot = [_holdout_row(sat=0.51) for _ in range(6)]
    fail_sat = c1_hygiene_pass(hot, kras)
    assert fail_sat["hygiene_pass"] is False
    assert fail_sat["mean_holdout_boundary_saturation"] >= 0.50

    monopoly = [_holdout_row(moe_min=0.0) for _ in range(3)] + [_holdout_row() for _ in range(3)]
    fail_moe = c1_hygiene_pass(monopoly, kras)
    assert fail_moe["hygiene_pass"] is False
    assert fail_moe["n_holdout_moe_spread"] == 3

    fail_kras = c1_hygiene_pass(holdouts, {"loaded": False})
    assert fail_kras["hygiene_pass"] is False
    assert fail_kras["kras_finite"] is False


def test_hygiene_derives_moe_min_from_expert_loads() -> None:
    rows = [
        {
            "loaded": True,
            "z_hyp_finite": True,
            "curvature_finite": True,
            "boundary_saturation": 0.1,
            "moe_load_e0": 0.25,
            "moe_load_e1": 0.25,
            "moe_load_e2": 0.25,
            "moe_load_e3": 0.25,
        }
        for _ in range(6)
    ]
    assert row_moe_load_min(rows[0]) == 0.25
    kras = {"loaded": True, "z_hyp_finite": True, "curvature_finite": True}
    assert c1_hygiene_pass(rows, kras)["hygiene_pass"] is True


def test_epoch_eligible_and_theme_starve() -> None:
    assert C1_BEST_MOE_MIN == 0.02
    assert epoch_is_best_eligible(
        {"loss_total": 1.2, "diag_mean_radius": 0.7, "moe_load_min": 0.05, "z_hyp_finite": True}
    )
    assert not epoch_is_best_eligible(
        {"loss_total": 0.1, "diag_mean_radius": 0.7, "moe_load_min": 0.01, "z_hyp_finite": True}
    )
    assert not epoch_is_best_eligible(
        {"loss_total": float("nan"), "diag_mean_radius": 0.7, "moe_load_min": 0.05}
    )
    split = load_c1_split(MANIFEST)
    train = split["train"]
    assert theme_counts_viable(train, set()) is True
    ig = [(p["pdb_id"], p["chain"]) for p in train if p["theme"] == "ig_like"]
    # 4 Ig train chains; skipping 3 leaves 1 → abort
    assert theme_counts_viable(train, set(ig[:3])) is False


def test_c1_gumbel_stretched_across_150_not_mode_c_alpha() -> None:
    assert C1_EPOCHS == 150
    assert C1_TAU_END == 0.90
    assert C1_WRAP_MAX == 1
    assert C1_BOUNDARY_RADIUS == 0.80
    assert C1_GUMBEL_HALF_EPOCHS == 75
    c1 = GumbelTemperatureSchedule(
        1.0,
        0.3,
        C1_EPOCHS,
        schedule="exponential",
        alpha=None,
        half_epochs=C1_GUMBEL_HALF_EPOCHS,
    )
    assert abs(c1.temperature(75) - 0.3) < 1e-6
    assert c1.temperature(12) > 0.7
    mode_c = GumbelTemperatureSchedule(
        1.0, 0.3, 150, schedule="exponential", alpha=0.1002, half_epochs=12
    )
    assert mode_c.temperature(12) <= 0.31


def test_build_c1_stamp_biology_pass_false() -> None:
    holdouts = [_holdout_row() for _ in range(6)]
    for i, key in enumerate(sorted(HOLDOUT_KEYS)):
        holdouts[i]["pdb_id"], holdouts[i]["chain"] = key
        holdouts[i]["theme"] = "ig_like"
    kras = {
        "pdb_id": "4OBE",
        "chain": "A",
        "loaded": True,
        "z_hyp_finite": True,
        "curvature_finite": True,
    }
    stamp = build_c1_stamp(
        holdouts,
        kras,
        champion_sha256="507d54bd" + "0" * 56,
        champion_version=5,
        wrap_max=1,
        boundary_radius=0.80,
        tau_probe=0.90,
    )
    assert stamp["biology_pass"] is False
    assert stamp["biology_gate"] == "not_applicable"
    assert stamp["hygiene_pass"] is True
    assert stamp["theta"]["alias"] == "champion"
    assert stamp["graph_recipe"]["dehydron_wrap_max"] == 1
    assert "0.80" in stamp["h3_note"]
