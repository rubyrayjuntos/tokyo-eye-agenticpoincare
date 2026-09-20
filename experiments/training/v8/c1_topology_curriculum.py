"""C1 topology curriculum — panel split, frontend freeze, hygiene Pass (no GPU)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch.nn as nn

from experiments.training.v8.b0_topology_observation import HELD_OUT_PDBS, THEMES

DEFAULT_MANIFEST = Path("manifests/v8_c1_topology_curriculum_v1.json")
DEFAULT_STAMP = Path("data/gates/tokyo_eye_v8_c1_topology_curriculum.json")
TRAIN_THEME_COUNTS = {
    "ig_like": 4,
    "lysozyme_like": 4,
    "ubiquitin_grasp": 4,
    "tim_barrel": 5,
    "globin": 4,
    "ploop_ntpase": 4,
}

C1_TAU_START = 0.70
C1_TAU_END = 0.90
C1_WRAP_MAX = 1
C1_BOUNDARY_RADIUS = 0.80
C1_EPOCHS = 150
C1_GUMBEL_HALF_EPOCHS = 75
C1_BEST_MOE_MIN = 0.02
C1_HOLDOUT_PROBE_EVERY = 25
C1_MIN_THEME_TRAIN = 2


def _row_flags(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "grasp_cousin": bool(raw.get("grasp_cousin", False)),
        "nmr_model1": bool(raw.get("nmr_model1", False)),
        "first_ca_polymer": bool(raw.get("first_ca_polymer", False)),
        "fallback_pdb_id": str(raw.get("fallback_pdb_id") or "").strip().upper(),
        "fallback_chain": str(raw.get("fallback_chain") or "").strip(),
    }


def _as_entry(raw: Mapping[str, Any]) -> dict[str, Any]:
    pdb_id = str(raw.get("pdb_id") or "").strip().upper()
    chain = str(raw.get("chain") or "A").strip()
    theme = str(raw.get("theme") or "").strip()
    role = str(raw.get("role") or "").strip().lower()
    if not pdb_id or theme not in THEMES or role not in {"train", "holdout"}:
        raise ValueError(f"invalid C1 protein row: {raw}")
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "theme": theme,
        "gene": str(raw.get("gene") or ""),
        "role": role,
        "enabled": True,
        "flags": _row_flags(raw),
    }


def load_c1_split(
    manifest_path: Path | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{"train": [...], "holdout": [...]}`` from the C1 manifest."""
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    with path.open() as f:
        data = json.load(f)
    proteins = data.get("proteins", [])
    if not isinstance(proteins, list):
        raise ValueError(f"manifest {path} missing proteins[]")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for raw in proteins:
        if not raw.get("enabled", True):
            continue
        entry = _as_entry(raw)
        if entry["pdb_id"] in HELD_OUT_PDBS:
            raise ValueError(f"shock PDB {entry['pdb_id']} must not appear in C1")
        if entry["role"] == "train":
            train.append(entry)
        else:
            holdout.append(entry)
    if len(train) != 25:
        raise ValueError(f"C1 train must have 25 rows, got {len(train)}")
    if len(holdout) != 6:
        raise ValueError(f"C1 holdout must have 6 rows, got {len(holdout)}")
    train_ids = {(p["pdb_id"], p["chain"]) for p in train}
    hold_ids = {(p["pdb_id"], p["chain"]) for p in holdout}
    if train_ids & hold_ids:
        raise ValueError(f"train/holdout overlap: {train_ids & hold_ids}")
    for theme, n in TRAIN_THEME_COUNTS.items():
        got = sum(1 for p in train if p["theme"] == theme)
        if got != n:
            raise ValueError(f"theme {theme} train count {got} != {n}")
    return {"train": train, "holdout": holdout}


def freeze_entire_frontend(system: nn.Module) -> int:
    """Freeze Equiformer bank **and** adapters. Returns frozen tensor count."""
    frontend = getattr(system, "frontend", None)
    if not isinstance(frontend, nn.Module):
        raise TypeError("system has no frontend module")
    n = 0
    for p in frontend.parameters():
        p.requires_grad = False
        n += 1
    return n


def spine_param_group(system: nn.Module, *, lr: float) -> list[dict[str, Any]]:
    spine = getattr(system, "spine", None)
    if not isinstance(spine, nn.Module):
        raise TypeError("system has no spine module")
    params = [p for p in spine.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable spine parameters")
    return [{"params": params, "lr": float(lr), "name": "hyperbolic"}]


def row_moe_load_min(row: Mapping[str, Any]) -> float | None:
    if row.get("moe_load_min") is not None:
        return float(row["moe_load_min"])
    loads = [
        float(row[key])
        for key in (f"moe_load_e{i}" for i in range(4))
        if row.get(key) is not None
    ]
    if not loads:
        return None
    return float(min(loads))


def epoch_is_best_eligible(last_step: Mapping[str, Any]) -> bool:
    """Lowest epoch-mean loss only among finite last steps with MoE not dead."""
    if float(last_step.get("nan_abort") or 0.0) >= 1.0:
        return False
    loss = last_step.get("loss_total")
    if loss is None or not math.isfinite(float(loss)):
        return False
    z_finite = last_step.get("z_hyp_finite")
    if z_finite is None:
        mean_r = last_step.get("diag_mean_radius")
        z_finite = mean_r is not None and math.isfinite(float(mean_r))
    moe_min = last_step.get("moe_load_min")
    return bool(z_finite) and moe_min is not None and float(moe_min) >= C1_BEST_MOE_MIN


def theme_counts_viable(
    train: Sequence[Mapping[str, Any]],
    skipped: set[tuple[str, str]],
    *,
    min_per_theme: int = C1_MIN_THEME_TRAIN,
) -> bool:
    remaining = [
        e for e in train if (str(e["pdb_id"]), str(e["chain"])) not in skipped
    ]
    for theme in THEMES:
        n = sum(1 for e in remaining if e["theme"] == theme)
        if n < int(min_per_theme):
            return False
    return True


def c1_hygiene_pass(
    holdout_rows: Sequence[Mapping[str, Any]],
    kras_row: Mapping[str, Any] | None,
    *,
    sat_ceiling: float = 0.50,
    min_holdouts_with_moe: int = 4,
) -> dict[str, Any]:
    """Spec Pass: finite H2 on 6 holdouts + 4OBE; mean sat < 0.50; MoE not monopoly."""
    loaded = [r for r in holdout_rows if r.get("loaded")]
    finite = all(
        bool(r.get("z_hyp_finite")) and bool(r.get("curvature_finite")) for r in loaded
    )
    kras_ok = bool(
        kras_row
        and kras_row.get("loaded")
        and kras_row.get("z_hyp_finite")
        and kras_row.get("curvature_finite")
    )
    sats = [
        float(r["boundary_saturation"])
        for r in loaded
        if r.get("boundary_saturation") is not None
    ]
    mean_sat = float(sum(sats) / len(sats)) if sats else 1.0
    moe_spread = 0
    for r in loaded:
        moe_min = row_moe_load_min(r)
        if moe_min is not None and moe_min > 0.0:
            moe_spread += 1
    passed = (
        len(loaded) == 6
        and finite
        and kras_ok
        and mean_sat < float(sat_ceiling)
        and moe_spread >= int(min_holdouts_with_moe)
    )
    return {
        "hygiene_pass": passed,
        "n_holdout_loaded": len(loaded),
        "holdout_finite": finite,
        "kras_finite": kras_ok,
        "mean_holdout_boundary_saturation": mean_sat,
        "n_holdout_moe_spread": moe_spread,
        "biology_pass": False,
    }


def build_c1_stamp(
    holdout_rows: Sequence[Mapping[str, Any]],
    kras_row: Mapping[str, Any] | None,
    *,
    champion_sha256: str,
    champion_version: int | str,
    wrap_max: int,
    boundary_radius: float,
    tau_probe: float,
    champion_run_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hygiene = c1_hygiene_pass(holdout_rows, kras_row)
    stamp: dict[str, Any] = {
        "schema_version": 1,
        "gate": "tokyo_eye_v8_c1_topology_curriculum",
        "spec": "docs/superpowers/specs/2026-08-24-tokyoeye-c1-topology-curriculum-design.md",
        "biology_pass": False,
        "biology_gate": "not_applicable",
        **hygiene,
        "tau_probe": float(tau_probe),
        "h3_note": (
            f"boundary_saturation uses PoincareDiagnosticsEngine("
            f"boundary_radius={boundary_radius:.2f}); default 0.90 is vacuous "
            f"when tau_ceiling={tau_probe:.2f} clamps r ≤ tau."
        ),
        "graph_recipe": {
            "dehydron_wrap_max": int(wrap_max),
            "note": "Frozen wrap_max=1 (B0 Sprint-8 retune). Not re-medianed each epoch.",
        },
        "theta": {
            "alias": "champion",
            "uri": "models:/TokyoEye@champion",
            "version": champion_version,
            "sha256": champion_sha256,
            "run_id": champion_run_id,
        },
        "per_holdout": list(holdout_rows),
        "kras_home": kras_row,
        "forbidden": [
            "champion_retarget",
            "allosteric_site_claim",
            "kras_understanding_claim",
            "pathway_or_resistance",
            "disc_alone_hub_rank",
            "affinity_core_as_c1_pass",
            "hardcoded_curvature",
        ],
    }
    if extra:
        stamp.update(dict(extra))
    return stamp


__all__ = [
    "C1_BEST_MOE_MIN",
    "C1_BOUNDARY_RADIUS",
    "C1_EPOCHS",
    "C1_GUMBEL_HALF_EPOCHS",
    "C1_HOLDOUT_PROBE_EVERY",
    "C1_MIN_THEME_TRAIN",
    "C1_TAU_END",
    "C1_TAU_START",
    "C1_WRAP_MAX",
    "DEFAULT_MANIFEST",
    "DEFAULT_STAMP",
    "TRAIN_THEME_COUNTS",
    "build_c1_stamp",
    "c1_hygiene_pass",
    "epoch_is_best_eligible",
    "freeze_entire_frontend",
    "load_c1_split",
    "row_moe_load_min",
    "spine_param_group",
    "theme_counts_viable",
]
