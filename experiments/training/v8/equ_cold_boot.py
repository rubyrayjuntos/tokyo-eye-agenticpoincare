"""Tokyo Eye EQU cold boot — split parse, sha guards, hygiene (no GPU)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch.nn as nn

from experiments.training.v8.c1_topology_curriculum import (
    freeze_entire_frontend,
    row_moe_load_min,
    spine_param_group,
)

DEFAULT_MANIFEST = Path("manifests/equ_cold_boot_v1.json")
DEFAULT_STAMP = Path("data/gates/tokyo_eye_equ_cold_boot.json")
PINNED_FRONTEND_SHA256 = (
    "59c6c23573a3b05b347662f473209d1bb1ccb5b85f6624a8f87072e7c397addf"
)
FORBIDDEN_CKPT_SUBSTRINGS = (
    "507d54bd",
    "eqf_affinity",
    "eqf_c1_topology",
    "TokyoEye_champion",
    "v8_affinity_best",
    "tokyoeye_eqf_champion",
)

BOOT_EPOCHS = 40
BOOT_PROBE_EVERY = 10
BOOT_TAU_START = 0.70
BOOT_TAU_END = 0.90
BOOT_WRAP_MAX = 1
BOOT_BOUNDARY_RADIUS = 0.80
BOOT_SAT_CEILING = 0.50
BOOT_SPREAD_FLOOR = 0.05
BOOT_MOE_FLOOR = 0.05
BOOT_MIN_PROBE_MOE = 4
BOOT_MIN_TRAIN = 6
BOOT_BEST_MOE_MIN = 0.02
SHOCK_PDBS = frozenset({"1BG1", "2Z6H", "1IVO", "2SHP"})


class ColdBootGuardError(RuntimeError):
    """Refuses start when frontend sha or forbidden init paths fail."""


def sha256_path(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_frontend_bank(path: Path | str, *, expected: str = PINNED_FRONTEND_SHA256) -> str:
    digest = sha256_path(path)
    if digest != expected:
        raise ColdBootGuardError(
            f"frontend bank sha mismatch for {path}: got {digest}, want {expected}"
        )
    return digest


def assert_not_forbidden_ckpt(path: Path | str | None) -> None:
    if path is None:
        return
    text = str(path)
    low = text.lower()
    for token in FORBIDDEN_CKPT_SUBSTRINGS:
        if token.lower() in low:
            raise ColdBootGuardError(f"forbidden init checkpoint path: {path} (matched {token})")


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
    if not pdb_id or not theme or role not in {"train", "probe"}:
        raise ValueError(f"invalid cold-boot protein row: {raw}")
    return {
        "pdb_id": pdb_id,
        "chain": chain,
        "theme": theme,
        "gene": str(raw.get("gene") or ""),
        "role": role,
        "enabled": True,
        "flags": _row_flags(raw),
    }


def load_boot_split(manifest_path: Path | str | None = None) -> dict[str, list[dict[str, Any]]]:
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    with path.open() as f:
        data = json.load(f)
    proteins = data.get("proteins", [])
    train: list[dict[str, Any]] = []
    probe: list[dict[str, Any]] = []
    for raw in proteins:
        if not raw.get("enabled", True):
            continue
        entry = _as_entry(raw)
        if entry["pdb_id"] in SHOCK_PDBS:
            raise ValueError(f"shock PDB {entry['pdb_id']} must not appear in cold boot")
        if entry["role"] == "train":
            train.append(entry)
        else:
            probe.append(entry)
    if len(train) != 8:
        raise ValueError(f"cold boot train must have 8 rows, got {len(train)}")
    if len(probe) != 6:
        raise ValueError(f"cold boot probe must have 6 rows, got {len(probe)}")
    train_ids = {(p["pdb_id"], p["chain"]) for p in train}
    probe_ids = {(p["pdb_id"], p["chain"]) for p in probe}
    if train_ids & probe_ids:
        raise ValueError(f"train/probe overlap: {train_ids & probe_ids}")
    return {"train": train, "probe": probe}


def boot_hygiene_pass(
    probe_rows: Sequence[Mapping[str, Any]],
    home_row: Mapping[str, Any] | None,
    *,
    sat_ceiling: float = BOOT_SAT_CEILING,
    spread_floor: float = BOOT_SPREAD_FLOOR,
    moe_floor: float = BOOT_MOE_FLOOR,
    min_probe_moe: int = BOOT_MIN_PROBE_MOE,
) -> dict[str, Any]:
    loaded = [r for r in probe_rows if r.get("loaded")]
    finite = all(
        bool(r.get("z_hyp_finite")) and bool(r.get("curvature_finite")) for r in loaded
    )
    home_ok = bool(
        home_row
        and home_row.get("loaded")
        and home_row.get("z_hyp_finite")
        and home_row.get("curvature_finite")
    )
    sats = [
        float(r["boundary_saturation"])
        for r in loaded
        if r.get("boundary_saturation") is not None
    ]
    mean_sat = float(sum(sats) / len(sats)) if sats else 1.0
    spreads = [
        float(r["radius_spread"])
        for r in loaded
        if r.get("radius_spread") is not None and math.isfinite(float(r["radius_spread"]))
    ]
    mean_spread = float(sum(spreads) / len(spreads)) if spreads else 0.0
    moe_spread = 0
    for r in loaded:
        moe_min = row_moe_load_min(r)
        if moe_min is not None and float(moe_min) > float(moe_floor):
            moe_spread += 1
    passed = (
        len(loaded) == 6
        and finite
        and home_ok
        and mean_sat < float(sat_ceiling)
        and mean_spread >= float(spread_floor)
        and moe_spread >= int(min_probe_moe)
    )
    return {
        "hygiene_pass": passed,
        "n_probe_loaded": len(loaded),
        "probe_finite": finite,
        "home_finite": home_ok,
        "mean_probe_boundary_saturation": mean_sat,
        "mean_probe_radius_spread": mean_spread,
        "n_probe_moe_spread": moe_spread,
        "biology_pass": False,
        "affinity_pass": False,
    }


def epoch_is_best_eligible(last_step: Mapping[str, Any]) -> bool:
    if float(last_step.get("nan_abort") or 0.0) >= 1.0:
        return False
    loss = last_step.get("loss_total")
    if loss is None or not math.isfinite(float(loss)):
        return False
    mean_r = last_step.get("diag_mean_radius")
    z_finite = mean_r is not None and math.isfinite(float(mean_r))
    moe_min = last_step.get("moe_load_min")
    return bool(z_finite) and moe_min is not None and float(moe_min) >= BOOT_BEST_MOE_MIN


def build_boot_stamp(
    probe_rows: Sequence[Mapping[str, Any]],
    home_row: Mapping[str, Any] | None,
    *,
    frontend_sha256: str,
    wrap_max: int,
    boundary_radius: float,
    tau_probe: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hygiene = boot_hygiene_pass(probe_rows, home_row)
    stamp: dict[str, Any] = {
        "schema_version": 1,
        "gate": "tokyo_eye_equ_cold_boot",
        "spec": "docs/superpowers/specs/2026-09-15-tokyoeye-equ-cold-boot-design.md",
        "plan": "docs/superpowers/plans/2026-09-15-tokyoeye-equ-cold-boot.md",
        "display_lineage": "Tokyo Eye EQU",
        **hygiene,
        "tau_probe": float(tau_probe),
        "h3_note": (
            f"boundary_saturation uses PoincareDiagnosticsEngine("
            f"boundary_radius={boundary_radius:.2f}); spread floor "
            f"{BOOT_SPREAD_FLOOR}; MoE floor {BOOT_MOE_FLOOR}."
        ),
        "graph_recipe": {
            "dehydron_wrap_max": int(wrap_max),
            "note": "Frozen wrap_max=1 (Sprint-8 retune). Not re-medianed each epoch.",
        },
        "theta": {
            "init": "equiformer_mptrj_bank_plus_fresh_spine",
            "frontend_bank": "mirror-physics/equiformer_v3 checkpoint/mptrj_gradient.pt",
            "frontend_sha256": frontend_sha256,
            "spine": "fresh_random_init",
            "forbidden_init": list(FORBIDDEN_CKPT_SUBSTRINGS),
        },
        "per_probe": list(probe_rows),
        "kras_home": home_row,
        "forbidden": [
            "champion_retarget",
            "init_from_v5_or_affinity_or_c1",
            "affinity_core_as_boot_pass",
            "pathway_or_resistance",
            "unfreeze_equiformer",
            "hardcoded_curvature",
        ],
    }
    if extra:
        stamp.update(dict(extra))
    return stamp


__all__ = [
    "BOOT_BEST_MOE_MIN",
    "BOOT_BOUNDARY_RADIUS",
    "BOOT_EPOCHS",
    "BOOT_MIN_TRAIN",
    "BOOT_PROBE_EVERY",
    "BOOT_TAU_END",
    "BOOT_TAU_START",
    "BOOT_WRAP_MAX",
    "ColdBootGuardError",
    "DEFAULT_MANIFEST",
    "DEFAULT_STAMP",
    "PINNED_FRONTEND_SHA256",
    "assert_frontend_bank",
    "assert_not_forbidden_ckpt",
    "build_boot_stamp",
    "epoch_is_best_eligible",
    "freeze_entire_frontend",
    "load_boot_split",
    "sha256_path",
    "spine_param_group",
]
