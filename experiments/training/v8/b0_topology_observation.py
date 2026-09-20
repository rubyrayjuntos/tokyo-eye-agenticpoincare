"""TokyoEye B0 topology observation — frozen-weight panel hygiene + theme rhyme.

No optimizer. No 25×150 train. Biology Pass is not a gate on this card.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from science.tokyo_eye.sse_hierarchy import SSERange

THEMES: tuple[str, ...] = (
    "ig_like",
    "lysozyme_like",
    "ubiquitin_grasp",
    "tim_barrel",
    "globin",
    "ploop_ntpase",
)

HELD_OUT_PDBS: frozenset[str] = frozenset({"1BG1", "2Z6H", "1IVO", "2SHP"})

# Globin lesson is helix packing; sandwich / barrel / grasp / P-loop use sheets.
THEME_CLASS_METRIC: dict[str, str] = {
    "ig_like": "sheet",
    "lysozyme_like": "sheet",
    "ubiquitin_grasp": "sheet",
    "tim_barrel": "sheet",
    "globin": "helix",
    "ploop_ntpase": "sheet",
}

CONTRAST_THEME: dict[str, str] = {
    "ig_like": "globin",
    "lysozyme_like": "globin",
    "ubiquitin_grasp": "globin",
    "tim_barrel": "globin",
    "globin": "ig_like",
    "ploop_ntpase": "globin",
}

DEFAULT_MANIFEST = Path("manifests/v8_b0_topology_observation_v1.json")
DEFAULT_STAMP = Path("data/gates/tokyo_eye_v8_b0_topology_observation.json")
DEFAULT_MODE_C = Path(
    "checkpoints/tokyoeye/runs/eqf_mode_c_s9_20260823/tokyoeye_best.pt"
)

COLLAPSE_RANGE = 0.05
RHYME_WITHIN_STD = 0.12
CONTRAST_GAP = 0.08
SINGLETON_GAP = 0.20


def load_panel(manifest_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load enabled B0 rows with theme + flags intact."""
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    with path.open() as f:
        data = json.load(f)
    proteins = data.get("proteins", [])
    if not isinstance(proteins, list):
        raise ValueError(f"manifest {path} missing proteins[] list")
    out: list[dict[str, Any]] = []
    for raw in proteins:
        if not raw.get("enabled", True):
            continue
        pdb_id = str(raw.get("pdb_id") or "").strip().upper()
        chain = str(raw.get("chain") or "A").strip()
        theme = str(raw.get("theme") or "").strip()
        if not pdb_id or theme not in THEMES:
            raise ValueError(f"invalid B0 protein row in {path}: {raw}")
        flags = {
            "grasp_cousin": bool(raw.get("grasp_cousin", False)),
            "nmr_model1": bool(raw.get("nmr_model1", False)),
            "first_ca_polymer": bool(raw.get("first_ca_polymer", False)),
            "fallback_pdb_id": str(raw.get("fallback_pdb_id") or "").strip().upper(),
            "fallback_chain": str(raw.get("fallback_chain") or "").strip(),
        }
        out.append(
            {
                "pdb_id": pdb_id,
                "chain": chain,
                "theme": theme,
                "gene": str(raw.get("gene") or ""),
                "enabled": True,
                "flags": flags,
            }
        )
    if len(out) != 18:
        raise ValueError(f"B0 panel must have 18 enabled rows, got {len(out)} in {path}")
    return out


def ss_class_for_residue(
    resseq: int,
    chain: str,
    ranges: Sequence[SSERange],
) -> str:
    """Sheet wins on overlap; else helix; else coil. Deposited HELIX/SHEET only."""
    in_sheet = False
    in_helix = False
    chain = str(chain).strip()
    for sse in ranges:
        if str(sse.chain).strip() != chain:
            continue
        if not (int(sse.start_resseq) <= int(resseq) <= int(sse.end_resseq)):
            continue
        if sse.sse_type == "E":
            in_sheet = True
        elif sse.sse_type == "H":
            in_helix = True
    if in_sheet:
        return "sheet"
    if in_helix:
        return "helix"
    return "coil"


def _nanmean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def rho_by_ss(
    rho: Sequence[float],
    ss: Sequence[str],
    tau: Sequence[float] | None = None,
) -> dict[str, float]:
    """Mean ρ (and optional wrap τ) per DSSP class. Empty class → NaN, not 0."""
    if len(rho) != len(ss):
        raise ValueError("rho and ss must be the same length")
    if tau is not None and len(tau) != len(rho):
        raise ValueError("tau must match rho length")
    buckets: dict[str, list[tuple[float, float]]] = {
        "helix": [],
        "sheet": [],
        "coil": [],
    }
    for i, label in enumerate(ss):
        key = str(label)
        if key not in buckets:
            key = "coil"
        t = float(tau[i]) if tau is not None else float("nan")
        buckets[key].append((float(rho[i]), t))
    out: dict[str, float] = {}
    for cls in ("helix", "sheet", "coil"):
        pairs = buckets[cls]
        out[f"n_{cls}"] = float(len(pairs))
        out[f"mean_rho_{cls}"] = _nanmean(p[0] for p in pairs)
        out[f"mean_tau_{cls}"] = _nanmean(p[1] for p in pairs)
    return out


def _class_metric_value(row: Mapping[str, Any], theme: str) -> float:
    kind = THEME_CLASS_METRIC.get(theme, "sheet")
    return float(row.get(f"mean_rho_{kind}", float("nan")))


def _ss_rho_range(row: Mapping[str, Any]) -> float:
    vals = [
        float(row[k])
        for k in ("mean_rho_helix", "mean_rho_sheet", "mean_rho_coil")
        if k in row and row[k] is not None and math.isfinite(float(row[k]))
    ]
    if len(vals) < 2:
        return float("nan")
    return float(max(vals) - min(vals))


def _finite_std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return float("nan")
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return float(math.sqrt(var))


def theme_rhyme_narratives(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Observation-only labels: rhyme / singleton / collapse / mixed / insufficient."""
    loaded = [r for r in rows if r.get("loaded", True)]
    by_theme: dict[str, list[Mapping[str, Any]]] = {t: [] for t in THEMES}
    for row in loaded:
        theme = str(row.get("theme") or "")
        if theme in by_theme:
            by_theme[theme].append(row)

    theme_means: dict[str, float] = {}
    for theme, members in by_theme.items():
        theme_means[theme] = _nanmean(
            _class_metric_value(m, theme) for m in members
        )

    out: dict[str, dict[str, Any]] = {}
    for theme in THEMES:
        members = by_theme[theme]
        n = len(members)
        if n < 2:
            out[theme] = {
                "label": "insufficient",
                "n_loaded": n,
                "narrative": (
                    f"{theme}: {n} of 3 loaded — theme still recorded, "
                    "not graded (need ≥2)."
                ),
            }
            continue
        class_vals = [_class_metric_value(m, theme) for m in members]
        within = _finite_std(class_vals)
        ranges = [_ss_rho_range(m) for m in members]
        all_collapse = bool(ranges) and all(
            math.isfinite(r) and r < COLLAPSE_RANGE for r in ranges
        )
        contrast = CONTRAST_THEME.get(theme)
        contrast_mean = theme_means.get(contrast, float("nan")) if contrast else float("nan")
        own_mean = theme_means.get(theme, float("nan"))
        gap = (
            abs(own_mean - contrast_mean)
            if math.isfinite(own_mean) and math.isfinite(contrast_mean)
            else float("nan")
        )
        finite_vals = [v for v in class_vals if math.isfinite(v)]
        singleton = False
        singleton_pdb = ""
        if len(finite_vals) >= 2:
            for i, val in enumerate(class_vals):
                if not math.isfinite(val):
                    continue
                others = [v for j, v in enumerate(class_vals) if j != i and math.isfinite(v)]
                if not others:
                    continue
                others_sorted = sorted(others)
                mid = others_sorted[len(others_sorted) // 2]
                if abs(val - mid) >= SINGLETON_GAP:
                    singleton = True
                    singleton_pdb = str(members[i].get("pdb_id") or "")
                    break
        if all_collapse:
            label = "collapse"
            narrative = (
                f"{theme}: every SS class sits in the same ρ band "
                f"(range < {COLLAPSE_RANGE:.2f}) — no geometric differentiation."
            )
        elif singleton:
            label = "singleton"
            narrative = (
                f"{theme}: only {singleton_pdb or 'one chain'} sits in the "
                f"theme-class ρ band; the others do not rhyme."
            )
        elif (
            math.isfinite(within)
            and within < RHYME_WITHIN_STD
            and math.isfinite(gap)
            and gap >= CONTRAST_GAP
        ):
            label = "rhyme"
            narrative = (
                f"{theme}: three-chain {THEME_CLASS_METRIC[theme]} ρ band "
                f"std={within:.3f}; contrast vs {contrast} gap={gap:.3f}."
            )
        else:
            label = "mixed"
            narrative = (
                f"{theme}: within-std={within if math.isfinite(within) else float('nan'):.3f}, "
                f"contrast_gap={gap if math.isfinite(gap) else float('nan'):.3f} "
                "— not a clean rhyme, singleton, or collapse."
            )
        out[theme] = {
            "label": label,
            "n_loaded": n,
            "within_std": within if math.isfinite(within) else None,
            "contrast_theme": contrast,
            "contrast_gap": gap if math.isfinite(gap) else None,
            "class_metric": THEME_CLASS_METRIC[theme],
            "narrative": narrative,
        }
    return out


def kras_home_narrative(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Observation: 4OBE vs other P-loop chains. Not a pathway claim."""
    ploop = [
        r
        for r in rows
        if r.get("loaded", True)
        and str(r.get("theme")) == "ploop_ntpase"
        and str(r.get("theta", "champion")) == "champion"
    ]
    kras = [r for r in ploop if str(r.get("pdb_id")) == "4OBE"]
    others = [r for r in ploop if str(r.get("pdb_id")) != "4OBE"]
    if not kras:
        return {
            "label": "kras_unloaded",
            "narrative": "4OBE did not load; KRAS home check skipped.",
        }
    k = kras[0]
    k_ok = bool(k.get("z_hyp_finite")) and bool(k.get("curvature_finite"))
    other_ok = [
        bool(r.get("z_hyp_finite")) and bool(r.get("curvature_finite")) for r in others
    ]
    if k_ok and other_ok and all(other_ok):
        label = "home_and_cousins_finite"
        narrative = (
            "4OBE H2 is finite, and so are the other loaded P-loop chains "
            f"({', '.join(str(r.get('pdb_id')) for r in others)}). "
            "Not a pathway claim."
        )
    elif k_ok and (not others or not any(other_ok)):
        label = "kras_only_sane"
        narrative = (
            "4OBE is the only loaded P-loop with finite z_hyp/c. "
            "Observation only — not a KRAS-specialist conclusion."
        )
    else:
        label = "ploop_hygiene_mixed"
        narrative = "P-loop H2 hygiene is mixed across KRAS and cousins."
    return {"label": label, "narrative": narrative, "n_ploop_loaded": len(ploop)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def build_stamp(
    rows: Sequence[Mapping[str, Any]],
    *,
    champion_sha256: str,
    champion_version: int | str,
    mode_c_s9_status: str,
    champion_run_id: str | None = None,
) -> dict[str, Any]:
    """Hygiene stamp. Explicitly not a biology Pass."""
    champion_rows = [r for r in rows if str(r.get("theta", "champion")) == "champion"]
    loaded = [r for r in champion_rows if r.get("loaded")]
    skipped = [r for r in champion_rows if not r.get("loaded")]
    hygiene = True
    for row in loaded:
        if not row.get("z_hyp_finite") or not row.get("curvature_finite"):
            hygiene = False
            break
    if not loaded:
        hygiene = False
    narratives = theme_rhyme_narratives(champion_rows)
    stamp = {
        "schema_version": 1,
        "gate": "tokyo_eye_v8_b0_topology_observation",
        "spec": "docs/superpowers/specs/2026-08-24-tokyoeye-b0-topology-observation-design.md",
        "biology_pass": False,
        "biology_gate": "not_applicable",
        "hygiene_finite": hygiene,
        "n_panel": 18,
        "n_loaded": len(loaded),
        "n_skip": len(skipped),
        "held_out": sorted(HELD_OUT_PDBS),
        "theta": {
            "alias": "champion",
            "uri": "models:/TokyoEye@champion",
            "version": champion_version,
            "sha256": champion_sha256,
            "run_id": champion_run_id,
        },
        "mode_c_s9": {"status": mode_c_s9_status, "label": "compare_only"},
        "theme_narratives": narratives,
        "kras_home": kras_home_narrative(champion_rows),
        "per_pdb": list(rows),
        "forbidden": [
            "allosteric_site_claim",
            "kras_understanding_claim",
            "pathway_or_resistance",
            "disc_alone_hub_rank",
            "evidential_investigation_as_h5",
            "hardcoded_curvature",
        ],
    }
    return _json_safe(stamp)


__all__ = [
    "CONTRAST_THEME",
    "DEFAULT_MANIFEST",
    "DEFAULT_MODE_C",
    "DEFAULT_STAMP",
    "HELD_OUT_PDBS",
    "THEME_CLASS_METRIC",
    "THEMES",
    "build_stamp",
    "kras_home_narrative",
    "load_panel",
    "rho_by_ss",
    "ss_class_for_residue",
    "theme_rhyme_narratives",
]
