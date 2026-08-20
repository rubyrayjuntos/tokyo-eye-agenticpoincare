"""Cheap pre-reg: does monopole maj have low local-persistence variance?

Checks whether dehydron-side residues piled into the committed majority are
homogeneous on local H1 persistence / wrap-deficit (edge barcode channel) vs
genuinely varied — the lever test for board-enrichment with persistence, not
another loss-pressure mechanism.

Uses stack seed2 ge30 (relative purity clean + known monopole).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.diagnostics.prototype_repulsion_epoch import (
    committed_majority_partition,
)
from experiments.training.v6.corpus import load_training_proteins
from experiments.training.v6.train_loop import prepare_training_batch
from science.dtie.common.dehydron_barcode_features import EDGE_BARCODE_NAMES
from science.training.gnn_lineage import load_model_from_checkpoint

DEVICE = "cuda"
BARCODE_DIR = Path("/app/checkpoints/v66/dehydron_barcode_v1")
CKPT = Path(
    "/app/checkpoints/v66/runs/"
    "fix1_s4_proto_repulsion_scale_l2_stage_a12_cold_seed2_v1/epochs/epoch_030.pt"
)
OUT_CANDIDATES = [
    Path(
        "/app/checkpoints/v66/diagnostics/proto_repulsion_scale_l2_stack/"
        "monopole_persistence_variance_audit.json"
    ),
    Path("/tmp/dtie_pdb_cache/monopole_persistence_variance_audit.json"),
    Path("/tmp/monopole_persistence_variance_audit.json"),
]

H1_I = EDGE_BARCODE_NAMES.index("local_h1_persistence")
WRAP_I = EDGE_BARCODE_NAMES.index("wrap_deficit")
NN1_I = EDGE_BARCODE_NAMES.index("witness_nn1_norm")
DENS_I = EDGE_BARCODE_NAMES.index("log1p_local_density")


def _std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(x.std(ddof=1))


def _mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _cv(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    m = float(x.mean())
    if abs(m) < 1e-12:
        return float("nan")
    return float(x.std(ddof=1) / abs(m))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 4:
        return float("nan")
    if float(a[m].std()) < 1e-12 or float(b[m].std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / max(a.size + b.size - 2, 1)
    )
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def per_residue_edge_agg(
    n: int, edge_pairs: np.ndarray, edge_scalars: np.ndarray
) -> dict[str, np.ndarray]:
    """Max / mean of local edge barcode dims over dehydrons touching each node."""
    max_h1 = np.full(n, np.nan, dtype=np.float64)
    mean_h1 = np.full(n, np.nan, dtype=np.float64)
    max_wrap = np.full(n, np.nan, dtype=np.float64)
    mean_wrap = np.full(n, np.nan, dtype=np.float64)
    max_nn1 = np.full(n, np.nan, dtype=np.float64)
    max_dens = np.full(n, np.nan, dtype=np.float64)
    n_touch = np.zeros(n, dtype=np.int32)
    buckets: list[list[int]] = [[] for _ in range(n)]
    for e, (i, j) in enumerate(edge_pairs):
        i, j = int(i), int(j)
        if 0 <= i < n:
            buckets[i].append(e)
        if 0 <= j < n and j != i:
            buckets[j].append(e)
    for r, edges in enumerate(buckets):
        if not edges:
            continue
        n_touch[r] = len(edges)
        h1 = edge_scalars[edges, H1_I]
        wr = edge_scalars[edges, WRAP_I]
        nn = edge_scalars[edges, NN1_I]
        dens = edge_scalars[edges, DENS_I]
        max_h1[r] = float(np.max(h1))
        mean_h1[r] = float(np.mean(h1))
        max_wrap[r] = float(np.max(wr))
        mean_wrap[r] = float(np.mean(wr))
        max_nn1[r] = float(np.max(nn))
        max_dens[r] = float(np.max(dens))
    return {
        "max_local_h1": max_h1,
        "mean_local_h1": mean_h1,
        "max_wrap_deficit": max_wrap,
        "mean_wrap_deficit": mean_wrap,
        "max_witness_nn1": max_nn1,
        "max_local_density": max_dens,
        "n_dehydrons_touching_edge": n_touch.astype(np.float64),
    }


def load_barcode(pdb_id: str, chain: str) -> dict | None:
    path = BARCODE_DIR / f"{pdb_id.upper()}_{chain}_dehydron_barcode_v1.pt"
    if not path.exists():
        # try without forcing chain letter case
        cands = list(BARCODE_DIR.glob(f"{pdb_id.upper()}_*_dehydron_barcode_v1.pt"))
        if not cands:
            return None
        path = cands[0]
    z = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "edge_pairs": z["edge_pairs"].numpy(),
        "edge_scalars": z["edge_scalars"].numpy(),
        "scalars": z["scalars"].numpy(),
        "path": str(path),
    }


def main() -> None:
    proteins, _ = load_training_proteins(
        Path("/tmp/dtie_pdb_cache"),
        Path("/app/manifests/v6_corpus_stage_a_small_v1.json"),
        max_proteins=12,
    )
    model = load_model_from_checkpoint(CKPT, DEVICE)
    model.to(DEVICE)
    model.core_capacity_quota_tau = 0.0
    model.eval()

    rows: list[dict] = []
    inventory = {
        "residue_scalar_channel": {
            "exists": True,
            "path_pattern": str(BARCODE_DIR / "{PDB}_{CHAIN}_dehydron_barcode_v1.pt"),
            "names": [
                "n_bars",
                "n_h1_bars",
                "total_persistence",
                "max_persistence",
                "mean_persistence",
                "std_persistence",
                "frac_long_lived",
                "mean_birth_h1",
                "mean_death_h1",
                "max_h1_persistence",
                "n_dehydrons_touching",
            ],
            "caveat": (
                "Most residue scalars are structure-global broadcast to "
                "dehydron-touching residues; only n_dehydrons_touching is "
                "clearly local among the 11. Not a within-population splitter "
                "by itself."
            ),
        },
        "edge_local_channel": {
            "exists": True,
            "names": EDGE_BARCODE_NAMES,
            "note": (
                "Per-dehydron-pair witness features: local_h1_persistence, "
                "wrap_deficit, witness_nn1_norm, log1p_local_density — the "
                "real geometric persistence differentiation signal."
            ),
        },
        "cone_depth_x_epistemic_sieve": {
            "on_gate_board": True,
            "cone_depth": "already in topology board",
            "epistemic_sigma_sieve": (
                "narrative / abandoned uncertainty version — not a shipped "
                "structural-physics residue feature for board enrichment"
            ),
            "replacement": "dehydron barcode edge local_h1_persistence (+ wrap_deficit)",
        },
        "use_dehydron_barcode_on_stack_baseline": False,
    }

    with torch.no_grad():
        for prot in proteins:
            pdb = str(prot.get("pdb_id") or "?").upper()
            chain = str(prot.get("chain") or "A")
            data = prepare_training_batch(model, prot, DEVICE)
            in_f = int(getattr(model.node_emb, "in_features", data.x.size(-1)))
            if data.x.size(-1) > in_f:
                data.x = data.x[:, :in_f].contiguous()
            out = model(data)
            w = out["expert_weights"].float()
            n = int(w.shape[0])
            max_p = w.max(dim=-1).values.cpu().numpy()
            hard = w.argmax(dim=-1).cpu().numpy()
            committed = max_p >= 0.60
            dh = prot.get("target_dehydron")
            if dh is None and data.x.size(1) > 1:
                dh = data.x[:, 1]
            if dh is None:
                dh = torch.zeros(n)
            dh = dh.float().reshape(-1)[:n].cpu().numpy()
            is_dh = dh > 0.5

            part = committed_majority_partition(w.cpu().numpy(), dh)
            maj_e = part["majority_expert"]
            if maj_e is None or part["n_committed"] < 20:
                continue
            maj = committed & (hard == int(maj_e))
            oth = committed & (hard != int(maj_e))
            share = float(part["majority_share"])
            maj_dh = float(dh[maj].mean()) if maj.any() else float("nan")
            maj_class = "dehydron" if maj_dh >= 0.5 else "core"

            bc = load_barcode(pdb, chain)
            if bc is None:
                rows.append(
                    {
                        "pdb_id": pdb,
                        "committed_hard_share_max": share,
                        "monopole": share >= 0.56,
                        "maj_class": maj_class,
                        "maj_dh": maj_dh,
                        "barcode_missing": True,
                    }
                )
                continue

            agg = per_residue_edge_agg(n, bc["edge_pairs"], bc["edge_scalars"])
            # Restrict persistence stats to residues that touch ≥1 dehydron edge
            touch = agg["n_dehydrons_touching_edge"] > 0

            def stats(mask: np.ndarray, key: str) -> dict:
                vals = agg[key][mask & touch & np.isfinite(agg[key])]
                return {
                    "n": int(vals.size),
                    "mean": _mean(vals),
                    "std": _std(vals),
                    "cv": _cv(vals),
                    "p10": float(np.percentile(vals, 10)) if vals.size else float("nan"),
                    "p90": float(np.percentile(vals, 90)) if vals.size else float("nan"),
                    "range_p90_p10": (
                        float(np.percentile(vals, 90) - np.percentile(vals, 10))
                        if vals.size
                        else float("nan")
                    ),
                }

            # Also cone_depth from board if present (already on board — control axis)
            board = getattr(model.gate, "_last_topo_features", None)
            cone = None
            if board is not None and board.shape[1] >= 2:
                cone = board.float().cpu().numpy()[:n, 1]  # cone_depth index in BOARD_BASE

            maj_h1 = stats(maj, "max_local_h1")
            oth_h1 = stats(oth, "max_local_h1")
            all_dh_h1 = stats(is_dh, "max_local_h1")
            maj_wrap = stats(maj, "max_wrap_deficit")
            oth_wrap = stats(oth, "max_wrap_deficit")

            # Within dehydron population: does high vs low persistence route differently?
            dh_touch = is_dh & touch
            h1_dh = agg["max_local_h1"][dh_touch]
            hard_dh = hard[dh_touch]
            expert_sep = float("nan")
            if h1_dh.size >= 10 and len(np.unique(hard_dh)) >= 2:
                # correlation of persistence with one-hot maj membership among dehydrons
                on_maj = (hard_dh == int(maj_e)).astype(np.float64)
                expert_sep = _corr(h1_dh, on_maj)

            d_h1_maj_vs_oth = _cohen_d(
                agg["max_local_h1"][maj & touch],
                agg["max_local_h1"][oth & touch],
            )
            d_wrap_maj_vs_oth = _cohen_d(
                agg["max_wrap_deficit"][maj & touch],
                agg["max_wrap_deficit"][oth & touch],
            )

            row = {
                "pdb_id": pdb,
                "barcode_path": bc["path"],
                "committed_hard_share_max": share,
                "monopole": share >= 0.56,
                "maj_class": maj_class,
                "maj_dh": maj_dh,
                "n_maj": int(maj.sum()),
                "n_oth_committed": int(oth.sum()),
                "n_dh": int(is_dh.sum()),
                "n_maj_touch_dehydron": int((maj & touch).sum()),
                "frac_maj_touch_dehydron": float((maj & touch).mean()) if maj.any() else 0.0,
                "max_local_h1_maj": maj_h1,
                "max_local_h1_oth_committed": oth_h1,
                "max_local_h1_all_dehydron": all_dh_h1,
                "wrap_deficit_maj": maj_wrap,
                "wrap_deficit_oth_committed": oth_wrap,
                "cohen_d_h1_maj_vs_oth_touch": d_h1_maj_vs_oth,
                "cohen_d_wrap_maj_vs_oth_touch": d_wrap_maj_vs_oth,
                "corr_h1_vs_on_maj_among_dehydrons": expert_sep,
                "ratio_std_h1_maj_over_all_dh": (
                    maj_h1["std"] / (all_dh_h1["std"] + 1e-12)
                    if np.isfinite(maj_h1["std"]) and np.isfinite(all_dh_h1["std"])
                    else float("nan")
                ),
            }
            if cone is not None:
                row["cone_depth_std_maj"] = _std(cone[maj])
                row["cone_depth_std_oth"] = _std(cone[oth]) if oth.any() else float("nan")
                row["cone_depth_std_all"] = _std(cone)
            rows.append(row)

    # Aggregate verdicts
    mono = [r for r in rows if r.get("monopole") and not r.get("barcode_missing")]
    non = [r for r in rows if not r.get("monopole") and not r.get("barcode_missing")]
    dh_maj_mono = [r for r in mono if r.get("maj_class") == "dehydron"]
    core_maj_mono = [r for r in mono if r.get("maj_class") == "core"]

    def collect(subset: list[dict], key: str) -> np.ndarray:
        out = []
        for r in subset:
            # nested keys like max_local_h1_maj.std
            if "." in key:
                a, b = key.split(".", 1)
                v = r.get(a, {})
                if isinstance(v, dict):
                    out.append(v.get(b, float("nan")))
                else:
                    out.append(float("nan"))
            else:
                out.append(r.get(key, float("nan")))
        return np.asarray(out, dtype=np.float64)

    shares = collect(rows, "committed_hard_share_max")
    std_maj = collect(
        [r for r in rows if not r.get("barcode_missing")],
        "max_local_h1_maj.std",
    )
    std_all = collect(
        [r for r in rows if not r.get("barcode_missing")],
        "max_local_h1_all_dehydron.std",
    )
    ratio_std = collect(
        [r for r in rows if not r.get("barcode_missing")],
        "ratio_std_h1_maj_over_all_dh",
    )
    corr_sep = collect(
        [r for r in rows if not r.get("barcode_missing")],
        "corr_h1_vs_on_maj_among_dehydrons",
    )
    frac_touch = collect(
        [r for r in rows if not r.get("barcode_missing")],
        "frac_maj_touch_dehydron",
    )

    # Among dehydron-majority monopoles: is maj persistence std collapsed?
    dh_mono_std = collect(dh_maj_mono, "max_local_h1_maj.std")
    dh_mono_all = collect(dh_maj_mono, "max_local_h1_all_dehydron.std")
    dh_mono_ratio = collect(dh_maj_mono, "ratio_std_h1_maj_over_all_dh")

    corr_share_vs_maj_std = _corr(
        collect([r for r in rows if not r.get("barcode_missing")], "committed_hard_share_max"),
        std_maj,
    )
    corr_share_vs_ratio = _corr(
        collect([r for r in rows if not r.get("barcode_missing")], "committed_hard_share_max"),
        ratio_std,
    )

    mean_frac_touch_mono = float(np.nanmean(collect(mono, "frac_maj_touch_dehydron"))) if mono else float("nan")
    mean_frac_touch_core_mono = (
        float(np.nanmean(collect(core_maj_mono, "frac_maj_touch_dehydron")))
        if core_maj_mono
        else float("nan")
    )

    # Informative dehydron-maj monopoles: exclude structure-wide H1 collapse
    # (std≈0 for all dehydrons) — those are barcode-flat, not maj-homogeneous.
    dh_maj_mono_info = [
        r
        for r in dh_maj_mono
        if np.isfinite((r.get("max_local_h1_all_dehydron") or {}).get("std", float("nan")))
        and float(r["max_local_h1_all_dehydron"]["std"]) > 1e-6
    ]
    dh_maj_mono_deg = [r for r in dh_maj_mono if r not in dh_maj_mono_info]
    info_ratio = collect(dh_maj_mono_info, "ratio_std_h1_maj_over_all_dh")
    info_corr = collect(dh_maj_mono_info, "corr_h1_vs_on_maj_among_dehydrons")
    info_std_maj = collect(dh_maj_mono_info, "max_local_h1_maj.std")
    info_std_all = collect(dh_maj_mono_info, "max_local_h1_all_dehydron.std")

    mean_abs_corr = float(np.nanmean(np.abs(corr_sep))) if corr_sep.size else float("nan")
    mean_ratio_dh_mono = float(np.nanmean(dh_mono_ratio)) if dh_mono_ratio.size else float("nan")
    mean_std_dh_mono = float(np.nanmean(dh_mono_std)) if dh_mono_std.size else float("nan")
    mean_std_dh_all = float(np.nanmean(dh_mono_all)) if dh_mono_all.size else float("nan")
    mean_ratio_info = float(np.nanmean(info_ratio)) if info_ratio.size else float("nan")
    mean_abs_corr_info = float(np.nanmean(np.abs(info_corr))) if info_corr.size else float("nan")

    # Decision logic (refined):
    # A) Strongest monopoles are core maj with tiny dehydron touch → off-population.
    # B) Among informative dh-maj (nonzero structure H1 std): ratio≈1 + low corr
    #    → variance present unused (SASA-class), but may be minority of monopoles.
    # C) Degenerate H1 structures → local_h1 cannot help; wrap_deficit often ≈ rho.
    # D) Do not treat all-zero H1 as "maj homogeneous."

    if len(core_maj_mono) >= 3 and mean_frac_touch_core_mono < 0.20:
        core_worst = max(core_maj_mono, key=lambda r: r["committed_hard_share_max"])
        if (
            dh_maj_mono_info
            and np.isfinite(mean_ratio_info)
            and mean_ratio_info > 0.9
            and np.isfinite(mean_abs_corr_info)
            and mean_abs_corr_info < 0.25
        ):
            verdict = "MIXED_CORE_OFF_POP_AND_DH_VARIANCE_UNUSED"
            rationale = (
                f"Monopoles split: {len(core_maj_mono)} core-maj (mean touch="
                f"{mean_frac_touch_core_mono:.3f}; worst {core_worst['pdb_id']} "
                f"share={core_worst['committed_hard_share_max']:.3f}) where "
                "dehydron persistence cannot split maj; "
                f"{len(dh_maj_mono_info)} informative dh-maj with maj/all H1-std "
                f"ratio≈{mean_ratio_info:.3f} but |corr(h1,on_maj)|≈"
                f"{mean_abs_corr_info:.3f} (unused); "
                f"{len(dh_maj_mono_deg)} H1-flat. Hold blanket board registration."
            )
        else:
            verdict = "MONOPOLE_MOSTLY_CORE_PERSISTENCE_OFF_POPULATION"
            rationale = (
                "Committed majority on several monopoles is core "
                f"(mean frac_maj_touch on core-maj={mean_frac_touch_core_mono:.3f}). "
                "Local dehydron persistence is off-population for those piles."
            )
    elif (
        dh_maj_mono_info
        and np.isfinite(mean_ratio_info)
        and mean_ratio_info < 0.7
    ):
        verdict = "DEHYDRON_MAJ_HOMOGENEOUS_ON_PERSISTENCE"
        rationale = (
            f"Informative dehydron-majority monopoles show maj H1 std / all-dh "
            f"std ≈ {mean_ratio_info:.3f} (<0.7): maj pile disproportionately "
            "homogeneous on local persistence."
        )
    elif (
        dh_maj_mono_info
        and np.isfinite(mean_ratio_info)
        and mean_ratio_info > 0.9
        and np.isfinite(mean_abs_corr_info)
        and mean_abs_corr_info < 0.25
    ):
        verdict = "PERSISTENCE_VARIANCE_PRESENT_UNUSED"
        rationale = (
            f"Informative dh-maj monopoles retain local-H1 variance "
            f"(mean maj std={float(np.nanmean(info_std_maj)):.4g} vs all-dh "
            f"{float(np.nanmean(info_std_all)):.4g}, ratio≈{mean_ratio_info:.3f}) "
            f"but |corr(h1, on_maj)|≈{mean_abs_corr_info:.3f}: unused — SASA-class "
            "only on this slice."
        )
    elif np.isfinite(mean_abs_corr) and mean_abs_corr >= 0.35:
        verdict = "PERSISTENCE_ALREADY_PROXY_SEPARATED"
        rationale = (
            f"Among dehydrons, |corr(local_h1, on_maj)| mean abs={mean_abs_corr:.3f}: "
            "routing already tracks a persistence-like axis via existing board."
        )
    else:
        verdict = "WEAK_OR_MIXED_PERSISTENCE_MONOPOLE_LINK"
        rationale = (
            "No clean corpus-wide pattern across core-maj / H1-flat / informative "
            "dh-maj slices. Do not draft board registration yet."
        )

    report = {
        "checkpoint": str(CKPT),
        "inventory": inventory,
        "verdict": verdict,
        "rationale": rationale,
        "n_structures": len(rows),
        "n_monopole": len(mono),
        "n_dehydron_maj_monopole": len(dh_maj_mono),
        "n_core_maj_monopole": len(core_maj_mono),
        "aggregates": {
            "corr_share_vs_maj_h1_std": corr_share_vs_maj_std,
            "corr_share_vs_ratio_std_maj_over_all_dh": corr_share_vs_ratio,
            "mean_frac_maj_touch_dehydron_monopole": mean_frac_touch_mono,
            "mean_frac_maj_touch_dehydron_core_maj_monopole": mean_frac_touch_core_mono,
            "mean_abs_corr_h1_vs_on_maj_among_dehydrons": mean_abs_corr,
            "mean_ratio_std_h1_maj_over_all_dh_dehydron_maj_mono": mean_ratio_dh_mono,
            "mean_std_h1_maj_dehydron_maj_mono": mean_std_dh_mono,
            "mean_std_h1_all_dh_dehydron_maj_mono": mean_std_dh_all,
            "mean_maj_h1_std_all_structures": float(np.nanmean(std_maj)),
            "mean_all_dh_h1_std_all_structures": float(np.nanmean(std_all)),
            "n_dehydron_maj_monopole_informative_h1": len(dh_maj_mono_info),
            "n_dehydron_maj_monopole_h1_flat": len(dh_maj_mono_deg),
            "mean_ratio_std_h1_informative_only": mean_ratio_info,
            "mean_abs_corr_h1_on_maj_informative_only": mean_abs_corr_info,
        },
        "per_structure": rows,
    }
    summary = {
        "verdict": verdict,
        "rationale": rationale,
        "aggregates": report["aggregates"],
        "n_monopole": report["n_monopole"],
        "n_dehydron_maj_monopole": report["n_dehydron_maj_monopole"],
        "n_core_maj_monopole": report["n_core_maj_monopole"],
    }
    print(json.dumps(summary, indent=2))
    payload = json.dumps(report, indent=2)
    written = None
    errors = []
    for out in OUT_CANDIDATES:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload)
            written = out
            break
        except OSError as exc:
            errors.append(f"{out}: {exc}")
    if written is None:
        raise RuntimeError("could not write audit json: " + "; ".join(errors))
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
