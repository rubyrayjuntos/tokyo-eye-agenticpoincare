"""Frozen-weight B0 topology observation on models:/TokyoEye@champion.

Does not train, retarget aliases, or open Sprint 10.2 / B1.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.training.v8.b0_topology_observation import (
    DEFAULT_MANIFEST,
    DEFAULT_MODE_C,
    DEFAULT_STAMP,
    build_stamp,
    load_panel,
    rho_by_ss,
    ss_class_for_residue,
)
from experiments.training.v8.export_viewers import ball_to_disc
from science.dtie.common.curvature_values import require_learned_curvature
from science.tokyo_eye.governance.resolve import resolve_alias_checkpoint
from science.tokyo_eye.governance.vault import sha256_file
from science.tokyo_eye.sse_hierarchy import parse_pdb_helix_sheet
from science.tokyo_eye.v8.biophysics import parse_residue_records_from_pdb_chain
from science.tokyo_eye.v8.engine import PoincareDiagnosticsEngine
from science.tokyo_eye.v8.equiformer_frontend import (
    DEFAULT_WEIGHT_MAP,
    StubEquiformerFrontend,
    TokyoEyeV8WithFrontend,
    evaluate_geometry_health,
    load_weight_map,
)
from science.tokyo_eye.v8.loader import (
    DEFAULT_PDB_DIR,
    batch_from_records,
    ensure_pdb_cached,
    load_structure_batch,
)
from science.tokyo_eye.v8.metrics import binary_auprc
from science.tokyo_eye.v8.model import TokyoEyesHyperbolicV8
from science.tokyo_eye.v8.r0_r5_graph import (
    R1_HBOND,
    R2_DEHYDRON,
    build_r0_r5_graph,
    get_dehydron_wrap_max,
)
from science.tokyo_eye.v8.types import AtomRecord, ResidueRecord

MLFLOW_EXPERIMENT = "tokyoeye/equiformer-v3-moe/geometric/full-stack"
MLFLOW_RUN_NAME = "b0_topology_observation_champion"


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[b0] WARN: CUDA unavailable — falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def _learned_curvature(blob: Any, spine: TokyoEyesHyperbolicV8) -> float:
    if isinstance(blob, dict):
        for key in ("learned_curvature", "curvature", "curvature_c"):
            if blob.get(key) is not None:
                return require_learned_curvature(blob[key], context=f"ckpt.{key}")
        cfg = blob.get("cfg") or blob.get("config") or {}
        if isinstance(cfg, dict):
            for key in ("curvature", "curvature_c", "c"):
                if cfg.get(key) is not None:
                    return require_learned_curvature(
                        cfg[key], context=f"ckpt.cfg.{key}"
                    )
    return require_learned_curvature(getattr(spine, "c", None), context="spine.c")


def _load_blob(path: Path, device: torch.device) -> Any:
    return torch.load(path, map_location=device, weights_only=False)


def _build_system(
    cfg: dict[str, Any],
    blob: Any,
    device: torch.device,
) -> tuple[TokyoEyeV8WithFrontend, float]:
    c_init = None
    if isinstance(blob, dict):
        c_init = blob.get("curvature", blob.get("curvature_c"))
        nested = blob.get("cfg") or blob.get("config") or {}
        if c_init is None and isinstance(nested, dict):
            c_init = nested.get("curvature", nested.get("curvature_c", nested.get("c")))
    if c_init is None:
        c_init = cfg.get("curvature_c", cfg.get("c"))
    c = require_learned_curvature(c_init, context="b0 system init")
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
        num_attn_layers=int(cfg.get("num_attn_layers", 2)),
        num_sdrp_classes=int(cfg.get("num_sdrp_classes", 5)),
        c=c,
    )
    system = TokyoEyeV8WithFrontend(frontend, spine).to(device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    system.load_state_dict(state, strict=False)
    system.eval()
    learned = _learned_curvature(blob, spine)
    spine.c = learned
    return system, learned


def _first_ca_complete_chain(pdb_path: Path) -> str | None:
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("b0", str(pdb_path))
    model = next(structure.get_models())
    for chain in model:
        n_ca = 0
        for res in chain.get_residues():
            if res.get_id()[0] != " ":
                continue
            if "CA" in res:
                n_ca += 1
        if n_ca > 0:
            return str(chain.id)
    return None


def _parse_residues_model1(pdb_path: Path, chain_id: str) -> list[ResidueRecord]:
    """NMR: first MODEL only (B0 1A5R). Does not walk other models."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("b0_nmr", str(pdb_path))
    model = next(structure.get_models())
    residues_raw = [
        r
        for r in model.get_residues()
        if r.get_id()[0] == " " and r.parent.id == chain_id
    ]
    records: list[ResidueRecord] = []
    for res in residues_raw:
        res_name = res.get_resname().strip().upper()
        atoms = tuple(
            AtomRecord(
                atom_name=atom.name,
                element=(atom.element or "").upper(),
                coord=np.asarray(atom.coord, dtype=np.float64),
                parent_residue_name=res_name,
            )
            for atom in res.get_atoms()
        )
        records.append(
            ResidueRecord(
                chain_label=chain_id,
                residue_index=int(res.get_id()[1]),
                residue_name=res_name,
                atoms=atoms,
                residue_id=f"{chain_id}:{int(res.get_id()[1])}:",
            )
        )
    return records


def _ca_records(pdb_path: Path, chain: str, *, model1: bool = False):
    records = (
        _parse_residues_model1(pdb_path, chain)
        if model1
        else parse_residue_records_from_pdb_chain(pdb_path, chain)
    )
    return [r for r in records if r.get_atom("CA") is not None]


def _per_node_wrap_tau(records) -> np.ndarray:
    graph = build_r0_r5_graph(records)
    n = int(graph.num_nodes)
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    ei = graph.edge_index
    et = graph.edge_type
    attr = graph.edge_attr
    for e in range(int(et.shape[0])):
        if int(et[e]) not in (R1_HBOND, R2_DEHYDRON):
            continue
        i = int(ei[0, e])
        sums[i] += float(attr[e, 0])
        counts[i] += 1.0
    tau = np.full(n, np.nan, dtype=np.float64)
    mask = counts > 0
    tau[mask] = sums[mask] / counts[mask]
    return tau


def _auprc_or_na(scores: torch.Tensor, labels: torch.Tensor) -> float | str:
    if labels.numel() == 0 or float(labels.sum()) <= 0:
        return "auprc_na"
    value = binary_auprc(scores, labels)
    if not math.isfinite(value):
        return "auprc_na"
    return float(value)


def _helix_sheet_counts(ss: list[str]) -> dict[str, int]:
    return {
        "n_helix": int(sum(1 for x in ss if x == "helix")),
        "n_sheet": int(sum(1 for x in ss if x == "sheet")),
        "n_coil": int(sum(1 for x in ss if x == "coil")),
    }


def observe_one(
    *,
    entry: dict[str, Any],
    system: TokyoEyeV8WithFrontend,
    curvature: float,
    tau_ceil: float,
    device: torch.device,
    pdb_dir: Path,
    theta: str,
    diagnostics: PoincareDiagnosticsEngine | None = None,
) -> dict[str, Any]:
    pdb_id = str(entry["pdb_id"])
    chain = str(entry["chain"])
    flags = dict(entry.get("flags") or {})
    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "chain": chain,
        "theme": entry["theme"],
        "gene": entry.get("gene"),
        "theta": theta,
        "flags": flags,
        "loaded": False,
    }
    try:
        pdb_path = ensure_pdb_cached(pdb_id, pdb_dir)
        model1 = bool(flags.get("nmr_model1"))
        records = _ca_records(pdb_path, chain, model1=model1)
        if not records and flags.get("first_ca_polymer"):
            alt = _first_ca_complete_chain(pdb_path)
            if alt and alt != chain:
                chain = alt
                row["chain"] = chain
                row["chain_resolved"] = "first_ca_polymer"
                records = _ca_records(pdb_path, chain, model1=model1)
        if not records and flags.get("fallback_pdb_id"):
            fb_id = str(flags["fallback_pdb_id"])
            fb_chain = str(flags.get("fallback_chain") or "A")
            pdb_path = ensure_pdb_cached(fb_id, pdb_dir)
            records = _ca_records(pdb_path, fb_chain, model1=model1)
            if records:
                row["pdb_id_requested"] = pdb_id
                pdb_id = fb_id
                chain = fb_chain
                row["pdb_id"] = pdb_id
                row["chain"] = chain
                row["used_fallback"] = True
        if not records:
            row["skip_reason"] = "no_ca_complete"
            return row
        if model1:
            batch = batch_from_records(
                records, pdb_id=pdb_id, chain=chain, device=device
            )
        else:
            batch = load_structure_batch(
                pdb_id,
                chain,
                pdb_dir=pdb_dir,
                device=device,
                graph_cache_dir=pdb_dir / "v8_graph_cache",
            )
        if int(batch["num_nodes"]) != len(records):
            records = _ca_records(pdb_path, chain, model1=model1)
        if int(batch["num_nodes"]) != len(records):
            row["skip_reason"] = "residue_graph_mismatch"
            row["n_res_records"] = len(records)
            row["n_res_graph"] = int(batch["num_nodes"])
            return row
        row["dehydron_wrap_max"] = int(get_dehydron_wrap_max())
        pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
        ranges = parse_pdb_helix_sheet(pdb_text)
        ss = [
            ss_class_for_residue(int(rec.residue_index), rec.chain_label, ranges)
            for rec in records
        ]
        wrap_tau = _per_node_wrap_tau(records)
        with torch.no_grad():
            out = system(
                batch["x"],
                batch["edge_index"],
                batch["edge_type"],
                tau_ceiling=tau_ceil,
            )
        z = out["z_hyp"]
        z_finite = bool(torch.isfinite(z).all().item())
        c_finite = bool(math.isfinite(curvature) and curvature > 0)
        diag_engine = diagnostics or PoincareDiagnosticsEngine()
        diag = diag_engine.summarize(z)
        disc_xy = ball_to_disc(z)
        disc_r = np.linalg.norm(np.asarray(disc_xy, dtype=np.float64), axis=1)
        health = evaluate_geometry_health(
            {
                "diag_radial_entropy": float(diag["radial_entropy"]),
                "diag_radius_spread": float(diag["radius_spread"]),
                "diag_boundary_saturation": float(diag["boundary_saturation"]),
            }
        )
        load = out["moe_aux"].get("load")
        if load is None:
            load = out["moe_aux"]["routing"].mean(dim=0)
        load_list = [float(v) for v in load.detach().tolist()]
        moe = {f"moe_load_e{i}": v for i, v in enumerate(load_list)}
        moe["moe_load_min"] = float(min(load_list) if load_list else 0.0)
        rho = batch["dehydron_labels"].detach().float().cpu().numpy().reshape(-1)
        h5 = rho_by_ss(rho.tolist(), ss, wrap_tau.tolist())
        auprc = _auprc_or_na(out["mechanism_score"], batch["dehydron_labels"])
        row.update(
            {
                "loaded": True,
                "n_res": int(batch["num_nodes"]),
                "z_hyp_finite": z_finite,
                "curvature": float(curvature),
                "curvature_finite": c_finite,
                "curvature_source": "loaded_checkpoint_or_spine",
                "mean_radius": float(diag["mean_radius"]),
                "max_radius": float(diag["max_radius"]),
                "min_radius": float(diag["min_radius"]),
                "radius_spread": float(diag["radius_spread"]),
                "boundary_saturation": float(diag["boundary_saturation"]),
                "mean_disc_r": float(np.mean(disc_r)) if disc_r.size else float("nan"),
                "max_disc_r": float(np.max(disc_r)) if disc_r.size else float("nan"),
                "radial_entropy": float(diag["radial_entropy"]),
                "warn_boundary_saturation": bool(diag["warn_boundary_saturation"]),
                "warn_core_collapse": bool(diag["warn_core_collapse"]),
                "warn_oversmooth": float(health["warn_oversmooth"]),
                "warn_boundary_blowout": float(health["warn_boundary_blowout"]),
                "dehydron_frac": float(batch["dehydron_frac"]),
                "dehydron_auprc": auprc,
                **{k: float(v) for k, v in h5.items()},
                **_helix_sheet_counts(ss),
                **moe,
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001 — skip, do not swap topology
        row["skip_reason"] = f"{type(exc).__name__}: {exc}"
        row["skip_trace"] = traceback.format_exc(limit=4)
        return row
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _log_mlflow(stamp: dict[str, Any], tracking_uri: str | None) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("[b0] mlflow not installed — stamp only")
        return None
    uri = tracking_uri or "http://mlflow:5000"
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:
            mlflow.set_tags(
                {
                    "card": "b0_topology_observation",
                    "alias_untouched": "true",
                    "biology_pass": "false",
                }
            )
            mlflow.log_metrics(
                {
                    "hygiene_finite": 1.0 if stamp["hygiene_finite"] else 0.0,
                    "n_loaded": float(stamp["n_loaded"]),
                    "n_skip": float(stamp["n_skip"]),
                }
            )
            for row in stamp.get("per_pdb") or []:
                if str(row.get("theta")) != "champion" or not row.get("loaded"):
                    continue
                prefix = f"{row['pdb_id']}_{row['chain']}"
                for key in (
                    "mean_radius",
                    "boundary_saturation",
                    "mean_disc_r",
                    "mean_rho_sheet",
                    "mean_rho_helix",
                    "mean_rho_coil",
                    "moe_load_e0",
                    "moe_load_e1",
                    "moe_load_e2",
                    "moe_load_e3",
                ):
                    val = row.get(key)
                    if isinstance(val, (int, float)) and math.isfinite(float(val)):
                        mlflow.log_metric(f"{prefix}_{key}", float(val))
            return str(run.info.run_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[b0] MLflow log skipped: {exc}")
        return None


def run(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="B0 frozen-weight topology observation")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pdb-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tracking-uri", type=str, default="")
    p.add_argument("--out-stamp", type=Path, default=DEFAULT_STAMP)
    p.add_argument("--compare-mode-c", type=Path, default=DEFAULT_MODE_C)
    p.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--champion-ckpt", type=Path, default=None, help="Override only for tests")
    args = p.parse_args(argv)

    pdb_dir = args.pdb_dir
    if pdb_dir is None:
        docker_cache = Path("/tmp/dtie_pdb_cache")
        pdb_dir = docker_cache if docker_cache.is_dir() else DEFAULT_PDB_DIR
    device = _device(args.device)
    cfg = load_weight_map(args.weight_map)
    tau_ceil = float(cfg.get("tau_end", 0.995))
    panel = load_panel(args.manifest)

from experiments.training.v8.run_v8_experiment import _log_r0_r5_graph_recipe

    recipe = _log_r0_r5_graph_recipe(
        pdb_dir=pdb_dir, use_graph_cache=False
    )
    print(f"[b0] graph recipe wrap_max={get_dehydron_wrap_max()} recipe={recipe}")

    if args.champion_ckpt is not None:
        ckpt_path = Path(args.champion_ckpt)
        resolved = {
            "path": ckpt_path,
            "version": "override",
            "run_id": None,
            "alias": "champion",
        }
    else:
        tracking = args.tracking_uri or None
        resolved = resolve_alias_checkpoint(alias="champion", tracking_uri=tracking)
        ckpt_path = Path(resolved["path"])
    digest = sha256_file(ckpt_path)
    blob = _load_blob(ckpt_path, device)
    system, curvature = _build_system(cfg, blob, device)
    print(
        f"[b0] champion v{resolved.get('version')} sha256={digest[:16]}… "
        f"c={curvature} device={device}"
    )

    rows: list[dict[str, Any]] = []
    for entry in panel:
        print(f"[b0] champion {entry['pdb_id']}:{entry['chain']} ({entry['theme']})")
        rows.append(
            observe_one(
                entry=entry,
                system=system,
                curvature=curvature,
                tau_ceil=tau_ceil,
                device=device,
                pdb_dir=pdb_dir,
                theta="champion",
            )
        )

    mode_c_status = "mode_c_s9_absent"
    compare_path = Path(args.compare_mode_c)
    if compare_path.is_file():
        mode_c_status = "mode_c_s9_compare"
        compare_blob = _load_blob(compare_path, device)
        compare_system, compare_c = _build_system(cfg, compare_blob, device)
        print(f"[b0] mode_c_s9 compare c={compare_c}")
        for entry in panel:
            print(f"[b0] mode_c {entry['pdb_id']}:{entry['chain']}")
            rows.append(
                observe_one(
                    entry=entry,
                    system=compare_system,
                    curvature=compare_c,
                    tau_ceil=tau_ceil,
                    device=device,
                    pdb_dir=pdb_dir,
                    theta="mode_c_s9_compare",
                )
            )
        del compare_system
    else:
        print("[b0] mode_c_s9 absent — compare skipped")

    stamp = build_stamp(
        rows,
        champion_sha256=digest,
        champion_version=resolved.get("version"),
        mode_c_s9_status=mode_c_status,
        champion_run_id=resolved.get("run_id"),
    )
    stamp["graph_recipe"] = {
        "dehydron_wrap_max": int(get_dehydron_wrap_max()),
        "recipe": recipe,
        "note": "Freeze wrap ≤ 1 (tokyo_eye_equ_wrap_threshold §2.7 AMEND). Runtime retune is forbidden.",
    }
    args.out_stamp.parent.mkdir(parents=True, exist_ok=True)
    args.out_stamp.write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"[b0] wrote {args.out_stamp}")

    if not args.no_mlflow:
        run_id = _log_mlflow(stamp, args.tracking_uri or None)
        if run_id:
            stamp["mlflow_run_id"] = run_id
            args.out_stamp.write_text(json.dumps(stamp, indent=2) + "\n")
            print(f"[b0] mlflow run {run_id} (no alias move)")

    print(
        json.dumps(
            {
                "ok": True,
                "hygiene_finite": stamp["hygiene_finite"],
                "biology_pass": False,
                "n_loaded": stamp["n_loaded"],
                "n_skip": stamp["n_skip"],
                "mode_c_s9": mode_c_status,
                "stamp": str(args.out_stamp),
            },
            indent=2,
        )
    )
    return stamp


def main() -> None:
    run()


if __name__ == "__main__":
    main()
