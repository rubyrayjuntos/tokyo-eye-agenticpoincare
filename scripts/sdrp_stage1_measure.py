"""Stage 1 (offline) measurement for tokyo_eye_equ_sdrp_target_wrap1: C0, C1, C2 on cached wrap=1 Stage-A-12.

Approved scope: APPROVED_STAGE1_ONLY. No loader change, no training, no MLflow, no cache rewrite.
C3 excluded (parked). G4 waived for C2 ONLY (still binds C0/C1). Thresholds are the pre-registered ones.
Writes data/gates/tokyo_eye_equ_sdrp_target_wrap1_stage1_results.json (sibling; the stamp itself is
not writable by this account) and /tmp/merge_sdrp_stage1_into_stamp.py for the host agent.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/workspace")

import numpy as np

from science.tokyo_eye.v8 import loader
from science.tokyo_eye.v8.loader import TokyoEyeCuratedDataset
from science.tokyo_eye.v8.r0_r5_graph import get_dehydron_wrap_max

ROOT = Path("/workspace")
GATES = ROOT / "data" / "gates"
STAMP = GATES / "tokyo_eye_equ_sdrp_target_wrap1.json"
PREREG = GATES / "tokyo_eye_equ_sdrp_target_wrap1_prereg.json"
OUT = GATES / "tokyo_eye_equ_sdrp_target_wrap1_stage1_results.json"
CLASS_NAMES = ["core", "dehydron_rim", "salt", "hydrophobe", "neighborhood"]
REL = ["R0_COVALENT", "R1_HBOND", "R2_DEHYDRON", "R3_HYDROPHOBIC_PI", "R4_SALT_BRIDGE", "R5_LOCAL_NEIGHBORHOOD"]
K = 5
REL_TO_CLS = {2: 1, 1: 0, 0: 0, 4: 2, 3: 3, 5: 4}  # same map as sdrp_heuristic_from_edges
PRIORITY = [1, 2, 3, 4, 0]  # rim, salt, hydrophobe, neighborhood, core (existing tie-break order)

prereg = json.load(open(PREREG))
stamp0 = json.load(open(STAMP))
TH = prereg["stage_1_offline_measurement"]["thresholds"]
assert prereg["approval_status"] == "APPROVED_STAGE1_ONLY" and stamp0["approval_status"] == "APPROVED_STAGE1_ONLY"
assert get_dehydron_wrap_max() == 1

pins = stamp0["pins_at_draft"]
pins_now = {
    "sdrp_fn": hashlib.sha256(inspect.getsource(loader.sdrp_heuristic_from_edges).encode()).hexdigest(),
    "loader": hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(),
    "manifest": hashlib.sha256((ROOT / "manifests/v8_stage_a_small_v1.json").read_bytes()).hexdigest(),
}
assert pins_now["sdrp_fn"] == pins["sdrp_heuristic_from_edges_sha256"], "sdrp rule changed since pin"
assert pins_now["loader"] == pins["loader_py_sha256"], "loader.py changed since pin"
assert pins_now["manifest"] == pins["manifest_sha256"], "manifest changed since pin"


def ent_norm(h: np.ndarray) -> float:
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(K))


def prioritized_argmax(score: np.ndarray, enabled: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """argmax over columns in PRIORITY order (first max wins) — mirrors loader tie-break."""
    s = score.copy()
    if enabled is not None:
        s[:, ~enabled] = -np.inf
    p = s[:, PRIORITY]
    w = np.argmax(p, axis=1)
    top = p.max(axis=1, keepdims=True)
    n_tied = int(((p == top).sum(axis=1) > 1).sum())
    return np.asarray([PRIORITY[int(i)] for i in w], dtype=np.int64), n_tied


def node_features(b: dict) -> dict:
    et = b["edge_type"].numpy()
    ei = b["edge_index"].numpy()
    n = int(b["num_nodes"])
    counts = np.zeros((n, K), dtype=np.float64)
    inc = np.zeros((n, 6), dtype=bool)
    for r in range(6):
        m = et == r
        inc[ei[0, m], r] = True
        inc[ei[1, m], r] = True
        if r in REL_TO_CLS:
            np.add.at(counts[:, REL_TO_CLS[r]], ei[0, m], 1.0)
            np.add.at(counts[:, REL_TO_CLS[r]], ei[1, m], 1.0)
    return {"n": n, "counts": counts, "inc": inc}


def rule_c0(f: dict) -> tuple[np.ndarray, dict]:
    y, ties = prioritized_argmax(f["counts"])
    return y, {"argmax_tie_nodes": ties}


def rule_c1(f: dict) -> tuple[np.ndarray, dict]:
    counts = f["counts"]
    mean = counts.mean(axis=0)
    enabled = mean > 0
    score = np.zeros_like(counts)
    score[:, enabled] = counts[:, enabled] / mean[enabled]
    y, ties = prioritized_argmax(score, enabled)
    return y, {"argmax_tie_nodes": ties, "classes_disabled": [CLASS_NAMES[c] for c in range(K) if not enabled[c]],
               "all_zero_count_nodes": int((counts.sum(axis=1) == 0).sum())}


def rule_c2(f: dict, fallback: int = 4) -> tuple[np.ndarray, dict]:
    inc = f["inc"]
    order = [(2, 1), (1, 0), (3, 3), (4, 2), (5, 4)]  # (relation, class): R2 > R1 > R3 > R4 > R5; R0 not used
    y = np.full(f["n"], -1, dtype=np.int64)
    for r, c in order:
        m = (y < 0) & inc[:, r]
        y[m] = c
    n_fb = int((y < 0).sum())
    y[y < 0] = fallback
    return y, {"fallback_nodes_no_R1_to_R5_edge": n_fb, "fallback_class": CLASS_NAMES[fallback]}


def gate_table(per: dict, pooled: np.ndarray, y1_jaccard: float, waive_g4: bool, determ: bool) -> dict:
    frac = pooled / pooled.sum()
    n_ok = sum(1 for v in per.values() if v["max_class_frac"] <= TH["G5_struct_max_class_frac"])
    g = {
        "G1": {"value": float(frac.max()), "threshold_le": TH["G1_max_class_frac_pooled_le"],
               "pass": bool(frac.max() <= TH["G1_max_class_frac_pooled_le"])},
        "G2": {"value": int((frac >= TH["G2_class_frac_floor"]).sum()),
               "threshold_ge": TH["G2_min_classes_with_frac_ge_0p02_pooled"],
               "pass": bool((frac >= TH["G2_class_frac_floor"]).sum() >= TH["G2_min_classes_with_frac_ge_0p02_pooled"])},
        "G3": {"value": ent_norm(pooled.astype(float)), "threshold_ge": TH["G3_norm_entropy_pooled_ge"],
               "pass": bool(ent_norm(pooled.astype(float)) >= TH["G3_norm_entropy_pooled_ge"])},
        "G4": {"value": y1_jaccard, "threshold_le": TH["G4_class1_vs_dehydron_label_jaccard_le"],
               "raw_pass": bool(y1_jaccard <= TH["G4_class1_vs_dehydron_label_jaccard_le"]),
               "waived": bool(waive_g4)},
        "G5": {"value": int(n_ok), "threshold_ge": TH["G5_structures_with_max_class_frac_le_0p85_min"],
               "pass": bool(n_ok >= TH["G5_structures_with_max_class_frac_le_0p85_min"])},
        "G6": {"value": bool(determ), "pass": bool(determ)},
    }
    g["G4"]["pass"] = bool(g["G4"]["raw_pass"] or waive_g4)
    return g


def measure(name: str, rule, structs: list[dict], waive_g4: bool, cached_targets: list[np.ndarray] | None = None, **kw) -> dict:
    per, ys, labs, meta_sum = {}, [], [], {}
    determ = True
    xtab = np.zeros((6, K), dtype=np.int64)
    for s in structs:
        y, meta = rule(s["f"], **kw)
        y2, _ = rule(s["f"], **kw)
        determ &= bool(np.array_equal(y, y2))
        h = np.bincount(y, minlength=K)
        per[s["tag"]] = {"n": s["f"]["n"], "hist": h.tolist(), "max_class_frac": float(h.max() / s["f"]["n"])}
        ys.append(y)
        labs.append(s["lab"])
        for r in range(6):
            xtab[r] += np.bincount(y[s["f"]["inc"][:, r]], minlength=K)
        for k, v in meta.items():
            meta_sum[k] = (meta_sum.get(k, 0) + v) if isinstance(v, int) and not isinstance(v, bool) else v
    y = np.concatenate(ys)
    lab = np.concatenate(labs)
    pooled = np.bincount(y, minlength=K)
    c1 = y == 1
    inter = int((c1 & (lab == 1)).sum())
    union = int((c1 | (lab == 1)).sum())
    jac = inter / union if union else 0.0
    out = {
        "pooled_class_hist": dict(zip(CLASS_NAMES, pooled.tolist())),
        "pooled_class_frac": dict(zip(CLASS_NAMES, [float(x) for x in pooled / pooled.sum()])),
        "per_structure": per,
        "class1_vs_dehydron_label": {
            "class1_nodes": int(c1.sum()), "dehydron_label_nodes": int((lab == 1).sum()), "intersection": inter,
            "jaccard": jac, "class1_precision_vs_label": inter / max(int(c1.sum()), 1),
            "class1_recall_of_label": inter / max(int((lab == 1).sum()), 1),
        },
        "assigned_class_of_relation_incident_nodes": {
            REL[r]: dict(zip(CLASS_NAMES, xtab[r].tolist())) for r in (1, 2, 3, 4)
        },
        "rule_meta": meta_sum,
        "gates": gate_table(per, pooled, jac, waive_g4, determ),
    }
    if cached_targets is not None:
        out["reproduces_cached_sdrp_target_bitwise"] = bool(all(np.array_equal(a, b) for a, b in zip(ys, cached_targets)))
    g = out["gates"]
    out["passes_all_hard_gates"] = bool(all(g[k]["pass"] for k in ("G1", "G2", "G3", "G4", "G5", "G6")))
    out["passes_all_hard_gates_without_waiver"] = bool(
        all(g[k]["pass"] for k in ("G1", "G2", "G3", "G5", "G6")) and g["G4"]["raw_pass"]
    )
    out["failed_gates"] = [k for k in ("G1", "G2", "G3", "G4", "G5", "G6") if not g[k]["pass"]]
    return out


def main() -> None:
    d = TokyoEyeCuratedDataset(
        manifest_path=ROOT / "manifests/v8_stage_a_small_v1.json", pdb_dir=ROOT / "pdb_cache",
        use_graph_cache=True, graph_cache_dir=ROOT / "pdb_cache/v8_graph_cache",
    )
    assert d.mode == "manifest" and len(d) == 12
    structs, cached, from_cache = [], [], []
    for i in range(len(d)):
        b = d.get_on_device(i, "cpu")
        from_cache.append(bool(b.get("from_graph_cache")))
        structs.append({"tag": f"{b['pdb_id']}:{b['chain']}", "f": node_features(b),
                        "lab": b["dehydron_labels"].numpy().astype(np.int64)})
        cached.append(b["sdrp_target"].numpy().astype(np.int64))
    assert all(from_cache), "not all graphs came from the wrap=1 cache"

    res = {
        "C0_status_quo": measure("C0", rule_c0, structs, waive_g4=False, cached_targets=cached),
        "C1_enrichment_normalized_argmax": measure("C1", rule_c1, structs, waive_g4=False),
        "C2_exclusivity_order": measure("C2", rule_c2, structs, waive_g4=True, fallback=4),
    }
    # C2 is silent on nodes with no R1-R5 edge; sensitivity to the fallback choice (reported, not selected on)
    sens = measure("C2_alt", rule_c2, structs, waive_g4=True, fallback=0)
    res["C2_exclusivity_order"]["fallback_sensitivity_core"] = {
        "failed_gates": sens["failed_gates"], "passes_all_hard_gates": sens["passes_all_hard_gates"],
        "pooled_class_hist": sens["pooled_class_hist"],
        "gate_values": {k: sens["gates"][k].get("value") for k in ("G1", "G2", "G3", "G5")},
    }
    res["C2_exclusivity_order"]["fallback_choice_note"] = (
        "prereg C2 definition is silent on nodes with no R1-R5 edge; measured with fallback=neighborhood (terminal class of the "
        "exclusivity chain); alternate fallback=core reported as sensitivity only"
    )

    passing = [c for c, r in res.items() if c != "C0_status_quo" and r["passes_all_hard_gates"]]
    if not passing:
        selected, reason = None, "NO_CANDIDATE: no C1/C2 passes all hard gates -> card closes NO_CANDIDATE, SDRP stays diagnostic-only"
    elif len(passing) == 1:
        selected, reason = passing[0], "sole passing candidate"
    else:
        selected = "C2_exclusivity_order"
        reason = ("multiple pass; leak tier equal (both T0) -> prereg selection_rule prefers simpler rule (C2 parameter-free priority chain "
                  "over C1 normalization)")
    out = {
        "gate_id": "tokyo_eye_equ_sdrp_target_wrap1",
        "artifact": "stage1_results",
        "stage": 1,
        "execution_state": "STAGE1_COMPLETE",
        "approval_scope": "APPROVED_STAGE1_ONLY",
        "stage_2_approved": False,
        "promotion_eligible": False,
        "do_not_promote": True,
        "executed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "/tmp/sdrp_stage1_measure.py",
        "dehydron_wrap_max": get_dehydron_wrap_max(),
        "n_structures": 12, "n_nodes": int(sum(s["f"]["n"] for s in structs)),
        "all_graphs_from_cache": True,
        "pins_verified_unchanged": True,
        "excluded": {"C3_edge_type_independent_context": "parked by operator"},
        "g4_waiver": "C2 only (pre-authorized); C0/C1 evaluated with G4 binding",
        "thresholds": TH,
        "candidates": res,
        "candidates_passing_all_hard_gates": passing,
        "selected_candidate": selected,
        "selection_reason": reason,
        "leak_note": "all candidates are T0 (edge_type-derived); any Stage 2 must include the blind-arm control",
        "stamp_merge_pending": "stamp file not writable by assistant (owner uid 1000, 0644); host agent: python3 /tmp/merge_sdrp_stage1_into_stamp.py",
    }
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("wrote", OUT)
    for c, r in res.items():
        g = r["gates"]
        print(f"{c}: pass_all={r['passes_all_hard_gates']} failed={r['failed_gates']} hist={list(r['pooled_class_hist'].values())} "
              f"G1={g['G1']['value']:.3f} G2={g['G2']['value']} G3={g['G3']['value']:.3f} "
              f"G4j={g['G4']['value']:.3f}(waived={g['G4'].get('waived')}) G5={g['G5']['value']} G6={g['G6']['value']}")
    print("C0 reproduces cached:", res["C0_status_quo"]["reproduces_cached_sdrp_target_bitwise"])
    print("C1 meta:", res["C1_enrichment_normalized_argmax"]["rule_meta"])
    print("C2 meta:", res["C2_exclusivity_order"]["rule_meta"], "sens:", res["C2_exclusivity_order"]["fallback_sensitivity_core"]["failed_gates"])
    print("selected:", selected, "|", reason)

    merge = f'''"""Host agent: merge Stage 1 results into the (operator-owned) stamp. Run as a user with write access."""
import json
p = "{STAMP}"; r = json.load(open("{OUT}")); s = json.load(open(p))
s["execution_state"] = "STAGE1_COMPLETE"
s["candidates"] = {{c: {{"stage_1": v, "gates": v["gates"], "advances": v["passes_all_hard_gates"]}} for c, v in r["candidates"].items() if c != "C0_status_quo"}}
s["candidates"]["C0_status_quo"] = {{"stage_1": r["candidates"]["C0_status_quo"], "gates": r["candidates"]["C0_status_quo"]["gates"], "advances": False}}
s["gates_stage_1"] = {{c: v["gates"] for c, v in r["candidates"].items()}}
s["selected_candidate"] = r["selected_candidate"]
s["stage_1_results_file"] = "data/gates/tokyo_eye_equ_sdrp_target_wrap1_stage1_results.json"
s["stage_1_executed_at_utc"] = r["executed_at_utc"]
s["stage_2_approved"] = False
json.dump(s, open(p, "w"), indent=2, allow_nan=False); open(p, "a").write("\\n")
print("merged into", p)
'''
    Path("/tmp/merge_sdrp_stage1_into_stamp.py").write_text(merge)


if __name__ == "__main__":
    main()
