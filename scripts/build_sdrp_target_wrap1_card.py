"""Builds the DRAFT SDRP-target pre-registration card + DRAFT stamp under data/gates/.

Read-only: loads cached wrap=1 graphs for Stage-A-12 (enabled:true) and measures the
STATUS-QUO rule only. No candidate rule is evaluated here; no code/label change; no training;
no MLflow writes.
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

from science.tokyo_eye.v8 import biophysics, loader
from science.tokyo_eye.v8.loader import TokyoEyeCuratedDataset
from science.tokyo_eye.v8.r0_r5_graph import get_dehydron_wrap_max

ROOT = Path("/workspace")
GATES = ROOT / "data" / "gates"
GATE_ID = "tokyo_eye_equ_sdrp_target_wrap1"
CLASS_NAMES = ["core", "dehydron_rim", "salt", "hydrophobe", "neighborhood"]
REL = ["R0_COVALENT", "R1_HBOND", "R2_DEHYDRON", "R3_HYDROPHOBIC_PI", "R4_SALT_BRIDGE", "R5_LOCAL_NEIGHBORHOOD"]
K = 5

# ---- pre-registered gate thresholds (fixed here, before any candidate is measured) ----
TH = {
    "G1_max_class_frac_pooled_le": 0.60,
    "G2_min_classes_with_frac_ge_0p02_pooled": 4,
    "G2_class_frac_floor": 0.02,
    "G3_norm_entropy_pooled_ge": 0.50,
    "G4_class1_vs_dehydron_label_jaccard_le": 0.80,
    "G5_structures_with_max_class_frac_le_0p85_min": 10,
    "G5_struct_max_class_frac": 0.85,
}


def ent_norm(h: np.ndarray) -> float:
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(K))


def measure_status_quo() -> dict:
    d = TokyoEyeCuratedDataset(
        manifest_path=ROOT / "manifests" / "v8_stage_a_small_v1.json",
        pdb_dir=ROOT / "pdb_cache", use_graph_cache=True,
        graph_cache_dir=ROOT / "pdb_cache" / "v8_graph_cache",
    )
    assert len(d) == 12 and d.mode == "manifest"
    edges_by_rel = np.zeros(6, dtype=np.int64)
    nodes_incident_by_rel = np.zeros(6, dtype=np.int64)
    pooled = np.zeros(K, dtype=np.int64)
    per, y_lab, y_cls = {}, [], []
    from_cache = []
    n_tied = 0
    n_nodes = 0
    incident_class_xtab = np.zeros((6, K), dtype=np.int64)  # rows: relation-incident nodes, cols: assigned class
    for i in range(len(d)):
        b = d.get_on_device(i, "cpu")
        from_cache.append(bool(b.get("from_graph_cache")))
        et = b["edge_type"].numpy()
        ei = b["edge_index"].numpy()
        n = int(b["num_nodes"])
        t = b["sdrp_target"].numpy()
        lab = b["dehydron_labels"].numpy().astype(np.int64)
        edges_by_rel += np.bincount(et, minlength=6)[:6]
        inc = np.zeros((n, 6), dtype=bool)
        counts = np.zeros((n, K), dtype=np.float64)
        for r in range(6):
            m = et == r
            inc[ei[0, m], r] = True
            inc[ei[1, m], r] = True
        rel_to_cls = {2: 1, 1: 0, 0: 0, 4: 2, 3: 3, 5: 4}
        for r, c in rel_to_cls.items():
            m = et == r
            np.add.at(counts[:, c], ei[0, m], 1.0)
            np.add.at(counts[:, c], ei[1, m], 1.0)
        top = counts.max(axis=1, keepdims=True)
        n_tied += int(((counts == top).sum(axis=1) > 1).sum())
        n_nodes += n
        nodes_incident_by_rel += inc.sum(0)
        for r in range(6):
            incident_class_xtab[r] += np.bincount(t[inc[:, r]], minlength=K)
        h = np.bincount(t, minlength=K)
        pooled += h
        y_lab.append(lab)
        y_cls.append(t)
        per[f"{b['pdb_id']}:{b['chain']}"] = {
            "n": n, "hist": h.tolist(), "max_class_frac": float(h.max() / n),
        }
    lab = np.concatenate(y_lab)
    cls = np.concatenate(y_cls)
    c1 = cls == 1
    inter = int((c1 & (lab == 1)).sum())
    union = int((c1 | (lab == 1)).sum())
    frac = pooled / pooled.sum()
    n_ok_struct = sum(1 for v in per.values() if v["max_class_frac"] <= TH["G5_struct_max_class_frac"])
    gates = {
        "G1_max_class_frac_pooled": {
            "value": float(frac.max()), "threshold_le": TH["G1_max_class_frac_pooled_le"],
            "pass": bool(frac.max() <= TH["G1_max_class_frac_pooled_le"]),
        },
        "G2_classes_with_frac_ge_floor": {
            "value": int((frac >= TH["G2_class_frac_floor"]).sum()),
            "threshold_ge": TH["G2_min_classes_with_frac_ge_0p02_pooled"],
            "pass": bool((frac >= TH["G2_class_frac_floor"]).sum() >= TH["G2_min_classes_with_frac_ge_0p02_pooled"]),
        },
        "G3_norm_entropy_pooled": {
            "value": ent_norm(pooled.astype(float)), "threshold_ge": TH["G3_norm_entropy_pooled_ge"],
            "pass": bool(ent_norm(pooled.astype(float)) >= TH["G3_norm_entropy_pooled_ge"]),
        },
        "G4_class1_dehydron_label_jaccard": {
            "value": inter / union if union else float("nan"),
            "threshold_le": TH["G4_class1_vs_dehydron_label_jaccard_le"],
            "pass": bool((inter / union if union else 0.0) <= TH["G4_class1_vs_dehydron_label_jaccard_le"]),
        },
        "G5_structures_with_max_class_frac_le_0p85": {
            "value": int(n_ok_struct), "threshold_ge": TH["G5_structures_with_max_class_frac_le_0p85_min"],
            "pass": bool(n_ok_struct >= TH["G5_structures_with_max_class_frac_le_0p85_min"]),
        },
    }
    return {
        "n_structures": 12, "n_nodes": int(n_nodes),
        "all_graphs_from_cache": all(from_cache),
        "edges_by_relation": dict(zip(REL, edges_by_rel.tolist())),
        "nodes_incident_by_relation": dict(zip(REL, nodes_incident_by_rel.tolist())),
        "r5_edge_ratio_vs": {
            REL[r]: float(edges_by_rel[5] / edges_by_rel[r]) for r in range(5)
        },
        "pooled_class_hist": dict(zip(CLASS_NAMES, pooled.tolist())),
        "pooled_class_frac": dict(zip(CLASS_NAMES, [float(x) for x in frac])),
        "per_structure": per,
        "argmax_tie_nodes": int(n_tied),
        "assigned_class_of_relation_incident_nodes": {
            REL[r]: dict(zip(CLASS_NAMES, incident_class_xtab[r].tolist())) for r in range(6)
        },
        "class1_vs_dehydron_label": {
            "class1_nodes": int(c1.sum()), "dehydron_label_nodes": int((lab == 1).sum()),
            "intersection": inter, "jaccard": inter / union if union else None,
            "class1_precision_vs_label": inter / max(int(c1.sum()), 1),
            "class1_recall_of_label": inter / max(int((lab == 1).sum()), 1),
        },
        "gates_applied_to_status_quo": gates,
        "status_quo_fails_G1_G2_G3_G5": bool(not any(g["pass"] for k, g in gates.items() if k != "G4_class1_dehydron_label_jaccard")),
    }


def main() -> None:
    assert get_dehydron_wrap_max() == 1
    src = inspect.getsource(loader.sdrp_heuristic_from_edges)
    pins = {
        "wrap": get_dehydron_wrap_max(),
        "sdrp_heuristic_from_edges_sha256": hashlib.sha256(src.encode()).hexdigest(),
        "loader_py_sha256": hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(),
        "manifest": "manifests/v8_stage_a_small_v1.json",
        "manifest_sha256": hashlib.sha256((ROOT / "manifests" / "v8_stage_a_small_v1.json").read_bytes()).hexdigest(),
        "BIOPHYS_CACHE_VERSION": biophysics.BIOPHYS_CACHE_VERSION,
        "graph_cache_hash_wrap1": loader.graph_cache_hash(),
        "graph_cache_hash_covers_sdrp_rule": False,
        "SDRP_LOSS_COEFF": 0.1,
        "num_sdrp_classes": K,
    }
    base = measure_status_quo()

    prereg = {
        "schema_version": 1,
        "gate_id": f"{GATE_ID}_prereg",
        "display_lineage": "Tokyo Eye EQU",
        "status": "DRAFT",
        "approval_status": "DRAFT_NOT_APPROVED",
        "approver": None,
        "approved_at": None,
        "drafted_by": "Claude Sonnet 5 (assistant) — draft for operator review",
        "date_drafted": "2026-09-18",
        "sibling_stamp": f"data/gates/{GATE_ID}.json",
        "predecessor_evidence": {
            "wrap1_biology_rescore_stamp": "data/gates/tokyo_eye_equ_wrap1_biology_rescore.json",
            "mlflow_run_id": "170ff7b27ef04205bff6c93d19eaf940",
            "finding": "SDRP majority-collapse: pred all class 4, acc == majority (0.9248), macro-F1 0.3203; target hist [170,85,0,0,3138]",
            "wrap19_numbers_cited": False,
        },
        "constraints": {
            "no_code_or_label_change_until_approved": True,
            "no_promote": True,
            "no_alias_moves": True,
            "no_silent_retune": True,
            "unchanged_by_this_card": [
                "dehydron_labels / wrap SSOT (DEHYDRON_WRAP_MAX=1)",
                "SDRP_LOSS_COEFF=0.1, lr, dehydron loss, all MoE/eps schedule and coefficients",
                "num_sdrp_classes=5 (head shape)",
                "graph construction R0-R5",
            ],
            "not_claims": [
                "Stage-1 pass is NOT a biology pass",
                "no CASF/affinity/cryptic-pocket implication",
            ],
        },
        "problem_statement": {
            "rule": "sdrp_heuristic_from_edges: argmax over per-node endpoint counts of relation groups "
                    "(R2->rim, R0|R1->core, R4->salt, R3->hydrophobe, R5->neighborhood); the 'priority order' only breaks exact ties",
            "diagnosis": [
                "Raw-count argmax is dominated by relation prevalence, not node-level enrichment: R5 edges outnumber R2 by "
                "%.1fx, R3 by %.1fx, R4 by %.1fx pooled over Stage-A-12."
                % (base["r5_edge_ratio_vs"]["R2_DEHYDRON"], base["r5_edge_ratio_vs"]["R3_HYDROPHOBIC_PI"], base["r5_edge_ratio_vs"]["R4_SALT_BRIDGE"]),
                "Classes salt (2) and hydrophobe (3) are empty although %d R4-incident and %d R3-incident nodes exist."
                % (base["nodes_incident_by_relation"]["R4_SALT_BRIDGE"], base["nodes_incident_by_relation"]["R3_HYDROPHOBIC_PI"]),
                "class 1 (dehydron_rim) has %d nodes vs %d dehydron-label (R2-incident) nodes: a strict subset (precision %.2f) recovering only %.1f%% of them; "
                "%d R2-incident nodes are assigned neighborhood."
                % (base["class1_vs_dehydron_label"]["class1_nodes"], base["class1_vs_dehydron_label"]["dehydron_label_nodes"],
                   base["class1_vs_dehydron_label"]["class1_precision_vs_label"], 100.0 * base["class1_vs_dehydron_label"]["class1_recall_of_label"],
                   base["assigned_class_of_relation_incident_nodes"]["R2_DEHYDRON"]["neighborhood"]),
                "core (0) mixes universal R0 backbone connectivity with wrapped H-bonds (R1).",
                "Tie-break priority list only decides %d of %d nodes (%.1f%%); the other nodes are decided by raw relation counts."
                % (base["argmax_tie_nodes"], base["n_nodes"], 100.0 * base["argmax_tie_nodes"] / base["n_nodes"]),
                "Target is a function of batch['edge_type'], which is also a model input (see wrap1 leakage_audit): any edge-type-derived "
                "target is leak-confounded and needs the blind-arm control.",
            ],
            "cache_hazard": "Cached graphs store sdrp_target; graph_cache_hash does not include the SDRP rule. Changing the rule "
                            "without a cache-key change would silently serve stale labels from pdb_cache/v8_graph_cache.",
        },
        "pins_at_draft": pins,
        "candidates": {
            "C0_status_quo": {
                "definition": "current sdrp_heuristic_from_edges (reference arm; measured in sibling stamp)",
                "leak_tier": "T0_edge_type_derived",
            },
            "C1_enrichment_normalized_argmax": {
                "definition": "score_c(node)=count_c(node)/mean_over_nodes_in_same_structure(count_c); argmax over the 5 class groups "
                              "(same relation->class map as C0); class disabled in a structure if its mean count is 0; "
                              "ties -> existing priority order. Parameter-free, deterministic.",
                "class_semantics": "unchanged ids 0-4",
                "leak_tier": "T0_edge_type_derived",
                "known_risk": "any node with a single rare-relation edge is pushed to that rare class (over-assignment)",
            },
            "C2_exclusivity_order": {
                "definition": "node class = first relation present in descending exclusivity order R2 > R1 (R0 excluded from core) > R3 > R4 > R5, "
                              "mirroring the graph layer's own edge exclusivity",
                "class_semantics": "unchanged ids 0-4",
                "leak_tier": "T0_edge_type_derived",
                "expected_gate_interaction": "class 1 == dehydron label by construction, so G4 is expected to fail; listed so the operator can waive G4 "
                                             "via a signed amendment instead of the card silently excluding it",
            },
            "C3_edge_type_independent_context": {
                "definition": "OPEN — structural-context target not derived from edge_type (e.g. burial tier x DSSP class). Changes class "
                              "semantics from relation-mix to structural context; needs separate operator approval and a definition amendment "
                              "before any measurement.",
                "class_semantics": "REMAPPED (requires amendment)",
                "leak_tier": "T1_independent_of_edge_type",
            },
        },
        "stage_1_offline_measurement": {
            "executes_only_after": "approval_status == APPROVED",
            "no_training": True,
            "inputs": "cached wrap=1 graphs of Stage-A-12 (enabled:true), read-only; candidate rule evaluated in /tmp script, not in loader.py",
            "metrics": [
                "pooled + per-structure class histogram", "normalized entropy (pooled)", "max class fraction (pooled, per-structure)",
                "classes with pooled frac >= 0.02", "Jaccard/precision/recall of class 1 vs dehydron_labels",
                "argmax tie count", "bit-for-bit determinism across two evaluations",
                "assigned-class crosstab for R2/R3/R4-incident nodes",
            ],
            "hard_gates": {
                "G1": "max pooled class fraction <= 0.60",
                "G2": ">= 4 of 5 classes with pooled fraction >= 0.02",
                "G3": "pooled normalized entropy H/ln5 >= 0.50",
                "G4": "Jaccard(class==1, dehydron_labels==1) <= 0.80 (SDRP not a copy of the dehydron target)",
                "G5": ">= 10 of 12 structures with max class fraction <= 0.85",
                "G6": "deterministic (two evaluations identical)",
            },
            "thresholds": TH,
            "threshold_provenance": "fixed before any candidate is measured; chosen from generic non-degeneracy reasoning "
                                    "(status quo, measured, fails G1/G2/G3/G5 — see sibling stamp). Operator may amend before approval only.",
            "selection_rule": "candidates passing all hard gates advance; if more than one passes, prefer lower leak tier, then simpler rule; "
                              "if none passes, card closes NO_CANDIDATE and SDRP stays diagnostic-only",
        },
        "stage_2_training_read": {
            "executes_only_after": "Stage 1 selects a candidate AND operator approves code change + cache-key change",
            "protocol": "same harness/arms as tokyo_eye_equ_wrap1_biology_rescore (typed + type-blind + untrained-init control, in-sample, "
                        "seed 0, 60 epochs, schedules unchanged)",
            "reads_not_promote_gates": {
                "S1": "typed eval SDRP macro-F1 (present classes) >= majority-only macro-F1 + 0.10",
                "S2": "recall > 0 for >= 3 classes (typed arm)",
                "S3": "blind-arm result reported beside typed; leak gap = typed - blind, no threshold",
            },
            "conditioning_caveat": "router remains the parked COLLAPSED_POST_FLOOR eps config (eval_moe_liveness 0/60 in wrap1 rescore); "
                                   "Stage 2 must log liveness and cannot attribute a failure to the label alone",
        },
        "implementation_constraints_post_approval": [
            "change confined to sdrp_heuristic_from_edges (or a versioned SDRP_TARGET_RULE selector) in science/tokyo_eye/v8/loader.py",
            "add the rule id to graph_cache_hash (or bump BIOPHYS_CACHE_VERSION) so stale cached sdrp_target cannot be served",
            "class ids 0-4 keep meaning for C1/C2; record semantic change for consumers: scripts/stripe_membership_crosstab.py, "
            "governance/pyfunc_model.py (sdrp_logits), experiments/training/v8/run_affinity_s10.py (freezes sdrp_head)",
            "scripts/ is not writable for the assistant: code lands via host agent from /tmp",
        ],
        "forbidden": [
            "silent label rewrite", "changing SDRP_LOSS_COEFF/lr/eps/MoE coefficients under this card",
            "citing wrap=19 REVERT-era numbers", "promote or alias moves",
            "reading any result of this card as affinity/CASF/cryptic-pocket evidence",
        ],
        "operator_decisions_needed": [
            "approve/amend thresholds G1-G6, S1-S3",
            "keep or drop C3 (semantic change)",
            "keep C2 with expected G4 failure, or pre-authorize a G4 waiver",
            "approve Stage 1 execution",
        ],
    }

    stamp = {
        "schema_version": 1,
        "gate_id": GATE_ID,
        "display_lineage": "Tokyo Eye EQU",
        "status": "DRAFT",
        "approval_status": "DRAFT_NOT_APPROVED",
        "execution_state": "NOT_RUN",
        "promotion_eligible": False,
        "do_not_promote": True,
        "prereg": f"data/gates/{GATE_ID}_prereg.json",
        "dehydron_wrap_max": get_dehydron_wrap_max(),
        "pins_at_draft": pins,
        "baseline_status_quo_measurement": {
            "note": "read-only measurement of the CURRENT rule on cached wrap=1 Stage-A-12 graphs; not a candidate result; no training",
            **base,
        },
        "candidates": {c: {"stage_1": None, "gates": None, "advances": None} for c in prereg["candidates"] if c != "C0_status_quo"},
        "gates_stage_1": {g: {"pass": None} for g in ("G1", "G2", "G3", "G4", "G5", "G6")},
        "gates_stage_2": {g: {"pass": None} for g in ("S1", "S2", "S3")},
        "selected_candidate": None,
        "mlflow_run_id": None,
        "note": "DRAFT stamp. All candidate/gate fields are null until an approved Stage 1 run; no MLflow run exists for this card.",
    }
    for name, obj in ((f"{GATE_ID}_prereg.json", prereg), (f"{GATE_ID}.json", stamp)):
        p = GATES / name
        p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
        print("wrote", p)
    print(json.dumps({k: (v["value"], v["pass"]) for k, v in base["gates_applied_to_status_quo"].items()}, indent=1))
    print("class hist", base["pooled_class_hist"], "ties", base["argmax_tie_nodes"], "from_cache", base["all_graphs_from_cache"])
    print("class1 vs label", base["class1_vs_dehydron_label"])
    print("xtab R2/R3/R4", {k: v for k, v in base["assigned_class_of_relation_incident_nodes"].items() if k[:2] in ("R2", "R3", "R4")})


if __name__ == "__main__":
    main()
