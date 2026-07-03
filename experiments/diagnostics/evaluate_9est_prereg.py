"""
evaluate_9est_prereg.py
=======================

Executable lock for PREREG_9EST_1FLE_cryptic_interface.md.

The verdict for the single-case cryptic-interface test is computed HERE from
frozen constants, not by eye. Freeze this file in the same commit as the
pre-registration. After Phase 3 (Tokyo Eye output viewed), do not edit the
LOCKED block, the metric functions, or `decide()`.

Run the self-tests now (pre-data) to prove the metric/decision code is correct:
    python -m experiments.diagnostics.evaluate_9est_prereg --selftest

Phase 1 (output-blind — safe before / after freeze):
    python -m experiments.diagnostics.evaluate_9est_prereg --phase1

Run the real evaluation in Phase 4 (after Phase 3):
    python -m experiments.diagnostics.evaluate_9est_prereg --run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_PDB_DIR = Path("/tmp/dtie_pdb_cache")
DEFAULT_LOCK_PATH = Path("data/benchmarks/9est_1fle_interface_lock.json")
DEFAULT_PESTO_OUTPUT = Path("data/benchmarks/9est_pesto_scores.json")
DEFAULT_PHASE2_OUTPUT = Path("data/benchmarks/9est_phase2_precondition.json")
DEFAULT_PHASE34_OUTPUT = Path("data/benchmarks/9est_phase34_results.json")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"
)
# Epistemic-gated residues must rank last; finite sentinel (recovery@k uses argsort(-score)).
_GATE_SENTINEL = -1e30

# ===========================================================================
# LOCKED CONSTANTS  (frozen at pre-registration — do not edit after Phase 3)
# ===========================================================================
CONTACT_CUTOFF_A = 5.0  # heavy-atom min-atom-pair interface cutoff (§3)
K_PRIMARY = 3  # recovery@k, PeSTo parity (§5)
AUC_SUCCESS = 0.80  # H1 bar = PeSTo bound-state median (§6)
AUC_FLOOR = 0.65  # refutation ceiling (§6)
# Tokyo Eye S1 weights + epistemic gate — frozen at pre-registration:
W_DEHYDRON = 0.5
W_SHELL = 0.5
TAU_EPISTEMIC = 1.5896  # p90 epistemic on lever_a holdout (11QE,4OBE,1IVO,4MNE); no 9EST
W_ALEATORIC = 0.5  # S2 only (exploratory)

# Complex chain assignment (verified Phase 1 from 1FLE header — §2)
RECEPTOR_CHAIN_1FLE = "E"  # elastase
PARTNER_CHAIN_1FLE = "I"  # elafin
RECEPTOR_CHAIN_9EST = "A"  # unbound elastase monomer


# ===========================================================================
# Metric functions  (LOCKED)
# ===========================================================================
def recovery_at_k(scores: np.ndarray, gold_mask: np.ndarray, k: int) -> int:
    """1 iff the top-k residues by score are ALL true interface residues.

    Ties at the k-boundary are resolved pessimistically (a tie that lets a
    non-interface residue into the top-k counts as failure).
    """
    n = len(scores)
    if k <= 0 or k > n:
        raise ValueError(f"k={k} out of range for n={n}")
    order = np.argsort(-scores, kind="stable")
    topk = order[:k]
    if not gold_mask[topk].all():
        return 0
    kth = scores[order[k - 1]]
    tied_beyond = order[k:][scores[order[k:]] == kth]
    if len(tied_beyond) and not gold_mask[tied_beyond].all():
        return 0
    return 1


def roc_auc(scores: np.ndarray, gold_mask: np.ndarray) -> float:
    """ROC AUC via the rank (Mann-Whitney U) formula, with tie correction."""
    pos = gold_mask.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ===========================================================================
# Verdict  (LOCKED — mirrors §8 of the pre-registration)
# ===========================================================================
@dataclass
class Metrics:
    recovery_at_3: int
    auc: float
    sasa_margin: float  # auc(S1) - auc(SASA baseline)


def decide(te: Metrics, pesto_recovery_at_3: int) -> str:
    if pesto_recovery_at_3 == 1:
        return (
            "VOID: PeSTo recovers the interface on the static structure — "
            "not a cryptic-on-static discriminator. Pick another case."
        )
    if te.recovery_at_3 == 1 and te.auc >= AUC_SUCCESS and te.sasa_margin > 0:
        return (
            "CONFIRM (proof-of-concept): trigger the full PPDB5/MaSIF "
            "benchmark. No performance claim from n=1."
        )
    if te.recovery_at_3 == 0 and te.auc < AUC_FLOOR:
        return (
            "REFUTE: static dehydron/shell signal does not carry this "
            "cryptic interface. Record the negative."
        )
    return (
        "PARTIAL/INCONCLUSIVE: signal may be exposure-driven or weak. "
        "Exploratory only; inspect before benchmarking."
    )


# ===========================================================================
# Phase 1 — structure / label (output-blind; safe pre-Phase 3)
# ===========================================================================
def _download_pdb(pdb_id: str, pdb_dir: Path) -> Path:
    pdb_dir.mkdir(parents=True, exist_ok=True)
    path = pdb_dir / f"{pdb_id.upper()}.pdb"
    if path.exists():
        return path
    import urllib.request

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, path)
    return path


def _chain_compounds(pdb_path: Path) -> dict[str, str]:
    """Parse COMPND records for chain → molecule name."""
    compounds: dict[str, list[str]] = {}
    current_mol: list[str] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("COMPND"):
            continue
        text = line[10:].strip().rstrip(";")
        if text.startswith("MOL_ID:"):
            current_mol = []
            compounds[f"mol_{text.split(':')[1].strip()}"] = current_mol
        elif text.startswith("MOLECULE:"):
            current_mol.append(text.split(":", 1)[1].strip())
        elif text.startswith("CHAIN:"):
            chains = text.split(":", 1)[1].replace(" ", "").split(",")
            name = current_mol[0] if current_mol else "unknown"
            for ch in chains:
                compounds[ch] = [name]
    return {k: v[0] if isinstance(v, list) and v else str(v) for k, v in compounds.items() if len(k) == 1}


def _seq1(chain) -> tuple[str, list[int]]:
    from Bio.SeqUtils import seq1

    resnums: list[int] = []
    letters: list[str] = []
    for res in chain:
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        resnums.append(int(res.id[1]))
        letters.append(seq1(res.get_resname()))
    return "".join(letters), resnums


def interface_label_resnums(
    complex_pdb: str | Path,
    receptor_chain: str,
    partner_chain: str,
    *,
    cutoff: float = CONTACT_CUTOFF_A,
) -> set[int]:
    """Heavy-atom min-pair interface label on the complex (§3)."""
    from Bio.PDB import NeighborSearch, PDBParser

    s = PDBParser(QUIET=True).get_structure("c", str(complex_pdb))[0]

    def heavy_atoms(chain_id: str):
        return [a for a in s[chain_id].get_atoms() if a.element != "H"]

    partner_atoms = heavy_atoms(partner_chain)
    ns = NeighborSearch(partner_atoms)
    gold: set[int] = set()
    for res in s[receptor_chain]:
        if res.id[0] != " ":
            continue
        for atom in res:
            if atom.element == "H":
                continue
            if ns.search(atom.coord, cutoff):
                gold.add(int(res.id[1]))
                break
    return gold


def align_gold_to_9est(
    complex_pdb: Path,
    unbound_pdb: Path,
    *,
    receptor_chain_complex: str = RECEPTOR_CHAIN_1FLE,
    partner_chain: str = PARTNER_CHAIN_1FLE,
    receptor_chain_unbound: str = RECEPTOR_CHAIN_9EST,
) -> dict:
    from Bio import pairwise2
    from Bio.PDB import PDBParser
    from Bio.Seq import Seq

    parser = PDBParser(QUIET=True)
    c_struct = parser.get_structure("c", str(complex_pdb))[0]
    u_struct = parser.get_structure("u", str(unbound_pdb))[0]

    seq_c, nums_c = _seq1(c_struct[receptor_chain_complex])
    seq_u, nums_u = _seq1(u_struct[receptor_chain_unbound])
    aln = pairwise2.align.globalxx(Seq(seq_c), Seq(seq_u), one_alignment_only=True)[0]
    ac, au = aln.seqA, aln.seqB

    gold_1fle = interface_label_resnums(
        complex_pdb, receptor_chain_complex, partner_chain
    )
    map_c_to_u: dict[int, int] = {}
    i = j = 0
    for a, b in zip(ac, au, strict=True):
        if a != "-" and b != "-":
            map_c_to_u[nums_c[i]] = nums_u[j]
        if a != "-":
            i += 1
        if b != "-":
            j += 1

    gold_9est: list[int] = []
    unmapped: list[int] = []
    for rn in sorted(gold_1fle):
        if rn in map_c_to_u:
            gold_9est.append(map_c_to_u[rn])
        else:
            unmapped.append(rn)

    eval_resnums = sorted(set(nums_u) & set(map_c_to_u.values()))
    return {
        "gold_1fle_elastase_resnums": sorted(gold_1fle),
        "gold_9est_resnums": sorted(gold_9est),
        "unmapped_gold_1fle": unmapped,
        "eval_universe_9est_resnums": eval_resnums,
        "alignment": {
            "1fle_chain": receptor_chain_complex,
            "9est_chain": receptor_chain_unbound,
            "1fle_seq_len": len(seq_c),
            "9est_seq_len": len(seq_u),
            "identity": float(sum(1 for x, y in zip(ac, au, strict=True) if x == y and x != "-"))
            / max(1, sum(1 for x in ac if x != "-")),
        },
        "map_1fle_to_9est": {str(k): v for k, v in sorted(map_c_to_u.items())},
    }


def calibrate_tau_epistemic(
    checkpoint: Path,
    structures: list[str],
    pdb_dir: Path,
    *,
    percentile: float = 90.0,
    device: str = "cpu",
) -> float:
    """Epistemic gate threshold from holdout structures (no 9EST)."""
    from experiments.diagnostics.embedding_occupancy_audit import (
        _forward_audit,
        load_audit_model,
    )
    from experiments.training.v6._data import load_protein_graph

    model, version = load_audit_model(checkpoint, device)
    epi_all: list[float] = []
    for sid in structures:
        prot = load_protein_graph(sid, "A", pdb_dir)
        if prot is None:
            continue
        out = _forward_audit(model, version, prot, device)
        epi = out["uncertainty"]["epistemic"].detach().cpu().numpy().reshape(-1)
        epi_all.extend(epi.tolist())
    if not epi_all:
        raise RuntimeError("no epistemic values for tau calibration")
    return float(np.percentile(np.asarray(epi_all, dtype=float), percentile))


def run_phase1(
    *,
    pdb_dir: Path = DEFAULT_PDB_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    checkpoint: Path = Path(
        "checkpoints/v6/runs/lever_a_clean_slate_v1/v6_best_disc.pt"
    ),
) -> dict:
    fle = _download_pdb("1FLE", pdb_dir)
    est = _download_pdb("9EST", pdb_dir)
    chains = _chain_compounds(fle)

    label = align_gold_to_9est(fle, est)
    tau = calibrate_tau_epistemic(
        checkpoint, ["11QE", "4OBE", "1IVO", "4MNE"], pdb_dir
    )

    payload = {
        "phase": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "contact_cutoff_a": CONTACT_CUTOFF_A,
        "chain_verification": {
            "1FLE_header_chains": chains,
            "receptor_chain": RECEPTOR_CHAIN_1FLE,
            "partner_chain": PARTNER_CHAIN_1FLE,
            "receptor_molecule": chains.get(RECEPTOR_CHAIN_1FLE),
            "partner_molecule": chains.get(PARTNER_CHAIN_1FLE),
        },
        "interface_label": label,
        "n_gold_9est": len(label["gold_9est_resnums"]),
        "n_eval_universe": len(label["eval_universe_9est_resnums"]),
        "tau_epistemic_calibration": {
            "checkpoint": str(checkpoint),
            "holdout_structures": ["11QE", "4OBE", "1IVO", "4MNE"],
            "percentile": 90.0,
            "tau_epistemic": tau,
            "note": "Calibrated without 9EST; must match LOCKED TAU_EPISTEMIC at freeze",
        },
        "locked_constants_snapshot": {
            "W_DEHYDRON": W_DEHYDRON,
            "W_SHELL": W_SHELL,
            "TAU_EPISTEMIC": TAU_EPISTEMIC,
            "K_PRIMARY": K_PRIMARY,
            "AUC_SUCCESS": AUC_SUCCESS,
            "AUC_FLOOR": AUC_FLOOR,
        },
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


# ===========================================================================
# Phase 2 — PeSTo precondition (before Tokyo Eye Phase 3)
# ===========================================================================
def load_lock(lock_path: Path = DEFAULT_LOCK_PATH) -> dict:
    with open(lock_path, encoding="utf-8") as f:
        return json.load(f)


def load_pesto_scores(path: str | Path) -> dict[int, float]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("scores", payload)
    return {int(k): float(v) for k, v in raw.items()}


def _eval_scores_on_universe(
    scores: dict[int, float],
    eval_resnums: list[int],
    gold_resnums: list[int],
) -> Metrics:
    eval_resnums = sorted(eval_resnums)
    gold_set = set(gold_resnums)
    sc = np.array([scores.get(r, 0.0) for r in eval_resnums], dtype=float)
    gold_mask = np.array([r in gold_set for r in eval_resnums], dtype=bool)
    rec = recovery_at_k(sc, gold_mask, K_PRIMARY)
    auc = roc_auc(sc, gold_mask)
    return Metrics(recovery_at_3=rec, auc=auc, sasa_margin=float("nan"))


def run_phase2(
    *,
    pdb_dir: Path = DEFAULT_PDB_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    pesto_output: Path = DEFAULT_PESTO_OUTPUT,
    phase2_output: Path = DEFAULT_PHASE2_OUTPUT,
    pesto_root: Path | None = None,
    device: str = "cpu",
    scores_path: Path | None = None,
) -> dict:
    """Run PeSTo on static 9EST and evaluate §7 precondition gate."""
    lock = load_lock(lock_path)
    label = lock["interface_label"]
    eval_resnums = label["eval_universe_9est_resnums"]
    gold_resnums = label["gold_9est_resnums"]

    if scores_path is not None:
        scores = load_pesto_scores(scores_path)
        meta = {"source": "imported", "path": str(scores_path)}
    else:
        from experiments.diagnostics.pesto_runner import (
            DEFAULT_PESTO_ROOT,
            run_pesto_pp_interface,
        )

        est = _download_pdb("9EST", pdb_dir)
        root = pesto_root or DEFAULT_PESTO_ROOT
        scores, meta = run_pesto_pp_interface(
            est, chain=RECEPTOR_CHAIN_9EST, pesto_root=root, device=device
        )
        pesto_output.parent.mkdir(parents=True, exist_ok=True)
        with open(pesto_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_utc": datetime.now(UTC).isoformat(),
                    "structure": "9EST",
                    "chain": RECEPTOR_CHAIN_9EST,
                    "model": "PeSTo i_v4_1 (protein-protein)",
                    "scores": {str(k): v for k, v in sorted(scores.items())},
                    "meta": meta,
                },
                f,
                indent=2,
            )

    metrics = _eval_scores_on_universe(scores, eval_resnums, gold_resnums)
    if metrics.recovery_at_3 == 1:
        gate = "VOID"
        interpretation = decide(metrics, pesto_recovery_at_3=1)
    else:
        gate = "PASS"
        interpretation = (
            "PeSTo does not recover interface on static 9EST at recovery@3 — "
            "precondition satisfied; Phase 3 (Tokyo Eye) may proceed."
        )

    payload = {
        "phase": 2,
        "generated_utc": datetime.now(UTC).isoformat(),
        "pesto_metrics": {
            "recovery_at_3": metrics.recovery_at_3,
            "roc_auc": metrics.auc,
            "k_primary": K_PRIMARY,
        },
        "n_gold": len(gold_resnums),
        "n_eval_universe": len(eval_resnums),
        "precondition_gate": gate,
        "interpretation": interpretation,
        "pesto_meta": meta,
        "scores_path": str(pesto_output if scores_path is None else scores_path),
    }
    phase2_output.parent.mkdir(parents=True, exist_ok=True)
    with open(phase2_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


# ===========================================================================
# Phase 3 — Tokyo Eye channels (forward pass; no stdout)
# ===========================================================================
def load_tokyo_eye_channels(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    pdb_dir: Path = DEFAULT_PDB_DIR,
    structure_id: str = "9EST",
    chain: str = RECEPTOR_CHAIN_9EST,
    eval_resnums: list[int],
    device: str = "cpu",
) -> dict[int, dict[str, float]]:
    """Per-residue physics channels from lever_a forward pass on static 9EST.

    Returns {pdb_resnum: {dehydron, shell, epistemic, aleatoric}} for eval universe only.
    """
    from experiments.diagnostics.embedding_occupancy_audit import (
        _forward_audit,
        load_audit_model,
    )
    from experiments.training.v6._data import TAU, load_protein_graph

    eval_set = set(eval_resnums)
    prot = load_protein_graph(structure_id, chain, pdb_dir)
    if prot is None:
        raise RuntimeError(f"failed to build graph for {structure_id} chain {chain}")

    model, version = load_audit_model(checkpoint, device)
    out = _forward_audit(model, version, prot, device)

    rho = prot["data"].x.detach().cpu().numpy()[:, 0]
    xy = out["hyp_projections_2d"].detach().cpu().numpy()
    disc_r = np.linalg.norm(xy, axis=1)
    epi = out["uncertainty"]["epistemic"].detach().cpu().numpy().reshape(-1)
    ale = out["uncertainty"]["aleatoric"].detach().cpu().numpy().reshape(-1)

    res_ids = prot["residue_ids"]
    channels: dict[int, dict[str, float]] = {}
    for i, rid in enumerate(res_ids):
        resnum = int(str(rid).split(":")[1])
        if resnum not in eval_set:
            continue
        channels[resnum] = {
            "dehydron": float(max(0.0, TAU - float(rho[i]))),
            "shell": float(disc_r[i]),
            "epistemic": float(epi[i]),
            "aleatoric": float(ale[i]),
        }
    return channels


def s1_s2_scores(
    channels: dict[int, dict[str, float]],
    eval_resnums: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Build locked S1 (physics) and S2 (physics+aleatoric) with epistemic gate."""
    resids = sorted(r for r in eval_resnums if r in channels)
    if not resids:
        raise ValueError("no eval residues present in Tokyo Eye channels")
    deh = np.array([channels[r]["dehydron"] for r in resids], float)
    shl = np.array([channels[r]["shell"] for r in resids], float)
    epi = np.array([channels[r]["epistemic"] for r in resids], float)
    ale = np.array([channels[r]["aleatoric"] for r in resids], float)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    s1 = W_DEHYDRON * z(deh) + W_SHELL * z(shl)
    s2 = s1 + W_ALEATORIC * z(ale)
    gate = epi > TAU_EPISTEMIC
    s1[gate] = _GATE_SENTINEL
    s2[gate] = _GATE_SENTINEL
    return (dict(zip(resids, s1)), dict(zip(resids, s2)))


# ===========================================================================
# Phase 4 — SASA baseline (freesasa, heavy atoms only)
# ===========================================================================
def compute_sasa_baseline_freesasa(
    pdb_path: Path,
    chain_id: str,
    eval_resnums: list[int],
) -> dict[int, float]:
    """Per-residue heavy-atom SASA on static structure (higher = more exposed)."""
    import freesasa
    from Bio.PDB import PDBParser

    _ELEMENT_RADII = {
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
    }
    default_radius = 1.70

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sasa", str(pdb_path))[0]
    chain = structure[chain_id]

    flat_coords: list[float] = []
    radii: list[float] = []
    atom_to_resnum: list[int] = []

    for res in chain:
        if res.id[0] != " ":
            continue
        resnum = int(res.id[1])
        for atom in res:
            element = (atom.element or "").upper()
            if element == "H" or (atom.name or "").startswith("H"):
                continue
            flat_coords.extend(atom.coord.tolist())
            radii.append(_ELEMENT_RADII.get(element, default_radius))
            atom_to_resnum.append(resnum)

    if not flat_coords:
        raise RuntimeError(f"no heavy atoms on chain {chain_id} in {pdb_path}")

    result = freesasa.calcCoord(flat_coords, radii)
    per_res = {r: 0.0 for r in eval_resnums}
    for i, resnum in enumerate(atom_to_resnum):
        if resnum in per_res:
            per_res[resnum] += float(result.atomArea(i))
    return per_res


def _metrics_with_sasa_margin(
    scores: dict[int, float],
    eval_resnums: list[int],
    gold_resnums: list[int],
    sasa_auc: float,
) -> Metrics:
    base = _eval_scores_on_universe(scores, eval_resnums, gold_resnums)
    return Metrics(
        recovery_at_3=base.recovery_at_3,
        auc=base.auc,
        sasa_margin=base.auc - sasa_auc,
    )


def _recovery_sweep(
    scores: dict[int, float],
    eval_resnums: list[int],
    gold_resnums: list[int],
    *,
    k_max: int = 10,
) -> dict[str, int]:
    eval_sorted = sorted(eval_resnums)
    gold_set = set(gold_resnums)
    sc = np.array([scores.get(r, 0.0) for r in eval_sorted], dtype=float)
    gold_mask = np.array([r in gold_set for r in eval_sorted], dtype=bool)
    sweep: dict[str, int] = {}
    for k in range(1, k_max + 1):
        sweep[f"recovery_at_{k}"] = recovery_at_k(sc, gold_mask, k)
    return sweep


def _print_verdict_table(payload: dict) -> None:
    """§12 verdict table — only stdout from blind Phase 4 run."""
    t = payload["table"]
    pesto = t["pesto"]
    s1 = t["tokyo_eye_s1"]
    s2 = t["tokyo_eye_s2"]
    sasa = t["sasa_baseline"]
    n_gold = payload["n_gold"]
    verdict = payload["verdict"]

    def _fmt_auc(v: float) -> str:
        return f"{v:.3f}" if np.isfinite(v) else "nan"

    print("## 12. Results (Phase 4 — blind evaluation)")
    print()
    print("| Metric | PeSTo (static 9EST) | Tokyo Eye S1 | Tokyo Eye S2 | SASA baseline |")
    print("|---|---|---|---|---|")
    print(
        f"| recovery@3 | {pesto['recovery_at_3']} | {s1['recovery_at_3']} | "
        f"{s2['recovery_at_3']} | {sasa['recovery_at_3']} |"
    )
    print(
        f"| ROC AUC | {_fmt_auc(pesto['roc_auc'])} | {_fmt_auc(s1['roc_auc'])} | "
        f"{_fmt_auc(s2['roc_auc'])} | {_fmt_auc(sasa['roc_auc'])} |"
    )
    print(f"| |I_gold| | {n_gold} | — | — | — |")
    print(
        f"| SASA margin | — | {_fmt_auc(s1['sasa_margin'])} | "
        f"{_fmt_auc(s2['sasa_margin'])} | — |"
    )
    print(f"| **Verdict (§8):** | — | {verdict} | — | — |")


def run_phase34(
    *,
    pdb_dir: Path = DEFAULT_PDB_DIR,
    lock_path: Path = DEFAULT_LOCK_PATH,
    phase2_output: Path = DEFAULT_PHASE2_OUTPUT,
    pesto_scores_path: Path = DEFAULT_PESTO_OUTPUT,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_path: Path = DEFAULT_PHASE34_OUTPUT,
    device: str = "cpu",
) -> dict:
    """Phase 3+4 blind driver: channels → scores → decide(); no S1 stdout until table."""
    with open(phase2_output, encoding="utf-8") as f:
        phase2 = json.load(f)
    if phase2.get("precondition_gate") != "PASS":
        raise SystemExit(
            f"Phase 2 precondition gate is {phase2.get('precondition_gate')!r} — abort."
        )

    lock = load_lock(lock_path)
    label = lock["interface_label"]
    eval_resnums: list[int] = label["eval_universe_9est_resnums"]
    gold_resnums: list[int] = label["gold_9est_resnums"]

    pesto_scores = load_pesto_scores(pesto_scores_path)
    pesto_m = _eval_scores_on_universe(pesto_scores, eval_resnums, gold_resnums)

    channels = load_tokyo_eye_channels(
        checkpoint=checkpoint,
        pdb_dir=pdb_dir,
        eval_resnums=eval_resnums,
        device=device,
    )
    missing = sorted(set(eval_resnums) - set(channels))
    if missing:
        raise SystemExit(
            f"Tokyo Eye channels missing {len(missing)} eval-universe residues "
            f"(first: {missing[:5]})"
        )

    s1_scores, s2_scores = s1_s2_scores(channels, eval_resnums)

    est_pdb = _download_pdb("9EST", pdb_dir)
    sasa_scores = compute_sasa_baseline_freesasa(
        est_pdb, RECEPTOR_CHAIN_9EST, eval_resnums
    )

    sasa_m = _eval_scores_on_universe(sasa_scores, eval_resnums, gold_resnums)
    te_m = _metrics_with_sasa_margin(s1_scores, eval_resnums, gold_resnums, sasa_m.auc)
    s2_m = _metrics_with_sasa_margin(s2_scores, eval_resnums, gold_resnums, sasa_m.auc)

    verdict = decide(te_m, pesto_m.recovery_at_3)
    s1_sweep = _recovery_sweep(s1_scores, eval_resnums, gold_resnums)

    payload = {
        "phase": "3+4",
        "generated_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(checkpoint),
        "n_gold": len(gold_resnums),
        "n_eval_universe": len(eval_resnums),
        "phase2_gate": phase2["precondition_gate"],
        "verdict": verdict,
        "table": {
            "pesto": {
                "recovery_at_3": pesto_m.recovery_at_3,
                "roc_auc": pesto_m.auc,
                "sasa_margin": None,
            },
            "tokyo_eye_s1": {
                "recovery_at_3": te_m.recovery_at_3,
                "roc_auc": te_m.auc,
                "sasa_margin": te_m.sasa_margin,
            },
            "tokyo_eye_s2": {
                "recovery_at_3": s2_m.recovery_at_3,
                "roc_auc": s2_m.auc,
                "sasa_margin": s2_m.sasa_margin,
            },
            "sasa_baseline": {
                "recovery_at_3": sasa_m.recovery_at_3,
                "roc_auc": sasa_m.auc,
                "sasa_margin": None,
            },
        },
        "s1_recovery_sweep": s1_sweep,
        "locked_constants": {
            "K_PRIMARY": K_PRIMARY,
            "AUC_SUCCESS": AUC_SUCCESS,
            "AUC_FLOOR": AUC_FLOOR,
            "W_DEHYDRON": W_DEHYDRON,
            "W_SHELL": W_SHELL,
            "TAU_EPISTEMIC": TAU_EPISTEMIC,
            "W_ALEATORIC": W_ALEATORIC,
        },
        "channels": {
            str(k): v for k, v in sorted(channels.items())
        },
        "scores": {
            "s1": {str(k): v for k, v in sorted(s1_scores.items())},
            "s2": {str(k): v for k, v in sorted(s2_scores.items())},
            "sasa": {str(k): v for k, v in sorted(sasa_scores.items())},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


# ===========================================================================
# Self-tests
# ===========================================================================
def _selftest() -> int:
    sc = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
    gold = np.array([1, 1, 1, 0, 0], bool)
    assert recovery_at_k(sc, gold, 3) == 1
    gold2 = np.array([1, 1, 0, 1, 0], bool)
    assert recovery_at_k(sc, gold2, 3) == 0
    sc_t = np.array([0.9, 0.8, 0.5, 0.5, 0.1])
    gold_t = np.array([1, 1, 1, 0, 0], bool)
    assert recovery_at_k(sc_t, gold_t, 3) == 0

    assert abs(roc_auc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0], bool)) - 1.0) < 1e-9
    assert abs(roc_auc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1, 1, 0, 0], bool)) - 0.0) < 1e-9
    assert abs(roc_auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 0, 0], bool)) - 0.5) < 1e-9
    assert abs(roc_auc(np.array([0.8, 0.4, 0.6, 0.2]), np.array([1, 0, 1, 0], bool)) - 1.0) < 1e-9

    assert decide(Metrics(1, 0.85, 0.10), pesto_recovery_at_3=0).startswith("CONFIRM")
    assert decide(Metrics(0, 0.55, -0.02), pesto_recovery_at_3=0).startswith("REFUTE")
    assert decide(Metrics(1, 0.85, -0.01), pesto_recovery_at_3=0).startswith("PARTIAL")
    assert decide(Metrics(0, 0.72, 0.05), pesto_recovery_at_3=0).startswith("PARTIAL")
    assert decide(Metrics(1, 0.99, 0.5), pesto_recovery_at_3=1).startswith("VOID")

    print("self-tests PASSED: recovery@k, roc_auc, decide() all correct")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--phase1", action="store_true", help="Compute I_gold lock (output-blind)")
    ap.add_argument("--phase2", action="store_true", help="PeSTo precondition on static 9EST")
    ap.add_argument("--pesto-scores", type=Path, default=None, help="Import PeSTo scores JSON")
    ap.add_argument("--pesto-root", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run", action="store_true", help="Phase 3+4 blind evaluation")
    ap.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    ap.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    ap.add_argument("--phase2-output", type=Path, default=DEFAULT_PHASE2_OUTPUT)
    ap.add_argument("--pesto-output", type=Path, default=DEFAULT_PESTO_OUTPUT)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--output", type=Path, default=DEFAULT_PHASE34_OUTPUT)
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.phase1:
        out = run_phase1(pdb_dir=args.pdb_dir, lock_path=args.lock_path)
        print(json.dumps(out, indent=2))
        print(f"Wrote → {args.lock_path}")
        return 0
    if args.phase2:
        out = run_phase2(
            pdb_dir=args.pdb_dir,
            lock_path=args.lock_path,
            phase2_output=args.phase2_output,
            pesto_root=args.pesto_root,
            device=args.device,
            scores_path=args.pesto_scores,
        )
        print(json.dumps(out, indent=2))
        print(f"Wrote → {args.phase2_output}")
        return 0 if out["precondition_gate"] == "PASS" else 2
    if args.run:
        payload = run_phase34(
            pdb_dir=args.pdb_dir,
            lock_path=args.lock_path,
            phase2_output=args.phase2_output,
            pesto_scores_path=args.pesto_output,
            checkpoint=args.checkpoint,
            output_path=args.output,
            device=args.device,
        )
        _print_verdict_table(payload)
        print(f"\nWrote → {args.output}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
