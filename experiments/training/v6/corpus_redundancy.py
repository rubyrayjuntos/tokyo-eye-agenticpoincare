#!/usr/bin/env python3
"""Corpus redundancy matrix + Stage A draft selection (spec §4)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from experiments.training.v6._data import _download_pdb, _extract_chain, _chain_cache_dir
from experiments.training.v6.cath_coverage_probe import author_chain_from_cache, resolve_fold_from_cache
from experiments.training.v6.corpus import load_corpus_manifest

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_MANIFEST = _REPO / "manifests" / "v6_corpus_120.json"
_CATH_CACHE = _REPO / "manifests" / "cache" / "cath_pdbe"

# §3.6 locked dispositions
PRE_MATRIX_EXCLUDE: set[tuple[str, str]] = {
    ("1D0T", "A"),
    ("1VII", "A"),
    ("1E0L", "A"),
    ("1BEN", "A"),
    ("1CHO", "A"),
}

MANUAL_ENTRIES: dict[tuple[str, str], dict[str, Any]] = {
    ("6OIM", "A"): {
        "fold_id": "3.40.50.300",
        "fold_id_tier": "topology",
        "fold_id_source": "manual",
        "disposition_rule": "G-domain sibling topology; eval-only per per-fold cap",
        "role": "eval_holdout",
    },
}

DEFAULT_MAX_TRAIN_PER_FOLD = 2
DEFAULT_IDENTITY_THRESHOLD_PCT = 30.0
CROSS_FOLD_TM_THRESHOLD = 0.5


def _entry_key(pdb_id: str, chain: str) -> tuple[str, str]:
    return pdb_id.upper(), chain.upper()


def _label(pdb_id: str, chain: str) -> str:
    return f"{pdb_id.upper()}:{chain.upper()}"


def matrix_input_entries(manifest_path: Path) -> list[dict[str, Any]]:
    data = load_corpus_manifest(manifest_path)
    out: list[dict[str, Any]] = []
    for entry in data["proteins"]:
        if not entry.get("enabled", True):
            continue
        key = _entry_key(str(entry["pdb_id"]), str(entry.get("chain", "A")))
        if key in PRE_MATRIX_EXCLUDE:
            continue
        row = dict(entry)
        row["pdb_id"] = key[0]
        row["chain"] = key[1]
        manual = MANUAL_ENTRIES.get(key)
        if manual:
            row.update(manual)
        else:
            cath = resolve_fold_from_cache(key[0], key[1], cache_dir=_CATH_CACHE)
            row["fold_id"] = cath.get("fold_id")
            row["fold_id_tier"] = cath.get("fold_id_tier")
            row["fold_id_source"] = "pdbe_cache"
            row["cath_status"] = cath.get("status")
        row["structure_key"] = _label(key[0], key[1])
        out.append(row)
    return out


def _effective_chain(pdb_path: Path, pdb_id: str, manifest_chain: str) -> str:
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure(pdb_id, str(pdb_path))
    model = next(structure.get_models())
    chains = list(model.get_chains())

    def _ca_count(chain_id: str) -> int:
        chain = model[chain_id]
        return sum(1 for r in chain if r.get_id()[0] == " " and "CA" in r)

    if manifest_chain in model.child_dict and _ca_count(manifest_chain) > 0:
        return manifest_chain
    alt = author_chain_from_cache(pdb_id, manifest_chain, cache_dir=_CATH_CACHE)
    if alt and alt in model.child_dict and _ca_count(alt) > 0:
        return alt
    best = max(chains, key=lambda c: _ca_count(c.id), default=None)
    if best is None or _ca_count(best.id) == 0:
        raise ValueError(f"No CA atoms for {pdb_id}:{manifest_chain}")
    return best.id


def _chain_sequence(pdb_dir: Path, pdb_id: str, chain: str) -> str:
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    pdb_path = _download_pdb(pdb_id, pdb_dir)
    effective = _effective_chain(pdb_path, pdb_id, chain)
    chain_path = _extract_chain(pdb_path, effective, _chain_cache_dir(pdb_dir))
    structure = PDBParser(QUIET=True).get_structure(pdb_id, str(chain_path))
    residues = [r for r in structure.get_residues() if r.get_id()[0] == " " and "CA" in r]
    return "".join(seq1(r.get_resname()) for r in residues)


def _sequence_identity_pct(seq_a: str, seq_b: str) -> float:
    from Bio import pairwise2

    if not seq_a or not seq_b:
        return 0.0
    alns = pairwise2.align.globalxx(seq_a, seq_b, one_alignment_only=True)
    if not alns:
        return 0.0
    aligned_a, aligned_b, _, _, _ = alns[0]
    matches = sum(1 for a, b in zip(aligned_a, aligned_b, strict=False) if a == b and a != "-")
    denom = max(len(seq_a), len(seq_b))
    return 100.0 * matches / denom if denom else 0.0


def _ca_coords(pdb_dir: Path, pdb_id: str, chain: str) -> np.ndarray:
    from Bio.PDB import PDBParser

    pdb_path = _download_pdb(pdb_id, pdb_dir)
    effective = _effective_chain(pdb_path, pdb_id, chain)
    chain_path = _extract_chain(pdb_path, effective, _chain_cache_dir(pdb_dir))
    structure = PDBParser(QUIET=True).get_structure(pdb_id, str(chain_path))
    coords: list[np.ndarray] = []
    for residue in structure.get_residues():
        if residue.get_id()[0] != " " or "CA" not in residue:
            continue
        coords.append(np.asarray(residue["CA"].get_coord(), dtype=np.float64))
    if not coords:
        raise ValueError(f"No CA atoms for {pdb_id}:{chain}")
    return np.stack(coords)


def _tmalign_version() -> str:
    for binary in ("TMalign", "USalign"):
        path = shutil.which(binary)
        if path:
            try:
                proc = subprocess.run([path], capture_output=True, text=True, timeout=5)
                first = (proc.stdout or proc.stderr or "").splitlines()[:1]
                return f"{binary}:{first[0] if first else 'unknown'}"
            except (subprocess.SubprocessError, OSError):
                return binary
    return "unavailable"


def _tm_score_tmtools(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    seq_a: str,
    seq_b: str,
) -> float | None:
    try:
        from tmtools import tm_align
    except ImportError:
        return None
    try:
        result = tm_align(coords_a.astype(np.float64), coords_b.astype(np.float64), seq_a, seq_b)
        return float(result.tm_norm_chain1)
    except (TypeError, ValueError, RuntimeError):
        return None


def _tm_score_tmalign(coords_a: np.ndarray, coords_b: np.ndarray, pdb_dir: Path) -> float | None:
    binary = shutil.which("TMalign") or shutil.which("USalign")
    if not binary:
        return None
    with tempfile.TemporaryDirectory(prefix="tmalign_") as tmp:
        tmp_path = Path(tmp)
        path_a = tmp_path / "a.pdb"
        path_b = tmp_path / "b.pdb"
        _write_ca_pdb(coords_a, path_a)
        _write_ca_pdb(coords_b, path_b)
        proc = subprocess.run(
            [binary, str(path_a), str(path_b)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        text = proc.stdout or ""
        for line in text.splitlines():
            if "TM-score=" in line and "Chain_1" in line:
                # TM-score= 0.12345 (if normalized by length of Chain_1, ...)
                part = line.split("TM-score=")[1].split()[0]
                return float(part)
    return None


def _write_ca_pdb(coords: np.ndarray, path: Path) -> None:
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _tm_score_biotite(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """CA-only TM-score proxy via biotite superimpose (fallback when TMalign absent)."""
    import biotite.structure as struc

    n = min(len(coords_a), len(coords_b))
    if n < 10:
        return 0.0
    a = coords_a[:n]
    b = coords_b[:n]
    _, transform = struc.superimpose(a, b)
    b_fit = transform.apply(b)
    diff = a - b_fit
    d_i2 = np.sum(diff * diff, axis=1)
    l = n
    d0 = 1.24 * (l - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    tm = 1.0 / (1.0 + (d_i2.sum() / l) / (d0 * d0))
    return float(tm)


def _tm_score(
    pdb_dir: Path,
    pdb_a: str,
    chain_a: str,
    pdb_b: str,
    chain_b: str,
    *,
    cache: dict[tuple[str, str, str, str], float],
    sequences: dict[str, str],
    structural_metric: str,
) -> tuple[float, str]:
    key = (pdb_a, chain_a, pdb_b, chain_b)
    if key in cache:
        return cache[key], structural_metric
    ca_a = _ca_coords(pdb_dir, pdb_a, chain_a)
    ca_b = _ca_coords(pdb_dir, pdb_b, chain_b)
    la, lb = _label(pdb_a, chain_a), _label(pdb_b, chain_b)
    seq_a = sequences.get(la, "")
    seq_b = sequences.get(lb, "")
    score = _tm_score_tmalign(ca_a, ca_b, pdb_dir)
    metric = "tm_align_binary"
    if score is None:
        score = _tm_score_tmtools(ca_a, ca_b, seq_a, seq_b)
        metric = "tm_align_tmtools"
    if score is None:
        score = _tm_score_biotite(ca_a, ca_b)
        metric = "biotite_ca_proxy"
    cache[key] = score
    return score, metric


def _farthest_first_train(
    members: list[dict[str, Any]],
    k: int,
    tm_lookup: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Pick up to k structures maximizing structural spread (min TM among selected)."""
    if len(members) <= k:
        return list(members)
    if k <= 0:
        return []
    if k == 1:
        return [members[0]]

    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_tm = 2.0
    for a, b in combinations(members, 2):
        la, lb = a["structure_key"], b["structure_key"]
        tm = tm_lookup.get((la, lb)) or tm_lookup.get((lb, la)) or 1.0
        if tm < best_tm:
            best_tm = tm
            best_pair = (a, b)
    if best_pair is None:
        return members[:k]
    selected = list(best_pair)
    if k == 2:
        return selected
    remaining = [m for m in members if m not in selected]
    while len(selected) < k and remaining:
        best_cand = None
        best_min_tm = -1.0
        for cand in remaining:
            min_tm = min(
                tm_lookup.get((cand["structure_key"], s["structure_key"]))
                or tm_lookup.get((s["structure_key"], cand["structure_key"]))
                or 0.0
                for s in selected
            )
            if min_tm > best_min_tm:
                best_min_tm = min_tm
                best_cand = cand
        if best_cand is None:
            break
        selected.append(best_cand)
        remaining.remove(best_cand)
    return selected


def _identity_dedup_drops(
    structures: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> set[str]:
    """Within-fold high-identity clusters: keep one survivor per component (stage0 preferred)."""
    struct_by_key = {s["structure_key"]: s for s in structures}
    by_fold_edges: dict[str, list[tuple[str, str]]] = {}
    for p in pairs:
        if p["decision"] != "WITHIN_FOLD_HIGH_IDENTITY":
            continue
        a, b = p["a"], p["b"]
        sa, sb = struct_by_key[a], struct_by_key[b]
        if sa.get("role") == "eval_holdout" or sb.get("role") == "eval_holdout":
            continue
        fid = p["fold_id_a"]
        if fid:
            by_fold_edges.setdefault(fid, []).append((a, b))

    drop_keys: set[str] = set()
    for fold_id, edges in by_fold_edges.items():
        fold_keys = {
            s["structure_key"]
            for s in structures
            if s.get("fold_id") == fold_id and s.get("role") != "eval_holdout"
        }
        parent: dict[str, str] = {k: k for k in fold_keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for a, b in edges:
            if a in parent and b in parent:
                union(a, b)

        components: dict[str, list[str]] = {}
        for k in fold_keys:
            components.setdefault(find(k), []).append(k)

        for members in components.values():
            if len(members) <= 1:
                continue
            keeper = min(
                members,
                key=lambda k: (not struct_by_key[k].get("stage0", False), k),
            )
            for k in members:
                if k != keeper:
                    drop_keys.add(k)
    return drop_keys


def run_redundancy_matrix(
    manifest_path: Path,
    *,
    pdb_dir: Path,
    identity_threshold_pct: float = DEFAULT_IDENTITY_THRESHOLD_PCT,
    max_train_per_fold_id: int = DEFAULT_MAX_TRAIN_PER_FOLD,
) -> dict[str, Any]:
    entries = matrix_input_entries(manifest_path)
    structural_metric = "tm_align_binary" if shutil.which("TMalign") or shutil.which("USalign") else "tm_align_tmtools"
    try:
        import tmtools  # noqa: F401
    except ImportError:
        structural_metric = "biotite_ca_proxy"

    sequences: dict[str, str] = {}
    structures: list[dict[str, Any]] = []
    for e in entries:
        seq = _chain_sequence(pdb_dir, e["pdb_id"], e["chain"])
        sequences[e["structure_key"]] = seq
        structures.append(
            {
                **e,
                "sequence_length": len(seq),
                "flags": [],
            }
        )

    pairs: list[dict[str, Any]] = []
    tm_cache: dict[tuple[str, str, str, str], float] = {}
    tm_pair: dict[tuple[str, str], float] = {}

    for i, a in enumerate(structures):
        for b in structures[i + 1 :]:
            la, lb = a["structure_key"], b["structure_key"]
            ident = _sequence_identity_pct(sequences[la], sequences[lb])
            tm, pair_metric = _tm_score(
                pdb_dir,
                a["pdb_id"],
                a["chain"],
                b["pdb_id"],
                b["chain"],
                cache=tm_cache,
                sequences=sequences,
                structural_metric=structural_metric,
            )
            tm_pair[(la, lb)] = tm
            same_fold = a.get("fold_id") and a.get("fold_id") == b.get("fold_id")
            decision = "CROSS_FOLD_DISTINCT"
            action = "ok"
            metric_used = "tm_score"
            if same_fold:
                if ident >= identity_threshold_pct:
                    decision = "WITHIN_FOLD_HIGH_IDENTITY"
                    action = "drop_one_for_stage_a"
                    metric_used = "sequence_identity"
                else:
                    decision = "WITHIN_FOLD_DISTINCT"
                    action = "ok"
                    metric_used = "tm_score"
            elif tm >= CROSS_FOLD_TM_THRESHOLD:
                decision = "CROSS_FOLD_HIGH_TM"
                action = "review_fold_assignment"
                metric_used = "tm_score"
            pairs.append(
                {
                    "a": la,
                    "b": lb,
                    "fold_id_a": a.get("fold_id"),
                    "fold_id_b": b.get("fold_id"),
                    "same_fold_id": bool(same_fold),
                    "sequence_identity_pct": round(ident, 2),
                    "tm_score": round(tm, 4),
                    "decision": decision,
                    "action": action,
                    "metric_used": pair_metric,
                }
            )

    drop_keys = _identity_dedup_drops(structures, pairs)

    by_fold: dict[str, list[dict[str, Any]]] = {}
    for s in structures:
        if s["structure_key"] in drop_keys:
            s["flags"].append("WITHIN_FOLD_HIGH_IDENTITY_DROPPED")
            continue
        if s.get("role") == "eval_holdout":
            continue
        fid = s.get("fold_id") or "UNVERIFIED"
        by_fold.setdefault(fid, []).append(s)

    train: list[dict[str, Any]] = []
    eval_holdout: list[dict[str, Any]] = []
    violations: list[str] = []

    for s in structures:
        if s["structure_key"] in drop_keys or s.get("role") == "eval_holdout":
            row = {**s, "role": "eval_holdout", "disposition_rule": s.get("disposition_rule", "dedup or manual")}
            eval_holdout.append(row)

    for fold_id, members in sorted(by_fold.items()):
        if fold_id == "UNVERIFIED":
            violations.append("unverified fold_id in pool")
            continue
        picked = _farthest_first_train(members, max_train_per_fold_id, tm_pair)
        for p in members:
            if p in picked:
                train.append({**p, "role": "train", "disposition_rule": "per_fold_cap_structural_spread"})
            else:
                eval_holdout.append(
                    {**p, "role": "eval_holdout", "disposition_rule": "per_fold_cap_overflow"}
                )
        if len(picked) > max_train_per_fold_id:
            violations.append(f"FOLD_TRAIN_CAP_EXCEEDED:{fold_id}")

    fold_vocab = sorted({s["fold_id"] for s in structures if s.get("fold_id")})

    fold_counts = {}
    for fid in fold_vocab:
        fold_counts[fid] = {
            "pool": sum(1 for s in structures if s.get("fold_id") == fid),
            "train": sum(1 for s in train if s.get("fold_id") == fid),
            "eval_holdout": sum(1 for s in eval_holdout if s.get("fold_id") == fid),
        }

    is_provisional = structural_metric == "biotite_ca_proxy"
    report = {
        "spec_version": "stage-a-corpus-redundancy:2026-07-03",
        "status": "provisional" if is_provisional else "validated",
        "locking_artifact": not is_provisional,
        "superseded_by": "TMalign" if is_provisional else None,
        "input_manifest": str(manifest_path.relative_to(_REPO)),
        "structural_metric": structural_metric,
        "tm_align_version": _tmalign_version(),
        "within_fold_spread_metric": "tm_score_minimize",
        "sequence_identity_threshold_pct": identity_threshold_pct,
        "cross_fold_tm_threshold": CROSS_FOLD_TM_THRESHOLD,
        "max_train_per_fold_id": max_train_per_fold_id,
        "input_structure_count": len(structures),
        "excluded_pre_matrix": [_label(p, c) for p, c in sorted(PRE_MATRIX_EXCLUDE)],
        "eval_only_manual": [_label(p, c) for p, c in sorted(MANUAL_ENTRIES)],
        "structures": structures,
        "pairs": pairs,
        "fold_vocabulary": fold_vocab,
        "fold_distribution": fold_counts,
        "stage_a_draft": {
            "train_count": len(train),
            "eval_holdout_count": len(eval_holdout),
            "fold_count": len({s.get("fold_id") for s in train}),
            "violations": violations,
            "train": [{k: v for k, v in s.items() if k != "flags"} for s in train],
            "eval_holdout": [{k: v for k, v in s.items() if k != "flags"} for s in eval_holdout],
        },
        "pass": len(violations) == 0 and len(train) > 0,
    }
    return report


def write_draft_manifest(report: dict[str, Any], out_path: Path) -> None:
    train = report["stage_a_draft"]["train"]
    holdout = report["stage_a_draft"]["eval_holdout"]
    proteins = []
    for row in train + holdout:
        proteins.append(
            {
                "pdb_id": row["pdb_id"],
                "chain": row["chain"],
                "gene": row.get("gene"),
                "fold_id": row.get("fold_id"),
                "fold_id_tier": row.get("fold_id_tier"),
                "fold_id_source": row.get("fold_id_source"),
                "role": row["role"],
                "disposition_rule": row.get("disposition_rule"),
                "enabled": row["role"] == "train",
            }
        )
    payload = {
        "version": "1.0",
        "description": "Stage A draft — human review required before lock",
        "source_report": "corpus_redundancy_report.json",
        "max_sequence_identity_pct": report["sequence_identity_threshold_pct"],
        "proteins": proteins,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Corpus redundancy matrix + Stage A draft")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--pdb-dir", type=Path, default=Path("/tmp/dtie_pdb_cache"))
    parser.add_argument("--report-out", type=Path, default=_REPO / "manifests" / "corpus_redundancy_report.json")
    parser.add_argument("--draft-out", type=Path, default=_REPO / "manifests" / "v6_corpus_stage_a_draft.json")
    parser.add_argument("--identity-threshold", type=float, default=DEFAULT_IDENTITY_THRESHOLD_PCT)
    parser.add_argument("--max-train-per-fold", type=int, default=DEFAULT_MAX_TRAIN_PER_FOLD)
    args = parser.parse_args()

    args.pdb_dir.mkdir(parents=True, exist_ok=True)
    report = run_redundancy_matrix(
        args.manifest,
        pdb_dir=args.pdb_dir,
        identity_threshold_pct=args.identity_threshold,
        max_train_per_fold_id=args.max_train_per_fold,
    )
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    write_draft_manifest(report, args.draft_out)

    draft = report["stage_a_draft"]
    print(f"Input structures: {report['input_structure_count']}")
    print(f"Structural metric: {report['structural_metric']} ({report['tm_align_version']})")
    print(f"Fold vocabulary: {len(report['fold_vocabulary'])} folds")
    print(f"Train: {draft['train_count']} | Eval/holdout: {draft['eval_holdout_count']}")
    print(f"Pass: {report['pass']}")
    if draft["violations"]:
        print("Violations:", draft["violations"])
    print(f"Report: {args.report_out}")
    print(f"Draft:  {args.draft_out}")


if __name__ == "__main__":
    main()
