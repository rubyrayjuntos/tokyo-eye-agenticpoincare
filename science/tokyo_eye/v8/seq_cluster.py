"""Sequence clustering / identity holdout for v8 PDB-Bind splits (Sprint 10).

Backends (priority):
1. ``mmseqs`` easy-search / easy-cluster if on PATH
2. Exact-sequence match + k-mer Jaccard purge (conservative Core wall)
3. Optional Biopython pairwise confirm for borderline pairs
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def normalize_seq(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c.isalpha())


def kmer_jaccard(a: str, b: str, *, k: int = 5) -> float:
    a = normalize_seq(a)
    b = normalize_seq(b)
    if len(a) < k or len(b) < k:
        return 1.0 if a == b and a else 0.0
    sa = {a[i : i + k] for i in range(len(a) - k + 1)}
    sb = {b[i : i + k] for i in range(len(b) - k + 1)}
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 0.0


def mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def _write_fasta(path: Path, items: Mapping[str, str]) -> None:
    with path.open("w") as f:
        for sid, seq in items.items():
            f.write(f">{sid}\n{normalize_seq(seq)}\n")


def cluster_sequences_mmseqs(
    sequences: Mapping[str, str],
    *,
    min_seq_id: float = 0.30,
    coverage: float = 0.8,
    threads: int = 4,
) -> dict[str, str]:
    """Return ``{member_id: cluster_rep_id}`` via mmseqs easy-cluster."""
    if not mmseqs_available():
        raise RuntimeError("mmseqs not on PATH")
    with tempfile.TemporaryDirectory(prefix="v8_mmseqs_") as tmp:
        td = Path(tmp)
        fasta = td / "seqs.fasta"
        _write_fasta(fasta, sequences)
        prefix = td / "clu"
        cmd = [
            "mmseqs",
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(td / "tmp"),
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--threads",
            str(threads),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tsv = Path(str(prefix) + "_cluster.tsv")
        if not tsv.is_file():
            # mmseqs naming variants
            cands = list(td.glob("*_cluster.tsv"))
            if not cands:
                raise FileNotFoundError("mmseqs cluster TSV missing")
            tsv = cands[0]
        mapping: dict[str, str] = {}
        with tsv.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[1]] = parts[0]
        for sid in sequences:
            mapping.setdefault(sid, sid)
        return mapping


def cluster_sequences_exact(sequences: Mapping[str, str]) -> dict[str, str]:
    """Cluster by exact normalized sequence; rep = lexicographically smallest id."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for sid, seq in sequences.items():
        buckets[normalize_seq(seq)].append(sid)
    mapping: dict[str, str] = {}
    for _seq, ids in buckets.items():
        rep = sorted(ids)[0]
        for sid in ids:
            mapping[sid] = rep
    return mapping


def _kmer_set(seq: str, k: int) -> set[str]:
    seq = normalize_seq(seq)
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def ids_hitting_core(
    query_seqs: Mapping[str, str],
    core_seqs: Mapping[str, str],
    *,
    identity_floor: float = 0.30,
    kmer_k: int = 5,
    kmer_jaccard_floor: float = 0.45,
) -> set[str]:
    """Return query IDs that must be purged (exact or ≥ identity / k-mer wall).

    When mmseqs is available, uses search at ``identity_floor``.
    Otherwise: exact match OR k-mer Jaccard ≥ ``kmer_jaccard_floor`` via inverted index.
    """
    hit: set[str] = set()
    core_norm = {cid: normalize_seq(s) for cid, s in core_seqs.items()}
    core_by_seq = {s: cid for cid, s in core_norm.items()}
    for qid, qseq in query_seqs.items():
        nq = normalize_seq(qseq)
        if nq in core_by_seq:
            hit.add(qid)

    if mmseqs_available() and query_seqs and core_seqs:
        with tempfile.TemporaryDirectory(prefix="v8_mmseqs_search_") as tmp:
            td = Path(tmp)
            qf, cf = td / "q.fasta", td / "c.fasta"
            _write_fasta(qf, query_seqs)
            _write_fasta(cf, core_seqs)
            out = td / "hits.m8"
            cmd = [
                "mmseqs",
                "easy-search",
                str(qf),
                str(cf),
                str(out),
                str(td / "tmp"),
                "--min-seq-id",
                str(identity_floor),
                "-c",
                "0.5",
                "--threads",
                "4",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out.is_file():
                with out.open() as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if parts:
                            hit.add(parts[0])
        return hit

    # Inverted k-mer index → candidate Core IDs (avoids O(|Q|·|C|) full scans)
    inv: dict[str, set[str]] = defaultdict(set)
    core_kmers: dict[str, set[str]] = {}
    for cid, nc in core_norm.items():
        ks = _kmer_set(nc, kmer_k)
        core_kmers[cid] = ks
        for km in ks:
            inv[km].add(cid)

    floor = float(kmer_jaccard_floor)
    for qid, qseq in query_seqs.items():
        if qid in hit:
            continue
        nq = normalize_seq(qseq)
        qks = _kmer_set(nq, kmer_k)
        if not qks:
            continue
        counts: dict[str, int] = defaultdict(int)
        for km in qks:
            for cid in inv.get(km, ()):
                counts[cid] += 1
        for cid, inter in counts.items():
            cks = core_kmers[cid]
            union = len(qks) + len(cks) - inter
            if union > 0 and (inter / union) >= floor:
                hit.add(qid)
                break
    return hit


def split_reps_train_val(
    rep_ids: Sequence[str],
    *,
    val_fraction: float = 0.20,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Deterministic shuffle of cluster representatives → train/val."""
    import random

    ids = sorted(rep_ids)
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * float(val_fraction)))) if ids else 0
    # Keep at least one train when possible
    if len(ids) >= 2:
        n_val = min(n_val, len(ids) - 1)
    val = sorted(ids[:n_val])
    train = sorted(ids[n_val:])
    return train, val


def assert_no_core_leak(
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    core_ids: Iterable[str],
) -> None:
    """Hard containment: train∪val must be disjoint from core PDB IDs."""
    core = {str(x).strip().upper() for x in core_ids}
    train = {str(x).strip().upper() for x in train_ids}
    val = {str(x).strip().upper() for x in val_ids}
    leak = (train | val) & core
    if leak:
        raise AssertionError(
            f"CORE LEAK WALL VIOLATED: {len(leak)} IDs in train/val ∩ core "
            f"(examples: {sorted(leak)[:10]})"
        )


def content_hash(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "assert_no_core_leak",
    "cluster_sequences_exact",
    "cluster_sequences_mmseqs",
    "content_hash",
    "ids_hitting_core",
    "kmer_jaccard",
    "mmseqs_available",
    "normalize_seq",
    "split_reps_train_val",
]
