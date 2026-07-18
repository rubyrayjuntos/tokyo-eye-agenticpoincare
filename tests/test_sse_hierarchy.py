"""Deposited HELIX/SHEET parse + residue mapping (v6.6 containment Path B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from science.dtie.v66.sse_hierarchy import (
    map_sse_ranges_to_node_indices,
    parse_pdb_helix_sheet,
)

UBQ_HELIX = """\
HELIX    1   1 ILE A   23  GLY A   34  1                                  12
ATOM      1  CA  ILE A  23      0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLY A  34      1.000   0.000   0.000  1.00  0.00           C
END
"""


def test_parse_helix_range_inclusive() -> None:
    ranges = parse_pdb_helix_sheet(UBQ_HELIX)
    assert len(ranges) == 1
    assert ranges[0].sse_type == "H"
    assert ranges[0].chain == "A"
    assert ranges[0].start_resseq == 23
    assert ranges[0].end_resseq == 34


def test_map_range_to_residue_ids() -> None:
    residue_ids = [f"A:{i}:" for i in range(23, 35)]
    mapped = map_sse_ranges_to_node_indices(parse_pdb_helix_sheet(UBQ_HELIX), residue_ids)
    assert len(mapped) == 1
    assert mapped[0][1] == list(range(12))


def _residue_ids_from_ca_atoms(pdb_text: str) -> list[str]:
    """Build training-style residue_ids from CA ATOM records."""
    ids: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain = line[21].strip() or " "
        resseq = int(line[22:26].strip())
        ids.append(f"{chain}:{resseq}:")
    return ids


def test_1ubq_smoke_helix_and_sheet_mapped() -> None:
    pdb_path = Path(__file__).resolve().parents[1] / "pdb_cache" / "1UBQ.pdb"
    if not pdb_path.is_file():
        pytest.skip(f"missing fixture {pdb_path}")

    pdb_text = pdb_path.read_text()
    ranges = parse_pdb_helix_sheet(pdb_text)
    helix_ranges = [r for r in ranges if r.sse_type == "H"]
    sheet_ranges = [r for r in ranges if r.sse_type == "E"]
    assert len(helix_ranges) >= 1
    assert len(sheet_ranges) >= 1

    residue_ids = _residue_ids_from_ca_atoms(pdb_text)
    assert residue_ids

    mapped = map_sse_ranges_to_node_indices(ranges, residue_ids)
    mapped_helix = [m for m in mapped if m[0].sse_type == "H"]
    mapped_sheet = [m for m in mapped if m[0].sse_type == "E"]
    assert len(mapped_helix) >= 1
    assert len(mapped_sheet) >= 1
    assert all(len(indices) > 0 for _, indices in mapped)
