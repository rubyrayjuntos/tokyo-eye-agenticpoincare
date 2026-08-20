"""Parse deposited HELIX/SHEET secondary-structure ranges from PDB text.

Path B containment uses these ranges to define SSE parent nodes over residue
children. Residue ``ss`` / ``ss_type`` from biotite may disagree at boundaries
(design §9.1); this module reads **deposited** HELIX/SHEET records only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# PDB column slices (0-based, end-exclusive) for deposited HELIX/SHEET records.
# Chain/resseq columns follow classic fixed-width layout as seen in pdb_cache/*.pdb
# (chain sits after the blank separator following the 3-letter residue name).
_HELIX_ID_SLICE = slice(11, 14)
_HELIX_INIT_CHAIN_SLICE = slice(19, 20)
_HELIX_INIT_RESSEQ_SLICE = slice(22, 26)
_HELIX_END_CHAIN_SLICE = slice(31, 32)
_HELIX_END_RESSEQ_SLICE = slice(33, 37)

_SHEET_ID_SLICE = slice(11, 14)
_SHEET_INIT_CHAIN_SLICE = slice(21, 22)
_SHEET_INIT_RESSEQ_SLICE = slice(22, 26)
_SHEET_END_CHAIN_SLICE = slice(32, 33)
_SHEET_END_RESSEQ_SLICE = slice(33, 37)


@dataclass(frozen=True)
class SSERange:
    """Inclusive deposited SSE span on one chain."""

    sse_type: str  # "H" | "E"
    chain: str
    start_resseq: int
    end_resseq: int
    helix_id: str | None = None
    sheet_id: str | None = None


def _parse_resseq(field: str) -> int:
    return int(field.strip())


def _parse_chain(field: str) -> str:
    return field.strip() or " "


def _parse_helix_line(line: str) -> SSERange | None:
    if len(line) < 37:
        return None
    init_chain = _parse_chain(line[_HELIX_INIT_CHAIN_SLICE])
    end_chain = _parse_chain(line[_HELIX_END_CHAIN_SLICE])
    if init_chain != end_chain:
        logger.debug("Skipping cross-chain HELIX record: %s", line.rstrip())
        return None
    start_resseq = _parse_resseq(line[_HELIX_INIT_RESSEQ_SLICE])
    end_resseq = _parse_resseq(line[_HELIX_END_RESSEQ_SLICE])
    if end_resseq < start_resseq:
        start_resseq, end_resseq = end_resseq, start_resseq
    helix_id = line[_HELIX_ID_SLICE].strip() or None
    return SSERange(
        sse_type="H",
        chain=init_chain,
        start_resseq=start_resseq,
        end_resseq=end_resseq,
        helix_id=helix_id,
    )


def _parse_sheet_line(line: str) -> SSERange | None:
    if len(line) < 37:
        return None
    init_chain = _parse_chain(line[_SHEET_INIT_CHAIN_SLICE])
    end_chain = _parse_chain(line[_SHEET_END_CHAIN_SLICE])
    if init_chain != end_chain:
        logger.debug("Skipping cross-chain SHEET record: %s", line.rstrip())
        return None
    start_resseq = _parse_resseq(line[_SHEET_INIT_RESSEQ_SLICE])
    end_resseq = _parse_resseq(line[_SHEET_END_RESSEQ_SLICE])
    if end_resseq < start_resseq:
        start_resseq, end_resseq = end_resseq, start_resseq
    sheet_id = line[_SHEET_ID_SLICE].strip() or None
    return SSERange(
        sse_type="E",
        chain=init_chain,
        start_resseq=start_resseq,
        end_resseq=end_resseq,
        sheet_id=sheet_id,
    )


def parse_pdb_helix_sheet(pdb_text: str) -> list[SSERange]:
    """Parse classic PDB HELIX and SHEET records into inclusive residue ranges."""
    ranges: list[SSERange] = []
    for line in pdb_text.splitlines():
        if line.startswith("HELIX"):
            parsed = _parse_helix_line(line)
            if parsed is not None:
                ranges.append(parsed)
        elif line.startswith("SHEET"):
            parsed = _parse_sheet_line(line)
            if parsed is not None:
                ranges.append(parsed)
    return ranges


def map_sse_ranges_to_node_indices(
    ranges: list[SSERange],
    residue_ids: Sequence[str],
) -> list[tuple[SSERange, list[int]]]:
    """Map SSE ranges to residue node indices via exact ``f\"{chain}:{resseq}:`` keys."""
    key_to_index = {rid: idx for idx, rid in enumerate(residue_ids)}
    mapped: list[tuple[SSERange, list[int]]] = []
    dropped = 0

    for sse_range in ranges:
        indices: list[int] = []
        for resseq in range(sse_range.start_resseq, sse_range.end_resseq + 1):
            key = f"{sse_range.chain}:{resseq}:"
            idx = key_to_index.get(key)
            if idx is not None:
                indices.append(idx)
        if indices:
            mapped.append((sse_range, indices))
        else:
            dropped += 1

    if dropped:
        logger.info("Dropped %d SSE range(s) with zero residue_id hits", dropped)
    return mapped
