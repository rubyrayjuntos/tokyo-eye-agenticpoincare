"""Protein chain eligibility gates for computation scope selection.

Thresholds are SSOT for ingest chain selection — see
``docs/specs/structure-ingestion-normalization/design.md`` §4.1 (CA eligibility).

Calibrated against the v6 Stage-A corpus: must accept 1UBQ (76 CA) and reject
peptide ligands / nucleic-acid polymers mis-tagged as protein (e.g. 1BG1 DNA).
"""

from __future__ import annotations

from science.dtie.ingest.parser import ParsedChain, ParsedResidue

# Minimum resolved Cα-bearing residues for primary protein selection.
# Below 1UBQ (76); rejects short peptide ligands that pass fraction alone.
PROTEIN_CHAIN_MIN_CA_COUNT: int = 15

# Minimum fraction of chain residues with Cα (or complete resolved backbone).
# Rejects long nucleic-acid chains with zero Cα (1BG1 DNA: 639 res, 0 CA).
PROTEIN_CHAIN_MIN_CA_FRACTION: float = 0.25

# When duplicate entity instances score within this band, treat them as equivalent
# and prefer lexicographically first auth_asym_id (4OBE KRAS A vs B: Δscore≈0.001).
DUPLICATE_REPRESENTATIVE_SCORE_EPSILON: float = 0.05


def count_ca_residues(chain: ParsedChain) -> int:
    """Count residues with a resolved Cα atom (or complete backbone proxy)."""
    n = 0
    for residue in chain.residues:
        if _residue_has_ca(residue):
            n += 1
    return n


def ca_fraction(chain: ParsedChain) -> float:
    """Fraction of chain residues with Cα / complete resolved backbone."""
    total = len(chain.residues)
    if total == 0:
        return 0.0
    return count_ca_residues(chain) / total


def passes_ca_eligibility(chain: ParsedChain) -> bool:
    """True when chain meets minimum Cα count and fraction for protein scope."""
    ca_count = count_ca_residues(chain)
    if ca_count < PROTEIN_CHAIN_MIN_CA_COUNT:
        return False
    return ca_fraction(chain) >= PROTEIN_CHAIN_MIN_CA_FRACTION


def _residue_has_ca(residue: ParsedResidue) -> bool:
    if any(atom.atom_name == "CA" for atom in residue.atoms):
        return True
    # Parser sets partial_backbone when N/CA/C not all present; resolved + complete
    # backbone implies Cα even when atom list is omitted (synthetic/test fixtures).
    return bool(residue.is_resolved and not residue.partial_backbone)
