"""Chain scorer — multi-factor quality scoring and computation scope assignment.

Scores each chain in a parsed structure using multiple quality signals, collapses
duplicate entity instances to select one representative per entity_id, and produces
a ComputationScope specifying which chains to use for downstream computation.

Requirements: 4.1, 4.2, 4.5, 4.6
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from science.dtie.ingest.chain_eligibility import (
    DUPLICATE_REPRESENTATIVE_SCORE_EPSILON,
    PROTEIN_CHAIN_MIN_CA_COUNT,
    PROTEIN_CHAIN_MIN_CA_FRACTION,
    ca_fraction,
    count_ca_residues,
    passes_ca_eligibility,
)
from science.dtie.ingest.metadata import EntityMetadata, StructureMetadata
from science.dtie.ingest.parser import ParsedChain, ParsedStructure

logger = logging.getLogger(__name__)


class NoEligibleProteinChainError(ValueError):
    """Raised when no chain passes protein + Cα eligibility for scope selection."""

    def __init__(
        self,
        pdb_id: str,
        chain_summaries: list[dict[str, object]],
    ) -> None:
        self.pdb_id = pdb_id
        self.chain_summaries = chain_summaries
        super().__init__(
            f"No eligible protein chain for {pdb_id} "
            f"(min_ca_count={PROTEIN_CHAIN_MIN_CA_COUNT}, "
            f"min_ca_fraction={PROTEIN_CHAIN_MIN_CA_FRACTION}); "
            f"chains={chain_summaries}"
        )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class QualityFilters:
    """Quality filter thresholds for computation scope."""

    max_b_factor_threshold: float = 100.0
    min_resolution: float | None = None


@dataclass
class ChainScore:
    """Scoring result for a single chain."""

    chain_id: str
    auth_asym_id: str
    score: float
    factors: dict[str, float] = field(default_factory=dict)
    is_representative: bool = False


@dataclass
class ComputationScope:
    """Defines which chains and parameters to use for downstream computation."""

    primary_chain_ids: list[str]
    reference_chain: str
    exclude_chain_ids: list[str]
    scope_source: str  # 'auto' or 'user'
    selection_reason: str
    normalization_protocol: str  # default: 'graph_default'
    quality_filters: QualityFilters | None = None


# ---------------------------------------------------------------------------
# Scoring weights (from design spec §4)
# ---------------------------------------------------------------------------

SCORING_WEIGHTS: dict[str, float] = {
    "is_protein": 10.0,
    "uniprot_coverage": 3.0,
    "resolved_fraction": 2.0,
    "b_factor_quality": 1.5,
    "not_duplicate": 2.0,
    "low_mutation_burden": 1.0,
    "sequence_length": 0.5,
}

# Entity types considered "protein"
_PROTEIN_ENTITY_TYPES = frozenset({
    "polymer",
    "polypeptide(l)",
    "polypeptide(d)",
    "protein",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_chains(
    parsed: ParsedStructure,
    metadata: StructureMetadata | None = None,
) -> ComputationScope:
    """Score all chains and select primary + reference for computation scope."""
    scope, _ = score_chains_detailed(parsed, metadata)
    return scope


def score_chains_detailed(
    parsed: ParsedStructure,
    metadata: StructureMetadata | None = None,
) -> tuple[ComputationScope, list[ChainScore]]:
    """Score all chains; returns scope and per-chain scores.

    Raises:
        NoEligibleProteinChainError: No chain passes protein + Cα eligibility.
    """
    if not parsed.chains:
        empty = ComputationScope(
            primary_chain_ids=[],
            reference_chain="",
            exclude_chain_ids=[],
            scope_source="auto",
            selection_reason="no chains in structure",
            normalization_protocol="graph_default",
        )
        return empty, []

    # Build metadata lookups
    entity_uniprot = _build_entity_uniprot_map(metadata)
    duplicate_entity_ids = _detect_duplicate_entities(parsed.chains)

    # Score each chain
    chain_scores: list[ChainScore] = []
    for chain in parsed.chains:
        score = _score_single_chain(chain, entity_uniprot, duplicate_entity_ids)
        chain_scores.append(score)

    # Collapse duplicate entities → select one representative per entity_id
    representatives = _select_representatives(parsed.chains, chain_scores, duplicate_entity_ids)

    # Mark representative status on scores
    rep_set = {cs.auth_asym_id for cs in representatives}
    for cs in chain_scores:
        cs.is_representative = cs.auth_asym_id in rep_set

    # Select primary chain: highest scoring eligible protein representative
    protein_reps = _eligible_protein_representatives(parsed.chains, representatives)

    if not protein_reps:
        summaries = [_chain_eligibility_summary(c) for c in parsed.chains]
        raise NoEligibleProteinChainError(parsed.pdb_id, summaries)

    protein_reps.sort(key=lambda cs: cs.score, reverse=True)
    primary = protein_reps[0]
    primary_chain_ids = [primary.auth_asym_id]

    # Exclude non-representatives from duplicate groups
    exclude_chain_ids = []
    for chain in parsed.chains:
        if chain.auth_asym_id == primary.auth_asym_id:
            continue
        cs = next(s for s in chain_scores if s.auth_asym_id == chain.auth_asym_id)
        if not cs.is_representative:
            exclude_chain_ids.append(chain.auth_asym_id)

    # Build reason string
    reason_parts = []
    reason_parts.append(f"chain {primary.auth_asym_id} scored {primary.score:.2f}")
    top_factors = sorted(primary.factors.items(), key=lambda kv: kv[1], reverse=True)[:3]
    factor_str = ", ".join(f"{k}={v:.2f}" for k, v in top_factors)
    reason_parts.append(f"top factors: {factor_str}")
    if duplicate_entity_ids:
        reason_parts.append(
            f"collapsed {len(duplicate_entity_ids)} duplicate entity group(s)"
        )
    selection_reason = "; ".join(reason_parts)

    scope = ComputationScope(
        primary_chain_ids=primary_chain_ids,
        reference_chain=primary.auth_asym_id,
        exclude_chain_ids=sorted(exclude_chain_ids),
        scope_source="auto",
        selection_reason=selection_reason,
        normalization_protocol="graph_default",
        quality_filters=QualityFilters(),
    )
    return scope, chain_scores


def score_chains_interface(
    parsed: ParsedStructure,
    metadata: StructureMetadata | None = None,
) -> ComputationScope:
    """Score chains for interface analysis (multi-chain scope).

    Selects all protein representative chains for interface computation.
    Entity-instance collapse is disabled for interface protocol.

    Args:
        parsed: ParsedStructure from the BinaryCIF parser.
        metadata: Optional StructureMetadata from enrichment.

    Returns:
        ComputationScope with multiple primary_chain_ids for interface analysis.
    """
    if not parsed.chains:
        return ComputationScope(
            primary_chain_ids=[],
            reference_chain="",
            exclude_chain_ids=[],
            scope_source="auto",
            selection_reason="no chains for interface analysis",
            normalization_protocol="interface",
        )

    entity_uniprot = _build_entity_uniprot_map(metadata)
    duplicate_entity_ids = _detect_duplicate_entities(parsed.chains)

    # Score all chains
    chain_scores: list[ChainScore] = []
    for chain in parsed.chains:
        score = _score_single_chain(chain, entity_uniprot, duplicate_entity_ids)
        chain_scores.append(score)

    # For interface: include all eligible protein chains (no entity collapse)
    protein_scores = _eligible_protein_scores(parsed.chains, chain_scores)

    if not protein_scores:
        summaries = [_chain_eligibility_summary(c) for c in parsed.chains]
        raise NoEligibleProteinChainError(parsed.pdb_id, summaries)

    protein_scores.sort(key=lambda cs: cs.score, reverse=True)

    primary_chain_ids = [cs.auth_asym_id for cs in protein_scores]
    reference_chain = primary_chain_ids[0] if primary_chain_ids else ""

    # Exclude non-protein chains
    non_protein = [
        c.auth_asym_id for c in parsed.chains
        if not _is_protein_chain(c)
    ]

    return ComputationScope(
        primary_chain_ids=primary_chain_ids,
        reference_chain=reference_chain,
        exclude_chain_ids=sorted(non_protein),
        scope_source="auto",
        selection_reason=f"interface mode: {len(primary_chain_ids)} protein chains selected",
        normalization_protocol="interface",
        quality_filters=QualityFilters(),
    )


# ---------------------------------------------------------------------------
# Internal: Scoring
# ---------------------------------------------------------------------------


def _score_single_chain(
    chain: ParsedChain,
    entity_uniprot: dict[str, list[str]],
    duplicate_entity_ids: set[str],
) -> ChainScore:
    """Compute a multi-factor score for a single chain."""
    factors: dict[str, float] = {}

    # Factor 1: Is protein (10 points)
    is_protein = _is_protein_chain(chain)
    factors["is_protein"] = SCORING_WEIGHTS["is_protein"] if is_protein else 0.0

    # Factor 2: UniProt coverage (up to 3 points)
    uniprot_accessions = entity_uniprot.get(chain.entity_id, [])
    if uniprot_accessions and chain.residues:
        # Having any UniProt mapping gives full credit
        factors["uniprot_coverage"] = SCORING_WEIGHTS["uniprot_coverage"]
    else:
        factors["uniprot_coverage"] = 0.0

    # Factor 3: Resolved fraction (up to 2 points)
    if chain.residues:
        resolved_count = sum(1 for r in chain.residues if r.is_resolved)
        total_count = len(chain.residues)
        resolved_frac = resolved_count / total_count if total_count > 0 else 0.0
        factors["resolved_fraction"] = SCORING_WEIGHTS["resolved_fraction"] * resolved_frac
    else:
        factors["resolved_fraction"] = 0.0

    # Factor 4: B-factor quality (up to 1.5 points)
    # Lower mean B-factor → higher quality
    resolved_residues = [r for r in chain.residues if r.is_resolved and r.max_b_factor is not None]
    if resolved_residues:
        mean_b = sum(r.max_b_factor for r in resolved_residues) / len(resolved_residues)  # type: ignore[arg-type]
        # Normalize: 0-50 Å² is excellent (1.0), 50-100 is moderate (0.5), >100 is poor (0.0)
        if mean_b <= 50.0:
            b_quality = 1.0
        elif mean_b <= 100.0:
            b_quality = 1.0 - (mean_b - 50.0) / 100.0
        else:
            b_quality = 0.0
        factors["b_factor_quality"] = SCORING_WEIGHTS["b_factor_quality"] * b_quality
    else:
        factors["b_factor_quality"] = 0.0

    # Factor 5: Not a duplicate entity instance (2 points if unique)
    is_duplicate = chain.entity_id in duplicate_entity_ids
    factors["not_duplicate"] = SCORING_WEIGHTS["not_duplicate"] if not is_duplicate else 0.0

    # Factor 6: Low mutation burden (up to 1 point)
    # Modified residues as proxy for engineered mutations
    if chain.residues:
        modified_count = sum(1 for r in chain.residues if r.is_modified)
        total_count = len(chain.residues)
        mutation_frac = modified_count / total_count if total_count > 0 else 0.0
        # Fewer modifications → higher score
        factors["low_mutation_burden"] = SCORING_WEIGHTS["low_mutation_burden"] * (
            1.0 - mutation_frac
        )
    else:
        factors["low_mutation_burden"] = 0.0

    # Factor 7: Sequence length (up to 0.5 points, tiebreaker)
    # Normalize to 0-1 range: longer chains slightly preferred
    seq_len = len(chain.residues)
    # Cap at 500 residues for normalization
    norm_len = min(seq_len / 500.0, 1.0)
    factors["sequence_length"] = SCORING_WEIGHTS["sequence_length"] * norm_len

    total_score = sum(factors.values())

    return ChainScore(
        chain_id=f"{chain.auth_asym_id}",  # Will be qualified by caller if needed
        auth_asym_id=chain.auth_asym_id,
        score=total_score,
        factors=factors,
    )


# ---------------------------------------------------------------------------
# Internal: Entity duplicate handling
# ---------------------------------------------------------------------------


def _detect_duplicate_entities(chains: list[ParsedChain]) -> set[str]:
    """Detect entity_ids that appear on multiple chains."""
    entity_counts: dict[str, int] = defaultdict(int)
    for chain in chains:
        entity_counts[chain.entity_id] += 1
    return {eid for eid, count in entity_counts.items() if count > 1}


def _select_representatives(
    chains: list[ParsedChain],
    chain_scores: list[ChainScore],
    duplicate_entity_ids: set[str],
) -> list[ChainScore]:
    """Select one representative chain per entity_id from duplicates.

    For non-duplicate entities, the chain is automatically a representative.
    For duplicate entities, pick the highest-scoring chain; when scores are
    within DUPLICATE_REPRESENTATIVE_SCORE_EPSILON, prefer lexicographically
    first auth_asym_id (deterministic, no caller hints).
    """
    # Build score lookup
    score_by_chain = {cs.auth_asym_id: cs for cs in chain_scores}

    # Group chains by entity_id
    entity_chains: dict[str, list[ParsedChain]] = defaultdict(list)
    for chain in chains:
        entity_chains[chain.entity_id].append(chain)

    representatives: list[ChainScore] = []

    for entity_id, entity_chain_list in entity_chains.items():
        if entity_id in duplicate_entity_ids:
            group_scores = [score_by_chain[c.auth_asym_id] for c in entity_chain_list]
            representatives.append(_pick_near_tied_representative(group_scores))
        else:
            # Single chain for this entity — it's automatically representative
            representatives.append(score_by_chain[entity_chain_list[0].auth_asym_id])

    return representatives


# ---------------------------------------------------------------------------
# Internal: Helpers
# ---------------------------------------------------------------------------


def _pick_near_tied_representative(group_scores: list[ChainScore]) -> ChainScore:
    """Pick representative; near-tied duplicate instances prefer first auth_asym_id."""
    if not group_scores:
        raise ValueError("group_scores must not be empty")
    max_score = max(cs.score for cs in group_scores)
    near_top = [
        cs for cs in group_scores
        if max_score - cs.score <= DUPLICATE_REPRESENTATIVE_SCORE_EPSILON
    ]
    near_top.sort(key=lambda cs: cs.auth_asym_id)
    return near_top[0]


def _is_protein_chain(chain: ParsedChain) -> bool:
    """Check if a chain is a protein chain based on entity_type."""
    return chain.entity_type.lower() in _PROTEIN_ENTITY_TYPES


def _is_eligible_protein_chain(chain: ParsedChain) -> bool:
    """Protein entity type plus Cα count/fraction floors."""
    return _is_protein_chain(chain) and passes_ca_eligibility(chain)


def _eligible_protein_representatives(
    chains: list[ParsedChain],
    representatives: list[ChainScore],
) -> list[ChainScore]:
    chain_by_label = {c.auth_asym_id: c for c in chains}
    return [
        cs for cs in representatives
        if _is_eligible_protein_chain(chain_by_label[cs.auth_asym_id])
    ]


def _eligible_protein_scores(
    chains: list[ParsedChain],
    chain_scores: list[ChainScore],
) -> list[ChainScore]:
    chain_by_label = {c.auth_asym_id: c for c in chains}
    return [
        cs for cs in chain_scores
        if _is_eligible_protein_chain(chain_by_label[cs.auth_asym_id])
    ]


def _chain_eligibility_summary(chain: ParsedChain) -> dict[str, object]:
    return {
        "label": chain.auth_asym_id,
        "entity_type": chain.entity_type,
        "residue_count": len(chain.residues),
        "ca_count": count_ca_residues(chain),
        "ca_fraction": round(ca_fraction(chain), 4),
        "is_protein_entity": _is_protein_chain(chain),
        "eligible": _is_eligible_protein_chain(chain),
    }


def _build_entity_uniprot_map(
    metadata: StructureMetadata | None,
) -> dict[str, list[str]]:
    """Build entity_id → UniProt accessions mapping from metadata."""
    if metadata is None:
        return {}
    mapping: dict[str, list[str]] = {}
    for entity in metadata.entities:
        if entity.uniprot_accessions:
            mapping[entity.entity_id] = entity.uniprot_accessions
    return mapping
