"""Base protocol definitions and registry for normalization protocols.

Each protocol is a versioned, named configuration that defines the exact
transform applied JIT before a specific computation task. Protocols produce
a NormalizedView from a ParsedStructure + ComputationScope.

Requirements: 5.1, 5.6
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from science.dtie.ingest.chain_scorer import ComputationScope
from science.dtie.ingest.parser import ParsedAtom, ParsedChain, ParsedResidue, ParsedStructure

logger = logging.getLogger(__name__)


class ProtocolName(str, Enum):
    """Named normalization protocols."""

    GRAPH_DEFAULT = "graph_default"
    FAMILY_COMPARE = "family_compare"
    BINDING_SITE = "binding_site"
    INTERFACE = "interface"


@dataclass
class NormalizedResidue:
    """A residue after protocol filtering, with only retained atoms."""

    residue_id: str | None  # Canonical key (set by caller if needed)
    auth_seq_id: int
    label_seq_id: int | None
    insertion_code: str | None
    residue_name: str
    residue_name_3: str
    comp_id: str
    parent_comp_id: str | None
    is_modified: bool
    atoms: list[ParsedAtom] = field(default_factory=list)


@dataclass
class NormalizedView:
    """The output of applying a normalization protocol to a parsed structure.

    This is a task-specific, filtered view of the raw ingested data.
    """

    structure_id: str
    protocol_name: str
    protocol_version: int
    chains: dict[str, list[NormalizedResidue]]  # auth_asym_id → residues
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class NormalizationProtocol(ABC):
    """Abstract base class for normalization protocols.

    Each protocol:
    - Has a unique name (ProtocolName enum)
    - Is versioned (version integer)
    - Carries protocol-specific parameters
    - Implements apply() to produce a NormalizedView
    """

    name: ProtocolName
    version: int
    parameters: dict

    @abstractmethod
    def apply(
        self,
        parsed: ParsedStructure,
        scope: ComputationScope,
    ) -> NormalizedView:
        """Apply this protocol to a parsed structure within the given scope.

        Args:
            parsed: Raw parsed structure from BinaryCIF.
            scope: Computation scope defining chains and filters.

        Returns:
            NormalizedView with filtered, protocol-specific data.
        """
        ...

    def get_provenance(self) -> dict:
        """Return provenance information for this protocol application."""
        return {
            "protocol_name": self.name.value,
            "protocol_version": self.version,
            "parameters": self.parameters,
        }


def _select_highest_occupancy_altloc(atoms: list[ParsedAtom]) -> list[ParsedAtom]:
    """Select highest-occupancy alternate location for atoms with altlocs.

    For atoms without altlocs (altloc=None), keep them as-is.
    For atoms with altlocs, group by atom_name and keep the one with
    highest occupancy.
    """
    # Separate atoms with and without altlocs
    no_altloc = [a for a in atoms if a.altloc is None]
    with_altloc = [a for a in atoms if a.altloc is not None]

    if not with_altloc:
        return atoms

    # Group altloc atoms by atom_name, select highest occupancy
    by_name: dict[str, list[ParsedAtom]] = {}
    for atom in with_altloc:
        by_name.setdefault(atom.atom_name, []).append(atom)

    selected: list[ParsedAtom] = list(no_altloc)
    for atom_name, group in by_name.items():
        # Check if we already have this atom_name from no_altloc
        if any(a.atom_name == atom_name for a in no_altloc):
            continue
        best = max(group, key=lambda a: a.occupancy)
        selected.append(best)

    return selected


# ---------------------------------------------------------------------------
# Protocol registry (populated by protocol modules)
# ---------------------------------------------------------------------------

PROTOCOL_REGISTRY: dict[ProtocolName, NormalizationProtocol] = {}


def register_protocol(protocol: NormalizationProtocol) -> None:
    """Register a protocol instance in the global registry."""
    PROTOCOL_REGISTRY[protocol.name] = protocol


def get_protocol(name: str | ProtocolName) -> NormalizationProtocol:
    """Look up a protocol by name.

    Args:
        name: Protocol name (string or ProtocolName enum).

    Returns:
        The registered NormalizationProtocol instance.

    Raises:
        KeyError: If the protocol is not registered.
    """
    if isinstance(name, str):
        name = ProtocolName(name)
    if name not in PROTOCOL_REGISTRY:
        raise KeyError(f"Protocol '{name.value}' not registered. Available: {list(PROTOCOL_REGISTRY.keys())}")
    return PROTOCOL_REGISTRY[name]
