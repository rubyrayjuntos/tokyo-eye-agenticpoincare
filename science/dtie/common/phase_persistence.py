"""Phase persistence protocol and adapter registry.

Defines the contract that all phase persistence adapters must satisfy,
along with a central registry mapping phase names to their adapters.
This enables the orchestrator to persist phase outputs through a
consistent, discoverable pattern.

See: .kiro/specs/phase-output-persistence/design.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from science.dtie.common.interfaces import PhaseResult
from science.dtie.common.normalizer_payloads import NormalizerResult, ProvenanceContext

logger = logging.getLogger(__name__)


@dataclass
class PhasePersistenceSpec:
    """Declares a phase's persistence characteristics.

    Attributes:
        phase_name: Identifier matching PhaseResult.phase_name.
        tier: Priority tier (1 = must persist, 2 = should persist, 3 = optional).
        produces_residue_level_data: Whether outputs include per-residue records.
        schema_version: Version of the persistence schema for this phase.
    """

    phase_name: str
    tier: int
    produces_residue_level_data: bool
    schema_version: str = "1.0"


@runtime_checkable
class PhasePersistenceAdapter(Protocol):
    """Protocol that all phase persistence adapters must implement.

    Each adapter converts a PhaseResult into Normalizer payloads and
    writes them through the governed data layer.
    """

    spec: PhasePersistenceSpec

    async def persist(
        self,
        phase_result: PhaseResult,
        provenance: ProvenanceContext,
        normalizer: Any,
    ) -> NormalizerResult:
        """Persist a phase result through the Normalizer.

        Args:
            phase_result: The completed phase output to persist.
            provenance: Provenance context linking this write to its producing run.
            normalizer: The Normalizer instance for governed writes.

        Returns:
            NormalizerResult indicating success/failure and created assets.
        """
        ...


# Registry: phase_name → adapter instance
PERSISTENCE_ADAPTERS: dict[str, PhasePersistenceAdapter] = {}

_registry_initialized: bool = False


def register_adapter(adapter: PhasePersistenceAdapter) -> None:
    """Register a persistence adapter for a phase.

    Args:
        adapter: An adapter instance conforming to PhasePersistenceAdapter.

    Raises:
        ValueError: If the adapter's spec.phase_name is empty.
    """
    if not adapter.spec.phase_name:
        raise ValueError("Adapter spec.phase_name must not be empty")
    if adapter.spec.phase_name in PERSISTENCE_ADAPTERS:
        logger.warning(
            "Overwriting existing adapter for phase '%s'", adapter.spec.phase_name
        )
    PERSISTENCE_ADAPTERS[adapter.spec.phase_name] = adapter


def ensure_adapters_registered() -> None:
    """Register all built-in persistence adapters if not already registered.

    This is called automatically by the orchestrator before persistence.
    It is idempotent — calling it multiple times is safe.
    """
    global _registry_initialized
    if _registry_initialized:
        return

    from science.dtie.common.adapters.phase3_adapter import Phase3PersistenceAdapter
    from science.dtie.common.adapters.source_leak_adapter import SourceLeakAdapter
    from science.dtie.common.adapters.allosteric_site_adapter import AllostericSiteAdapter
    from science.dtie.common.adapters.phase2_vulnerability_adapter import Phase2VulnerabilityAdapter
    from science.dtie.common.adapters.phase35_lift_adapter import Phase35LiftAdapter
    from science.dtie.common.adapters.phase4_resistance_adapter import Phase4ResistanceAdapter
    from science.dtie.common.adapters.phase5_pharmacophore_adapter import Phase5PharmacophoreAdapter
    from science.dtie.common.adapters.phase6_drug_discovery_adapter import Phase6DrugDiscoveryAdapter
    from science.dtie.common.adapters.buffering_atlas_adapter import BufferingAtlasAdapter

    register_adapter(Phase3PersistenceAdapter())
    register_adapter(SourceLeakAdapter())
    register_adapter(AllostericSiteAdapter())
    register_adapter(Phase2VulnerabilityAdapter())
    register_adapter(Phase35LiftAdapter())
    register_adapter(Phase4ResistanceAdapter())
    register_adapter(Phase5PharmacophoreAdapter())
    register_adapter(Phase6DrugDiscoveryAdapter())
    register_adapter(BufferingAtlasAdapter())

    _registry_initialized = True
    logger.info(
        "Registered %d persistence adapters: %s",
        len(PERSISTENCE_ADAPTERS),
        list(PERSISTENCE_ADAPTERS.keys()),
    )


def reset_registry() -> None:
    """Clear the registry and reset initialization flag.

    Intended for use in tests only.
    """
    global _registry_initialized
    PERSISTENCE_ADAPTERS.clear()
    _registry_initialized = False
