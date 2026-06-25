"""Phase persistence adapters for the DTIE pipeline.

This package contains:
- Legacy adapters (GNNOutputAdapter, Phase3Adapter) for direct normalizer use
- New protocol-conforming adapters for registry-based orchestrator invocation

Importing this module registers all protocol adapters in the persistence registry.
"""

# Re-export legacy adapters for backward compatibility
from science.dtie.common.adapters.legacy import GNNOutputAdapter, Phase3Adapter

# New protocol-conforming adapters
from science.dtie.common.adapters.allosteric_site_adapter import AllostericSiteAdapter
from science.dtie.common.adapters.phase3_adapter import Phase3PersistenceAdapter
from science.dtie.common.adapters.source_leak_adapter import SourceLeakAdapter
from science.dtie.common.phase_persistence import register_adapter

__all__ = [
    "GNNOutputAdapter",
    "Phase3Adapter",
    "Phase3PersistenceAdapter",
    "SourceLeakAdapter",
    "AllostericSiteAdapter",
]

# Register all Tier 1 adapters on import
register_adapter(Phase3PersistenceAdapter())
register_adapter(SourceLeakAdapter())
register_adapter(AllostericSiteAdapter())
