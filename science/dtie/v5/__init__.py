# PRIMARY — this is the production model and pipeline.
"""V5 DTIE — GOSPConeMapper-v5 (Decoupled Radial-Angular Hyperbolic GNN).

This is the CURRENT, PRODUCTION model. All new analysis uses v5.

Key features:
- Decoupled radial/angular heads (no training conflict)
- Native 2D disc + 3D ball projections
- Full phase coverage (1-6d) via the DTIEOrchestrator
- Source-leak detection and allosteric site identification

V3 and V4 remain in the repo for provenance only.
"""

from science.dtie.v5.orchestrator.pipeline import DTIEOrchestrator, PipelineConfig

__all__ = ["DTIEOrchestrator", "PipelineConfig"]
