"""DTIE — Dynamic Topology Inference Engine.

The production pipeline uses V5 (decoupled radial-angular GNN).
V3 and V4 are retained for provenance and backward compatibility only.

Usage:
    from science.dtie import DTIEOrchestrator, PipelineConfig

    orchestrator = DTIEOrchestrator(db=connection)
    result = await orchestrator.run(PipelineConfig(structure_id="4obe"))
"""

from science.dtie.v5 import DTIEOrchestrator, PipelineConfig

__all__ = ["DTIEOrchestrator", "PipelineConfig"]
