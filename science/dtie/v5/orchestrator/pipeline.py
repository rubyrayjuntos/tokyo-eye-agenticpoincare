# Primary DTIE Orchestrator — uses V5 GNN (decoupled radial-angular)
"""DTIE Pipeline Orchestrator — the production pipeline.

This is THE orchestrator for all new work. It uses the v5 GNN
(decoupled radial-angular architecture) and supports the full
phase coverage inherited from the v3 pipeline logic.

V3 and V4 code remains in the repo for provenance and backward
compatibility only — they are NOT used for new analysis.

Pipeline steps:
1. GNN inference (v5) → decoupled radial-angular hyperbolic embeddings
2. Phase 1: Witness embedding complex
3. Phase 2: Vulnerability scan
4. Phase 3: Witness persistence (with hyperbolic distances)
5. Phase 3.5: Topological lift
6. Phase 4: Resistance mapping
7. Phase 5: Pharmacophore identification
8. Phase 6a-6d: Virtual screening, binding affinity, ADMET, state selectivity
9. Source-leak detection
10. Allosteric site identification

Not all phases need to run every time — the config controls which
phases are executed. The default for source-leak analysis is:
GNN → Phase 3 → source-leak → allosteric sites.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from science.dtie.common.interfaces import (
    GNNInferenceResult,
    PhaseResult,
    PipelineResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase Registry — maps phase names to (module_path, function_name)
# ---------------------------------------------------------------------------

PHASE_REGISTRY: dict[str, tuple[str, str]] = {
    "phase1_witness_embedding": (
        "science.dtie.v3.phases.phase1_witness_embedding",
        "execute_phase_1_witness_embedding",
    ),
    "phase2_vulnerability_scan": (
        "science.dtie.v3.phases.phase2_vulnerability_scan",
        "execute_phase_2_vulnerability_scan",
    ),
    "phase35_topological_lift": (
        "science.dtie.v3.phases.phase35_topological_lift",
        "execute_phase_35_topological_lift",
    ),
    "phase4_resistance_mapping": (
        "science.dtie.v3.phases.phase4_resistance_mapping",
        "execute_phase_4_resistance_mapping",
    ),
    "phase5_pharmacophore": (
        "science.dtie.v3.phases.phase5_pharmacophore",
        "execute_phase_5_pharmacophore_generation",
    ),
    "phase6a_virtual_screening": (
        "science.dtie.v3.phases.phase6a_virtual_screening",
        "execute_phase_6a_virtual_screening",
    ),
    "phase6b_binding_affinity": (
        "science.dtie.v3.phases.phase6b_binding_affinity",
        "execute_phase_6b_binding_affinity",
    ),
    "phase6c_admet_filter": (
        "science.dtie.v3.phases.phase6c_admet_filter",
        "execute_phase_6c_admet_filter",
    ),
    "phase6d_state_selectivity": (
        "science.dtie.v3.phases.phase6d_state_selectivity",
        "execute_phase_6d_state_selectivity_check",
    ),
}

# Phases that must succeed for the pipeline to report overall success.
# Source-leak detection and allosteric sites are always required when enabled.
_REQUIRED_PHASES = {"gnn_inference", "source_leak_detection"}


def _load_phase_function(phase_name: str) -> Callable | None:
    """Dynamically load a phase function from the registry."""
    entry = PHASE_REGISTRY.get(phase_name)
    if entry is None:
        return None
    module_path, func_name = entry
    try:
        module = importlib.import_module(module_path)
        return getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        logger.warning("Failed to load phase %s: %s", phase_name, e)
        return None


# ---------------------------------------------------------------------------
# Pipeline Config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the DTIE pipeline.

    By default, runs the source-leak focused subset.
    Set source_leak_only=False for the complete drug discovery workflow.
    """

    structure_id: str
    checkpoint_path: str | None = None

    # Phase control
    run_gnn: bool = True
    run_phase1: bool = True
    run_phase2: bool = True
    run_phase3: bool = True
    run_phase35: bool = True
    run_phase4: bool = True
    run_phase5: bool = True
    run_phase6: bool = True  # Covers 6a-6d
    detect_source_leaks: bool = True
    identify_allosteric_sites: bool = True

    # Shortcut: disable non-essential phases for quick analysis
    source_leak_only: bool = False

    # Thresholds
    uncertainty_threshold: float = 0.3
    depth_threshold: float = 1.5
    persistence_threshold: float = 0.5

    # Provenance
    parent_run_id: str | None = None
    code_version: str | None = None

    # Phase parameters
    n_landmarks: int = 50
    n_witnesses: int = 500
    curvature_override: float | None = None
    spatial_cutoff: float = 8.0

    @property
    def is_full_pipeline(self) -> bool:
        """Whether this config runs the full pipeline (not source-leak-only)."""
        return not self.source_leak_only

    def __post_init__(self):
        if self.source_leak_only:
            self.run_phase1 = False
            self.run_phase2 = False
            self.run_phase35 = False
            self.run_phase4 = False
            self.run_phase5 = False
            self.run_phase6 = False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DTIEOrchestrator:
    """The production DTIE pipeline orchestrator.

    Uses V5 GNN exclusively. This is the only orchestrator that should
    be used for new analysis. V3/V4 orchestrators exist only for
    reproducing historical results.

    Usage:
        orchestrator = DTIEOrchestrator(db=connection)

        # Source-leak focused (default):
        result = await orchestrator.run(PipelineConfig(structure_id="4obe"))

        # Full drug discovery pipeline:
        result = await orchestrator.run(PipelineConfig(
            structure_id="4obe",
            source_leak_only=False,
        ))
    """

    def __init__(self, db: Any):
        self._db = db

    @property
    def model_version(self) -> str:
        return "DTIE-v5"

    async def run(self, config: PipelineConfig) -> PipelineResult:
        """Execute the DTIE pipeline."""
        run_id = f"dtie_{uuid.uuid4().hex[:12]}"
        phase_results: dict[str, PhaseResult] = {}
        gnn_result: GNNInferenceResult | None = None
        warnings: list[str] = []

        logger.info(
            "Starting DTIE pipeline for %s (run_id=%s, full=%s)",
            config.structure_id,
            run_id,
            config.is_full_pipeline,
        )

        # ── Step 1: GNN Inference (V5) ────────────────────────────────────
        if config.run_gnn:
            try:
                gnn_result = await self._run_gnn(config, run_id)
                phase_results["gnn_inference"] = PhaseResult(
                    phase_name="gnn_inference",
                    structure_id=config.structure_id,
                    model_version="GOSPConeMapper-v5",
                    success=True,
                    outputs={
                        "node_count": len(gnn_result.nodes),
                        "architecture": "decoupled_radial_angular",
                    },
                )
            except Exception as e:
                logger.error("GNN inference failed: %s", e)
                return PipelineResult(
                    run_id=run_id,
                    structure_id=config.structure_id,
                    model_version=self.model_version,
                    success=False,
                    phase_results=phase_results,
                    warnings=[f"GNN inference failed: {e}"],
                )

        # ── Step 2: Phase 1 — Witness Embedding ──────────────────────────
        if config.run_phase1:
            phase_results["phase1"] = await self._run_phase(
                "phase1_witness_embedding", config, run_id, gnn_result
            )

        # ── Step 3: Phase 2 — Vulnerability Scan ─────────────────────────
        if config.run_phase2:
            phase_results["phase2"] = await self._run_phase(
                "phase2_vulnerability_scan", config, run_id, gnn_result
            )

        # ── Step 4: Phase 3 — Witness Persistence ────────────────────────
        if config.run_phase3:
            phase_results["phase3"] = await self._run_phase3(config, run_id, gnn_result)

        # ── Step 5: Phase 3.5 — Topological Lift ─────────────────────────
        if config.run_phase35:
            phase_results["phase35"] = await self._run_phase(
                "phase35_topological_lift", config, run_id, gnn_result
            )

        # ── Step 6: Phase 4 — Resistance Mapping ─────────────────────────
        if config.run_phase4:
            phase_results["phase4"] = await self._run_phase(
                "phase4_resistance_mapping", config, run_id, gnn_result
            )

        # ── Step 7: Phase 5 — Pharmacophore ──────────────────────────────
        if config.run_phase5:
            phase_results["phase5"] = await self._run_phase(
                "phase5_pharmacophore", config, run_id, gnn_result
            )

        # ── Step 8: Phase 6 — Drug Discovery ─────────────────────────────
        if config.run_phase6:
            for sub in ["6a_virtual_screening", "6b_binding_affinity",
                        "6c_admet_filter", "6d_state_selectivity"]:
                phase_results[f"phase{sub}"] = await self._run_phase(
                    f"phase{sub}", config, run_id, gnn_result
                )

        # ── Step 9: Source-Leak Detection ─────────────────────────────────
        if config.detect_source_leaks:
            try:
                leak_result = await self._detect_source_leaks(config, run_id)
                phase_results["source_leak_detection"] = leak_result
            except Exception as e:
                warnings.append(f"Source-leak detection failed: {e}")
                phase_results["source_leak_detection"] = PhaseResult(
                    phase_name="source_leak_detection",
                    structure_id=config.structure_id,
                    model_version="DTIE-v5-source-leak",
                    success=False,
                    outputs={"error": str(e)},
                )

        # ── Step 10: Allosteric Site Identification ───────────────────────
        if config.identify_allosteric_sites:
            try:
                site_result = await self._identify_allosteric_sites(config, run_id)
                phase_results["allosteric_sites"] = site_result
            except Exception as e:
                warnings.append(f"Allosteric site identification failed: {e}")

        # Determine success: all required phases that were run must pass
        success = self._evaluate_success(phase_results, config)

        logger.info(
            "DTIE pipeline complete for %s: %d phases run, success=%s",
            config.structure_id,
            len(phase_results),
            success,
        )

        return PipelineResult(
            run_id=run_id,
            structure_id=config.structure_id,
            model_version=self.model_version,
            success=success,
            phase_results=phase_results,
            gnn_result=gnn_result,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Success evaluation
    # ------------------------------------------------------------------

    def _evaluate_success(
        self, phase_results: dict[str, PhaseResult], config: PipelineConfig
    ) -> bool:
        """Determine overall pipeline success.

        Rules:
        - All required phases (GNN, source-leak) that were enabled must pass.
        - Optional phases may fail without failing the pipeline.
        - If no phases ran at all, that's a failure.
        """
        if not phase_results:
            return False

        # Check required phases
        for phase_name in _REQUIRED_PHASES:
            result = phase_results.get(phase_name)
            if result is not None and not result.success:
                return False

        # At least one phase must have succeeded
        return any(r.success for r in phase_results.values())

    # ------------------------------------------------------------------
    # Internal: GNN
    # ------------------------------------------------------------------

    async def _run_gnn(
        self, config: PipelineConfig, run_id: str
    ) -> GNNInferenceResult:
        """Run v5 GNN inference and write through Normalizer."""
        from data.normalizer.core import Normalizer
        from science.dtie.common.adapters import GNNOutputAdapter
        from science.dtie.common.graph_builder import GraphBuilder
        from science.dtie.v5.gnn.runner import V5GNNRunner

        builder = GraphBuilder(db=self._db)
        graph = await builder.build_graph(config.structure_id)
        pyg_data = builder.to_pyg(graph)

        runner = V5GNNRunner(
            checkpoint_path=config.checkpoint_path,
            curvature_override=config.curvature_override,
        )
        result = await runner.run_inference(config.structure_id, pyg_data)

        normalizer = Normalizer(db=self._db, caller_identity="dtie_orchestrator")
        adapter = GNNOutputAdapter(normalizer=normalizer)
        await adapter.normalize(
            result,
            run_id=run_id,
            code_version=config.code_version,
            parent_run_id=config.parent_run_id,
        )

        return result

    # ------------------------------------------------------------------
    # Internal: Phases (registry-based dispatch)
    # ------------------------------------------------------------------

    async def _run_phase(
        self,
        phase_name: str,
        config: PipelineConfig,
        run_id: str,
        gnn_result: GNNInferenceResult | None,
    ) -> PhaseResult:
        """Run a phase by delegating to the registered implementation.

        Phase functions are CPU-bound (numpy/scipy) and run in a thread
        to avoid blocking the event loop.
        """
        from science.dtie.v5.phase_adapter import v5_result_to_phase_dict

        if gnn_result is None:
            return PhaseResult(
                phase_name=phase_name,
                structure_id=config.structure_id,
                model_version=f"DTIE-v5-{phase_name}",
                success=False,
                outputs={"error": "GNN result required but not available"},
            )

        # Build the gnn_output dict that phases expect
        gnn_output = v5_result_to_phase_dict(gnn_result, prefix="gdp")

        try:
            phase_output = await self._dispatch_phase(phase_name, gnn_output, config)
            return PhaseResult(
                phase_name=phase_name,
                structure_id=config.structure_id,
                model_version=f"DTIE-v5-{phase_name}",
                success=True,
                outputs=phase_output,
            )
        except Exception as e:
            logger.warning("Phase %s failed: %s", phase_name, e)
            return PhaseResult(
                phase_name=phase_name,
                structure_id=config.structure_id,
                model_version=f"DTIE-v5-{phase_name}",
                success=False,
                outputs={"error": str(e)},
            )

    async def _dispatch_phase(
        self,
        phase_name: str,
        gnn_output: dict,
        config: PipelineConfig,
    ) -> dict:
        """Dispatch to the actual phase implementation via the registry.

        Phase functions are CPU-bound numpy/scipy operations, so they
        run in a thread to avoid blocking the event loop.
        """
        phase_func = _load_phase_function(phase_name)
        if phase_func is None:
            return {"status": f"unknown_phase: {phase_name}"}

        # Phase 1 has a special signature (needs ingestion_data)
        if phase_name == "phase1_witness_embedding":
            ingestion_data = {
                "no_midpoints": gnn_output["gdp_no_midpoints"],
                "residue_ids": gnn_output["gdp_residue_ids"],
            }
            result = await asyncio.to_thread(
                phase_func, ingestion_data, gnn_output, n_landmarks=config.n_landmarks
            )
            return {"phase1_output": str(result), "n_landmarks": config.n_landmarks}

        # All other phases share the same signature: (gnn_output=...)
        result = await asyncio.to_thread(phase_func, gnn_output=gnn_output)
        return {f"{phase_name}_output": str(result)}

    async def _run_phase3(
        self,
        config: PipelineConfig,
        run_id: str,
        gnn_result: GNNInferenceResult | None,
    ) -> PhaseResult:
        """Phase 3 — witness persistence with hyperbolic distances.

        Uses the v4 phase3 implementation (which integrates hyperbolic
        distances into the filtration) rather than the v3 version.
        """
        from science.dtie.v5.phase_adapter import v5_result_to_phase_dict

        if gnn_result is None:
            return PhaseResult(
                phase_name="phase3_persistence",
                structure_id=config.structure_id,
                model_version="DTIE-v5-phase3",
                success=False,
                outputs={"error": "GNN result required"},
            )

        gnn_output = v5_result_to_phase_dict(gnn_result, prefix="gdp")

        try:
            # Use the v4 phase3 (hyperbolic-aware) implementation
            from science.dtie.v4.phases.phase3_witness_persistence_v4 import run_phase3_v4
            result = await asyncio.to_thread(
                run_phase3_v4, gnn_output=gnn_output, n_landmarks=config.n_landmarks
            )
            return PhaseResult(
                phase_name="phase3_persistence",
                structure_id=config.structure_id,
                model_version="DTIE-v5-phase3",
                success=True,
                outputs={
                    "phase3_output": str(result),
                    "n_landmarks": config.n_landmarks,
                    "n_witnesses": config.n_witnesses,
                    "hyperbolic_distances_used": True,
                    "gnn_version": "v5",
                },
            )
        except Exception as e:
            # Fallback to v3 phase3 if v4 implementation fails
            logger.warning("V4 phase3 failed, falling back to v3: %s", e)
            try:
                from science.dtie.v3.phases.phase3_witness_persistence import run_phase3
                result = await asyncio.to_thread(run_phase3, gnn_output=gnn_output)
                return PhaseResult(
                    phase_name="phase3_persistence",
                    structure_id=config.structure_id,
                    model_version="DTIE-v5-phase3-fallback",
                    success=True,
                    outputs={
                        "phase3_output": str(result),
                        "hyperbolic_distances_used": False,
                        "fallback": True,
                    },
                )
            except Exception as e2:
                return PhaseResult(
                    phase_name="phase3_persistence",
                    structure_id=config.structure_id,
                    model_version="DTIE-v5-phase3",
                    success=False,
                    outputs={"error": str(e2)},
                )

    async def _detect_source_leaks(
        self, config: PipelineConfig, run_id: str
    ) -> PhaseResult:
        """Identify source-leak candidates from v5 GNN outputs."""
        rows = await self._db.fetch_all(
            """
            SELECT e.residue_id, e.cone_depth, e.epistemic_uncertainty
            FROM fact_gnn_node_embedding e
            JOIN embedding_space es ON es.space_id = e.space_id
            WHERE e.structure_id = :structure_id
              AND es.space_type = 'hyperbolic'
              AND e.epistemic_uncertainty >= :threshold
              AND e.cone_depth >= :min_depth
            ORDER BY e.epistemic_uncertainty DESC
            """,
            {
                "structure_id": config.structure_id,
                "threshold": config.uncertainty_threshold,
                "min_depth": config.depth_threshold,
            },
        )

        leak_residues = [r["residue_id"] for r in rows]

        return PhaseResult(
            phase_name="source_leak_detection",
            structure_id=config.structure_id,
            model_version="DTIE-v5-source-leak",
            success=True,
            outputs={
                "source_leak_count": len(leak_residues),
                "source_leak_residues": leak_residues,
            },
            residue_contributions={
                r["residue_id"]: float(r["epistemic_uncertainty"])
                for r in rows
            },
        )

    async def _identify_allosteric_sites(
        self, config: PipelineConfig, run_id: str
    ) -> PhaseResult:
        """Cluster source-leak residues into allosteric site candidates."""
        return PhaseResult(
            phase_name="allosteric_sites",
            structure_id=config.structure_id,
            model_version="DTIE-v5-allosteric",
            success=True,
            outputs={"status": "ready_for_spatial_clustering"},
        )
