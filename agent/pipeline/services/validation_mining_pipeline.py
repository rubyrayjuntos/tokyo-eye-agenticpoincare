"""
Validation Mining Pipeline

Orchestrates dehydron/void analysis, correlation, glue site detection,
and wrapper design.  In the Tier-2 async data-layer job the upstream
detection results arrive pre-computed from the DB; use
``run_with_precomputed()`` in that path so the expensive service calls
are never duplicated.

The legacy ``run()`` entry-point is preserved for backward compatibility.
"""

from __future__ import annotations

import importlib
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from gosp.models.data_models import (
    Dehydron,
    GatingAuditRecord,
    StructureData,
    ValidationMiningConclusion,
    ValidationMiningResponse,
    ValidationMiningTiming,
    Void,
    WrapperSuggestion,
)


def _try_import(module_name: str) -> Any:
    """Return the module object or None if unavailable."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


# Module handles are resolved lazily at call time so that monkeypatching
# attributes on the module objects is honoured correctly in tests.
_dehydron_mod = _try_import("gosp.services.dehydron_detection")
_void_mod = _try_import("gosp.services.void_detection")
_correlation_mod = _try_import("gosp.services.outlier_dehydron_correlation")
_glue_mod = _try_import("gosp.services.glue_site_detection")
_wrapper_mod = _try_import("gosp.services.wrapper_design")
_cdd_mod = _try_import("gosp.services.cdd_annotations")


class ValidationMiningPipelineError(Exception):
    """Raised when pipeline execution fails."""


# ---------------------------------------------------------------------------
# Pure helper functions (shared by run() and run_with_precomputed())
# ---------------------------------------------------------------------------


def _parse_focus_range(focus: Optional[str]) -> Optional[Tuple[int, int]]:
    if not focus:
        return None
    parts = focus.split("-")
    if len(parts) != 2:
        raise ValueError("focus_residues must be in 'start-end' format")
    return int(parts[0]), int(parts[1])


def _filter_dehydrons_by_focus(
    dehydrons: List[Dehydron],
    focus: Optional[Tuple[int, int]],
) -> List[Dehydron]:
    if not focus:
        return dehydrons
    start, end = focus
    return [
        d for d in dehydrons
        if start <= d.donor_res_id <= end or start <= d.acceptor_res_id <= end
    ]


def _build_conclusion(
    correlations: List[Any],
    warnings: List[str],
) -> ValidationMiningConclusion:
    if not correlations:
        return ValidationMiningConclusion(
            summary="No correlations found between outliers and dehydrons",
            supported=False,
            confidence="low",
            warnings=warnings,
        )
    best = correlations[0]
    if best.distance < 5.0 and best.correlation_score > 0:
        summary = "Strong correlation between validation outlier and nearby dehydron"
        confidence = "high"
        supported = True
    elif best.distance < 8.0 and best.correlation_score > 0:
        summary = "Moderate correlation between validation outlier and nearby dehydron"
        confidence = "moderate"
        supported = True
    else:
        summary = "Weak or no correlation between outliers and dehydrons"
        confidence = "low"
        supported = False

    return ValidationMiningConclusion(
        summary=summary,
        supported=supported,
        confidence=confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Core pipeline class
# ---------------------------------------------------------------------------


class ValidationMiningPipeline:
    """
    Validation mining pipeline.

    Two entry-points:

    * ``run()``  – fetches dehydrons/voids/CDD gates live (legacy path).
    * ``run_with_precomputed()``  – accepts DB-resident results and skips
      the expensive detection calls entirely (Tier-2 async job path).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        pdb_id: str,
        structure: StructureData,
        chain: Optional[str] = None,
        focus_residues: Optional[str] = None,
        rho_threshold: int = 19,
        correlation_radius: float = 5.0,
        top_n_correlations: int = 3,
        min_dehydrons_per_site: int = 3,
    ) -> ValidationMiningResponse:
        """
        Execute validation mining pipeline, computing dehydrons/voids/CDD
        gates from scratch.

        Preserved for backward compatibility.
        """
        start_time = time.time()
        warnings: List[str] = []

        # --- Dehydron detection ---
        dehydron_start = time.time()
        if _dehydron_mod is None:
            raise ValidationMiningPipelineError(
                "dehydron_detection module is not available"
            )
        dehydron_result = _dehydron_mod.detect_dehydrons(
            structure, wrapping_threshold=rho_threshold
        )
        dehydron_duration = time.time() - dehydron_start

        # --- Void detection ---
        if _void_mod is None:
            raise ValidationMiningPipelineError(
                "void_detection module is not available"
            )
        void_result = _void_mod.detect_voids(
            structure, dehydrons=dehydron_result.dehydrons
        )

        # --- CDD gates ---
        run_id = f"{datetime.utcnow().isoformat()}-{uuid.uuid4().hex[:8]}"
        gating_audit = self._evaluate_cdd_gates(
            run_id=run_id,
            pdb_id=pdb_id,
            chain_id=chain,
            structure=structure,
        )

        return self._assemble_response(
            pdb_id=pdb_id,
            structure=structure,
            chain=chain,
            focus_residues=focus_residues,
            rho_threshold=rho_threshold,
            correlation_radius=correlation_radius,
            top_n_correlations=top_n_correlations,
            min_dehydrons_per_site=min_dehydrons_per_site,
            dehydron_result_list=dehydron_result.dehydrons,
            dehydron_count=dehydron_result.dehydron_count,
            void_list=void_result.voids,
            void_total_volume=void_result.total_volume,
            gating_audit=gating_audit,
            warnings=warnings,
            start_time=start_time,
            dehydron_duration=dehydron_duration,
            run_id=run_id,
        )

    async def run_with_precomputed(
        self,
        pdb_id: str,
        structure: StructureData,
        precomputed_dehydrons: List[Dehydron],
        precomputed_voids: List[Void],
        precomputed_cdd_annotations: List[GatingAuditRecord],
        chain: Optional[str] = None,
        focus_residues: Optional[str] = None,
        rho_threshold: int = 19,
        correlation_radius: float = 5.0,
        top_n_correlations: int = 3,
        min_dehydrons_per_site: int = 3,
    ) -> ValidationMiningResponse:
        """
        Execute validation mining using pre-computed detection results.

        This method does NOT call ``detect_dehydrons()``,
        ``detect_voids()``, or ``_evaluate_cdd_gates()`` internally.
        It uses the supplied lists in their place, enabling the Tier-2
        async job to read from the DB without duplicating work.
        """
        start_time = time.time()
        warnings: List[str] = []
        run_id = f"{datetime.utcnow().isoformat()}-{uuid.uuid4().hex[:8]}"

        void_total_volume = sum(
            getattr(v, "volume", 0.0) for v in precomputed_voids
        )
        dehydron_count = sum(
            1 for d in precomputed_dehydrons if getattr(d, "is_dehydron", True)
        )

        return self._assemble_response(
            pdb_id=pdb_id,
            structure=structure,
            chain=chain,
            focus_residues=focus_residues,
            rho_threshold=rho_threshold,
            correlation_radius=correlation_radius,
            top_n_correlations=top_n_correlations,
            min_dehydrons_per_site=min_dehydrons_per_site,
            dehydron_result_list=precomputed_dehydrons,
            dehydron_count=dehydron_count,
            void_list=precomputed_voids,
            void_total_volume=void_total_volume,
            gating_audit=precomputed_cdd_annotations,
            warnings=warnings,
            start_time=start_time,
            dehydron_duration=0.0,
            run_id=run_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_response(
        self,
        *,
        pdb_id: str,
        structure: StructureData,
        chain: Optional[str],
        focus_residues: Optional[str],
        rho_threshold: int,
        correlation_radius: float,
        top_n_correlations: int,
        min_dehydrons_per_site: int,
        dehydron_result_list: List[Dehydron],
        dehydron_count: int,
        void_list: List[Void],
        void_total_volume: float,
        gating_audit: List[GatingAuditRecord],
        warnings: List[str],
        start_time: float,
        dehydron_duration: float,
        run_id: str,
    ) -> ValidationMiningResponse:
        """Build the ``ValidationMiningResponse`` from fully-resolved inputs."""
        focus_range = _parse_focus_range(focus_residues)
        focused_dehydrons = _filter_dehydrons_by_focus(dehydron_result_list, focus_range)

        # Correlation -------------------------------------------------------
        correlation_start = time.time()
        correlations: List[Any] = []
        if _correlation_mod is not None:
            outliers: List[Any] = []
            try:
                correlation_result = _correlation_mod.correlate_outliers_to_dehydrons(
                    outliers=outliers,
                    dehydrons=focused_dehydrons,
                    structure=structure,
                    radius=correlation_radius,
                    rho_threshold=rho_threshold,
                    top_n=top_n_correlations,
                )
                correlations = correlation_result.correlations
                warnings.extend(correlation_result.warnings)
            except Exception as exc:
                warnings.append(f"Correlation step failed: {exc}")
        correlation_duration = time.time() - correlation_start

        # Glue sites --------------------------------------------------------
        glue_start = time.time()
        glue_sites: List[Any] = []
        if _glue_mod is not None:
            try:
                glue_sites = _glue_mod.detect_glueable_sites(
                    focused_dehydrons,
                    rho_threshold=rho_threshold,
                    min_dehydrons=min_dehydrons_per_site,
                )
            except Exception as exc:
                warnings.append(f"Glue site detection failed: {exc}")
        glue_duration = time.time() - glue_start

        # Wrapper suggestions -----------------------------------------------
        wrapper_suggestions: List[WrapperSuggestion] = []
        if _wrapper_mod is not None and glue_sites:
            try:
                wrapper_suggestions.append(
                    _wrapper_mod.build_wrapper_suggestion(glue_sites[0])
                )
            except Exception as exc:
                warnings.append(f"Wrapper suggestion failed: {exc}")

        conclusion = _build_conclusion(correlations, warnings)

        total_duration = time.time() - start_time

        pipeline_input: Dict[str, object] = {
            "pdb_id": pdb_id,
            "chain": chain,
            "focus_residues": focus_residues,
            "correlation_radius": correlation_radius,
            "top_n_correlations": top_n_correlations,
            "rho_threshold": rho_threshold,
            "min_dehydrons_per_site": min_dehydrons_per_site,
        }

        return ValidationMiningResponse(
            status="success",
            run_id=run_id,
            timestamp=datetime.utcnow(),
            input=pipeline_input,
            structure_summary={
                "pdb_id": structure.pdb_id,
                "resolution": structure.resolution,
                "chains": [c.chain_id for c in structure.chains],
                "residues": structure.total_residues,
            },
            validation_outliers={
                "bond_outliers": 0,
                "angle_outliers": 0,
                "severe": 0,
                "total": 0,
            },
            dehydrons={
                "total": len(dehydron_result_list),
                "in_focus": len(focused_dehydrons),
                "dehydron_count": dehydron_count,
            },
            voids={
                "total": len(void_list),
                "total_volume": void_total_volume,
            },
            correlations=correlations,
            glueable_sites=glue_sites,
            wrapper_suggestions=wrapper_suggestions,
            gating_audit=gating_audit,
            conclusion=conclusion,
            timings=ValidationMiningTiming(
                total=total_duration,
                fetch=0.0,
                dehydrons=dehydron_duration,
                correlation=correlation_duration,
                glue_detection=glue_duration,
            ),
        )

    def _evaluate_cdd_gates(
        self,
        run_id: str,
        pdb_id: str,
        chain_id: Optional[str],
        structure: StructureData,
    ) -> List[GatingAuditRecord]:
        """Evaluate CDD-based gating checks.  Returns an empty list when the
        ``cdd_annotations`` service is unavailable."""
        if _cdd_mod is None:
            return []

        gates: List[GatingAuditRecord] = []
        if not chain_id:
            for gate in ("CDD_CONFIDENCE", "LINKER_SNAP", "STERIC_OVERLAP", "ACCESSIBILITY"):
                gates.append(
                    GatingAuditRecord(
                        state_id=run_id,
                        gate=gate,
                        status="skipped",
                        reason="CDD gating requires explicit chain",
                        details={"pdb_id": pdb_id},
                        provenance={},
                    )
                )
            return gates

        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                new_loop = asyncio.new_event_loop()
                try:
                    annotation = new_loop.run_until_complete(
                        _cdd_mod.fetch_cdd_annotations(pdb_id, chain_id)
                    )
                finally:
                    new_loop.close()
            else:
                annotation = asyncio.run(
                    _cdd_mod.fetch_cdd_annotations(pdb_id, chain_id)
                )
        except Exception as exc:
            for gate in ("CDD_CONFIDENCE", "LINKER_SNAP", "STERIC_OVERLAP", "ACCESSIBILITY"):
                gates.append(
                    GatingAuditRecord(
                        state_id=run_id,
                        gate=gate,
                        status="skipped",
                        reason=f"CDD annotation fetch failed: {exc}",
                        details={"pdb_id": pdb_id, "chain": chain_id},
                        provenance={},
                    )
                )
            return gates

        _ = annotation  # reserved for future CDD gate evaluations
        return gates


def run_validation_mining_pipeline(
    pdb_id: str,
    chain: Optional[str] = None,
    focus_residues: Optional[str] = None,
    correlation_radius: float = 5.0,
    top_n_correlations: int = 3,
    rho_threshold: int = 19,
    min_dehydrons_per_site: int = 3,
    cache_dir: Optional[Any] = None,
) -> "ValidationMiningResponse":
    """Module-level entry point; fetches structure then delegates to pipeline class."""
    from gosp.services.structure_ingestion import fetch_structure_from_rcsb

    structure = fetch_structure_from_rcsb(pdb_id, format="cif", chain_id=chain)
    return ValidationMiningPipeline().run(
        pdb_id=pdb_id,
        structure=structure,
        chain=chain,
        focus_residues=focus_residues,
        rho_threshold=rho_threshold,
        correlation_radius=correlation_radius,
        top_n_correlations=top_n_correlations,
        min_dehydrons_per_site=min_dehydrons_per_site,
    )
