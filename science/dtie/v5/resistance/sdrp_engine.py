"""SDRP Engine — orchestrates multi-structure ensemble profiling.

This module wraps the existing ResistanceProfiler to run it independently
against multiple PDB structures (each representing a different drug-binding
state), then computes the State-Sensitivity Score (SSS), stability scores,
mechanism shifts, and categorization to produce an EnsembleProfile.

Key design principle: divergence is data, not error. A mutation that classifies
differently across PDBs is a "Conformational Switch" — a biologically meaningful
signal indicating state-dependent resistance.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from science.dtie.v5.resistance.models import (
    ClassifierConfig,
    MutationSpec,
    ResistanceReport,
)
from science.dtie.v5.resistance.sdrp_categorizer import (
    categorize,
    generate_clinical_relevance,
)
from science.dtie.v5.resistance.sdrp_models import (
    EnsembleProfile,
    KinaseStateSet,
    MechanismShift,
    StateProfileEntry,
    StructureEntry,
    ensemble_profile_to_dict,
)
from science.dtie.v5.resistance.sdrp_shifts import detect_mechanism_shifts
from science.dtie.v5.resistance.sdrp_sss import compute_sss
from science.dtie.v5.resistance.sdrp_stability import compute_stability_score

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "checkpoints_v5/v5_stage4_11prot.pt"


class SDRPEngine:
    """Orchestrates multi-structure ensemble profiling.

    Accepts variable-cardinality structure sets. If a KinaseStateSet is
    provided, the output state_profile will include None entries for
    canonical states not represented in the input structures, enabling
    consistent feature vectors for downstream ML.
    """

    def __init__(
        self,
        db: Any,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str = "cpu",
        classifier_config: ClassifierConfig | None = None,
        kinase_state_set: KinaseStateSet | None = None,
    ):
        from science.dtie.v5.resistance.profiler import ResistanceProfiler

        self._profiler = ResistanceProfiler(
            db=db,
            checkpoint_path=checkpoint_path,
            device=device,
            classifier_config=classifier_config,
        )
        self._db = db
        self._config = classifier_config or ClassifierConfig()
        self._kinase_state_set = kinase_state_set

    async def profile_variant(
        self,
        variant: MutationSpec,
        structures: list[StructureEntry],
        hub_residues: list[tuple[str, int]] | None = None,
    ) -> EnsembleProfile:
        """Profile a single variant across multiple structures.

        Each structure is profiled independently using the existing
        ResistanceProfiler. Per-structure baselines are cached and
        reused across variants in batch mode.

        Args:
            variant: The mutation to profile.
            structures: List of StructureEntry with binding contexts.
            hub_residues: Optional explicit hub residues.

        Returns:
            EnsembleProfile with per-state classifications and SSS.

        Raises:
            ValueError: If fewer than 2 structures are provided.
        """
        if len(structures) < 2:
            raise ValueError(
                "At least two structures required for ensemble profiling"
            )

        # Run ResistanceProfiler independently per structure with error isolation
        state_profile: dict[str, StateProfileEntry | None] = {}
        error_structures: dict[str, str] = {}
        per_structure_run_ids: list[str] = []

        for entry in structures:
            try:
                report = await self._profiler.profile_mutation(
                    structure_id=entry.structure_id,
                    mutation=variant,
                    hub_residues=hub_residues,
                )
                # Extract StateProfileEntry from the ResistanceReport
                state_entry = self._report_to_state_entry(report)
                state_profile[entry.binding_context.label] = state_entry

                # Collect run_id for provenance
                run_id = self._profiler._baseline_run_ids.get(
                    entry.structure_id, f"unknown_{uuid.uuid4().hex[:8]}"
                )
                per_structure_run_ids.append(run_id)

            except Exception as e:
                logger.warning(
                    "Structure %s failed for %s: %s",
                    entry.structure_id,
                    variant.variant_name,
                    e,
                )
                error_structures[entry.structure_id] = str(e)
                per_structure_run_ids.append("")

        # Fill None for missing canonical states if KinaseStateSet is provided
        if self._kinase_state_set is not None:
            for canonical_state in self._kinase_state_set.canonical_states:
                if canonical_state not in state_profile:
                    state_profile[canonical_state] = None

        # Compute SSS from populated state entries
        state_entries = list(state_profile.values())
        sss_score = compute_sss(state_entries)

        # Detect mechanism shifts
        mechanism_shifts = detect_mechanism_shifts(state_profile)

        # Categorize
        category = categorize(sss_score)

        # Generate clinical relevance
        clinical_relevance = generate_clinical_relevance(
            category=category,
            variant=variant.variant_name,
            state_profile=state_profile,
            mechanism_shifts=mechanism_shifts,
        )

        ensemble = EnsembleProfile(
            variant=variant.variant_name,
            state_profile=state_profile,
            conformational_sensitivity=sss_score,
            sss_score=sss_score,
            category=category,
            clinical_relevance=clinical_relevance,
            mechanism_shifts=mechanism_shifts,
            structure_entries=structures,
            per_structure_run_ids=per_structure_run_ids,
            error_structures=error_structures,
        )

        # Persist to database (best-effort — failure is logged, not raised)
        await self._persist_ensemble_profile(ensemble)

        return ensemble

    async def profile_batch(
        self,
        variants: list[MutationSpec],
        structures: list[StructureEntry],
        hub_residues: list[tuple[str, int]] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[EnsembleProfile]:
        """Profile multiple variants across the same structure ensemble.

        Baselines are computed once per structure and shared across all
        variants. Progress callback receives (variants_completed, total).

        Args:
            variants: List of mutations to profile.
            structures: List of StructureEntry with binding contexts.
            hub_residues: Optional explicit hub residues.
            on_progress: Optional callback(completed, total).

        Returns:
            List of EnsembleProfiles in same order as input variants.

        Raises:
            ValueError: If variants list is empty or fewer than 2 structures.
        """
        if not variants:
            raise ValueError("At least one variant required")
        if len(structures) < 2:
            raise ValueError(
                "At least two structures required for ensemble profiling"
            )

        total = len(variants)

        # Pre-warm baselines for all structures (computed once, shared across variants)
        for entry in structures:
            try:
                await self._profiler._ensure_baseline(entry.structure_id)
            except Exception as e:
                logger.warning(
                    "Baseline computation failed for %s: %s",
                    entry.structure_id,
                    e,
                )

        # Profile each variant
        results: list[EnsembleProfile] = []
        for idx, variant in enumerate(variants):
            profile = await self.profile_variant(
                variant=variant,
                structures=structures,
                hub_residues=hub_residues,
            )
            results.append(profile)

            if on_progress is not None:
                on_progress(idx + 1, total)

        return results

    def _report_to_state_entry(self, report: ResistanceReport) -> StateProfileEntry:
        """Convert a ResistanceReport to a StateProfileEntry.

        Extracts raw metrics from the report and computes the stability score.
        """
        site_delta = report.metrics.get("site_uncertainty_delta", 0.0)
        max_hub_delta = report.metrics.get("max_hub_delta", 0.0)
        propagation_radius = report.metrics.get("propagation_radius", 0)

        stability = compute_stability_score(
            site_delta=site_delta,
            max_hub_delta=max_hub_delta,
            propagation_radius=propagation_radius,
            mechanism_class=report.mechanism_class,
            config=self._config,
        )

        return StateProfileEntry(
            mechanism_class=report.mechanism_class,
            stability_score=stability,
            confidence_score=report.confidence_score,
            site_uncertainty_delta=site_delta,
            max_hub_delta=max_hub_delta,
            propagation_radius=propagation_radius,
        )

    async def _persist_ensemble_profile(self, profile: EnsembleProfile) -> None:
        """Persist an EnsembleProfile to fact_ensemble_resistance_profile.

        Uses upsert semantics: if a profile for the same variant and structure
        ensemble already exists, it is updated rather than duplicated.

        Records provenance by storing all per-structure run_ids in the row,
        linking the ensemble profile back to the individual resistance profiles.

        Args:
            profile: The EnsembleProfile to persist.
        """
        # Build a deterministic ensemble_id from variant + sorted structure_ids
        structure_ids = sorted(
            e.structure_id for e in profile.structure_entries
        )
        ensemble_id = f"ep_{profile.variant}_{'_'.join(structure_ids)}"

        # Serialize state_profile and mechanism_shifts to JSON
        profile_dict = ensemble_profile_to_dict(profile)
        state_profile_json = json.dumps(profile_dict["state_profile"])
        mechanism_shifts_json = json.dumps(profile_dict["mechanism_shifts"])

        # Filter out empty run_ids (from failed structures)
        run_ids = [rid for rid in profile.per_structure_run_ids if rid]

        try:
            await self._db.execute(
                """
                INSERT INTO fact_ensemble_resistance_profile (
                    ensemble_id, variant, sss_score, category,
                    state_profile, mechanism_shifts, clinical_relevance,
                    structure_ids, run_ids, computed_at
                ) VALUES (
                    :ensemble_id, :variant, :sss_score, :category,
                    :state_profile, :mechanism_shifts, :clinical_relevance,
                    :structure_ids, :run_ids, :computed_at
                )
                ON CONFLICT (variant, structure_ids) DO UPDATE SET
                    ensemble_id = EXCLUDED.ensemble_id,
                    sss_score = EXCLUDED.sss_score,
                    category = EXCLUDED.category,
                    state_profile = EXCLUDED.state_profile,
                    mechanism_shifts = EXCLUDED.mechanism_shifts,
                    clinical_relevance = EXCLUDED.clinical_relevance,
                    run_ids = EXCLUDED.run_ids,
                    computed_at = EXCLUDED.computed_at
                """,
                {
                    "ensemble_id": ensemble_id,
                    "variant": profile.variant,
                    "sss_score": float(profile.sss_score),
                    "category": profile.category,
                    "state_profile": state_profile_json,
                    "mechanism_shifts": mechanism_shifts_json,
                    "clinical_relevance": profile.clinical_relevance,
                    "structure_ids": structure_ids,
                    "run_ids": run_ids,
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            await self._db.commit()
            logger.info(
                "Persisted ensemble profile for %s (ensemble_id=%s)",
                profile.variant,
                ensemble_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to persist ensemble profile for %s: %s",
                profile.variant,
                e,
            )
            try:
                await self._db.rollback()
            except Exception:
                pass
