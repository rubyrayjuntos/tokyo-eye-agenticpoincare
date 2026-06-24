"""Resistance Profiler — orchestrator with baseline caching.

This module provides the main entry point for resistance profiling:
- Caches wild-type baseline inference results per structure
- Auto-detects propagation hubs from source leaks + Phase 4 pathway targets
- Profiles individual mutations against the cached baseline
- Records provenance for both baseline and mutant runs

The profiler CONSUMES Phase 4 outputs (pathway targets, spectral gap) from
the WT baseline run to identify hubs. It does NOT re-run Phase 4 on mutant
graphs — the mutation effect is measured at the GNN embedding level.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from science.dtie.common.graph_builder import GraphBuilder
from science.dtie.common.interfaces import GNNInferenceResult, GNNNodeOutput
from science.dtie.common.normalizer_payloads import (
    ProvenanceContext,
    RunType,
    SourceType,
)
from science.dtie.v5.gnn.runner import V5GNNRunner
from science.dtie.v5.resistance.models import (
    BatchReport,
    ClassifierConfig,
    HubMetrics,
    MutationSpec,
    ResistanceReport,
    mutant_structure_id,
)
from science.dtie.v5.resistance.operator import MutationOperator

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "checkpoints_v5/v5_stage4_11prot.pt"

# Source leak threshold: residues with epistemic uncertainty above this
# are considered source-leak candidates for hub detection.
SOURCE_LEAK_THRESHOLD = 0.3

# Top-N pathway targets to include as hubs
TOP_PATHWAY_TARGETS = 5


class ResistanceProfiler:
    """Main entry point for resistance profiling.

    Orchestrates:
    1. Wild-type baseline inference (cached per structure_id)
    2. Hub auto-detection from source leaks + Phase 4 pathways
    3. Virtual mutation application via MutationOperator
    4. Mutant GNN inference
    5. WT vs mutant comparison for classification

    Usage:
        profiler = ResistanceProfiler(db=connection)
        report = await profiler.profile_mutation("4obe", mutation)
    """

    def __init__(
        self,
        db: Any,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str = "cpu",
        classifier_config: ClassifierConfig | None = None,
    ):
        self._db = db
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._config = classifier_config or ClassifierConfig()

        # Core components
        self._runner = V5GNNRunner(
            checkpoint_path=checkpoint_path, device=device
        )
        self._builder = GraphBuilder(db=db)
        self._operator = MutationOperator(self._builder)

        # Baseline cache: structure_id → (gnn_result, hub_residues, phase4_outputs)
        self._baseline_cache: dict[
            str, tuple[GNNInferenceResult, list[tuple[str, int]], dict]
        ] = {}

        # Edge index cache for path interception analysis
        self._edge_index_cache: dict[str, np.ndarray] = {}

        # Baseline run_ids for provenance linking
        self._baseline_run_ids: dict[str, str] = {}

    async def profile_mutation(
        self,
        structure_id: str,
        mutation: MutationSpec,
        hub_residues: list[tuple[str, int]] | None = None,
    ) -> ResistanceReport:
        """Profile a single mutation against a structure.

        Args:
            structure_id: Canonical structure_id for the wild-type protein.
            mutation: The amino acid substitution to apply.
            hub_residues: Optional explicit hub residues. If None, auto-detected
                from WT source leaks + Phase 4 pathway targets.

        Returns:
            ResistanceReport with classification and metrics.
        """
        # 1. Ensure WT baseline is cached
        wt_result, detected_hubs, phase4_outputs = await self._ensure_baseline(
            structure_id
        )

        # Use provided hubs or auto-detected ones
        hubs = hub_residues if hub_residues is not None else detected_hubs

        # 2. Build mutant graph
        mutant_graph, node_idx = await self._operator.build_mutant_graph(
            structure_id, mutation
        )

        # 3. Run GNN inference on mutant graph
        mut_sid = mutant_structure_id(structure_id, mutation)
        mut_result = await self._runner.run_inference(
            structure_id=mut_sid,
            graph_data=mutant_graph,
        )

        # 4. Record provenance for the mutant run
        mutant_run_id = f"mut_{uuid.uuid4().hex[:12]}"
        baseline_run_id = self._baseline_run_ids.get(structure_id)
        await self._record_provenance(
            run_id=mutant_run_id,
            structure_id=mut_sid,
            run_type=RunType.IN_SILICO_MUTATION,
            parent_run_id=baseline_run_id,
            parameters={
                "mutation": mutation.variant_name,
                "chain": mutation.chain,
                "residue_index": mutation.residue_index,
                "wild_type_aa": mutation.wild_type_aa,
                "mutant_aa": mutation.mutant_aa,
            },
        )

        # 5. Compare WT vs mutant at hub residues and mutation site
        report = self._compare_and_classify(
            wt_result=wt_result,
            mut_result=mut_result,
            mutation=mutation,
            hub_residues=hubs,
            phase4_outputs=phase4_outputs,
            edge_index=self._edge_index_cache.get(structure_id),
        )

        # 6. Persist the resistance report
        await self._persist_report(
            report=report,
            run_id=mutant_run_id,
            structure_id=structure_id,
        )

        return report

    async def profile_batch(
        self,
        structure_id: str,
        mutations: list[MutationSpec],
        hub_residues: list[tuple[str, int]] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BatchReport:
        """Profile a batch of mutations against a single structure.

        Runs the WT baseline once, then applies each mutation independently.
        Failed mutations get an error recorded in their report; the batch
        continues processing remaining mutations.

        Args:
            structure_id: Canonical structure_id for the wild-type protein.
            mutations: List of amino acid substitutions to scan.
            hub_residues: Optional explicit hub residues. If None, auto-detected.
            on_progress: Optional callback(completed, total) for progress reporting.

        Returns:
            BatchReport with per-mutation reports and summary counts.

        Raises:
            ValueError: If mutations list is empty.
        """

        if not mutations:
            raise ValueError("At least one mutation required")

        total = len(mutations)

        # 1. Ensure WT baseline is cached (single computation for the batch)
        wt_result, detected_hubs, phase4_outputs = await self._ensure_baseline(
            structure_id
        )
        hubs = hub_residues if hub_residues is not None else detected_hubs

        # Extract baseline metrics for the report
        baseline_metrics = self._extract_baseline_metrics(wt_result, phase4_outputs)

        # 2. Process each mutation independently with error isolation
        reports: list[ResistanceReport] = []
        for idx, mutation in enumerate(mutations):
            try:
                report = await self._profile_single_mutation(
                    structure_id=structure_id,
                    mutation=mutation,
                    wt_result=wt_result,
                    hub_residues=hubs,
                    phase4_outputs=phase4_outputs,
                )
            except Exception as e:
                logger.warning(
                    "Mutation %s failed: %s", mutation.variant_name, e
                )
                report = ResistanceReport(
                    variant=mutation.variant_name,
                    mechanism_class="Error",
                    confidence_score=0.0,
                    metrics={},
                    affected_pathways=[],
                    structural_impact="",
                    hub_details=[],
                    error=str(e),
                )
            reports.append(report)

            if on_progress is not None:
                on_progress(idx + 1, total)

        # 3. Compute summary counts
        type_i_count = sum(
            1 for r in reports if r.mechanism_class == "Type_I_Steric"
        )
        type_ii_count = sum(
            1 for r in reports if r.mechanism_class == "Type_II_Allosteric"
        )
        hybrid_count = sum(
            1 for r in reports if r.mechanism_class == "Hybrid"
        )
        error_count = sum(1 for r in reports if r.error is not None)

        return BatchReport(
            structure_id=structure_id,
            total_variants=total,
            type_i_count=type_i_count,
            type_ii_count=type_ii_count,
            hybrid_count=hybrid_count,
            error_count=error_count,
            baseline_metrics=baseline_metrics,
            reports=reports,
        )

    async def _profile_single_mutation(
        self,
        structure_id: str,
        mutation: MutationSpec,
        wt_result: GNNInferenceResult,
        hub_residues: list[tuple[str, int]],
        phase4_outputs: dict,
    ) -> ResistanceReport:
        """Profile a single mutation using pre-computed baseline.

        This is the inner loop of profile_batch — it skips the baseline
        computation and uses the provided WT result directly.
        """
        # Build mutant graph
        mutant_graph, node_idx = await self._operator.build_mutant_graph(
            structure_id, mutation
        )

        # Run GNN inference on mutant graph
        mut_sid = mutant_structure_id(structure_id, mutation)
        mut_result = await self._runner.run_inference(
            structure_id=mut_sid,
            graph_data=mutant_graph,
        )

        # Record provenance for the mutant run
        mutant_run_id = f"mut_{uuid.uuid4().hex[:12]}"
        baseline_run_id = self._baseline_run_ids.get(structure_id)
        await self._record_provenance(
            run_id=mutant_run_id,
            structure_id=mut_sid,
            run_type=RunType.IN_SILICO_MUTATION,
            parent_run_id=baseline_run_id,
            parameters={
                "mutation": mutation.variant_name,
                "chain": mutation.chain,
                "residue_index": mutation.residue_index,
                "wild_type_aa": mutation.wild_type_aa,
                "mutant_aa": mutation.mutant_aa,
            },
        )

        # Compare WT vs mutant at hub residues and mutation site
        report = self._compare_and_classify(
            wt_result=wt_result,
            mut_result=mut_result,
            mutation=mutation,
            hub_residues=hub_residues,
            phase4_outputs=phase4_outputs,
            edge_index=self._edge_index_cache.get(structure_id),
        )

        # Persist the resistance report
        await self._persist_report(
            report=report,
            run_id=mutant_run_id,
            structure_id=structure_id,
        )

        return report

    def _extract_baseline_metrics(
        self,
        wt_result: GNNInferenceResult,
        phase4_outputs: dict,
    ) -> dict:
        """Extract WT baseline metrics for inclusion in batch output.

        Includes source leaks, spectral gap, and top pathways.
        """
        # Source leak residues
        source_leaks = [
            {"chain": n.chain_label, "residue_index": n.residue_index,
             "epistemic": n.epistemic_uncertainty}
            for n in wt_result.nodes
            if n.epistemic_uncertainty >= SOURCE_LEAK_THRESHOLD
        ]

        # Spectral gap from Phase 4
        spectral = phase4_outputs.get("spectral", {})
        spectral_gap = spectral.get("lambda_2")

        # Top pathways from Phase 4
        pathways = phase4_outputs.get("pathways", [])
        top_pathways = [
            {
                "source_residue": p.get("source_residue"),
                "target_residue": p.get("target_residue"),
                "coupling_strength": p.get("coupling_strength"),
            }
            for p in pathways[:TOP_PATHWAY_TARGETS]
        ]

        return {
            "source_leaks": source_leaks,
            "spectral_gap": spectral_gap,
            "top_pathways": top_pathways,
            "node_count": len(wt_result.nodes),
        }

    async def _ensure_baseline(
        self,
        structure_id: str,
        force_refresh: bool = False,
    ) -> tuple[GNNInferenceResult, list[tuple[str, int]], dict]:
        """Get or compute the WT baseline and auto-detect hubs.

        Caches the result so subsequent mutations against the same
        structure reuse the baseline without re-running inference.

        Args:
            structure_id: Canonical structure_id.
            force_refresh: If True, discard cache and recompute.

        Returns:
            Tuple of (wt_gnn_result, hub_residues, phase4_outputs).
        """
        if not force_refresh and structure_id in self._baseline_cache:
            return self._baseline_cache[structure_id]

        logger.info("Computing WT baseline for %s", structure_id)

        # Build WT graph
        protein_graph = await self._builder.build_graph(structure_id)
        pyg_data = self._builder.to_pyg(protein_graph)

        # Cache edge_index for path interception analysis
        self._edge_index_cache[structure_id] = protein_graph.edge_index

        # Run GNN inference on WT
        wt_result = await self._runner.run_inference(
            structure_id=structure_id,
            graph_data=pyg_data,
        )

        # Run Phase 4 for spectral analysis + pathway targets
        phase4_outputs = await self._run_phase4(structure_id, wt_result)

        # Auto-detect hubs from source leaks + Phase 4 pathway targets
        hub_residues = self._auto_detect_hubs(wt_result, phase4_outputs)

        # Record provenance for the baseline run
        baseline_run_id = f"baseline_{uuid.uuid4().hex[:12]}"
        self._baseline_run_ids[structure_id] = baseline_run_id
        await self._record_provenance(
            run_id=baseline_run_id,
            structure_id=structure_id,
            run_type=RunType.RESISTANCE_BASELINE,
            parent_run_id=None,
            parameters={
                "checkpoint_path": self._checkpoint_path,
                "device": self._device,
                "hub_count": len(hub_residues),
                "node_count": len(wt_result.nodes),
                "spectral_gap": phase4_outputs.get("spectral", {}).get(
                    "lambda_2", None
                ),
            },
        )

        # Cache the baseline
        self._baseline_cache[structure_id] = (wt_result, hub_residues, phase4_outputs)

        logger.info(
            "WT baseline cached for %s: %d nodes, %d hubs detected",
            structure_id,
            len(wt_result.nodes),
            len(hub_residues),
        )

        return wt_result, hub_residues, phase4_outputs

    async def _run_phase4(
        self,
        structure_id: str,
        gnn_result: GNNInferenceResult,
    ) -> dict:
        """Run Phase 4 resistance mapping on the WT structure.

        Returns the phase4 outputs dict containing pathways and spectral data.
        """
        from science.dtie.v5.phases.phase4_resistance import run_phase4_resistance

        phase4_result = await run_phase4_resistance(
            db=self._db,
            gnn_result=gnn_result,
            structure_id=structure_id,
        )

        if phase4_result.success:
            return phase4_result.outputs
        else:
            logger.warning(
                "Phase 4 failed for %s: %s",
                structure_id,
                phase4_result.outputs.get("error", "unknown"),
            )
            return {"pathways": [], "spectral": {}}

    def _auto_detect_hubs(
        self,
        wt_result: GNNInferenceResult,
        phase4_outputs: dict,
    ) -> list[tuple[str, int]]:
        """Identify propagation hubs from WT source leaks + Phase 4 targets.

        Hub detection strategy:
        1. Source leak residues: nodes with high epistemic uncertainty
           (these are where information "leaks" from the hyperbolic embedding)
        2. Phase 4 pathway targets: the deepest residues that serve as
           effector endpoints in the conductance graph

        The union of these sets forms the hub residues for comparison.

        Args:
            wt_result: Wild-type GNN inference result.
            phase4_outputs: Phase 4 outputs dict with pathways and spectral data.

        Returns:
            List of (chain, residue_index) tuples identifying hub residues.
        """
        hubs: set[tuple[str, int]] = set()

        # 1. Source leak residues (high epistemic uncertainty)
        for node in wt_result.nodes:
            if node.epistemic_uncertainty >= SOURCE_LEAK_THRESHOLD:
                hubs.add((node.chain_label, node.residue_index))

        # 2. Phase 4 pathway targets (top coupling targets)
        pathways = phase4_outputs.get("pathways", [])
        target_residues: set[int] = set()
        for pathway in pathways[:TOP_PATHWAY_TARGETS]:
            target_node = pathway.get("target_node")
            if target_node is not None:
                target_residues.add(target_node)

        # Map target node indices back to (chain, residue_index)
        if target_residues:
            for node in wt_result.nodes:
                # The target_node from phase4 is a graph index
                # We need to find the corresponding node in wt_result
                node_key = (node.chain_label, node.residue_index)
                # Phase 4 uses node indices that correspond to wt_result.nodes order
                node_idx = next(
                    (
                        i
                        for i, n in enumerate(wt_result.nodes)
                        if n.chain_label == node.chain_label
                        and n.residue_index == node.residue_index
                    ),
                    None,
                )
                if node_idx is not None and node_idx in target_residues:
                    hubs.add(node_key)

        if not hubs:
            # Fallback: use top-5 highest epistemic uncertainty nodes
            sorted_nodes = sorted(
                wt_result.nodes,
                key=lambda n: n.epistemic_uncertainty,
                reverse=True,
            )
            for node in sorted_nodes[:5]:
                hubs.add((node.chain_label, node.residue_index))

        return list(hubs)

    def _compare_and_classify(
        self,
        wt_result: GNNInferenceResult,
        mut_result: GNNInferenceResult,
        mutation: MutationSpec,
        hub_residues: list[tuple[str, int]],
        phase4_outputs: dict,
        edge_index: np.ndarray | None = None,
    ) -> ResistanceReport:
        """Compare WT and mutant outputs, classify resistance mechanism.

        Decision tree:
        1. Compute site_delta (Δeps at mutation site)
        2. Compute hub_deltas (Δeps at each hub)
        3. Compute propagation_radius (count of residues with |Δeps| > threshold)
        4. Classify:
           - If max(|hub_delta|) < HUB_THRESHOLD and site_delta significant → Type I
           - If max(|hub_delta|) >= HUB_THRESHOLD and radius > 5 → Type II
           - Otherwise → Hybrid

        Args:
            wt_result: Wild-type GNN inference result.
            mut_result: Mutant GNN inference result.
            mutation: The mutation specification.
            hub_residues: List of (chain, residue_index) hub residues.
            phase4_outputs: Phase 4 outputs for pathway annotation.

        Returns:
            ResistanceReport with classification and metrics.
        """
        config = self._config

        # Build lookup maps for WT and mutant nodes
        wt_map: dict[tuple[str, int], GNNNodeOutput] = {
            (n.chain_label, n.residue_index): n for n in wt_result.nodes
        }
        mut_map: dict[tuple[str, int], GNNNodeOutput] = {
            (n.chain_label, n.residue_index): n for n in mut_result.nodes
        }

        # Compute site delta (Δeps at mutation site)
        site_key = (mutation.chain, mutation.residue_index)
        wt_site = wt_map.get(site_key)
        mut_site = mut_map.get(site_key)

        if wt_site is None or mut_site is None:
            return ResistanceReport(
                variant=mutation.variant_name,
                mechanism_class="Unknown",
                confidence_score=0.0,
                metrics={"error": "Mutation site not found in results"},
                affected_pathways=[],
                structural_impact="Unable to compute — site not found",
                hub_details=[],
                error="Mutation site not found in WT or mutant results",
            )

        site_delta = mut_site.epistemic_uncertainty - wt_site.epistemic_uncertainty

        # Compute hub deltas
        hub_details: list[HubMetrics] = []
        for chain, res_idx in hub_residues:
            hub_key = (chain, res_idx)
            wt_hub = wt_map.get(hub_key)
            mut_hub = mut_map.get(hub_key)
            if wt_hub is None or mut_hub is None:
                continue

            delta_eps = mut_hub.epistemic_uncertainty - wt_hub.epistemic_uncertainty
            delta_depth = (
                mut_hub.cone_depth - wt_hub.cone_depth
            )

            hub_details.append(
                HubMetrics(
                    residue_index=res_idx,
                    chain=chain,
                    wt_epistemic=wt_hub.epistemic_uncertainty,
                    mut_epistemic=mut_hub.epistemic_uncertainty,
                    delta_epistemic=delta_eps,
                    wt_cone_depth=wt_hub.cone_depth,
                    mut_cone_depth=mut_hub.cone_depth,
                    delta_cone_depth=delta_depth,
                )
            )

        # Compute propagation radius
        propagation_count = 0
        for key in wt_map:
            wt_node = wt_map[key]
            mut_node = mut_map.get(key)
            if mut_node is None:
                continue
            delta = abs(
                mut_node.epistemic_uncertainty - wt_node.epistemic_uncertainty
            )
            if delta > abs(config.hub_allosteric_threshold):
                propagation_count += 1

        # Classification decision tree
        max_hub_delta = 0.0
        if hub_details:
            max_hub_delta = max(abs(h.delta_epistemic) for h in hub_details)

        # Step 1: NOISE FILTER — if both metrics are below the noise floor,
        # the mutation has no significant effect on the GNN topology
        noise_floor = config.noise_floor
        if abs(site_delta) < noise_floor and max_hub_delta < noise_floor:
            mechanism_class = "Neutral"
            structural_impact = (
                f"No significant perturbation detected for {mutation.variant_name}. "
                f"Site Δeps={site_delta:.4f}, max hub Δeps={max_hub_delta:.4f} "
                f"(below noise floor {noise_floor})."
            )
        else:
            # Step 2: CHEMICAL BOOST — amplify site_delta based on mutation severity
            # GUARD: Only boost when hub signal does NOT dominate (steric context)
            from science.dtie.v5.resistance.classifier import MechanismClassifier
            _boost_clf = MechanismClassifier(config)
            if max_hub_delta <= abs(site_delta):
                boosted_site_delta = _boost_clf._apply_chemical_boost(site_delta, mutation)
            else:
                boosted_site_delta = site_delta

            # Step 3: Compute significance flags using boosted site delta
            site_significant = abs(boosted_site_delta) > abs(config.site_perturbation_threshold)
            hub_significant = max_hub_delta >= abs(config.hub_allosteric_threshold)
            radius_significant = (
                propagation_count > config.propagation_radius_threshold
            )

            # Step 3: EXCLUSIONARY classification
            if site_significant and not hub_significant:
                mechanism_class = "Type_I_Steric"
                structural_impact = (
                    f"Localized steric perturbation at {mutation.variant_name}. "
                    f"Site Δeps={site_delta:.4f}, no significant hub disruption."
                )
            elif hub_significant and radius_significant:
                mechanism_class = "Type_II_Allosteric"
                structural_impact = (
                    f"Distributed allosteric perturbation from {mutation.variant_name}. "
                    f"Max hub Δeps={max_hub_delta:.4f}, "
                    f"propagation radius={propagation_count} residues."
                )
            elif site_significant and hub_significant and not radius_significant:
                # Check path interception using the classifier's method
                from science.dtie.v5.resistance.classifier import MechanismClassifier
                _clf = MechanismClassifier(config)
                path_intercepts = _clf._check_path_interception(
                    wt_nodes=wt_result.nodes,
                    mut_nodes=mut_result.nodes,
                    mutation=mutation,
                    hub_residues=hub_residues,
                    edge_index=edge_index,
                    threshold=abs(config.hub_allosteric_threshold),
                )
                if path_intercepts:
                    mechanism_class = "Type_II_Allosteric"
                    structural_impact = (
                        f"Narrow allosteric pathway from {mutation.variant_name}. "
                        f"Perturbation intercepts hub pathway. "
                        f"Max hub Δeps={max_hub_delta:.4f}, radius={propagation_count}."
                    )
                else:
                    mechanism_class = "Type_I_Steric"
                    structural_impact = (
                        f"Steric perturbation at {mutation.variant_name} with local ripples. "
                        f"Site Δeps={site_delta:.4f}, hub Δeps={max_hub_delta:.4f}, "
                        f"radius={propagation_count} (no pathway interception)."
                    )
            else:
                mechanism_class = "Hybrid"
                structural_impact = (
                    f"Mixed steric/allosteric signature for {mutation.variant_name}. "
                    f"Site Δeps={site_delta:.4f}, max hub Δeps={max_hub_delta:.4f}, "
                    f"radius={propagation_count}."
                )

        # Compute confidence score
        confidence = self._compute_confidence(
            site_delta=site_delta,
            max_hub_delta=max_hub_delta,
            propagation_radius=propagation_count,
        )

        # TOPOLOGY TIE-BREAKER: If GNN classification is ambiguous, use the
        # GraphComparer evidence layer to break the tie.
        # Triggers on: Hybrid, low confidence, or Type_I_Steric with hub disruption
        # (the latter catches narrow-pathway allosteric mutations misclassified as steric)
        needs_topology = (
            mechanism_class == "Hybrid"
            or confidence < 0.5
            or (mechanism_class == "Type_I_Steric"
                and max_hub_delta >= abs(config.hub_allosteric_threshold))
        )
        if edge_index is not None and needs_topology:
            try:
                from science.dtie.v5.resistance.analysis.graph_comparer import GraphComparer

                # Build node key → index mapping
                node_keys = [(n.chain_label, n.residue_index) for n in wt_result.nodes]
                key_to_idx = {k: i for i, k in enumerate(node_keys)}

                site_key = (mutation.chain, mutation.residue_index)
                site_idx = key_to_idx.get(site_key)
                hub_indices = [
                    key_to_idx[k] for k in
                    [(c, r) for c, r in hub_residues]
                    if k in key_to_idx
                ]

                if site_idx is not None and hub_indices:
                    comparer = GraphComparer(
                        edge_index=edge_index,
                        num_nodes=len(wt_result.nodes),
                    )
                    evidence = comparer.analyze_mutation(
                        mutation_node_idx=site_idx,
                        hub_node_indices=hub_indices,
                    )

                    # Use topology suggestion to override Hybrid classification
                    # Guard: only override to allosteric if GNN detected hub disruption
                    #   AND hub signal is at least as strong as site (pathway amplification)
                    # Guard: only override to steric if GNN detected site perturbation
                    noise_floor = config.noise_floor
                    has_hub_signal = max_hub_delta >= noise_floor
                    has_site_signal = abs(site_delta) >= noise_floor
                    hub_dominates = max_hub_delta > abs(site_delta)

                    if evidence.topology_confidence > 0.4:
                        if (evidence.topology_suggestion == "allosteric"
                                and has_hub_signal and hub_dominates):
                            mechanism_class = "Type_II_Allosteric"
                            structural_impact = (
                                f"Allosteric (topology-confirmed) for {mutation.variant_name}. "
                                f"Site betweenness={evidence.site_betweenness_percentile:.0f}th pctl, "
                                f"hub bridge paths={evidence.hub_paths_through_bridges}/{evidence.total_hubs}. "
                                f"GNN: Δsite={site_delta:.4f}, Δhub={max_hub_delta:.4f}."
                            )
                            confidence = min(1.0, confidence + 0.2 * evidence.topology_confidence)
                        elif (evidence.topology_suggestion == "steric"
                              and has_site_signal):
                            mechanism_class = "Type_I_Steric"
                            structural_impact = (
                                f"Steric (topology-confirmed) for {mutation.variant_name}. "
                                f"Low betweenness={evidence.site_betweenness_percentile:.0f}th pctl, "
                                f"local edge cluster. "
                                f"GNN: Δsite={site_delta:.4f}, Δhub={max_hub_delta:.4f}."
                            )
                            confidence = min(1.0, confidence + 0.2 * evidence.topology_confidence)

            except Exception as e:
                logger.debug("Topology tie-breaker skipped: %s", e)

        # Identify affected pathways from Phase 4 data
        affected_pathways = self._identify_affected_pathways(
            hub_details, phase4_outputs
        )

        metrics = {
            "site_uncertainty_delta": site_delta,
            "max_hub_delta": max_hub_delta,
            "propagation_radius": propagation_count,
            "site_wt_epistemic": wt_site.epistemic_uncertainty,
            "site_mut_epistemic": mut_site.epistemic_uncertainty,
        }

        return ResistanceReport(
            variant=mutation.variant_name,
            mechanism_class=mechanism_class,
            confidence_score=confidence,
            metrics=metrics,
            affected_pathways=affected_pathways,
            structural_impact=structural_impact,
            hub_details=hub_details,
        )

    def _compute_confidence(
        self,
        site_delta: float,
        max_hub_delta: float,
        propagation_radius: int,
    ) -> float:
        """Compute classification confidence based on metric magnitudes.

        Confidence is higher when the distinguishing metrics are far from
        the classification thresholds (clear separation).

        Returns:
            Float in [0.0, 1.0].
        """
        config = self._config

        # Site confidence: how far below the threshold
        site_ratio = abs(site_delta) / abs(config.site_perturbation_threshold)
        site_conf = min(site_ratio, 2.0) / 2.0  # Saturates at 2x threshold

        # Hub confidence: how far from the hub threshold
        hub_ratio = max_hub_delta / abs(config.hub_allosteric_threshold)
        hub_conf = min(hub_ratio, 3.0) / 3.0  # Saturates at 3x threshold

        # Radius confidence: how far above the radius threshold
        radius_ratio = propagation_radius / max(
            config.propagation_radius_threshold, 1
        )
        radius_conf = min(radius_ratio, 3.0) / 3.0

        # Combined confidence: weighted average
        confidence = 0.4 * site_conf + 0.35 * hub_conf + 0.25 * radius_conf

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _identify_affected_pathways(
        self,
        hub_details: list[HubMetrics],
        phase4_outputs: dict,
    ) -> list[str]:
        """Identify which Phase 4 pathways are affected by the mutation.

        A pathway is "affected" if either its source or target residue
        appears in the hub_details with a significant delta.

        Returns:
            List of pathway description strings.
        """
        pathways = phase4_outputs.get("pathways", [])
        affected_hub_indices = {
            h.residue_index
            for h in hub_details
            if abs(h.delta_epistemic) >= abs(self._config.hub_allosteric_threshold)
        }

        affected: list[str] = []
        for pathway in pathways:
            source_res = pathway.get("source_residue")
            target_res = pathway.get("target_residue")
            if source_res in affected_hub_indices or target_res in affected_hub_indices:
                coupling = pathway.get("coupling_strength", 0.0)
                affected.append(
                    f"Pathway {source_res}→{target_res} "
                    f"(coupling={coupling:.3f})"
                )

        return affected


    async def _record_provenance(
        self,
        run_id: str,
        structure_id: str,
        run_type: RunType,
        parent_run_id: str | None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Record a provenance_run entry for a resistance profiling run.

        WT baseline runs use run_type="resistance_baseline".
        Mutant runs use run_type="in_silico_mutation" with parent_run_id
        pointing to the baseline.

        Args:
            run_id: Unique run identifier.
            structure_id: Structure being profiled (may be virtual for mutants).
            run_type: RunType enum value.
            parent_run_id: Parent run for lineage (baseline run_id for mutants).
            parameters: Run parameters to store as JSONB.
        """
        try:
            await self._db.execute(
                """
                INSERT INTO provenance_run (
                    run_id, structure_id, model_version, checkpoint_uri,
                    run_type, source_type, parameters, parent_run_id, started_at
                ) VALUES (
                    :run_id, :structure_id, :model_version, :checkpoint_uri,
                    :run_type, :source_type, :parameters, :parent_run_id, :started_at
                )
                ON CONFLICT (run_id) DO NOTHING
                """,
                {
                    "run_id": run_id,
                    "structure_id": structure_id,
                    "model_version": "GOSPConeMapper-v5",
                    "checkpoint_uri": self._checkpoint_path,
                    "run_type": run_type.value,
                    "source_type": SourceType.PROBABILISTIC.value,
                    "parameters": json.dumps(parameters) if parameters else None,
                    "parent_run_id": parent_run_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.warning("Failed to record provenance for %s: %s", run_id, e)
            try:
                await self._db.rollback()
            except Exception:
                pass

    async def _persist_report(
        self,
        report: ResistanceReport,
        run_id: str,
        structure_id: str,
    ) -> None:
        """Persist a ResistanceReport to fact_resistance_profile.

        Args:
            report: The classification report to persist.
            run_id: The provenance run_id for this mutation.
            structure_id: The base structure_id (not the virtual mutant ID).
        """
        profile_id = f"rp_{run_id}_{report.variant}"

        hub_details_json = [
            {
                "residue_index": h.residue_index,
                "chain": h.chain,
                "wt_epistemic": h.wt_epistemic,
                "mut_epistemic": h.mut_epistemic,
                "delta_epistemic": h.delta_epistemic,
                "wt_cone_depth": h.wt_cone_depth,
                "mut_cone_depth": h.mut_cone_depth,
                "delta_cone_depth": h.delta_cone_depth,
            }
            for h in report.hub_details
        ]

        try:
            await self._db.execute(
                """
                INSERT INTO fact_resistance_profile (
                    profile_id, run_id, structure_id, variant,
                    mechanism_class, confidence_score,
                    site_uncertainty_delta, max_hub_delta,
                    propagation_radius, metrics, hub_details,
                    affected_pathways, structural_impact, error,
                    computed_at
                ) VALUES (
                    :profile_id, :run_id, :structure_id, :variant,
                    :mechanism_class, :confidence_score,
                    :site_uncertainty_delta, :max_hub_delta,
                    :propagation_radius, :metrics, :hub_details,
                    :affected_pathways, :structural_impact, :error,
                    :computed_at
                )
                ON CONFLICT (profile_id) DO UPDATE SET
                    mechanism_class = EXCLUDED.mechanism_class,
                    confidence_score = EXCLUDED.confidence_score,
                    site_uncertainty_delta = EXCLUDED.site_uncertainty_delta,
                    max_hub_delta = EXCLUDED.max_hub_delta,
                    propagation_radius = EXCLUDED.propagation_radius,
                    metrics = EXCLUDED.metrics,
                    hub_details = EXCLUDED.hub_details,
                    affected_pathways = EXCLUDED.affected_pathways,
                    structural_impact = EXCLUDED.structural_impact,
                    error = EXCLUDED.error,
                    computed_at = EXCLUDED.computed_at
                """,
                {
                    "profile_id": profile_id,
                    "run_id": run_id,
                    "structure_id": structure_id,
                    "variant": report.variant,
                    "mechanism_class": report.mechanism_class,
                    "confidence_score": report.confidence_score,
                    "site_uncertainty_delta": report.metrics.get(
                        "site_uncertainty_delta"
                    ),
                    "max_hub_delta": report.metrics.get("max_hub_delta"),
                    "propagation_radius": report.metrics.get("propagation_radius"),
                    "metrics": json.dumps(report.metrics, default=str),
                    "hub_details": json.dumps(hub_details_json),
                    "affected_pathways": json.dumps(report.affected_pathways),
                    "structural_impact": report.structural_impact,
                    "error": report.error,
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.warning(
                "Failed to persist resistance report for %s: %s",
                report.variant,
                e,
            )
            try:
                await self._db.rollback()
            except Exception:
                pass
