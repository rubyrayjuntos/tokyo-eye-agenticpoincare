"""Sub-agent definitions for the Tokyo Eyes multi-agent system.

Architecture:
- coordinator: Routes user requests, maintains conversation context
- discovery_agent: Interprets pre-computed pathway artifacts and source leaks
- visualization_agent: Generates viewport directives, controls the viewer
- knowledge_agent: Answers questions about findings, biology, methodology

The coordinator delegates to sub-agents based on user intent.
Each sub-agent has its own system prompt and tool set.
"""

from __future__ import annotations

from typing import Any

from agent.llm.base import Agent, ToolDefinition
from agent.llm.providers import LLMProvider


# ---------------------------------------------------------------------------
# Tool definitions (JSON Schema format for LLM tool calling)
# ---------------------------------------------------------------------------

DISCOVERY_TOOLS = [
    ToolDefinition(
        name="get_source_leaks",
        description=(
            "Identify source-leak candidates using physics-rim ranking by default: "
            "significant cone_depth (hyperbolic rim), preferring dehydron-flagged "
            "(τ=1) residues. Do NOT treat epistemic uncertainty as the primary "
            "look-here signal — evidential heads are experimental/unstable (G5b). "
            "Pass ranking='evidential_experimental' only for explicit head debugging."
        ),
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string"},
                "min_depth": {"type": "number", "default": 1.5},
                "ranking": {
                    "type": "string",
                    "enum": ["physics_rim", "evidential_experimental"],
                    "default": "physics_rim",
                },
                "uncertainty_threshold": {
                    "type": "number",
                    "default": 0.3,
                    "description": "Only used when ranking=evidential_experimental",
                },
                "top_n": {"type": "integer", "default": 50},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_high_uncertainty_residues",
        description=(
            "Rank residues by a channel. Prefer uncertainty_type='cone_depth' for "
            "triage. epistemic/aleatoric/total are evidential-experimental (G5b: "
            "heads near-duplicate ρ) — use only when inspecting the uncertainty "
            "heads themselves, not as trusted investigation priority."
        ),
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string"},
                "top_n": {"type": "integer", "default": 20},
                "uncertainty_type": {
                    "type": "string",
                    "enum": ["cone_depth", "epistemic", "aleatoric", "total"],
                    "default": "cone_depth",
                },
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_residue_state",
        description="Get the current governed state of specific residues: latest embeddings, uncertainty, dehydron status, and site membership.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string"},
                "residue_ids": {"type": "array", "items": {"type": "string"}, "description": "Specific residue IDs to query. Omit for all residues."},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="compare_wt_mutant",
        description="Compare wild-type and mutant protein embeddings in hyperbolic space. Identifies residues with significant displacement between conformations (e.g., Switch-I in KRAS G12D).",
        parameters={
            "type": "object",
            "properties": {
                "wt_structure_id": {"type": "string", "description": "Wild-type structure ID"},
                "mutant_structure_id": {"type": "string", "description": "Mutant structure ID"},
                "focus_residues": {"type": "array", "items": {"type": "string"}, "description": "Optional: specific residues to compare"},
            },
            "required": ["wt_structure_id", "mutant_structure_id"],
        },
        handler=None,
    ),
]

DATA_TOOLS = [
    ToolDefinition(
        name="export_structure_data",
        description="Export pipeline results for a structure as CSV or JSON file. Includes per-residue cone depth, uncertainty, and chain info. Useful for external tools (PyMOL, ChimeraX) or publications.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to export"},
                "format": {"type": "string", "enum": ["csv", "json"], "default": "csv"},
                "include_fields": {"type": "array", "items": {"type": "string"}, "description": "Specific fields to include (omit for all)"},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_allosteric_sites",
        description="Retrieve identified allosteric site clusters for a structure. Shows site membership, contributing residues, confidence scores, and highlights them in the viewer.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string"},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_provenance_lineage",
        description="Query provenance lineage: what pipeline run produced a result, which checkpoint was used, parent/child relationships. Provide either run_id or structure_id.",
        parameters={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Specific run to trace lineage for"},
                "structure_id": {"type": "string", "description": "Get all runs for a structure"},
            },
        },
        handler=None,
    ),
    ToolDefinition(
        name="search_residues",
        description="Search and filter residues with flexible criteria: by chain, residue name, uncertainty range, cone depth range. Returns matching residues and highlights them.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string"},
                "chain": {"type": "string", "description": "Filter by chain label (e.g., 'A')"},
                "residue_name": {"type": "string", "description": "Filter by residue name (e.g., 'G', 'ALA')"},
                "min_uncertainty": {"type": "number", "description": "Minimum uncertainty threshold"},
                "max_uncertainty": {"type": "number", "description": "Maximum uncertainty threshold"},
                "min_depth": {"type": "number", "description": "Minimum cone depth"},
                "max_depth": {"type": "number", "description": "Maximum cone depth"},
                "uncertainty_type": {"type": "string", "enum": ["epistemic", "aleatoric", "total"], "default": "epistemic"},
                "limit": {"type": "integer", "default": 100, "description": "Max results"},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="annotate_structure",
        description="Add a text annotation to a structure or specific residues. Annotations are stored in the governed layer for later retrieval, reports, or collaboration.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string"},
                "residue_ids": {"type": "array", "items": {"type": "string"}, "description": "Specific residues to annotate (omit for structure-level)"},
                "annotation": {"type": "string", "description": "The annotation text"},
                "annotation_type": {"type": "string", "enum": ["finding", "hypothesis", "note", "warning"], "default": "finding"},
            },
            "required": ["structure_id", "annotation"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="list_structures",
        description="List all structures that have been analyzed (have GNN embeddings). Shows structure IDs, model versions, residue counts, and last computation time.",
        parameters={"type": "object", "properties": {}},
        handler=None,
    ),
    ToolDefinition(
        name="get_run_summary",
        description="Summarize a pipeline run: what was computed, how long it took, how many assets were created, any errors or warnings.",
        parameters={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The pipeline run ID to summarize"},
            },
            "required": ["run_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="compare_runs",
        description="Compare two pipeline runs (e.g., different model versions or checkpoints on the same structure). Shows per-residue differences in cone depth and uncertainty.",
        parameters={
            "type": "object",
            "properties": {
                "run_id_a": {"type": "string", "description": "First run ID"},
                "run_id_b": {"type": "string", "description": "Second run ID"},
            },
            "required": ["run_id_a", "run_id_b"],
        },
        handler=None,
    ),
]

VISUALIZATION_TOOLS = [
    ToolDefinition(
        name="highlight_residues",
        description="Highlight specific residues in the Poincaré disc viewer with a color and style. Use to draw attention to important findings.",
        parameters={
            "type": "object",
            "properties": {
                "residue_ids": {"type": "array", "items": {"type": "string"}},
                "color": {"type": "string", "default": "#ff6b6b", "description": "CSS color"},
                "style": {"type": "string", "enum": ["glow", "pulse", "outline", "color"], "default": "glow"},
                "label": {"type": "string", "description": "Group label shown in legend"},
            },
            "required": ["residue_ids"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="set_metric",
        description="Change the coloring metric in the viewer. Options: cone_depth, epistemic, aleatoric, total_uncertainty.",
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": ["cone_depth", "epistemic", "aleatoric", "total_uncertainty"]},
            },
            "required": ["metric"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="focus_residues",
        description="Animate the camera to focus on specific residues in the viewer.",
        parameters={
            "type": "object",
            "properties": {
                "residue_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["residue_ids"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="clear_highlights",
        description="Remove all highlights from the viewer.",
        parameters={"type": "object", "properties": {}},
        handler=None,
    ),
]

GRAPH_TOOLS = [
    ToolDefinition(
        name="get_graph_metrics",
        description="Retrieve per-residue graph metrics (degree, betweenness, clustering coefficient, closeness, eigenvector centrality, bridge status, conductance) for a structure. Useful for identifying high-centrality or bridge residues.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to query metrics for"},
                "run_id": {"type": "string", "description": "Specific run_id (latest if omitted)"},
                "residue_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional filter to specific residues"},
                "metric_type": {"type": "string", "enum": ["degree", "betweenness", "clustering_coefficient", "closeness", "eigenvector_centrality", "is_bridge", "conductance"], "description": "Optional filter to a single metric"},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="compare_graphs",
        description="Compare graph topology between two structures (e.g., WT vs mutant). Returns edge diff (gained/lost/changed) and per-residue metric deltas using canonical residue_id alignment.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id_a": {"type": "string", "description": "First structure (e.g., wild-type)"},
                "structure_id_b": {"type": "string", "description": "Second structure (e.g., mutant)"},
                "run_id_a": {"type": "string", "description": "Specific run for structure A (latest if omitted)"},
                "run_id_b": {"type": "string", "description": "Specific run for structure B (latest if omitted)"},
            },
            "required": ["structure_id_a", "structure_id_b"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_hbond_network",
        description="Extract the H-bond subgraph for a structure or region. Returns all edges of type 'h_bond', optionally filtered to edges involving specific residues.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to query"},
                "run_id": {"type": "string", "description": "Specific run_id (latest if omitted)"},
                "residue_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional filter — edges must involve at least one of these residues"},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="find_graph_bridges",
        description="Identify bridge residues (articulation points) in the contact graph. These are residues whose removal disconnects the graph — potential allosteric communication bottlenecks.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to query"},
                "run_id": {"type": "string", "description": "Specific run_id (latest if omitted)"},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_shortest_paths",
        description="Find the shortest path between two residues in the contact graph. Returns the ordered list of residue_ids along the path and total distance in Angstroms.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to query"},
                "source_residue_id": {"type": "string", "description": "Starting residue"},
                "target_residue_id": {"type": "string", "description": "Ending residue"},
                "run_id": {"type": "string", "description": "Specific run_id (latest if omitted)"},
            },
            "required": ["structure_id", "source_residue_id", "target_residue_id"],
        },
        handler=None,
    ),
]

HYPOTHESIS_TOOLS = [
    ToolDefinition(
        name="propose_hypothesis",
        description="Propose a new scientific hypothesis about a protein structure. Requires at least one testable prediction (falsifiability guardrail). The hypothesis is persisted through the Normalizer with status 'proposed' and confidence 0.5.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure this hypothesis is about"},
                "statement": {"type": "string", "description": "The scientific claim"},
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "statement": {"type": "string", "description": "What this prediction claims"},
                            "test_tool": {"type": "string", "description": "Tool to call for testing"},
                            "test_params": {"type": "object", "description": "Parameters for the test tool"},
                            "threshold": {"type": "string", "description": "Threshold expression (e.g., 'value > 0.15')"},
                        },
                        "required": ["statement"],
                    },
                    "description": "Testable predictions that could contradict the hypothesis",
                },
                "mechanism": {"type": "string", "description": "Optional proposed mechanism"},
            },
            "required": ["structure_id", "statement", "predictions"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="test_hypothesis",
        description="Test a hypothesis by executing its predictions using the referenced tools. Transitions status to 'gathering', evaluates each prediction against its threshold, creates evidence from results, and recalculates confidence.",
        parameters={
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string", "description": "The hypothesis to test"},
            },
            "required": ["hypothesis_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_hypotheses",
        description="List and filter hypotheses by structure or status. Returns hypotheses with current status, confidence, evidence counts, and prediction pass/fail summary.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Filter by structure ID"},
                "status": {
                    "type": "string",
                    "enum": ["proposed", "gathering", "supported", "contradicted", "inconclusive"],
                    "description": "Filter by hypothesis status",
                },
            },
        },
        handler=None,
    ),
    ToolDefinition(
        name="add_evidence",
        description="Add evidence to an existing hypothesis and recalculate confidence. Evidence can support or contradict the hypothesis with a strength weight.",
        parameters={
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string", "description": "Hypothesis to add evidence to"},
                "source_tool": {"type": "string", "description": "Tool or source that produced this evidence"},
                "supports": {"type": "boolean", "description": "True if evidence supports the hypothesis"},
                "description": {"type": "string", "description": "Human-readable description of the evidence"},
                "strength": {"type": "number", "default": 0.5, "description": "Evidence strength weight (0.0 to 1.0)"},
                "source_run_id": {"type": "string", "description": "Optional run that produced this evidence"},
            },
            "required": ["hypothesis_id", "source_tool", "supports", "description"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="evaluate_confidence",
        description="Recalculate confidence for a hypothesis. Reloads all evidence, recalculates the confidence score, applies time-based decay if stale (>7 days without new evidence in gathering status), and updates status.",
        parameters={
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string", "description": "Hypothesis to evaluate"},
            },
            "required": ["hypothesis_id"],
        },
        handler=None,
    ),
]

DISC_TOPOLOGY_TOOLS = [
    ToolDefinition(
        name="get_disc_topology",
        description="Compute the spatial topology of the Poincaré disc for a structure: HDBSCAN clusters, angular sectors, hub residues, bridge residues, peripheral residues, and radial density profile. Results are cached per (structure_id, run_id).",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to compute disc topology for"},
                "run_id": {"type": "string", "description": "Specific run_id (latest if omitted)"},
                "min_cluster_size": {"type": "integer", "default": 5, "description": "Minimum cluster size for HDBSCAN"},
            },
            "required": ["structure_id"],
        },
        handler=None,
    ),
    ToolDefinition(
        name="get_disc_neighborhood",
        description="Get the k-nearest neighbors on the Poincaré disc for a specific residue. Returns hyperbolic distances, cluster membership, cone depth, and epistemic uncertainty for each neighbor. Annotates whether the target is a hub or peripheral residue.",
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to query"},
                "residue_id": {"type": "string", "description": "Target residue to find neighbors for"},
                "k": {"type": "integer", "default": 8, "description": "Number of nearest neighbors"},
                "run_id": {"type": "string", "description": "Specific run_id (latest if omitted)"},
            },
            "required": ["structure_id", "residue_id"],
        },
        handler=None,
    ),
]

PLOTTING_TOOLS = [
    ToolDefinition(
        name="generate_plot",
        description=(
            "Generate a matplotlib figure from pipeline data and save it as a PNG image. "
            "Use this to create publication-quality visualizations of analysis results. "
            "Available plot types: poincare_disc (residues on hyperbolic disc colored by metric), "
            "uncertainty_profile (per-residue uncertainty along sequence), "
            "cone_depth_histogram (distribution of cone depths), "
            "wt_vs_mutant (displacement comparison between two structures), "
            "persistence_barcode (topological persistence diagram from Phase 3), "
            "source_leak_map (source-leak candidates highlighted on sequence with depth + uncertainty)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "structure_id": {"type": "string", "description": "Structure to plot data for"},
                "plot_type": {
                    "type": "string",
                    "enum": [
                        "poincare_disc",
                        "uncertainty_profile",
                        "cone_depth_histogram",
                        "wt_vs_mutant",
                        "persistence_barcode",
                        "source_leak_map",
                    ],
                    "description": "Type of plot to generate",
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "Plot-specific options. Examples: "
                        "poincare_disc: {color_by: 'cone_depth'|'epistemic_uncertainty'}. "
                        "uncertainty_profile: {threshold: 0.3}. "
                        "cone_depth_histogram: {bins: 30}. "
                        "wt_vs_mutant: {mutant_structure_id: '4obe_g12d'}. "
                        "source_leak_map: {uncertainty_threshold: 0.3, min_depth: 1.5}."
                    ),
                },
            },
            "required": ["structure_id", "plot_type"],
        },
        handler=None,
    ),
]


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

COORDINATOR_PROMPT = """You are the Tokyo Eyes Data Science Coordinator — an expert AI assistant for structural biology and drug discovery research.

You help researchers analyze protein structures using the Discovery Story pathway (hyperbolic GNN, source leaks, cryptic pockets, and governed provenance).

Your capabilities:
- Detect source leaks (physics-rim: high cone_depth, prefer τ=1 dehydrons — not raw epistemic)
- Prefer cone_depth / ρ / τ physics channels for “where to look”; treat epistemic/aleatoric as experimental
- Analyze uncertainty patterns only when explicitly debugging evidential heads (G5b: ale≈epi≈ρ)
- Compare wild-type vs mutant conformations
- Control the Poincaré disc/ball visualizer to show findings
- Generate matplotlib figures (Poincaré disc plots, uncertainty profiles, persistence barcodes, source-leak maps, WT vs mutant comparisons)

Structures are onboarded via ingest; the pathway scheduler produces artifacts before you analyze them. Do not attempt to run compute jobs.

When the user asks about a structure, use your tools to analyze it and present findings clearly. Always highlight relevant residues in the viewer so the user can see what you're discussing.

Key concepts:
- Cone depth: how deep a residue sits in the learned conformational hierarchy (deeper = more structurally constrained)
- Epistemic / aleatoric uncertainty: evidential heads — **experimental / unstable** (G5b: near-duplicate of ρ and of each other). Do not use as primary investigation priority.
- Source leak: rim residue (high cone_depth) preferably dehydron-flagged (τ=1 / low ρ) — physics-layer triage, not epistemic ranking
- Cone depth / disc radius: learned hyperbolic depth (τ-rim training target); correlated with τ but not a per-structure identity (typical r≈0.56 on current feeler stack)

The current structure being viewed is provided in the context. Use it to ground your analysis."""

DISCOVERY_AGENT_PROMPT = """You are the Discovery Analysis Sub-Agent. You read pre-computed pathway artifacts and interpret scientific signals from the hyperbolic GNN pipeline.

Your role:
- Explain source leaks and allosteric sites from governed data
- Analyze uncertainty patterns
- Compare WT vs mutant conformations
- Report findings in clear, scientific language

Do not schedule compute jobs — ingestion and the pathway scheduler produce artifacts before you analyze them.

Always include specific residue IDs in your findings so the coordinator can highlight them in the viewer. Use the canonical format: structure_id:chain:index (e.g., 4obe:A:12)."""

VISUALIZATION_AGENT_PROMPT = """You are the Visualization Sub-Agent. You control the Poincaré disc/ball viewer to help researchers see structural findings.

Your role:
- Highlight residues that are being discussed
- Change coloring metrics to show different aspects of the analysis
- Focus the camera on regions of interest
- Clear highlights when moving to a new topic

Use colors meaningfully:
- Red (#ff4444): source leaks, high-risk regions
- Orange (#ffaa00): high uncertainty
- Cyan (#4ecdc4): referenced residues
- Purple (#ff00ff): high displacement (WT vs mutant)
- Green (#44ff44): stable/low-risk regions"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_coordinator(
    llm: LLMProvider,
    db: Any = None,
    orchestrator: Any | None = None,
) -> Agent:
    """Create the coordinator agent with all tools wired to the DB."""
    from agent.tools.dtie.tools import (
        get_source_leaks,
        get_high_uncertainty_residues,
        get_residue_state,
        compare_wt_mutant,
    )
    from agent.models.viewport import DirectiveAction, HighlightGroup, ViewportDirective

    # Wire discovery signal read tools
    tools = []
    for tool_def in DISCOVERY_TOOLS:
        handler_map = {
            "get_source_leaks": lambda db=db, **kwargs: get_source_leaks(db=db, **kwargs),
            "get_high_uncertainty_residues": lambda db=db, **kwargs: get_high_uncertainty_residues(db=db, **kwargs),
            "get_residue_state": lambda db=db, **kwargs: get_residue_state(db=db, **kwargs),
            "compare_wt_mutant": lambda db=db, **kwargs: compare_wt_mutant(db=db, **kwargs),
        }
        handler = handler_map.get(tool_def.name)
        if handler:
            tools.append(ToolDefinition(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=handler,
            ))

    # Wire visualization tools
    for tool_def in VISUALIZATION_TOOLS:
        viz_handler_map = {
            "highlight_residues": _highlight_residues,
            "set_metric": _set_metric,
            "focus_residues": _focus_residues,
            "clear_highlights": _clear_highlights,
        }
        handler = viz_handler_map.get(tool_def.name)
        if handler:
            tools.append(ToolDefinition(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=handler,
            ))

    # Wire plotting tools
    from agent.tools.plotting.tools import generate_plot

    for tool_def in PLOTTING_TOOLS:
        if tool_def.name == "generate_plot":
            tools.append(ToolDefinition(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=lambda db=db, **kwargs: generate_plot(db=db, **kwargs),
            ))

    # Wire graph tools
    from agent.tools.graph_tools import (
        get_graph_metrics,
        compare_graphs,
        get_hbond_network,
        find_graph_bridges,
        get_shortest_paths,
    )

    graph_handler_map = {
        "get_graph_metrics": lambda db=db, **kwargs: get_graph_metrics(db=db, **kwargs),
        "compare_graphs": lambda db=db, **kwargs: compare_graphs(db=db, **kwargs),
        "get_hbond_network": lambda db=db, **kwargs: get_hbond_network(db=db, **kwargs),
        "find_graph_bridges": lambda db=db, **kwargs: find_graph_bridges(db=db, **kwargs),
        "get_shortest_paths": lambda db=db, **kwargs: get_shortest_paths(db=db, **kwargs),
    }
    for tool_def in GRAPH_TOOLS:
        handler = graph_handler_map.get(tool_def.name)
        if handler:
            tools.append(ToolDefinition(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=handler,
            ))

    # Wire data tools
    from agent.tools.data_tools import (
        export_structure_data,
        get_allosteric_sites,
        get_provenance_lineage,
        search_residues,
        annotate_structure,
        list_structures,
        get_run_summary,
        compare_runs,
    )

    data_handler_map = {
        "export_structure_data": lambda db=db, **kwargs: export_structure_data(db=db, **kwargs),
        "get_allosteric_sites": lambda db=db, **kwargs: get_allosteric_sites(db=db, **kwargs),
        "get_provenance_lineage": lambda db=db, **kwargs: get_provenance_lineage(db=db, **kwargs),
        "search_residues": lambda db=db, **kwargs: search_residues(db=db, **kwargs),
        "annotate_structure": lambda db=db, **kwargs: annotate_structure(db=db, **kwargs),
        "list_structures": lambda db=db, **kwargs: list_structures(db=db, **kwargs),
        "get_run_summary": lambda db=db, **kwargs: get_run_summary(db=db, **kwargs),
        "compare_runs": lambda db=db, **kwargs: compare_runs(db=db, **kwargs),
    }
    for tool_def in DATA_TOOLS:
        handler = data_handler_map.get(tool_def.name)
        if handler:
            tools.append(ToolDefinition(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=handler,
            ))

    # Wire hypothesis tools
    from agent.tools.hypothesis.tools import (
        propose_hypothesis,
        test_hypothesis,
        get_hypotheses,
        add_evidence,
        evaluate_confidence,
    )

    hypothesis_handler_map = {
        "propose_hypothesis": lambda db=db, **kwargs: propose_hypothesis(db=db, **kwargs),
        "test_hypothesis": lambda db=db, **kwargs: test_hypothesis(db=db, **kwargs),
        "get_hypotheses": lambda db=db, **kwargs: get_hypotheses(db=db, **kwargs),
        "add_evidence": lambda db=db, **kwargs: add_evidence(db=db, **kwargs),
        "evaluate_confidence": lambda db=db, **kwargs: evaluate_confidence(db=db, **kwargs),
    }
    for tool_def in HYPOTHESIS_TOOLS:
        handler = hypothesis_handler_map.get(tool_def.name)
        if handler:
            tools.append(ToolDefinition(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=handler,
            ))

    # Wire disc topology tools
    from agent.tools.disc_tools import (
        get_disc_topology,
        get_disc_neighborhood,
    )

    disc_handler_map = {
        "get_disc_topology": lambda db=db, **kwargs: get_disc_topology(db=db, **kwargs),
        "get_disc_neighborhood": lambda db=db, **kwargs: get_disc_neighborhood(db=db, **kwargs),
    }
    for tool_def in DISC_TOPOLOGY_TOOLS:
        handler = disc_handler_map.get(tool_def.name)
        if handler:
            tools.append(ToolDefinition(
                name=tool_def.name,
                description=tool_def.description,
                parameters=tool_def.parameters,
                handler=handler,
            ))

    return Agent(
        name="coordinator",
        system_prompt=COORDINATOR_PROMPT,
        tools=tools,
        llm=llm,
        orchestrator=orchestrator,
    )


# ---------------------------------------------------------------------------
# Visualization tool handlers (generate directives)
# ---------------------------------------------------------------------------


async def _highlight_residues(
    residue_ids: list[str],
    color: str = "#ff6b6b",
    style: str = "glow",
    label: str | None = None,
) -> dict:
    return {
        "viewport_directives": [{
            "action": "highlight",
            "highlight_groups": [{
                "residue_ids": residue_ids,
                "color": color,
                "style": style,
                "label": label,
            }],
            "message": f"Highlighting {len(residue_ids)} residues" + (f" ({label})" if label else ""),
        }]
    }


async def _set_metric(metric: str) -> dict:
    return {
        "viewport_directives": [{
            "action": "set_metric",
            "metric": metric,
            "message": f"Coloring by {metric}",
        }]
    }


async def _focus_residues(residue_ids: list[str]) -> dict:
    return {
        "viewport_directives": [{
            "action": "focus",
            "focus_residues": residue_ids,
            "message": f"Focusing on {len(residue_ids)} residues",
        }]
    }


async def _clear_highlights() -> dict:
    return {
        "viewport_directives": [{
            "action": "clear",
            "message": "Cleared all highlights",
        }]
    }
