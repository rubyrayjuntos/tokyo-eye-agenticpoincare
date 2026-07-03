"""Compute job catalog schema — field metadata, execution, and destinations.

Canonical JSON: ``science/compute/job_schema.json``
Python loader validates registry sync at import/test time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from science.compute.registry import ACT_JOB_MAP, JOB_REGISTRY, PATHWAY_JOBS
from science.contracts.onboard_contract import validate_normalizer_destinations

SCHEMA_PATH = Path(__file__).with_name("job_schema.json")

# Shared destination field templates reused across jobs.
_GNN_EMBEDDING_FIELDS = [
    {
        "name": "residue_id",
        "type": "string",
        "description": "Canonical residue identifier (chain:index:name)",
        "nullable": False,
    },
    {
        "name": "run_id",
        "type": "string",
        "description": "Provenance run for this embedding write",
        "nullable": False,
    },
    {
        "name": "space_id",
        "type": "uuid",
        "description": "FK to embedding_space",
        "nullable": False,
    },
    {
        "name": "coordinates",
        "type": "jsonb",
        "description": "Hyperbolic (Poincaré) or Euclidean coordinates per residue",
        "nullable": False,
    },
    {
        "name": "uncertainty",
        "type": "float",
        "description": "Model uncertainty score",
        "nullable": True,
    },
    {
        "name": "cone_depth",
        "type": "float",
        "description": "Radial cone depth in embedding space",
        "nullable": True,
    },
]

_GRAPH_METRIC_FIELDS = [
    {"name": "residue_id", "type": "string", "description": "Node residue id", "nullable": False},
    {"name": "degree", "type": "integer", "description": "Graph degree", "nullable": False},
    {"name": "betweenness", "type": "float", "description": "Betweenness centrality", "nullable": False},
    {
        "name": "clustering_coefficient",
        "type": "float",
        "description": "Local clustering coefficient",
        "nullable": False,
    },
    {"name": "closeness", "type": "float", "description": "Closeness centrality", "nullable": False},
    {
        "name": "eigenvector_centrality",
        "type": "float",
        "description": "Eigenvector centrality",
        "nullable": False,
    },
    {"name": "is_bridge", "type": "boolean", "description": "Articulation point flag", "nullable": False},
    {"name": "conductance", "type": "float", "description": "Fiedler-based conductance proxy", "nullable": False},
]


def _execution(
    job_id: str,
    *,
    runner: str | None = None,
    job_module: str | None = None,
    aliases: list[str] | None = None,
    external: bool = False,
) -> dict[str, Any]:
    job = JOB_REGISTRY[job_id]
    triggers = ["scheduler", f"POST /compute/jobs/{job_id}"]
    if aliases:
        triggers.extend(aliases)
    if external:
        triggers = ["POST /compute/ingest-full"]
    return {
        "triggers": triggers,
        "runner": runner or job.runner,
        "job_module": job_module,
        "resource_class": job.resource_class,
        "discovery_act": job.discovery_act,
        "tier": job.tier,
        "status": job.status,
        "priority_group": job.priority_group,
        "legacy_alias": job.legacy_alias,
    }


def _dest(
    table: str,
    *,
    normalizer_path: str,
    write_mode: str = "upsert",
    key_columns: list[str] | None = None,
    fields: list[dict[str, Any]] | None = None,
    bypass: bool = False,
) -> dict[str, Any]:
    dest = {
        "table": table,
        "normalizer_path": normalizer_path,
        "write_mode": write_mode,
        "key_columns": key_columns or [],
        "fields": fields or [],
    }
    if bypass:
        dest["bypass"] = True
    return dest


def build_job_catalog() -> dict[str, Any]:
    """Build the full job catalog document."""
    jobs: dict[str, Any] = {
        "ingest_dims": {
            "job_id": "ingest_dims",
            "requires": [],
            "produces": ["dims"],
            "preconditions": [
                {"check": "pdb_download", "description": "Valid PDB ID and RCSB availability"}
            ],
            "execution": _execution("ingest_dims", external=True),
            "output_fields": [
                {"name": "structure_id", "type": "string", "role": "response"},
                {"name": "chain_count", "type": "integer", "role": "response"},
                {"name": "residue_count", "type": "integer", "role": "response"},
                {"name": "atom_count", "type": "integer", "role": "response"},
            ],
            "destinations": [
                _dest("dim_structure", normalizer_path="normalize_ingest_dimensions"),
                _dest("dim_chain", normalizer_path="normalize_ingest_dimensions"),
                _dest("dim_residue", normalizer_path="normalize_ingest_dimensions"),
                _dest("dim_atom", normalizer_path="normalize_ingest_dimensions"),
                _dest("fact_covalent_bond", normalizer_path="normalize_ingest_dimensions"),
            ],
        },
        "assign_computation_scope": {
            "job_id": "assign_computation_scope",
            "requires": ["ingest_dims"],
            "produces": ["scope"],
            "preconditions": [{"check": "artifact:dims", "description": "dim_residue populated"}],
            "execution": _execution("assign_computation_scope", external=True),
            "output_fields": [
                {"name": "computation_scope", "type": "object", "role": "response"},
            ],
            "destinations": [
                _dest(
                    "structure_computation_scope",
                    normalizer_path="normalize_computation_scope",
                    key_columns=["structure_id"],
                ),
            ],
        },
        "alignment_sidecar": {
            "job_id": "alignment_sidecar",
            "requires": ["ingest_dims"],
            "produces": ["alignment"],
            "preconditions": [{"check": "artifact:dims", "description": "Chains present in dim_chain"}],
            "execution": _execution("alignment_sidecar", external=True),
            "output_fields": [
                {"name": "alignment_status", "type": "string", "role": "response"},
            ],
            "destinations": [
                _dest("fact_residue_alignment", normalizer_path="normalize_alignment"),
                _dest("fact_structural_alignment", normalizer_path="normalize_alignment"),
            ],
        },
        "gnn_inference": {
            "job_id": "gnn_inference",
            "requires": ["assign_computation_scope"],
            "produces": ["gnn_hyp", "gnn_euc"],
            "preconditions": [
                {"check": "artifact:dims", "description": "Residue coordinates in dim_residue"},
                {"check": "artifact:scope", "description": "structure_computation_scope assigned"},
                {"check": "checkpoint", "description": "V6 checkpoint file present on disk"},
            ],
            "execution": _execution(
                "gnn_inference",
                job_module="science/compute/jobs/gnn_inference.py",
                aliases=["POST /compute/gnn"],
            ),
            "output_fields": [
                {"name": "node_count", "type": "integer", "role": "job_output"},
                {"name": "embedding_run_id", "type": "string", "role": "provenance"},
                {"name": "gnn_run_id", "type": "string", "role": "provenance"},
                {"name": "curvature", "type": "float", "role": "job_output"},
                {"name": "hyperbolic_distances", "type": "object", "role": "job_output", "nullable": True},
            ],
            "destinations": [
                _dest(
                    "fact_gnn_node_embedding",
                    normalizer_path="normalize_gnn_output",
                    key_columns=["run_id", "residue_id", "space_id"],
                    fields=_GNN_EMBEDDING_FIELDS,
                ),
                _dest("embedding_space", normalizer_path="normalize_gnn_output"),
                _dest("provenance_run", normalizer_path="_ensure_provenance_run"),
                _dest(
                    "fact_hyperbolic_distance",
                    normalizer_path="hyperbolic_distance_populator",
                    key_columns=["residue_id_a", "residue_id_b", "run_id"],
                    bypass=True,
                ),
            ],
        },
        "graph_topology": {
            "job_id": "graph_topology",
            "requires": ["gnn_inference"],
            "produces": ["graph"],
            "preconditions": [{"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"}],
            "execution": _execution(
                "graph_topology",
                job_module="science/compute/jobs/graph_topology.py",
                aliases=["POST /compute/graph-topology"],
            ),
            "job_params": [
                {"name": "contact_cutoff_angstrom", "type": "float", "default": 8.0},
                {"name": "chain_filter", "type": "string", "nullable": True},
            ],
            "output_fields": [
                {"name": "node_count", "type": "integer", "role": "job_output"},
                {"name": "edge_count", "type": "integer", "role": "job_output"},
                {"name": "bridge_count", "type": "integer", "role": "job_output"},
                {"name": "metrics_computed", "type": "array", "role": "job_output"},
                {"name": "run_id", "type": "string", "role": "provenance"},
            ],
            "destinations": [
                _dest(
                    "fact_graph_edge",
                    normalizer_path="normalize_graph_topology",
                    key_columns=["run_id", "source_residue_id", "target_residue_id", "edge_type"],
                ),
                _dest(
                    "fact_graph_node_metrics",
                    normalizer_path="normalize_graph_topology",
                    key_columns=["run_id", "residue_id"],
                    fields=_GRAPH_METRIC_FIELDS,
                ),
            ],
        },
        "witness_embedding": {
            "job_id": "witness_embedding",
            "requires": ["gnn_inference"],
            "produces": ["witness_embedding"],
            "preconditions": [{"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"}],
            "execution": _execution(
                "witness_embedding",
                job_module="science/compute/jobs/witness_embedding.py",
            ),
            "output_fields": [
                {"name": "n_witnesses", "type": "integer", "role": "job_output"},
                {"name": "n_landmarks", "type": "integer", "role": "job_output"},
            ],
            "destinations": [
                _dest("fact_phase_output", normalizer_path="normalize_phase_output", key_columns=["run_id"]),
            ],
        },
        "strain_vulnerability_scan": {
            "job_id": "strain_vulnerability_scan",
            "requires": ["gnn_inference"],
            "produces": ["strain_vulnerability"],
            "preconditions": [{"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"}],
            "execution": _execution(
                "strain_vulnerability_scan",
                job_module="science/compute/jobs/strain_vulnerability_scan.py",
            ),
            "output_fields": [{"name": "doorway_count", "type": "integer", "role": "job_output"}],
            "destinations": [
                _dest("fact_phase2_vulnerability", normalizer_path="normalize_phase2_vulnerability"),
            ],
        },
        "source_leak_detection": {
            "job_id": "source_leak_detection",
            "requires": ["gnn_inference"],
            "produces": ["source_leaks"],
            "preconditions": [{"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"}],
            "execution": _execution(
                "source_leak_detection",
                job_module="science/compute/jobs/source_leak_detection.py",
            ),
            "output_fields": [
                {"name": "source_leak_count", "type": "integer", "role": "job_output"},
                {"name": "source_leak_residues", "type": "array", "role": "job_output"},
            ],
            "destinations": [
                _dest("fact_source_leak", normalizer_path="normalize_source_leaks"),
            ],
        },
        "hyperbolic_motifs": {
            "job_id": "hyperbolic_motifs",
            "requires": ["gnn_inference"],
            "produces": ["motifs"],
            "preconditions": [{"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"}],
            "execution": _execution(
                "hyperbolic_motifs",
                job_module="science/compute/jobs/hyperbolic_motifs.py",
                aliases=["POST /compute/motif-analysis"],
            ),
            "job_params": [
                {"name": "min_cluster_size", "type": "integer", "default": 5},
                {"name": "min_samples", "type": "integer", "default": 3},
            ],
            "output_fields": [
                {"name": "motif_count", "type": "integer", "role": "job_output"},
                {"name": "motifs", "type": "array", "role": "job_output"},
            ],
            "destinations": [
                _dest("fact_hyperbolic_motif", normalizer_path="normalize_hyperbolic_motifs"),
            ],
        },
        "binding_site_scan": {
            "job_id": "binding_site_scan",
            "requires": ["gnn_inference", "graph_topology"],
            "produces": ["binding_scan"],
            "preconditions": [
                {"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"},
                {"check": "artifact:graph", "description": "Graph node metrics exist"},
            ],
            "execution": _execution(
                "binding_site_scan",
                job_module="science/compute/jobs/binding_site_scan.py",
                aliases=["POST /compute/cryptic-scan"],
            ),
            "job_params": [
                {"name": "cluster_distance_angstrom", "type": "float", "default": 8.0},
                {"name": "min_cluster_size", "type": "integer", "default": 3},
                {"name": "max_pockets", "type": "integer", "default": 25},
            ],
            "output_fields": [
                {"name": "sites_found", "type": "integer", "role": "job_output"},
                {"name": "scan_run_id", "type": "string", "role": "provenance"},
                {"name": "scan_status", "type": "string", "role": "job_output"},
            ],
            "destinations": [
                _dest("fact_cryptic_site", normalizer_path="normalize_binding_site_scan"),
                _dest("fact_binding_site_scan", normalizer_path="normalize_binding_site_scan"),
            ],
        },
        "pocket_pharmacophore_map": {
            "job_id": "pocket_pharmacophore_map",
            "requires": ["binding_site_scan"],
            "produces": ["pocket_pharmacophore"],
            "preconditions": [{"check": "artifact:binding_scan", "description": "Binding scan complete"}],
            "execution": _execution(
                "pocket_pharmacophore_map",
                job_module="science/compute/jobs/pocket_pharmacophore_map.py",
            ),
            "output_fields": [{"name": "pharmacophore_count", "type": "integer", "role": "job_output"}],
            "destinations": [_dest("fact_pharmacophore", normalizer_path="normalize_pharmacophores")],
        },
        "md_validate_top_n": {
            "job_id": "md_validate_top_n",
            "requires": ["binding_site_scan"],
            "produces": ["md_validation"],
            "preconditions": [{"check": "artifact:binding_scan", "description": "Cryptic sites persisted"}],
            "execution": _execution(
                "md_validate_top_n",
                job_module="science/compute/cryptic/md_validate.py",
                aliases=["POST /compute/md-validate"],
            ),
            "job_params": [
                {"name": "top_n", "type": "integer", "default": 5},
                {"name": "site_id", "type": "string", "nullable": True},
                {"name": "dry_run", "type": "boolean", "default": False},
            ],
            "output_fields": [
                {"name": "validated", "type": "integer", "role": "job_output"},
                {"name": "passed", "type": "integer", "role": "job_output"},
                {"name": "results", "type": "array", "role": "job_output"},
            ],
            "destinations": [
                _dest(
                    "fact_cryptic_site",
                    normalizer_path="md_status_update",
                    key_columns=["site_id"],
                    bypass=True,
                ),
            ],
        },
        "pharmacophore_identification": {
            "job_id": "pharmacophore_identification",
            "requires": ["binding_site_scan"],
            "produces": ["pharmacophores"],
            "preconditions": [{"check": "artifact:binding_scan", "description": "Binding scan complete"}],
            "execution": _execution(
                "pharmacophore_identification",
                job_module="science/compute/jobs/pharmacophore_identification.py",
            ),
            "output_fields": [{"name": "pharmacophore_count", "type": "integer", "role": "job_output"}],
            "destinations": [_dest("fact_pharmacophore", normalizer_path="normalize_pharmacophores")],
        },
        "fragment_screen": {
            "job_id": "fragment_screen",
            "requires": ["pharmacophore_identification"],
            "produces": ["fragment_hits"],
            "preconditions": [{"check": "artifact:pharmacophores", "description": "Pharmacophores identified"}],
            "execution": _execution("fragment_screen", job_module="science/compute/runners/fragment_screen.py"),
            "output_fields": [{"name": "status", "type": "string", "role": "job_output"}],
            "destinations": [_dest("provenance_run", normalizer_path="_ensure_provenance_run")],
        },
        "drug_candidate_ranking": {
            "job_id": "drug_candidate_ranking",
            "requires": ["pharmacophore_identification"],
            "produces": ["drug_candidates"],
            "preconditions": [{"check": "artifact:pharmacophores", "description": "Pharmacophores identified"}],
            "execution": _execution(
                "drug_candidate_ranking",
                job_module="science/compute/jobs/drug_candidate_ranking.py",
            ),
            "output_fields": [{"name": "candidate_count", "type": "integer", "role": "job_output"}],
            "destinations": [_dest("fact_drug_candidate", normalizer_path="normalize_drug_candidates")],
        },
        "topological_lift": {
            "job_id": "topological_lift",
            "requires": ["gnn_inference"],
            "produces": ["buffering_atlas"],
            "preconditions": [{"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"}],
            "execution": _execution(
                "topological_lift",
                job_module="science/compute/jobs/topological_lift.py",
            ),
            "output_fields": [{"name": "lifted_sites", "type": "integer", "role": "job_output"}],
            "destinations": [
                _dest("fact_topological_lift", normalizer_path="normalize_topological_lift"),
                _dest("fact_phase_output", normalizer_path="buffering_atlas", bypass=True),
            ],
        },
        "resistance_pathway_map": {
            "job_id": "resistance_pathway_map",
            "requires": ["gnn_inference", "graph_topology"],
            "produces": ["resistance_pathway"],
            "preconditions": [
                {"check": "artifact:gnn_hyp", "description": "Hyperbolic embeddings exist"},
                {"check": "artifact:graph", "description": "Graph metrics exist"},
            ],
            "execution": _execution(
                "resistance_pathway_map",
                job_module="science/compute/jobs/resistance_pathway_map.py",
            ),
            "output_fields": [{"name": "pathway_count", "type": "integer", "role": "job_output"}],
            "destinations": [
                _dest("fact_resistance_pathway", normalizer_path="normalize_resistance_pathways"),
                _dest("fact_resistance_spectral", normalizer_path="normalize_resistance_pathways"),
            ],
        },
        "allosteric_site_detection": {
            "job_id": "allosteric_site_detection",
            "requires": ["source_leak_detection", "resistance_pathway_map"],
            "produces": ["allosteric_sites"],
            "preconditions": [
                {"check": "artifact:source_leaks", "description": "Source leaks detected"},
                {"check": "artifact:resistance_pathway", "description": "Resistance pathways mapped"},
            ],
            "execution": _execution(
                "allosteric_site_detection",
                job_module="science/compute/jobs/allosteric_site_detection.py",
            ),
            "output_fields": [{"name": "site_count", "type": "integer", "role": "job_output"}],
            "destinations": [
                _dest("fact_allosteric_site", normalizer_path="normalize_allosteric_sites"),
                _dest("fact_allosteric_site_residue", normalizer_path="normalize_allosteric_sites"),
            ],
        },
        "allele_selectivity_assessment": {
            "job_id": "allele_selectivity_assessment",
            "requires": ["drug_candidate_ranking", "resistance_pathway_map"],
            "produces": ["allele_selectivity"],
            "preconditions": [
                {"check": "artifact:drug_candidates", "description": "Drug candidates ranked"},
                {"check": "artifact:resistance_pathway", "description": "Resistance pathways mapped"},
            ],
            "execution": _execution("allele_selectivity_assessment"),
            "output_fields": [{"name": "selectivity_ratio", "type": "float", "role": "job_output"}],
            "destinations": [
                _dest(
                    "fact_drug_candidate",
                    normalizer_path="phase6d_selectivity",
                    key_columns=["candidate_id"],
                    bypass=True,
                ),
            ],
        },
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "tokyo-eye/compute-job-catalog",
        "title": "Tokyo Eye Compute Job Catalog",
        "version": "1.0.0",
        "description": (
            "Field-level metadata for all discovery pathway jobs: preconditions, "
            "execution triggers, job outputs, and governed write destinations."
        ),
        "canonical_dispatch": "POST /compute/jobs/{job_id}",
        "onboard_trigger": "POST /api/ingest → run_onboard_compute → execute_pathway",
        "pathways": {
            pid: sorted(jids) for pid, jids in PATHWAY_JOBS.items()
        },
        "acts": {act: list(jobs) for act, jobs in ACT_JOB_MAP.items()},
        "jobs": jobs,
    }


def load_job_schema() -> dict[str, Any]:
    if SCHEMA_PATH.is_file():
        with SCHEMA_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    catalog = build_job_catalog()
    write_job_schema(catalog)
    return catalog


def write_job_schema(catalog: dict[str, Any] | None = None) -> None:
    catalog = catalog or build_job_catalog()
    SCHEMA_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


def validate_schema_against_registry() -> list[str]:
    """Return validation errors when schema and JOB_REGISTRY diverge."""
    schema = load_job_schema()
    errors: list[str] = []
    schema_jobs = schema.get("jobs", {})
    reg_ids = set(JOB_REGISTRY.keys())
    schema_ids = set(schema_jobs.keys())
    if reg_ids != schema_ids:
        errors.append(f"job id mismatch registry-only={sorted(reg_ids - schema_ids)} schema-only={sorted(schema_ids - reg_ids)}")
    for job_id, job in JOB_REGISTRY.items():
        entry = schema_jobs.get(job_id, {})
        if set(entry.get("requires", [])) != set(job.requires):
            errors.append(f"{job_id}: requires mismatch")
        if set(entry.get("produces", [])) != set(job.produces):
            errors.append(f"{job_id}: produces mismatch")
    errors.extend(validate_normalizer_destinations(schema))
    return errors
