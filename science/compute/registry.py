"""Normative compute job catalog for Discovery Story pathways.

Canonical source of truth for job metadata. Mirror in:
  docs/specs/discovery-story-pathway/design.md §2.2

See: docs/specs/discovery-story-pathway/design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ResourceClass = Literal["cpu_light", "cpu_heavy", "gpu"]
JobStatus = Literal["implemented", "partial", "planned"]
DiscoveryAct = Literal[
    "foundation",
    "sidecar",
    "signal",
    "persistent_leak",
    "cryptic_pocket",
    "fragment",
    "verdict",
]

DEFAULT_PATHWAY = "discovery_story"


@dataclass(frozen=True)
class ComputeJob:
    job_id: str
    discovery_act: DiscoveryAct
    resource_class: ResourceClass
    priority_group: int
    requires: frozenset[str]
    produces: frozenset[str]
    tier: int
    status: JobStatus
    legacy_alias: str | None = None
    runner: str | None = None


def _job(
    job_id: str,
    discovery_act: DiscoveryAct,
    resource_class: ResourceClass,
    priority_group: int,
    requires: tuple[str, ...],
    produces: tuple[str, ...],
    tier: int,
    status: JobStatus,
    legacy_alias: str | None = None,
    runner: str | None = None,
) -> ComputeJob:
    return ComputeJob(
        job_id=job_id,
        discovery_act=discovery_act,
        resource_class=resource_class,
        priority_group=priority_group,
        requires=frozenset(requires),
        produces=frozenset(produces),
        tier=tier,
        status=status,
        legacy_alias=legacy_alias,
        runner=runner,
    )


JOB_REGISTRY: dict[str, ComputeJob] = {
    j.job_id: j
    for j in (
        _job(
            "ingest_dims",
            "foundation",
            "cpu_light",
            0,
            (),
            ("dims",),
            1,
            "implemented",
            runner="science/api/routers/ingest.py",
        ),
        _job(
            "assign_computation_scope",
            "foundation",
            "cpu_light",
            0,
            ("ingest_dims",),
            ("scope",),
            1,
            "implemented",
        ),
        _job(
            "alignment_sidecar",
            "sidecar",
            "cpu_light",
            0,
            ("ingest_dims",),
            ("alignment",),
            2,
            "implemented",
        ),
        _job(
            "gnn_inference",
            "foundation",
            "gpu",
            0,
            ("assign_computation_scope",),
            ("gnn_hyp", "gnn_euc"),
            1,
            "implemented",
            legacy_alias="gnn_inference_v6",
            runner="science/compute/runners/gnn_inference.py",
        ),
        _job(
            "graph_topology",
            "signal",
            "cpu_heavy",
            1,
            ("gnn_inference",),
            ("graph",),
            1,
            "implemented",
            runner="science/compute/runners/graph_topology.py",
        ),
        _job(
            "witness_embedding",
            "signal",
            "cpu_heavy",
            1,
            ("gnn_inference",),
            ("witness_embedding",),
            1,
            "implemented",
            legacy_alias="phase1_witness_embedding",
            runner="science/compute/runners/witness_embedding.py",
        ),
        _job(
            "strain_vulnerability_scan",
            "signal",
            "cpu_heavy",
            1,
            ("gnn_inference",),
            ("strain_vulnerability",),
            1,
            "implemented",
            legacy_alias="phase2_vulnerability_scan",
            runner="science/compute/runners/strain_vulnerability_scan.py",
        ),
        _job(
            "source_leak_detection",
            "persistent_leak",
            "cpu_light",
            2,
            ("gnn_inference",),
            ("source_leaks",),
            1,
            "implemented",
            legacy_alias="dtie_core",
            runner="science/compute/runners/source_leak_detection.py",
        ),
        _job(
            "hyperbolic_motifs",
            "persistent_leak",
            "cpu_heavy",
            2,
            ("gnn_inference",),
            ("motifs",),
            2,
            "implemented",
            runner="science/compute/runners/hyperbolic_motifs.py",
        ),
        _job(
            "binding_site_scan",
            "cryptic_pocket",
            "cpu_heavy",
            3,
            ("gnn_inference", "graph_topology"),
            ("binding_scan",),
            1,
            "implemented",
            legacy_alias="Scan_Phase",
            runner="agent/tools/cryptic/scan_phase.run_full_structure_scan",
        ),
        _job(
            "pocket_pharmacophore_map",
            "cryptic_pocket",
            "cpu_light",
            3,
            ("binding_site_scan",),
            ("pocket_pharmacophore",),
            2,
            "partial",
            legacy_alias="phase5 subset",
        ),
        _job(
            "md_validate_top_n",
            "cryptic_pocket",
            "gpu",
            3,
            ("binding_site_scan",),
            ("md_validation",),
            2,
            "partial",
            runner="science/dtie/cryptic/smd_runner",
        ),
        _job(
            "pharmacophore_identification",
            "fragment",
            "cpu_heavy",
            4,
            ("binding_site_scan",),
            ("pharmacophores",),
            2,
            "implemented",
            legacy_alias="phase5_pharmacophore",
            runner="science/dtie/v5/phases/phase5_pharmacophore",
        ),
        _job(
            "fragment_screen",
            "fragment",
            "gpu",
            4,
            ("pharmacophore_identification",),
            ("fragment_hits",),
            2,
            "planned",
        ),
        _job(
            "drug_candidate_ranking",
            "fragment",
            "cpu_heavy",
            4,
            ("pharmacophore_identification",),
            ("drug_candidates",),
            2,
            "partial",
            legacy_alias="phase6a-6d",
            runner="science/dtie/v5/phases/phase6_drug_discovery",
        ),
        _job(
            "topological_lift",
            "verdict",
            "cpu_heavy",
            5,
            ("gnn_inference",),
            ("buffering_atlas",),
            2,
            "implemented",
            legacy_alias="phase35_topological_lift",
            runner="science/dtie/v5/phases/phase35_topological_lift",
        ),
        _job(
            "resistance_pathway_map",
            "verdict",
            "cpu_heavy",
            5,
            ("gnn_inference", "graph_topology"),
            ("resistance_pathway",),
            2,
            "implemented",
            legacy_alias="phase4_resistance_mapping",
            runner="science/dtie/v5/phases/phase4_resistance",
        ),
        _job(
            "allosteric_site_detection",
            "verdict",
            "cpu_light",
            5,
            ("source_leak_detection", "resistance_pathway_map"),
            ("allosteric_sites",),
            2,
            "implemented",
        ),
        _job(
            "allele_selectivity_assessment",
            "verdict",
            "cpu_light",
            5,
            ("drug_candidate_ranking", "resistance_pathway_map"),
            ("allele_selectivity",),
            2,
            "partial",
            legacy_alias="phase6d / ASAR",
        ),
    )
}


ACT_JOB_MAP: dict[str, tuple[str, ...]] = {
    "signal": (
        "graph_topology",
        "witness_embedding",
        "strain_vulnerability_scan",
    ),
    "persistent_leak": (
        "source_leak_detection",
        "hyperbolic_motifs",
    ),
    "cryptic_pocket": (
        "binding_site_scan",
        "pocket_pharmacophore_map",
        "md_validate_top_n",
    ),
    "fragment": (
        "pharmacophore_identification",
        "fragment_screen",
        "drug_candidate_ranking",
    ),
    "verdict": (
        "topological_lift",
        "resistance_pathway_map",
        "allosteric_site_detection",
        "allele_selectivity_assessment",
    ),
}

FOUNDATION_JOB_IDS: frozenset[str] = frozenset(
    job_id
    for job_id, job in JOB_REGISTRY.items()
    if job.discovery_act in ("foundation", "sidecar")
)

PATHWAY_JOBS: dict[str, frozenset[str]] = {
    DEFAULT_PATHWAY: frozenset(JOB_REGISTRY.keys()),
    "viewport_explore": FOUNDATION_JOB_IDS
    | frozenset(ACT_JOB_MAP["signal"])
    | frozenset(ACT_JOB_MAP["persistent_leak"]),
}


def topological_order(
    job_ids: frozenset[str] | None = None,
) -> list[str]:
    """Return job IDs in dependency order (requires before dependents)."""
    ids = job_ids if job_ids is not None else frozenset(JOB_REGISTRY.keys())
    unknown = ids - set(JOB_REGISTRY.keys())
    if unknown:
        raise ValueError(f"Unknown job IDs: {sorted(unknown)}")

    completed: set[str] = set()
    ordered: list[str] = []

    while len(completed) < len(ids):
        progress = False
        for job_id in sorted(ids - completed):
            job = JOB_REGISTRY[job_id]
            if job.requires <= completed:
                ordered.append(job_id)
                completed.add(job_id)
                progress = True
        if not progress:
            remaining = ids - completed
            raise ValueError(f"Unsatisfiable or cyclic job dependencies: {sorted(remaining)}")
    return ordered


def validate_registry() -> None:
    """Raise if registry invariants are violated."""
    for job_id, job in JOB_REGISTRY.items():
        if job_id != job.job_id:
            raise ValueError(f"Registry key mismatch: {job_id} != {job.job_id}")
        unknown = job.requires - set(JOB_REGISTRY.keys())
        if unknown:
            raise ValueError(f"Job {job_id} requires unknown jobs: {sorted(unknown)}")

    for act_id, act_jobs in ACT_JOB_MAP.items():
        unknown = set(act_jobs) - set(JOB_REGISTRY.keys())
        if unknown:
            raise ValueError(f"Act {act_id} references unknown jobs: {sorted(unknown)}")

    for pathway_id, pathway_job_ids in PATHWAY_JOBS.items():
        unknown = pathway_job_ids - set(JOB_REGISTRY.keys())
        if unknown:
            raise ValueError(f"Pathway {pathway_id} references unknown jobs: {sorted(unknown)}")

    topological_order()


validate_registry()
