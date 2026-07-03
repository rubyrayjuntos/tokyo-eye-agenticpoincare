"""Structure readiness probes and status derivation.

Implements the ingest–compute contract artifact checklist. Read-only DB access.
See: docs/specs/ingest-compute-contract/requirements.md
      docs/specs/discovery-story-pathway/requirements.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from data.act_readiness import (
    derive_act_readiness,
    derive_current_act,
    derive_foundation_status,
)
from science.compute.registry import DEFAULT_PATHWAY
from science.contracts.onboard_contract import (
    derive_geometric_readiness,
    get_artifact_aliases,
    get_artifact_labels,
    get_extended_artifact_keys,
    get_probe_id,
    get_tier_artifact_keys,
)
from science.contracts.geometric_runtime import load_structure_learned_curvature

ARTIFACT_ALIASES: dict[str, str] = get_artifact_aliases()

TIER1_ARTIFACTS = get_tier_artifact_keys(1)
TIER2_ARTIFACTS = get_tier_artifact_keys(2)
EXTENDED_ARTIFACTS = get_extended_artifact_keys()
ARTIFACT_LABELS: dict[str, str] = get_artifact_labels()

PROBE_ERROR_PREFIX = "Probe error:"


class ProbeInfrastructureError(RuntimeError):
    """Probe could not run (DB adapter mismatch, connection error, etc.)."""

    def __init__(self, artifact: str, cause: Exception) -> None:
        self.artifact = artifact
        self.cause = cause
        super().__init__(
            f"{PROBE_ERROR_PREFIX} {artifact} — {cause.__class__.__name__}: {cause}"
        )


class ReadinessDB(Protocol):
    async def fetch_one(self, query: str, params: dict[str, Any]) -> dict[str, Any] | None: ...


@dataclass
class StructureReadiness:
    structure_id: str
    readiness_status: str
    computation_run_id: str | None = None
    tier1: dict[str, bool] = field(default_factory=dict)
    tier2: dict[str, bool] = field(default_factory=dict)
    missing_artifacts: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    pipeline_job: dict[str, Any] | None = None
    pathway: str = DEFAULT_PATHWAY
    current_act: int = 0
    foundation: dict[str, bool] = field(default_factory=dict)
    acts: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, bool] = field(default_factory=dict)
    geometric_readiness: dict[str, Any] = field(default_factory=dict)
    probe_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "structure_id": self.structure_id,
            "readiness_status": self.readiness_status,
            "computation_run_id": self.computation_run_id,
            "pathway": self.pathway,
            "current_act": self.current_act,
            "foundation": self.foundation,
            "acts": self.acts,
            "artifacts": self.artifacts,
            "geometric_readiness": self.geometric_readiness,
            "tier1": self._tier_with_aliases(self.tier1),
            "tier2": self._tier_with_aliases(self.tier2),
            "missing_artifacts": self.missing_artifacts,
            "degraded_reasons": self.degraded_reasons,
            "probe_errors": self.probe_errors,
            "pipeline_job": self.pipeline_job,
        }
        return payload

    @staticmethod
    def _tier_with_aliases(tier: dict[str, bool]) -> dict[str, bool]:
        merged = dict(tier)
        for legacy, canonical in ARTIFACT_ALIASES.items():
            if canonical in tier:
                merged[canonical] = tier[canonical]
            if legacy in tier:
                merged[legacy] = tier[legacy]
                merged[canonical] = tier[legacy]
        return merged


BINDING_SCAN_COMPLETE_STATUSES = frozenset(
    {"complete", "no_sites_found", "no_coordinates"}
)


async def probe_dims(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT COUNT(*)::int AS count
        FROM dim_residue r
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
        """,
        {"structure_id": structure_id},
    )
    return bool(row and row.get("count", 0) > 0)


async def probe_scope(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM structure_computation_scope
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_gnn_hyp(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_gnn_node_embedding f
        JOIN embedding_space es ON es.space_id = f.space_id
        JOIN provenance_run p ON p.run_id = f.run_id
        WHERE f.structure_id = :structure_id
          AND es.space_type = 'hyperbolic'
          AND COALESCE(p.parameters->>'audit_only', 'false') != 'true'
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_graph(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_graph_node_metrics
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_dtie_core(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_source_leak
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_binding_scan(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT status
        FROM fact_binding_site_scan
        WHERE structure_id = :structure_id
        ORDER BY created_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    if row is None:
        return False
    return row.get("status") in BINDING_SCAN_COMPLETE_STATUSES


async def probe_alignment(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_residue_alignment a
        JOIN dim_residue r ON r.residue_id = a.residue_id
        JOIN dim_chain c ON c.chain_id = r.chain_id
        WHERE c.structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_dtie_phases(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_phase2_vulnerability
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_strain_vulnerability(structure_id: str, db: ReadinessDB) -> bool:
    return await probe_dtie_phases(structure_id, db)


async def probe_witness_embedding(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_phase_output
        WHERE structure_id = :structure_id
          AND (
            phase_name IN ('phase1_witness_embedding', 'witness_embedding')
            OR phase IN ('1', 'phase1')
          )
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_source_leaks(structure_id: str, db: ReadinessDB) -> bool:
    return await probe_dtie_core(structure_id, db)


async def probe_resistance_pathway(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_resistance_pathway
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_pharmacophores(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_pharmacophore
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_pocket_pharmacophore(structure_id: str, db: ReadinessDB) -> bool:
    return await probe_pharmacophores(structure_id, db)


async def probe_drug_candidates(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_drug_candidate
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_fragment_hits(structure_id: str, db: ReadinessDB) -> bool:
    """Fragment screen hits — planned until fragment_screen ships for real."""
    row = await db.fetch_one(
        """
        SELECT 1
        FROM provenance_run
        WHERE structure_id = :structure_id
          AND parameters->>'job_id' = 'fragment_screen'
          AND COALESCE(parameters->>'status', '') = 'complete'
          AND COALESCE((parameters->>'hits')::int, 0) > 0
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_allosteric_sites(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_allosteric_site
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_buffering_atlas(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_phase_output
        WHERE structure_id = :structure_id
          AND (
            phase_name = 'buffering_atlas'
            OR phase = 'buffering_atlas'
          )
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_allele_selectivity(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_drug_candidate
        WHERE structure_id = :structure_id
          AND selectivity_ratio IS NOT NULL
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_motifs(structure_id: str, db: ReadinessDB) -> bool:
    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_hyperbolic_motif
        WHERE structure_id = :structure_id
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


async def probe_md_validation(structure_id: str, db: ReadinessDB) -> bool:
    """True when scan ran with zero sites, or at least one site left pending-only."""
    scan_row = await db.fetch_one(
        """
        SELECT sites_found, status
        FROM fact_binding_site_scan
        WHERE structure_id = :structure_id
        ORDER BY created_at DESC
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    if scan_row and scan_row.get("sites_found", 0) == 0:
        if scan_row.get("status") in BINDING_SCAN_COMPLETE_STATUSES:
            return True

    row = await db.fetch_one(
        """
        SELECT 1
        FROM fact_cryptic_site
        WHERE structure_id = :structure_id
          AND md_validation_status IN ('passed', 'failed', 'timeout')
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    return row is not None


PROBE_BY_ID: dict[str, Any] = {
    "dim_residue_exists": probe_dims,
    "structure_computation_scope_exists": probe_scope,
    "gnn_hyperbolic_embedding_exists": probe_gnn_hyp,
    "graph_node_metrics_exists": probe_graph,
    "source_leak_exists": probe_source_leaks,
    "binding_site_scan_complete": probe_binding_scan,
    "residue_alignment_exists": probe_alignment,
    "phase2_vulnerability_exists": probe_dtie_phases,
    "hyperbolic_motif_exists": probe_motifs,
    "md_validation_complete": probe_md_validation,
    "strain_vulnerability_exists": probe_strain_vulnerability,
    "witness_phase_output_exists": probe_witness_embedding,
    "pharmacophore_exists": probe_pharmacophores,
    "drug_candidate_exists": probe_drug_candidates,
    "fragment_hits_exists": probe_fragment_hits,
    "resistance_pathway_exists": probe_resistance_pathway,
    "buffering_atlas_exists": probe_buffering_atlas,
    "allosteric_site_exists": probe_allosteric_sites,
    "allele_selectivity_exists": probe_allele_selectivity,
}


def _probes_for_artifacts(artifact_keys: tuple[str, ...]) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    for key in artifact_keys:
        probe_id = get_probe_id(key)
        if probe_id is None:
            continue
        fn = PROBE_BY_ID.get(probe_id)
        if fn is not None:
            probes[key] = fn
    return probes


TIER1_PROBES = _probes_for_artifacts(TIER1_ARTIFACTS)
TIER2_PROBES = _probes_for_artifacts(TIER2_ARTIFACTS)
EXTENDED_PROBES = _probes_for_artifacts(EXTENDED_ARTIFACTS)

ARTIFACT_PROBES: dict[str, Any] = {
    **TIER1_PROBES,
    **TIER2_PROBES,
    **EXTENDED_PROBES,
    "gnn_euc": probe_gnn_hyp,
}
for legacy, canonical in ARTIFACT_ALIASES.items():
    if canonical in ARTIFACT_PROBES:
        ARTIFACT_PROBES[legacy] = ARTIFACT_PROBES[canonical]


async def probe_artifact(db: ReadinessDB, structure_id: str, artifact: str) -> bool:
    """Return whether a registry artifact is present for a structure."""
    canonical = ARTIFACT_ALIASES.get(artifact, artifact)
    probe = ARTIFACT_PROBES.get(canonical)
    if probe is None:
        return True
    try:
        return bool(await probe(structure_id, db))
    except ProbeInfrastructureError:
        raise
    except Exception as exc:
        raise ProbeInfrastructureError(canonical, exc) from exc


async def _probe_tier(
    structure_id: str,
    db: ReadinessDB,
    probes: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, str]]:
    results: dict[str, bool] = {}
    probe_errors: dict[str, str] = {}
    for key, probe in probes.items():
        try:
            results[key] = bool(await probe(structure_id, db))
        except Exception as exc:
            results[key] = False
            probe_errors[key] = f"{exc.__class__.__name__}: {exc}"
    return results, probe_errors


async def fetch_latest_pipeline_job(
    structure_id: str,
    db: ReadinessDB,
) -> dict[str, Any] | None:
    row = await db.fetch_one(
        """
        SELECT job_id, structure_id, status, current_step, progress,
               modules, error, started_at, completed_at
        FROM pipeline_job
        WHERE structure_id = :structure_id
        ORDER BY
          CASE status
            WHEN 'running' THEN 0
            WHEN 'queued' THEN 1
            WHEN 'failed' THEN 2
            WHEN 'complete' THEN 3
            ELSE 4
          END,
          started_at DESC NULLS LAST
        LIMIT 1
        """,
        {"structure_id": structure_id},
    )
    if row is None:
        return None
    return {
        "job_id": str(row["job_id"]),
        "structure_id": str(row["structure_id"]),
        "status": row["status"],
        "current_step": row["current_step"],
        "progress": row["progress"],
        "modules": row.get("modules") or [],
        "error": row.get("error"),
        "started_at": str(row["started_at"]) if row.get("started_at") else None,
        "completed_at": str(row["completed_at"]) if row.get("completed_at") else None,
    }


_ACTIVE_PIPELINE_STATUSES = frozenset({"queued", "running"})


async def should_requeue_discovery_after_audit_ingest(
    structure_id: str,
    db: ReadinessDB,
) -> bool:
    """Return whether audit-only ingest should queue a new discovery pathway job.

    Re-queue when foundation (dims + scope) is present and hyperbolic compute is
    incomplete or the latest pipeline job failed/timed out. Skip when a job is
    already active.
    """
    structure_id = structure_id.strip().lower()

    if not await probe_artifact(db, structure_id, "dims"):
        return False
    if not await probe_artifact(db, structure_id, "scope"):
        return False

    job = await fetch_latest_pipeline_job(structure_id, db)
    if job and job.get("status") in _ACTIVE_PIPELINE_STATUSES:
        return False

    if not await probe_artifact(db, structure_id, "gnn_hyp"):
        return True

    if job is None:
        return True

    return job.get("status") in ("failed", "timed_out")


def derive_readiness_status(
    tier1: dict[str, bool],
    tier2: dict[str, bool],
    pipeline_job: dict[str, Any] | None,
) -> str:
    job_status = (pipeline_job or {}).get("status")
    if job_status in ("queued", "running"):
        return "running"

    tier1_complete = all(tier1.get(key, False) for key in TIER1_ARTIFACTS)
    if not tier1_complete:
        if job_status == "failed":
            return "failed"
        if job_status in ("complete", "timed_out"):
            return "degraded" if any(tier1.values()) else "failed"
        return "failed"

    tier2_complete = all(tier2.get(key, False) for key in TIER2_ARTIFACTS)
    if not tier2_complete:
        return "degraded"
    return "ready"


def _missing_artifacts(tier1: dict[str, bool], tier2: dict[str, bool]) -> list[str]:
    missing: list[str] = []
    for key in TIER1_ARTIFACTS:
        if not tier1.get(key, False):
            missing.append(key)
    for key in TIER2_ARTIFACTS:
        if not tier2.get(key, False):
            missing.append(key)
    return missing


def _degraded_reasons(
    tier1: dict[str, bool],
    tier2: dict[str, bool],
    pipeline_job: dict[str, Any] | None,
    probe_errors: dict[str, str] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if pipeline_job and pipeline_job.get("error"):
        reasons.append(f"Pipeline job error: {pipeline_job['error']}")
    for key, message in sorted((probe_errors or {}).items()):
        reasons.append(f"{PROBE_ERROR_PREFIX} {key} — {message}")
    for key, ok in tier1.items():
        if not ok and key not in (probe_errors or {}):
            reasons.append(f"Missing tier-1 artifact: {key} — {ARTIFACT_LABELS.get(key, key)}")
    for key, ok in tier2.items():
        if not ok and key not in (probe_errors or {}):
            reasons.append(f"Missing tier-2 artifact: {key} — {ARTIFACT_LABELS.get(key, key)}")
    return reasons


async def assess_structure_readiness(
    structure_id: str,
    db: ReadinessDB,
    *,
    pathway: str = DEFAULT_PATHWAY,
) -> StructureReadiness:
    """Run all probes and derive readiness for a structure."""
    structure_id = structure_id.strip().lower()

    exists = await db.fetch_one(
        "SELECT 1 FROM dim_structure WHERE structure_id = :structure_id",
        {"structure_id": structure_id},
    )
    if exists is None:
        return StructureReadiness(
            structure_id=structure_id,
            readiness_status="failed",
            pathway=pathway,
            missing_artifacts=list(TIER1_ARTIFACTS) + list(TIER2_ARTIFACTS),
            degraded_reasons=["Structure not found in dim_structure"],
        )

    tier1, tier1_errors = await _probe_tier(structure_id, db, TIER1_PROBES)
    tier2, tier2_errors = await _probe_tier(structure_id, db, TIER2_PROBES)
    extended, extended_errors = await _probe_tier(structure_id, db, EXTENDED_PROBES)
    probe_errors = {**tier1_errors, **tier2_errors, **extended_errors}
    pipeline_job = await fetch_latest_pipeline_job(structure_id, db)

    artifacts: dict[str, bool] = {}
    artifacts.update(tier1)
    artifacts.update(tier2)
    artifacts.update(extended)
    for legacy, canonical in ARTIFACT_ALIASES.items():
        if canonical in artifacts:
            artifacts[legacy] = artifacts[canonical]

    learned_curvature = await load_structure_learned_curvature(db, structure_id)
    geometric_readiness = derive_geometric_readiness(
        artifacts,
        learned_curvature=learned_curvature,
    )
    from shared.audit.instrumentation import audit_geometric_readiness

    await audit_geometric_readiness(
        db,
        structure_id=structure_id,
        geometric_readiness=geometric_readiness,
    )

    foundation = derive_foundation_status(artifacts)
    act_models = derive_act_readiness(
        artifacts,
        foundation=foundation,
        pipeline_job=pipeline_job,
    )
    acts = {act_id: act.to_dict() for act_id, act in act_models.items()}
    current_act = derive_current_act(act_models)

    status = derive_readiness_status(tier1, tier2, pipeline_job)
    missing = _missing_artifacts(tier1, tier2)
    for key in probe_errors:
        if key in missing:
            missing.remove(key)
    reasons = _degraded_reasons(tier1, tier2, pipeline_job, probe_errors)

    return StructureReadiness(
        structure_id=structure_id,
        readiness_status=status,
        computation_run_id=(pipeline_job or {}).get("job_id"),
        tier1=tier1,
        tier2=tier2,
        missing_artifacts=missing,
        degraded_reasons=reasons if status in ("degraded", "failed") or probe_errors else [],
        pipeline_job=pipeline_job,
        pathway=pathway,
        current_act=current_act,
        foundation=foundation,
        acts=acts,
        artifacts=artifacts,
        geometric_readiness=geometric_readiness,
        probe_errors=probe_errors,
    )
