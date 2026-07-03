"""Derive artifact_availability and hydrate_meta for HydrateBundle responses."""

from __future__ import annotations

from typing import Any

from data.readiness import BINDING_SCAN_COMPLETE_STATUSES
from science.compute.registry import JOB_REGISTRY
from science.contracts.onboard_contract import get_artifact_catalog, load_contract

# Artifacts surfaced directly in HydrateBundle (field → canonical artifact keys).
_HYDRATE_FIELD_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "embeddings": ("gnn_hyp",),
    "graph_metrics": ("graph",),
    "source_leaks": ("source_leaks",),
    "allosteric_sites": ("allosteric_sites",),
    "binding_scan": ("binding_scan",),
    "phase2_vulnerability": ("strain_vulnerability",),
    "phase4_resistance": ("resistance_pathway",),
    "phase5_pharmacophore": ("pharmacophores", "pocket_pharmacophore"),
    "phase6_drug_candidates": ("drug_candidates",),
    "buffering_atlas": ("buffering_atlas",),
}

_SNAPSHOT_ARTIFACTS = ("dims", "scope")


def _is_present(payload: dict[str, Any], artifact_key: str) -> bool:
    if artifact_key == "dims":
        snap = payload.get("structure_snapshot")
        return bool(snap and snap.get("structure"))
    if artifact_key == "scope":
        snap = payload.get("structure_snapshot") or {}
        scope = snap.get("scope") or {}
        return bool(scope.get("primary_chain_ids"))
    if artifact_key == "gnn_hyp":
        residues = (payload.get("embeddings") or {}).get("residues") or []
        if residues:
            return True
        snap = payload.get("structure_snapshot") or {}
        return bool(snap.get("residues"))
    if artifact_key == "graph":
        return bool(payload.get("graph_metrics"))
    if artifact_key == "source_leaks":
        sl = payload.get("source_leaks")
        if not sl:
            return False
        leaks = sl.get("source_leaks") or sl.get("leaks") or []
        return len(leaks) > 0
    if artifact_key == "strain_vulnerability":
        return bool(payload.get("phase2_vulnerability"))
    if artifact_key == "resistance_pathway":
        p4 = payload.get("phase4_resistance")
        return bool(p4 and (p4.get("pathways") or p4.get("spectral")))
    if artifact_key == "pharmacophores":
        p5 = payload.get("phase5_pharmacophore")
        pockets = (p5 or {}).get("pharmacophores") or (p5 or {}).get("pockets") or []
        return len(pockets) > 0
    if artifact_key == "pocket_pharmacophore":
        return _is_present(payload, "pharmacophores")
    if artifact_key == "drug_candidates":
        p6 = payload.get("phase6_drug_candidates")
        candidates = (p6 or {}).get("candidates") or (p6 or {}).get("drug_candidates") or []
        return len(candidates) > 0
    if artifact_key == "allosteric_sites":
        sites = payload.get("allosteric_sites")
        return bool(sites and sites.get("sites"))
    if artifact_key == "binding_scan":
        binding_scan = payload.get("binding_scan")
        if not binding_scan:
            snap = payload.get("structure_snapshot") or {}
            binding_scan = (snap.get("findings") or {}).get("binding_scan")
        if not binding_scan:
            return False
        status = str(binding_scan.get("status") or "")
        if status not in BINDING_SCAN_COMPLETE_STATUSES:
            return False
        return bool(binding_scan.get("sites") or binding_scan.get("count", 0) > 0)
    if artifact_key == "buffering_atlas":
        return bool(payload.get("buffering_atlas"))
    return False


def _artifact_in_hydrate_bundle(artifact_key: str) -> bool:
    if artifact_key in _SNAPSHOT_ARTIFACTS:
        return True
    for keys in _HYDRATE_FIELD_ARTIFACTS.values():
        if artifact_key in keys:
            return True
    return False


def _default_reason(artifact_key: str, *, present: bool) -> str | None:
    if present:
        return None
    spec = get_artifact_catalog().get(artifact_key, {})
    producing = spec.get("producing_jobs") or []
    for job_id in producing:
        job = JOB_REGISTRY.get(job_id)
        if job and job.status == "planned":
            return "job_planned"
        if job and job.status == "partial":
            return "job_partial"
    if not _artifact_in_hydrate_bundle(artifact_key):
        return "not_in_hydrate_bundle"
    return "absent"


def build_artifact_availability(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return per-artifact availability keyed by canonical artifact id."""
    availability: dict[str, dict[str, Any]] = {}
    for artifact_key, spec in get_artifact_catalog().items():
        tier = spec.get("tier")
        present = _is_present(payload, artifact_key)
        availability[artifact_key] = {
            "present": present,
            "tier": tier if isinstance(tier, int) else None,
            "reason": _default_reason(artifact_key, present=present),
        }
    return availability


def build_hydrate_meta(
    payload: dict[str, Any],
    availability: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize bundle health for the frontend."""
    contract = load_contract()
    tier1_missing = [
        key
        for key, entry in availability.items()
        if entry.get("tier") == 1 and not entry.get("present")
    ]
    return {
        "contract_version": str(contract.get("version", "")),
        "degraded": len(tier1_missing) > 0,
        "missing_tier1_count": len(tier1_missing),
        "missing_keys": sorted(tier1_missing),
    }


def enrich_hydrate_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach artifact_availability and hydrate_meta to a hydrate payload."""
    availability = build_artifact_availability(payload)
    meta = build_hydrate_meta(payload, availability)
    return {
        **payload,
        "artifact_availability": availability,
        "hydrate_meta": meta,
    }
