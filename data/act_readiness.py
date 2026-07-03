"""Discovery Story act-scoped readiness derivation.

See: docs/specs/discovery-story-pathway/requirements.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from science.compute.registry import ACT_JOB_MAP, JOB_REGISTRY
from science.contracts.onboard_contract import (
    get_act_definitions,
    get_act_optional_artifacts,
    get_act_order,
    get_act_required_artifacts,
    get_artifact_probe_keys,
    get_foundation_artifacts,
    get_foundation_optional,
)

ACT_ORDER: tuple[str, ...] = get_act_order()
FOUNDATION_ARTIFACTS: tuple[str, ...] = get_foundation_artifacts()
FOUNDATION_OPTIONAL: tuple[str, ...] = get_foundation_optional()
ACT_DEFINITIONS: dict[str, dict[str, Any]] = get_act_definitions()
ACT_REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = get_act_required_artifacts()
ACT_OPTIONAL_ARTIFACTS: dict[str, tuple[str, ...]] = get_act_optional_artifacts()

# Canonical artifact key -> key in merged artifact probe results (contract-derived).
ARTIFACT_PROBE_KEYS: dict[str, str] = get_artifact_probe_keys()

PLANNED_ACTS: frozenset[str] = frozenset(
    act_id
    for act_id, job_ids in ACT_JOB_MAP.items()
    if all(JOB_REGISTRY[j].status == "planned" for j in job_ids)
)


@dataclass
class ActStatus:
    act_id: str
    number: int
    title: str
    question: str
    color: str
    status: str
    jobs_complete: int
    jobs_total: int
    required_artifacts: dict[str, bool]
    optional_artifacts: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "act_id": self.act_id,
            "number": self.number,
            "title": self.title,
            "question": self.question,
            "color": self.color,
            "status": self.status,
            "jobs_complete": self.jobs_complete,
            "jobs_total": self.jobs_total,
            "required_artifacts": self.required_artifacts,
            "optional_artifacts": self.optional_artifacts,
        }


def _artifact_value(artifacts: dict[str, bool], key: str) -> bool:
    probe_key = ARTIFACT_PROBE_KEYS.get(key, key)
    if artifacts.get(probe_key, False):
        return True
    if key == "pocket_pharmacophore":
        return bool(artifacts.get("pharmacophores", False))
    return False


def _prior_acts_complete(act_id: str, act_statuses: dict[str, str]) -> bool:
    idx = ACT_ORDER.index(act_id)
    if idx == 0:
        return True
    for prior in ACT_ORDER[:idx]:
        if prior in PLANNED_ACTS:
            continue
        if act_statuses.get(prior) not in ("complete", "degraded", "not_implemented"):
            return False
    return True


def _count_jobs_complete(act_id: str, artifacts: dict[str, bool]) -> int:
    complete = 0
    for job_id in ACT_JOB_MAP.get(act_id, ()):
        job = JOB_REGISTRY[job_id]
        if job.status == "planned":
            continue
        if job.produces and all(_artifact_value(artifacts, a) for a in job.produces):
            complete += 1
        elif not job.produces:
            complete += 1
    return complete


def _count_jobs_total(act_id: str) -> int:
    return sum(
        1
        for job_id in ACT_JOB_MAP.get(act_id, ())
        if JOB_REGISTRY[job_id].status != "planned"
    )


def derive_act_status(
    act_id: str,
    artifacts: dict[str, bool],
    *,
    foundation: dict[str, bool],
    pipeline_job: dict[str, Any] | None,
    prior_act_statuses: dict[str, str],
) -> str:
    """Derive a single act's readiness status."""
    if act_id in PLANNED_ACTS:
        return "not_implemented"

    if not foundation.get("gnn_hyp", False) and act_id in ACT_ORDER:
        return "pending"

    if not _prior_acts_complete(act_id, prior_act_statuses):
        return "pending"

    required = ACT_REQUIRED_ARTIFACTS.get(act_id, ())
    optional = ACT_OPTIONAL_ARTIFACTS.get(act_id, ())
    req_state = {k: _artifact_value(artifacts, k) for k in required}
    opt_state = {k: _artifact_value(artifacts, k) for k in optional}

    job_status = (pipeline_job or {}).get("status")
    job_running = job_status in ("queued", "running")

    if required:
        if all(req_state.values()):
            if optional and any(opt_state.values()) and not all(opt_state.values()):
                return "degraded"
            if optional and not any(opt_state.values()):
                return "complete" if not optional else "degraded"
            return "complete"
        if job_running:
            return "running"
        if job_status == "failed" and not any(req_state.values()):
            return "failed"
        if any(req_state.values()):
            return "running"
        return "pending"

    # Acts without tier-1 required artifacts (fragment, verdict).
    if act_id == "fragment":
        if not _artifact_value(artifacts, "binding_scan"):
            return "pending"
        if _artifact_value(artifacts, "pharmacophores") and _artifact_value(
            artifacts, "drug_candidates"
        ):
            return "complete"
        if _artifact_value(artifacts, "pharmacophores") or _artifact_value(
            artifacts, "drug_candidates"
        ):
            return "degraded"
        if all(
            JOB_REGISTRY[j].status == "planned"
            for j in ACT_JOB_MAP.get("fragment", ())
        ):
            return "not_implemented"
        return "pending"

    if act_id == "verdict":
        if not _artifact_value(artifacts, "binding_scan"):
            return "pending"
        present = sum(1 for k in optional if opt_state.get(k))
        if present == len(optional) and optional:
            return "complete"
        if present > 0:
            return "degraded"
        if job_running:
            return "running"
        if job_status == "failed":
            return "failed"
        return "pending"

    return "pending"


def derive_act_readiness(
    artifacts: dict[str, bool],
    *,
    foundation: dict[str, bool],
    pipeline_job: dict[str, Any] | None,
) -> dict[str, ActStatus]:
    """Derive all act statuses in story order."""
    statuses: dict[str, str] = {}
    result: dict[str, ActStatus] = {}

    for act_id in ACT_ORDER:
        meta = ACT_DEFINITIONS[act_id]
        status = derive_act_status(
            act_id,
            artifacts,
            foundation=foundation,
            pipeline_job=pipeline_job,
            prior_act_statuses=statuses,
        )
        statuses[act_id] = status
        required = ACT_REQUIRED_ARTIFACTS.get(act_id, ())
        optional = ACT_OPTIONAL_ARTIFACTS.get(act_id, ())
        result[act_id] = ActStatus(
            act_id=act_id,
            number=meta["number"],
            title=meta["title"],
            question=meta["question"],
            color=meta["color"],
            status=status,
            jobs_complete=_count_jobs_complete(act_id, artifacts),
            jobs_total=_count_jobs_total(act_id),
            required_artifacts={k: _artifact_value(artifacts, k) for k in required},
            optional_artifacts={k: _artifact_value(artifacts, k) for k in optional},
        )
    return result


def derive_current_act(acts: dict[str, ActStatus]) -> int:
    """Return the lowest act number that is not complete (0 = cover / pre-act)."""
    for act_id in ACT_ORDER:
        act = acts[act_id]
        if act.status not in ("complete",):
            return act.number
    return ACT_DEFINITIONS["verdict"]["number"]


def derive_foundation_status(artifacts: dict[str, bool]) -> dict[str, bool]:
    return {
        key: _artifact_value(artifacts, key)
        for key in FOUNDATION_ARTIFACTS + FOUNDATION_OPTIONAL
    }
