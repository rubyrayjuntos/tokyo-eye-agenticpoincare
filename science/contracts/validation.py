"""Runtime boundary validation against the onboard contract."""

from __future__ import annotations

from science.compute.runners.base import JobRunResult
from science.contracts.geometric_runtime import (
    apply_enforcement,
    validate_learned_curvature_in_result,
)
from science.contracts.onboard_contract import (
    get_artifact_aliases,
    get_artifact_geometric_space,
    get_canonical_artifact_keys,
    get_hyperbolic_jobs,
    job_requires_hyperbolic,
)


def validate_job_geometric_constraints(job_id: str) -> list[str]:
    """Return warnings when a job's declared geometry conflicts with policy."""
    messages: list[str] = []
    from science.contracts.onboard_contract import get_required_geometric_space

    job_space = get_required_geometric_space(job_id)
    if job_id in get_hyperbolic_jobs() and job_space != "hyperbolic":
        messages.append(
            f"job {job_id!r} is listed in hyperbolic_jobs but declares geometric_space {job_space!r}"
        )
    if job_requires_hyperbolic(job_id) and job_space != "hyperbolic":
        messages.append(
            f"job {job_id!r} requires_hyperbolic but declares geometric_space {job_space!r}"
        )
    return messages


def validate_job_run_result(
    result: JobRunResult,
    *,
    learned_curvature: float | None = None,
) -> list[str]:
    """Return warnings when a job result violates artifact or geometric rules."""
    messages: list[str] = []
    known = get_canonical_artifact_keys() | frozenset(get_artifact_aliases().keys())
    job_space = None
    from science.contracts.onboard_contract import get_required_geometric_space

    job_space = get_required_geometric_space(result.job_id)

    for artifact in result.artifacts_produced:
        canonical = get_artifact_aliases().get(artifact, artifact)
        if canonical not in known:
            messages.append(
                f"job {result.job_id!r} produced unknown artifact {artifact!r}"
            )
            continue
        if job_space == "hyperbolic" and canonical != "gnn_euc":
            art_space = get_artifact_geometric_space(canonical)
            if art_space not in ("hyperbolic", "mixed", None):
                messages.append(
                    f"hyperbolic job {result.job_id!r} produced artifact {canonical!r} "
                    f"with geometric_space {art_space!r}"
                )

    passthrough_c = learned_curvature
    if passthrough_c is None and result.outputs.get("curvature") is not None:
        passthrough_c = float(result.outputs["curvature"])
    messages.extend(
        validate_learned_curvature_in_result(
            result.job_id,
            success=result.success,
            outputs=result.outputs,
            learned_curvature=passthrough_c,
        )
    )
    messages.extend(validate_job_geometric_constraints(result.job_id))
    return messages


def validate_hydrate_bundle(payload: dict) -> list[str]:
    """Return warnings when hydrate response keys drift from the contract."""
    from science.contracts.onboard_contract import get_api_surface_fields

    allowed = frozenset(get_api_surface_fields("HydrateBundle"))
    extra = sorted(set(payload.keys()) - allowed)
    if extra:
        return [f"hydrate bundle has undocumented fields: {extra}"]
    return []
