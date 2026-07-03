"""Liveness canaries for high-value §0 gate claims.

Symbol-presence checks (``audit_gate_claims``) catch deletion/rename drift only.
These canaries prove the underlying gate still *fails* on known-bad input —
the difference between existence and liveness named in ENFORCEMENT_MATRIX §0.

Not mutation testing; a small fixed set of "the canary still dies" checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from data.audit.gate_claims import GATE_CLAIMS, GateClaim, audit_gate_claims
from data.audit.write_paths import WritePathHit, _gate_violation
from science.compute.registry import ComputeJob, JOB_REGISTRY
from science.compute.runner_dispatch import JOB_RUNNERS, validate_runners_against_registry
from science.contracts.onboard_contract import validate_registry_job_keys_match_contract

# gate_id values in GATE_CLAIMS that have a liveness canary below.
GATE_LIVENESS_CANARY_IDS: frozenset[str] = frozenset(
    {
        "onboard_contract_registry",
        "runner_registry_triangle",
        "write_path_inventory",
        "enforcement_meta_gate",
    }
)


@dataclass(frozen=True)
class GateLivenessCanary:
    gate_id: str
    description: str
    check: Callable[[], str | None]


def _canary_registry_job_keys() -> str | None:
    errors = validate_registry_job_keys_match_contract()
    if errors:
        return f"live repo failed registry↔contract check: {errors}"
    phantom_id = "__gate_canary_phantom_job__"
    phantom = ComputeJob(
        job_id=phantom_id,
        discovery_act="foundation",
        resource_class="cpu_light",
        priority_group=0,
        requires=frozenset(),
        produces=frozenset(("dims",)),
        tier=1,
        status="implemented",
    )
    JOB_REGISTRY[phantom_id] = phantom
    try:
        drift = validate_registry_job_keys_match_contract()
        if not drift:
            return "validate_registry_job_keys_match_contract did not fail on phantom registry job"
        if not any(phantom_id in msg for msg in drift):
            return f"phantom job not named in drift errors: {drift}"
    finally:
        JOB_REGISTRY.pop(phantom_id, None)
    return None


def _canary_runner_registry_triangle() -> str | None:
    errors = validate_runners_against_registry()
    if errors:
        return f"live repo failed runner↔registry check: {errors}"
    phantom_id = "__gate_canary_phantom_runner__"
    JOB_RUNNERS[phantom_id] = lambda *_a, **_k: None  # type: ignore[assignment]
    try:
        drift = validate_runners_against_registry()
        if not drift:
            return "validate_runners_against_registry did not fail on phantom JOB_RUNNERS key"
        if not any(phantom_id in msg for msg in drift):
            return f"phantom runner not named in drift errors: {drift}"
    finally:
        JOB_RUNNERS.pop(phantom_id, None)
    return None


def _canary_write_path_gate() -> str | None:
    violation = _gate_violation(
        WritePathHit(
            path="agent/tools/__gate_canary__.py",
            line_no=1,
            table="fact_residue_state",
            category="bypass_fact",
            operation="insert",
        )
    )
    if violation is None:
        return "write-path gate did not flag synthetic agent/tools fact_* INSERT bypass"
    allowed = _gate_violation(
        WritePathHit(
            path="science/compute/runners/__gate_canary__.py",
            line_no=1,
            table="provenance_run",
            category="bypass_governance",
            operation="insert",
        )
    )
    if allowed is not None:
        return f"write-path gate wrongly flagged allowed runner provenance_run insert: {allowed}"
    return None


def _canary_meta_gate_self() -> str | None:
    if audit_gate_claims() != []:
        return "audit_gate_claims failed on live GATE_CLAIMS registry"
    bogus = (
        GateClaim(
            "canary_bogus",
            "tests/test_enforcement_matrix_gate_claims.py",
            ("__symbol_that_must_not_exist__",),
        ),
    )
    drift = audit_gate_claims(claims=bogus)
    if not drift:
        return "audit_gate_claims did not fail on bogus missing symbol"
    return None


GATE_LIVENESS_CANARIES: tuple[GateLivenessCanary, ...] = (
    GateLivenessCanary("onboard_contract_registry", "registry ⊆ contract", _canary_registry_job_keys),
    GateLivenessCanary("runner_registry_triangle", "JOB_RUNNERS ↔ registry", _canary_runner_registry_triangle),
    GateLivenessCanary("write_path_inventory", "agent/tools INSERT bypass", _canary_write_path_gate),
    GateLivenessCanary("enforcement_meta_gate", "meta-gate detects missing symbol", _canary_meta_gate_self),
)


def audit_gate_liveness_canaries() -> list[str]:
    """Return errors when a liveness canary no longer fails on known-bad input."""
    errors: list[str] = []
    registered = {claim.gate_id for claim in GATE_CLAIMS}
    missing = GATE_LIVENESS_CANARY_IDS - {c.gate_id for c in GATE_LIVENESS_CANARIES}
    if missing:
        errors.append(f"liveness canary missing for gate_id(s): {sorted(missing)}")
    orphan = {c.gate_id for c in GATE_LIVENESS_CANARIES} - registered
    if orphan:
        errors.append(f"liveness canary gate_id(s) not in GATE_CLAIMS: {sorted(orphan)}")
    for canary in GATE_LIVENESS_CANARIES:
        try:
            failure = canary.check()
        except Exception as exc:
            errors.append(f"{canary.gate_id} liveness canary raised: {exc.__class__.__name__}: {exc}")
            continue
        if failure:
            errors.append(f"{canary.gate_id}: {failure}")
    return errors
