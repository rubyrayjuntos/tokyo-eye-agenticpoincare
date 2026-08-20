"""MLflow hierarchy taxonomy for Tokyo Eye governance.

Experiment path (locked)::

    tokyoeye/equiformer-v3-moe/{domain}/{subsystem}

See ``docs/superpowers/specs/2026-07-23-tokyoeye-mlflow-governance-design.md``.
"""

from __future__ import annotations

import re

REGISTERED_MODEL_NAME = "TokyoEye"
LINEAGE_ID = "equiformer-v3-moe"
EXPERIMENT_PREFIX = f"tokyoeye/{LINEAGE_ID}"

DOMAINS: frozenset[str] = frozenset({"geometric", "biologic", "chemical"})
SUBSYSTEMS: frozenset[str] = frozenset(
    {
        "equiformer-frontend",
        "hyperbolic-spine",
        "moe-router",
        "affinity-head",
        "full-stack",
    }
)

_RUN_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,199}$")


class TaxonomyError(ValueError):
    """Invalid domain, subsystem, or run name."""


def experiment_path(domain: str, subsystem: str) -> str:
    """Return nested MLflow experiment name for domain + subsystem."""
    d = domain.strip().lower()
    s = subsystem.strip().lower()
    if d not in DOMAINS:
        raise TaxonomyError(f"Unknown domain {domain!r}; allowed: {sorted(DOMAINS)}")
    if s not in SUBSYSTEMS:
        raise TaxonomyError(
            f"Unknown subsystem {subsystem!r}; allowed: {sorted(SUBSYSTEMS)}"
        )
    return f"{EXPERIMENT_PREFIX}/{d}/{s}"


def validate_run_name(name: str) -> str:
    """Capability+goal run names: alnum start, then alnum/._- (max 200)."""
    cleaned = name.strip()
    if not cleaned or not _RUN_NAME_RE.match(cleaned):
        raise TaxonomyError(
            f"Invalid run name {name!r}; use capability_goal style "
            "(e.g. affinity_core_pearson_ge_0.40)"
        )
    return cleaned


def mandatory_run_tags(
    *,
    domain: str,
    subsystem: str,
    capability_goal: str,
    git_sha: str | None = None,
    package_revision: str | None = None,
) -> dict[str, str]:
    """Tags every governed run must carry."""
    experiment_path(domain, subsystem)  # validate
    goal = validate_run_name(capability_goal)
    tags = {
        "model": REGISTERED_MODEL_NAME,
        "lineage": LINEAGE_ID,
        "domain": domain.strip().lower(),
        "subsystem": subsystem.strip().lower(),
        "capability_goal": goal,
        "git_sha": _resolve_git_sha(git_sha),
    }
    if package_revision:
        tags["package_revision"] = package_revision.strip()
    return tags


def _resolve_git_sha(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    try:
        from science.dtie.common.provenance_runtime import resolve_code_version

        return resolve_code_version(None) or "unknown"
    except Exception:
        return "unknown"
