"""Loader and validators for the master onboard contract."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml

from data.normalizer.core import Normalizer
from science.compute.registry import JOB_REGISTRY

CONTRACT_PATH = Path(__file__).with_name("onboard_contract.yaml")

GeometricSpace = Literal["hyperbolic", "euclidean", "mixed", "none"]

_VALID_GEOMETRIC_SPACES = frozenset({"hyperbolic", "euclidean", "mixed", "none"})

_VALID_NORMALIZER_METHODS = frozenset(
    name
    for name in dir(Normalizer)
    if name.startswith("normalize_") and callable(getattr(Normalizer, name, None))
)


@functools.lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    """Load and parse onboard_contract.yaml."""
    with CONTRACT_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid contract format in {CONTRACT_PATH}")
    return data


def get_artifact_catalog() -> dict[str, dict[str, Any]]:
    return dict(load_contract().get("artifacts", {}))


def get_artifact_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for spec in get_artifact_catalog().values():
        canonical = spec["canonical_key"]
        for legacy in spec.get("aliases", []):
            aliases[str(legacy)] = canonical
    return aliases


def get_artifact_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for spec in get_artifact_catalog().values():
        canonical = spec["canonical_key"]
        labels[canonical] = spec.get("label", canonical)
        for legacy in spec.get("aliases", []):
            labels[str(legacy)] = spec.get("label", canonical)
    return labels


def get_tier_artifact_keys(tier: int) -> tuple[str, ...]:
    return tuple(
        spec["canonical_key"]
        for spec in get_artifact_catalog().values()
        if spec.get("tier") == tier
    )


def get_extended_artifact_keys() -> tuple[str, ...]:
    return tuple(
        spec["canonical_key"]
        for spec in get_artifact_catalog().values()
        if spec.get("tier") == "extended"
    )


def get_probe_id(artifact_key: str) -> str | None:
    canonical = get_artifact_aliases().get(artifact_key, artifact_key)
    spec = get_artifact_catalog().get(canonical)
    if spec is None:
        return None
    return spec.get("probe")


def get_foundation_artifacts() -> tuple[str, ...]:
    readiness = load_contract().get("readiness", {})
    foundation = readiness.get("foundation", {})
    return tuple(foundation.get("required", ()))


def get_foundation_optional() -> tuple[str, ...]:
    readiness = load_contract().get("readiness", {})
    foundation = readiness.get("foundation", {})
    return tuple(foundation.get("optional", ()))


def get_act_order() -> tuple[str, ...]:
    return tuple(load_contract().get("readiness", {}).get("act_order", ()))


def get_act_definitions() -> dict[str, dict[str, Any]]:
    acts = load_contract().get("readiness", {}).get("acts", {})
    return {
        act_id: {
            "number": meta["number"],
            "title": meta["title"],
            "question": meta["question"],
            "color": meta["color"],
        }
        for act_id, meta in acts.items()
    }


def get_act_required_artifacts() -> dict[str, tuple[str, ...]]:
    acts = load_contract().get("readiness", {}).get("acts", {})
    return {act_id: tuple(meta.get("required", ())) for act_id, meta in acts.items()}


def get_act_optional_artifacts() -> dict[str, tuple[str, ...]]:
    acts = load_contract().get("readiness", {}).get("acts", {})
    return {act_id: tuple(meta.get("optional", ())) for act_id, meta in acts.items()}


def get_artifact_probe_keys() -> dict[str, str]:
    """Map canonical artifact keys to merged probe-result dict keys."""
    return {
        spec["canonical_key"]: spec["canonical_key"]
        for spec in get_artifact_catalog().values()
    }


def get_act_geometric_space(act_id: str) -> GeometricSpace | None:
    acts = load_contract().get("readiness", {}).get("acts", {})
    space = acts.get(act_id, {}).get("geometric_space")
    if space is None:
        return None
    return str(space)  # type: ignore[return-value]


def get_geometric_constraints() -> dict[str, Any]:
    return dict(load_contract().get("geometric_constraints", {}))


def get_hyperbolic_jobs() -> frozenset[str]:
    jobs = get_geometric_constraints().get("hyperbolic_jobs", [])
    return frozenset(str(job_id) for job_id in jobs)


def get_euclidean_exception_jobs() -> frozenset[str]:
    jobs = get_geometric_constraints().get("allowed_euclidean_exceptions", [])
    return frozenset(str(job_id) for job_id in jobs)


def get_job_geometric_catalog() -> dict[str, dict[str, Any]]:
    return dict(load_contract().get("jobs", {}))


def get_required_geometric_space(job_id: str) -> GeometricSpace | None:
    """Return the contract-declared geometric space for a compute job."""
    spec = get_job_geometric_catalog().get(job_id)
    if spec is None:
        if job_id in get_hyperbolic_jobs():
            return "hyperbolic"
        if job_id in get_euclidean_exception_jobs():
            return "none" if job_id in {"ingest_dims", "assign_computation_scope", "alignment_sidecar"} else "euclidean"
        return None
    space = spec.get("geometric_space")
    return str(space) if space is not None else None  # type: ignore[return-value]


def job_requires_hyperbolic(job_id: str) -> bool:
    spec = get_job_geometric_catalog().get(job_id, {})
    if "requires_hyperbolic" in spec:
        return bool(spec["requires_hyperbolic"])
    return job_id in get_hyperbolic_jobs()


def get_artifact_geometric_space(artifact_key: str) -> GeometricSpace | None:
    canonical = get_artifact_aliases().get(artifact_key, artifact_key)
    spec = get_artifact_catalog().get(canonical, {})
    space = spec.get("geometric_space")
    if space is None:
        return None
    return str(space)  # type: ignore[return-value]


def get_artifact_representation(artifact_key: str) -> str | None:
    canonical = get_artifact_aliases().get(artifact_key, artifact_key)
    spec = get_artifact_catalog().get(canonical, {})
    rep = spec.get("representation")
    return str(rep) if rep is not None else None


def requires_hyperbolic_readiness() -> bool:
    ready = load_contract().get("readiness", {}).get("global_status", {}).get("ready", {})
    return bool(ready.get("requires_hyperbolic", False))


def get_required_hyperbolic_artifacts() -> tuple[str, ...]:
    ready = load_contract().get("readiness", {}).get("global_status", {}).get("ready", {})
    return tuple(ready.get("required_hyperbolic_artifacts", ()))


def geometric_enforcement_level() -> str:
    enforcement = get_geometric_constraints().get("enforcement", {})
    return str(enforcement.get("level", "warning"))


def derive_geometric_readiness(
    artifacts: dict[str, bool],
    *,
    learned_curvature: float | None = None,
) -> dict[str, Any]:
    """Summarize whether required hyperbolic artifacts and learned curvature are present."""
    required = get_required_hyperbolic_artifacts()
    present = {key: bool(artifacts.get(key, False)) for key in required}
    complete = all(present.values()) if required else True
    has_gnn = present.get("gnn_hyp", False)
    curvature_ready = learned_curvature is not None and learned_curvature > 0 if has_gnn else None
    return {
        "requires_hyperbolic": requires_hyperbolic_readiness(),
        "required_hyperbolic_artifacts": present,
        "hyperbolic_ready": complete,
        "learned_curvature": learned_curvature,
        "curvature_ready": curvature_ready,
    }


def get_api_surface_fields(surface: str, *, variant: str | None = None) -> tuple[str, ...]:
    surfaces = load_contract().get("api_surfaces", {})
    spec = surfaces.get(surface, {})
    if variant is not None:
        fields = spec.get(variant, [])
    else:
        fields = spec.get("fields", [])
    return tuple(fields)


def validate_contract_against_registry() -> list[str]:
    """Ensure registry artifact keys are declared in the contract catalog."""
    errors: list[str] = []
    catalog = get_artifact_catalog()
    canonical_keys = {spec["canonical_key"] for spec in catalog.values()}
    alias_keys = set(get_artifact_aliases().keys())
    known = canonical_keys | alias_keys

    registry_produces: set[str] = set()
    for job in JOB_REGISTRY.values():
        registry_produces |= set(job.produces)

    for artifact in sorted(registry_produces):
        if artifact not in known:
            errors.append(f"registry produces unknown artifact {artifact!r}")

    for key, spec in catalog.items():
        if key != spec.get("canonical_key"):
            errors.append(f"artifact catalog key {key!r} != canonical_key {spec.get('canonical_key')!r}")
        for job_id in spec.get("producing_jobs", []):
            if job_id not in JOB_REGISTRY:
                errors.append(f"artifact {key!r} references unknown job {job_id!r}")
                continue
            produced = set(JOB_REGISTRY[job_id].produces)
            canonical = spec["canonical_key"]
            if canonical not in produced and canonical not in alias_keys:
                errors.append(
                    f"artifact {canonical!r} not in produces for job {job_id!r}"
                )

    for tier in (1, 2):
        for artifact in get_tier_artifact_keys(tier):
            if get_probe_id(artifact) is None:
                errors.append(f"tier {tier} artifact {artifact!r} missing probe id")

    errors.extend(validate_geometric_contract())
    errors.extend(validate_registry_job_keys_match_contract())
    from science.contracts.model_registry import validate_gnn_models_contract

    errors.extend(validate_gnn_models_contract())
    return errors


_EXTERNAL_REGISTRY_JOBS = frozenset(
    {"ingest_dims", "assign_computation_scope", "alignment_sidecar"}
)


def validate_registry_job_keys_match_contract() -> list[str]:
    """Every registry job (except external foundation jobs) must appear in contract ``jobs:``."""
    contract_jobs = set(get_job_geometric_catalog().keys())
    registry_keys = set(JOB_REGISTRY.keys())
    expected_in_contract = registry_keys - _EXTERNAL_REGISTRY_JOBS

    errors: list[str] = []
    missing_in_contract = sorted(expected_in_contract - contract_jobs)
    extra_in_contract = sorted(contract_jobs - registry_keys)
    if missing_in_contract:
        errors.append(
            "registry jobs missing from contract jobs: "
            + ", ".join(missing_in_contract)
        )
    if extra_in_contract:
        errors.append(
            "contract jobs missing from registry: " + ", ".join(extra_in_contract)
        )
    return errors


def validate_geometric_contract() -> list[str]:
    """Validate geometric metadata consistency across artifacts and jobs."""
    errors: list[str] = []
    catalog = get_artifact_catalog()
    hyperbolic_jobs = get_hyperbolic_jobs()
    job_catalog = get_job_geometric_catalog()

    for key, spec in catalog.items():
        space = spec.get("geometric_space")
        if space is None:
            errors.append(f"artifact {key!r} missing geometric_space")
        elif space not in _VALID_GEOMETRIC_SPACES:
            errors.append(f"artifact {key!r} has invalid geometric_space {space!r}")
        if not spec.get("representation"):
            errors.append(f"artifact {key!r} missing representation")
        if space == "hyperbolic":
            curvature = spec.get("curvature")
            if not isinstance(curvature, dict) or not curvature.get("source"):
                errors.append(
                    f"hyperbolic artifact {key!r} must declare curvature.source "
                    "(learned_at_inference or inherited_from_gnn_hyp)"
                )
            elif curvature.get("source") not in (
                "learned_at_inference",
                "inherited_from_gnn_hyp",
            ):
                errors.append(
                    f"artifact {key!r} has invalid curvature.source {curvature.get('source')!r}"
                )
            if isinstance(curvature, dict) and "passthrough_from" not in curvature and curvature.get("source") != "learned_at_inference":
                if curvature.get("source") == "inherited_from_gnn_hyp":
                    errors.append(
                        f"artifact {key!r} with inherited curvature must declare passthrough_from"
                    )

    for job_id in hyperbolic_jobs:
        if job_id not in JOB_REGISTRY:
            errors.append(f"hyperbolic_jobs lists unknown job {job_id!r}")
            continue
        if not job_requires_hyperbolic(job_id):
            errors.append(f"hyperbolic_jobs job {job_id!r} must have requires_hyperbolic: true")
        job_spec = get_job_geometric_catalog().get(job_id, {})
        if job_spec and not job_spec.get("requires_hyperbolic"):
            errors.append(f"jobs.{job_id}.requires_hyperbolic must be true")
        job_space = get_required_geometric_space(job_id)
        if job_space != "hyperbolic":
            errors.append(f"hyperbolic_jobs job {job_id!r} must declare geometric_space hyperbolic")
        for artifact in JOB_REGISTRY[job_id].produces:
            if artifact == "gnn_euc":
                continue
            art_space = get_artifact_geometric_space(artifact)
            if art_space not in ("hyperbolic", "mixed"):
                errors.append(
                    f"hyperbolic job {job_id!r} produces {artifact!r} "
                    f"with geometric_space {art_space!r}"
                )

    for job_id, spec in job_catalog.items():
        if job_id not in JOB_REGISTRY:
            errors.append(f"jobs geometric catalog references unknown job {job_id!r}")
            continue
        declared_produces = set(spec.get("produces", []))
        if declared_produces and declared_produces != set(JOB_REGISTRY[job_id].produces):
            errors.append(
                f"jobs.{job_id}.produces {sorted(declared_produces)} "
                f"!= registry {sorted(JOB_REGISTRY[job_id].produces)}"
            )

    for act_id, required in get_act_required_artifacts().items():
        act_space = get_act_geometric_space(act_id)
        if act_space == "hyperbolic":
            for artifact in required:
                art_space = get_artifact_geometric_space(artifact)
                if art_space not in ("hyperbolic", "mixed"):
                    errors.append(
                        f"act {act_id!r} is hyperbolic but requires non-hyperbolic artifact {artifact!r}"
                    )
        elif act_space == "mixed" and required:
            hyperbolic_required = [
                a for a in required if get_artifact_geometric_space(a) == "hyperbolic"
            ]
            if not hyperbolic_required:
                errors.append(
                    f"act {act_id!r} is mixed with required artifacts but none are hyperbolic"
                )

    return errors


def validate_normalizer_destinations(job_catalog: dict[str, Any]) -> list[str]:
    """Validate job_schema destination normalizer_path values."""
    errors: list[str] = []
    for job_id, job in job_catalog.get("jobs", {}).items():
        for dest in job.get("destinations", []):
            path = dest.get("normalizer_path")
            if not path:
                continue
            if dest.get("bypass"):
                continue
            if path.startswith("_"):
                continue
            if path not in _VALID_NORMALIZER_METHODS:
                errors.append(
                    f"job {job_id!r} destination {dest.get('table')!r} "
                    f"references unknown normalizer_path {path!r}"
                )
    return errors


def get_canonical_artifact_keys() -> frozenset[str]:
    return frozenset(spec["canonical_key"] for spec in get_artifact_catalog().values())
