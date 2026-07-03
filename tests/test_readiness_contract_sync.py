"""Gates: readiness module constants stay aligned with onboard_contract.yaml."""

from __future__ import annotations

from data.act_readiness import (
    ACT_DEFINITIONS,
    ACT_OPTIONAL_ARTIFACTS,
    ACT_ORDER,
    ACT_REQUIRED_ARTIFACTS,
    ARTIFACT_PROBE_KEYS,
    FOUNDATION_ARTIFACTS,
    FOUNDATION_OPTIONAL,
)
from data.readiness import EXTENDED_ARTIFACTS, TIER1_ARTIFACTS, TIER2_ARTIFACTS
from science.contracts.onboard_contract import (
    get_act_definitions,
    get_act_optional_artifacts,
    get_act_order,
    get_act_required_artifacts,
    get_artifact_probe_keys,
    get_extended_artifact_keys,
    get_foundation_artifacts,
    get_foundation_optional,
    get_tier_artifact_keys,
)


def test_tier_artifacts_match_contract() -> None:
    assert TIER1_ARTIFACTS == get_tier_artifact_keys(1)
    assert TIER2_ARTIFACTS == get_tier_artifact_keys(2)
    assert EXTENDED_ARTIFACTS == get_extended_artifact_keys()


def test_act_readiness_constants_match_contract() -> None:
    assert ACT_ORDER == get_act_order()
    assert FOUNDATION_ARTIFACTS == get_foundation_artifacts()
    assert FOUNDATION_OPTIONAL == get_foundation_optional()
    assert ACT_DEFINITIONS == get_act_definitions()
    assert ACT_REQUIRED_ARTIFACTS == get_act_required_artifacts()
    assert ACT_OPTIONAL_ARTIFACTS == get_act_optional_artifacts()
    assert ARTIFACT_PROBE_KEYS == get_artifact_probe_keys()


def test_artifact_probe_keys_cover_act_artifacts() -> None:
    for artifacts in get_act_required_artifacts().values():
        for key in artifacts:
            assert key in ARTIFACT_PROBE_KEYS, f"missing probe map for act artifact {key!r}"
    for artifacts in get_act_optional_artifacts().values():
        for key in artifacts:
            assert key in ARTIFACT_PROBE_KEYS, f"missing probe map for optional artifact {key!r}"
