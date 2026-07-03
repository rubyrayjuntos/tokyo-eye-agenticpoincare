"""§0 meta-gate: ENFORCEMENT_MATRIX GATED rows must reference real assertions."""

from __future__ import annotations

from data.audit.gate_canaries import (
    GATE_LIVENESS_CANARY_IDS,
    audit_gate_liveness_canaries,
)
from data.audit.gate_claims import GATE_CLAIMS, GateClaim, audit_gate_claims


def test_enforcement_matrix_gate_claims_pass() -> None:
    errors = audit_gate_claims()
    assert errors == [], "Gate-claim drift:\n" + "\n".join(errors)


def test_enforcement_matrix_gate_liveness_canaries_pass() -> None:
    errors = audit_gate_liveness_canaries()
    assert errors == [], "Gate liveness canary failure:\n" + "\n".join(errors)


def test_gate_claim_registry_is_non_empty_and_unique() -> None:
    assert len(GATE_CLAIMS) >= 20
    gate_ids = [claim.gate_id for claim in GATE_CLAIMS]
    assert len(gate_ids) == len(set(gate_ids)), "duplicate gate_id in GATE_CLAIMS"


def test_high_value_claims_have_liveness_canaries() -> None:
    assert GATE_LIVENESS_CANARY_IDS <= {claim.gate_id for claim in GATE_CLAIMS}


def test_write_path_claim_covers_zero_bypass_policy() -> None:
    """Write-path gate must still scan agent/coordinator surfaces."""
    from data.audit.write_paths import ZERO_BYPASS_PREFIXES

    claim = next(c for c in GATE_CLAIMS if c.gate_id == "write_path_inventory")
    text = (__import__("pathlib").Path(__file__).resolve().parents[1] / claim.artifact).read_text(
        encoding="utf-8"
    )
    assert "ZERO_BYPASS_PREFIXES" in text or "agent/tools/" in text
    assert "agent/tools/" in str(ZERO_BYPASS_PREFIXES)
