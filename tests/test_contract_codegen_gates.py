"""P3 codegen/version hygiene gates."""

from __future__ import annotations

from data.audit.audit_event_types import audit_event_type_literals
from data.audit.contract_codegen import audit_contract_sync, audit_contract_version_bump


def test_contract_sync_gate_passes() -> None:
    errors = audit_contract_sync()
    assert errors == [], "Stale generated contract artifacts:\n" + "\n".join(errors)


def test_contract_version_bump_gate_passes_against_head() -> None:
    """No-op when contract is unchanged vs HEAD (typical clean tree)."""
    errors = audit_contract_version_bump(base_ref="HEAD")
    assert errors == [], "Contract version gate:\n" + "\n".join(errors)


def test_contract_version_bump_requires_version_line_in_diff() -> None:
    from data.audit.contract_codegen import audit_contract_version_bump as gate

    # Simulate: body changed, version untouched.
    diff_without_version = (
        "--- a/science/contracts/onboard_contract.yaml\n"
        "+++ b/science/contracts/onboard_contract.yaml\n"
        "@@ -1,3 +1,3 @@\n"
        '-description: "old"\n'
        '+description: "new"\n'
    )

    def _fake_git_diff(ref: str, path):  # noqa: ANN001
        return diff_without_version

    import data.audit.contract_codegen as mod

    original = mod._git_diff
    mod._git_diff = _fake_git_diff
    try:
        errors = gate(base_ref="HEAD")
    finally:
        mod._git_diff = original
    assert errors


def test_audit_event_type_literal_gate_passes() -> None:
    errors = audit_event_type_literals()
    assert errors == [], "Audit event_type literal violations:\n" + "\n".join(errors)


def test_all_event_types_matches_constants() -> None:
    from shared.audit import events

    expected = {
        v
        for k, v in vars(events).items()
        if k.startswith("EVENT_") and isinstance(v, str)
    }
    assert events.ALL_EVENT_TYPES == frozenset(expected)
