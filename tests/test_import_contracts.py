"""P2 boundary gates: import contracts for agent/compute separation."""

from __future__ import annotations

from data.audit.import_contract import audit_import_contracts


def test_import_contract_gate_passes() -> None:
    errors = audit_import_contracts()
    assert errors == [], "Import-boundary violations:\n" + "\n".join(errors)


def test_agent_normalizer_allowlist_is_documented() -> None:
    from data.audit.import_contract import AGENT_NORMALIZER_IMPORT_ALLOWLIST

    assert "agent/tools/data_tools.py" in AGENT_NORMALIZER_IMPORT_ALLOWLIST
    assert "agent/tools/hypothesis/tools.py" in AGENT_NORMALIZER_IMPORT_ALLOWLIST


def test_agent_compute_persist_grandfather_is_frozen() -> None:
    """Legacy cryptic on-demand paths — shrink this set, never grow it."""
    from data.audit.import_contract import AGENT_COMPUTE_PERSIST_ALLOWLIST

    assert AGENT_COMPUTE_PERSIST_ALLOWLIST == frozenset(
        {
            "agent/tools/cryptic/scan_phase.py",
            "agent/tools/cryptic/smd_dispatch.py",
            "agent/tools/cryptic/tool.py",
        }
    )
