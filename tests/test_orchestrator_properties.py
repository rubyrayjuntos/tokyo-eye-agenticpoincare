"""Property-based tests for SessionOrchestrator.

Feature: backend-orchestration-layer
Properties: 1, 7, 8
Validates: Requirements 1.1, 4.1, 4.2, 4.5
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agent.orchestration.models import DiscoveryPhase, HypothesisLifecycleState
from agent.orchestration.orchestrator import SessionOrchestrator
from agent.orchestration.tool_policy import TOOL_POLICY, tool_policy_for_phase


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# All tool names that appear anywhere in the policy (allowed or blocked)
ALL_TOOL_NAMES: list[str] = sorted(
    {
        tool
        for phase_policy in TOOL_POLICY.values()
        for tool_list in phase_policy.values()
        for tool in tool_list
    }
)

# Add some tools that are not in any policy list (ungated tools)
UNGATED_TOOLS = ["unknown_tool", "custom_analysis", "fetch_data"]
ALL_TOOLS_INCLUDING_UNGATED = ALL_TOOL_NAMES + UNGATED_TOOLS

session_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=64,
)

phase_strategy = st.sampled_from(list(DiscoveryPhase))
tool_strategy = st.sampled_from(ALL_TOOLS_INCLUDING_UNGATED)
lifecycle_strategy = st.sampled_from(list(HypothesisLifecycleState))


# ---------------------------------------------------------------------------
# Property 1: Session initialization invariant
# Feature: backend-orchestration-layer, Property 1
# ---------------------------------------------------------------------------


class TestProperty1SessionInitialization:
    """Property 1: Session initialization invariant.

    For any newly created session (regardless of structure, user, or connection
    state), the orchestrator SHALL initialize with discovery phase = `residue`
    and hypothesis lifecycle = `emergent`.

    **Validates: Requirements 1.1**
    """

    @settings(max_examples=100)
    @given(session_id=session_id_strategy)
    def test_initialization_invariant(self, session_id: str):
        """Feature: backend-orchestration-layer, Property 1: Session initialization invariant

        **Validates: Requirements 1.1**
        """
        orchestrator = SessionOrchestrator(session_id)

        # Discovery phase must be 'residue'
        assert orchestrator.discovery_phase == DiscoveryPhase.RESIDUE, (
            f"Expected discovery_phase='residue', got '{orchestrator.discovery_phase.value}'"
        )

        # Hypothesis lifecycle must be 'emergent'
        assert orchestrator.hypothesis_lifecycle == HypothesisLifecycleState.EMERGENT, (
            f"Expected hypothesis_lifecycle='emergent', "
            f"got '{orchestrator.hypothesis_lifecycle.value}'"
        )

        # Tool policy must match the residue phase
        residue_policy = tool_policy_for_phase(DiscoveryPhase.RESIDUE)
        assert orchestrator.discovery_phase_state.allowed_tools == residue_policy["allowed"]
        assert orchestrator.discovery_phase_state.blocked_tools == residue_policy["blocked"]

        # State snapshot must reflect initialization
        snapshot = orchestrator.get_state_snapshot()
        assert snapshot["discovery_phase"] == "residue"
        assert snapshot["hypothesis_lifecycle"] == "emergent"
        assert snapshot["session_id"] == session_id


# ---------------------------------------------------------------------------
# Property 7: Tool gating correctness
# Feature: backend-orchestration-layer, Property 7
# ---------------------------------------------------------------------------


class TestProperty7ToolGating:
    """Property 7: Tool gating correctness.

    For any combination of discovery phase and tool name, `can_invoke_tool`
    SHALL return True if and only if the tool appears in the allowed list for
    that phase (as defined by TOOL_POLICY), and SHALL return False with an
    explanation if the tool appears in the blocked list.

    **Validates: Requirements 4.1, 4.2**
    """

    @settings(max_examples=200)
    @given(phase=phase_strategy, tool=tool_strategy)
    def test_tool_gating_correctness(self, phase: DiscoveryPhase, tool: str):
        """Feature: backend-orchestration-layer, Property 7: Tool gating correctness

        **Validates: Requirements 4.1, 4.2**
        """
        orchestrator = SessionOrchestrator("test-session")
        # Set phase directly
        orchestrator.transition_discovery({"type": "USER_SET_PHASE", "phase": phase.value})

        allowed, reason = orchestrator.can_invoke_tool(tool)
        policy = TOOL_POLICY[phase]

        if tool in policy["blocked"]:
            # Blocked tools must be rejected
            assert allowed is False, (
                f"Tool '{tool}' should be blocked in phase '{phase.value}' "
                f"but can_invoke_tool returned True"
            )
            assert reason is not None, (
                f"Blocked tool '{tool}' returned no reason for rejection"
            )
            assert "blocked" in reason.lower() or phase.value in reason.lower(), (
                f"Reason should mention blocking or phase, got: {reason}"
            )

        elif tool in policy["allowed"]:
            # Allowed tools must be permitted
            assert allowed is True, (
                f"Tool '{tool}' should be allowed in phase '{phase.value}' "
                f"but can_invoke_tool returned False"
            )
            assert reason is None, (
                f"Allowed tool '{tool}' should have no reason, got: {reason}"
            )

        else:
            # Tools not in either list are allowed by default (not explicitly gated)
            assert allowed is True, (
                f"Tool '{tool}' not in blocked list for phase '{phase.value}' "
                f"should be allowed by default"
            )


# ---------------------------------------------------------------------------
# Property 8: Tool policy equivalence
# Feature: backend-orchestration-layer, Property 8
# ---------------------------------------------------------------------------


class TestProperty8ToolPolicyEquivalence:
    """Property 8: Tool policy equivalence.

    For any discovery phase, the backend TOOL_POLICY derivation SHALL produce
    the same allowed and blocked tool lists as the `tool_policy_for_phase`
    function. The orchestrator's derived policy must be consistent with both.

    **Validates: Requirements 4.5**
    """

    @settings(max_examples=100)
    @given(phase=phase_strategy)
    def test_tool_policy_equivalence(self, phase: DiscoveryPhase):
        """Feature: backend-orchestration-layer, Property 8: Tool policy equivalence

        **Validates: Requirements 4.5**
        """
        # Direct lookup from TOOL_POLICY dict
        direct_policy = TOOL_POLICY[phase]

        # Function-based lookup
        function_policy = tool_policy_for_phase(phase)

        # They must produce identical results
        assert direct_policy["allowed"] == function_policy["allowed"], (
            f"Allowed tools mismatch for phase '{phase.value}': "
            f"direct={direct_policy['allowed']}, function={function_policy['allowed']}"
        )
        assert direct_policy["blocked"] == function_policy["blocked"], (
            f"Blocked tools mismatch for phase '{phase.value}': "
            f"direct={direct_policy['blocked']}, function={function_policy['blocked']}"
        )

        # Orchestrator's derived policy must also match
        orchestrator = SessionOrchestrator("test-session")
        orchestrator.transition_discovery({"type": "USER_SET_PHASE", "phase": phase.value})
        derived = orchestrator.derive_policy()

        assert derived.allowed_tools == direct_policy["allowed"], (
            f"Orchestrator allowed_tools mismatch for phase '{phase.value}'"
        )
        assert derived.blocked_tools == direct_policy["blocked"], (
            f"Orchestrator blocked_tools mismatch for phase '{phase.value}'"
        )

        # No tool should appear in both allowed and blocked
        overlap = set(direct_policy["allowed"]) & set(direct_policy["blocked"])
        assert len(overlap) == 0, (
            f"Tools appear in both allowed and blocked for phase '{phase.value}': {overlap}"
        )
