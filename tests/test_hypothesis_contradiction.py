"""Property-based tests for contradiction detection on prediction flip.

Feature: hypothesis-engine, Property 7: Contradiction detection on prediction flip

For any hypothesis with a previously-passing prediction, if re-evaluation
shows the prediction now fails, contradicting evidence should be added
and the confidence should decrease.

**Validates: Requirements 6.2**
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent.tools.hypothesis.confidence import calculate_confidence
from agent.tools.hypothesis.contradiction import check_contradictions
from agent.tools.hypothesis.models import Evidence, HypothesisStatus


# ---------------------------------------------------------------------------
# In-memory DB mock for contradiction tests
# ---------------------------------------------------------------------------


class ContradictionTestDB:
    """In-memory mock DB that simulates the hypothesis tables for
    testing contradiction detection logic.
    """

    def __init__(self):
        self.hypotheses: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self._provenance_runs: list[dict[str, Any]] = []
        self._governed_assets: list[dict[str, Any]] = []

    def add_hypothesis(
        self,
        hypothesis_id: str,
        structure_id: str,
        status: str,
        confidence: float = 0.5,
    ) -> None:
        self.hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "structure_id": structure_id,
            "statement": "test hypothesis",
            "mechanism": None,
            "status": status,
            "confidence": confidence,
            "created_by": "agent",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        })

    def add_prediction(
        self,
        prediction_id: str,
        hypothesis_id: str,
        statement: str = "test prediction",
        test_tool: str | None = None,
        test_params: dict | None = None,
        threshold: str | None = None,
        passed: bool | None = None,
    ) -> None:
        self.predictions.append({
            "prediction_id": prediction_id,
            "hypothesis_id": hypothesis_id,
            "statement": statement,
            "test_tool": test_tool,
            "test_params": test_params,
            "threshold": threshold,
            "passed": passed,
            "result": None,
            "tested_at": None,
        })

    def add_evidence(
        self,
        evidence_id: str,
        hypothesis_id: str,
        supports: bool,
        strength: float = 0.5,
    ) -> None:
        self.evidence.append({
            "evidence_id": evidence_id,
            "hypothesis_id": hypothesis_id,
            "source_tool": "test",
            "source_run_id": None,
            "supports": supports,
            "strength": strength,
            "description": "test evidence",
            "gathered_at": datetime.now(timezone.utc).isoformat(),
        })

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        query_lower = query.lower()

        if "update hypothesis_prediction" in query_lower:
            pred_id = params.get("prediction_id")
            for pred in self.predictions:
                if pred["prediction_id"] == pred_id:
                    if "passed" in params:
                        pred["passed"] = params["passed"]
                    if "result" in params:
                        pred["result"] = params["result"]
                    if "tested_at" in params:
                        pred["tested_at"] = params["tested_at"]
                    break

        elif "update hypothesis set" in query_lower:
            hyp_id = params.get("hypothesis_id")
            for hyp in self.hypotheses:
                if hyp["hypothesis_id"] == hyp_id:
                    if "confidence" in params:
                        hyp["confidence"] = params["confidence"]
                    if "status" in params:
                        hyp["status"] = params["status"]
                    if "updated_at" in params:
                        hyp["updated_at"] = params["updated_at"]
                    break

        elif "insert into hypothesis_evidence" in query_lower:
            self.evidence.append({
                "evidence_id": params.get("evidence_id"),
                "hypothesis_id": params.get("hypothesis_id"),
                "source_tool": params.get("source_tool"),
                "source_run_id": params.get("source_run_id"),
                "supports": params.get("supports"),
                "strength": params.get("strength"),
                "description": params.get("description"),
                "gathered_at": params.get("gathered_at"),
            })

        elif "insert into provenance_run" in query_lower:
            self._provenance_runs.append(params)

        elif "insert into governed_asset" in query_lower:
            self._governed_assets.append(params)

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        for params in params_list:
            await self.execute(query, params)

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return None

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        query_lower = query.lower()

        if "from hypothesis" in query_lower and "status in" in query_lower:
            structure_id = params.get("structure_id")
            return [
                h for h in self.hypotheses
                if h["structure_id"] == structure_id
                and h["status"] in ("supported", "gathering")
            ]

        if "hypothesis_prediction" in query_lower and "passed = true" in query_lower:
            hyp_id = params.get("hypothesis_id")
            return [
                p for p in self.predictions
                if p["hypothesis_id"] == hyp_id and p["passed"] is True
            ]

        if "hypothesis_evidence" in query_lower and "hypothesis_id" in params:
            hyp_id = params.get("hypothesis_id")
            return [
                e for e in self.evidence
                if e["hypothesis_id"] == hyp_id
            ]

        return []

    async def begin(self) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def contradiction_scenario_strategy(draw):
    """Generate a scenario where a hypothesis has previously-passing predictions
    that may now fail based on new tool results.

    Returns:
        Tuple of (structure_id, hypothesis_id, predictions, original_threshold_value,
                  new_result_value, should_contradict)
    """
    structure_id = f"struct_{draw(st.text(alphabet='abcdef0123456789', min_size=4, max_size=8))}"
    hypothesis_id = f"hyp_{draw(st.text(alphabet='abcdef0123456789', min_size=8, max_size=12))}"

    # Generate a threshold with a ">" operator and a threshold value
    threshold_value = draw(st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False))
    threshold_expr = f"value > {threshold_value}"

    # The original result passed (was above threshold)
    # The new result may or may not pass
    new_result = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    should_contradict = new_result <= threshold_value

    # Number of existing supporting evidence (to have a baseline confidence)
    num_supporting = draw(st.integers(min_value=1, max_value=5))
    supporting_strengths = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=num_supporting,
            max_size=num_supporting,
        )
    )

    return {
        "structure_id": structure_id,
        "hypothesis_id": hypothesis_id,
        "threshold_expr": threshold_expr,
        "threshold_value": threshold_value,
        "new_result": new_result,
        "should_contradict": should_contradict,
        "supporting_strengths": supporting_strengths,
    }


# ---------------------------------------------------------------------------
# Property 7: Contradiction detection on prediction flip
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    """Property 7: Contradiction detection on prediction flip.

    For any hypothesis with a previously-passing prediction, if re-evaluation
    shows the prediction now fails, contradicting evidence should be added
    and the confidence should decrease.
    """

    @settings(max_examples=100)
    @given(scenario=contradiction_scenario_strategy())
    @pytest.mark.asyncio
    async def test_contradiction_adds_evidence_and_decreases_confidence(self, scenario):
        """Feature: hypothesis-engine, Property 7: Contradiction detection on prediction flip

        For any hypothesis with a previously-passing prediction, if re-evaluation
        shows the prediction now fails, contradicting evidence should be added
        and the confidence should decrease.

        **Validates: Requirements 6.2**
        """
        structure_id = scenario["structure_id"]
        hypothesis_id = scenario["hypothesis_id"]
        threshold_expr = scenario["threshold_expr"]
        new_result = scenario["new_result"]
        should_contradict = scenario["should_contradict"]
        supporting_strengths = scenario["supporting_strengths"]

        # Set up mock DB with an active hypothesis
        mock_db = ContradictionTestDB()
        mock_db.add_hypothesis(
            hypothesis_id=hypothesis_id,
            structure_id=structure_id,
            status="supported",
            confidence=0.8,
        )

        # Add a prediction that previously passed
        pred_id = f"pred_{uuid.uuid4().hex[:12]}"
        mock_db.add_prediction(
            prediction_id=pred_id,
            hypothesis_id=hypothesis_id,
            statement="test metric should be high",
            test_tool="get_graph_metrics",
            test_params={"structure_id": structure_id},
            threshold=threshold_expr,
            passed=True,
        )

        # Add existing supporting evidence
        for i, strength in enumerate(supporting_strengths):
            mock_db.add_evidence(
                evidence_id=f"ev_existing_{i}",
                hypothesis_id=hypothesis_id,
                supports=True,
                strength=strength,
            )

        # Calculate confidence before contradiction check
        evidence_before = [
            Evidence(source_tool="db", supports=e["supports"], strength=e["strength"], description="")
            for e in mock_db.evidence
        ]
        confidence_before = calculate_confidence(evidence_before)

        # Create a tool dispatcher that returns the new result
        async def mock_dispatcher(tool_name: str, params: dict) -> dict:
            return {"value": new_result}

        # Run contradiction check
        contradictions = await check_contradictions(
            structure_id=structure_id,
            tool_dispatcher=mock_dispatcher,
            db=mock_db,
        )

        if should_contradict:
            # Contradiction should be detected
            assert len(contradictions) == 1
            c = contradictions[0]
            assert c["hypothesis_id"] == hypothesis_id
            assert c["prediction_id"] == pred_id

            # Contradicting evidence should have been added
            contradicting_evidence = [
                e for e in mock_db.evidence if not e["supports"]
            ]
            assert len(contradicting_evidence) >= 1

            # Confidence should decrease (new contradicting evidence added)
            assert c["new_confidence"] < confidence_before
        else:
            # No contradiction — prediction still passes
            assert len(contradictions) == 0

            # No contradicting evidence should have been added
            contradicting_evidence = [
                e for e in mock_db.evidence if not e["supports"]
            ]
            assert len(contradicting_evidence) == 0
