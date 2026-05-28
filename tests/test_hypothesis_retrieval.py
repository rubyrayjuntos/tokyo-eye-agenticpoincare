"""Property-based tests for hypothesis retrieval with filtering.

Feature: hypothesis-engine, Property 6: Hypothesis retrieval with filtering

For any set of hypotheses stored for a structure, querying by structure_id
should return all of them. Querying with a status filter should return
exactly those hypotheses matching that status.

**Validates: Requirements 5.1, 5.2**
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent.tools.hypothesis.models import HypothesisStatus
from agent.tools.hypothesis.tools import get_hypotheses


# ---------------------------------------------------------------------------
# In-memory DB mock for retrieval tests
# ---------------------------------------------------------------------------


class HypothesisRetrievalDB:
    """In-memory mock DB that stores hypotheses, predictions, and evidence
    for testing retrieval/filtering logic without a real database.
    """

    def __init__(self):
        self.hypotheses: list[dict[str, Any]] = []
        self.predictions: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []

    def add_hypothesis(
        self,
        hypothesis_id: str,
        structure_id: str,
        status: str,
        statement: str = "test",
        confidence: float = 0.5,
    ) -> None:
        self.hypotheses.append({
            "hypothesis_id": hypothesis_id,
            "structure_id": structure_id,
            "statement": statement,
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
        passed: bool | None = None,
    ) -> None:
        self.predictions.append({
            "prediction_id": prediction_id,
            "hypothesis_id": hypothesis_id,
            "statement": "test prediction",
            "passed": passed,
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
            "supports": supports,
            "strength": strength,
            "cnt": 1,
        })

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        pass

    async def execute_many(self, query: str, params_list: list[dict[str, Any]]) -> None:
        pass

    async def fetch_one(self, query: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return None

    async def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        query_lower = query.lower()

        # Order matters: check more specific patterns first
        if "hypothesis_evidence" in query_lower and "group by" in query_lower:
            hyp_id = params.get("hypothesis_id")
            # Aggregate evidence by supports flag
            supporting = sum(1 for e in self.evidence if e["hypothesis_id"] == hyp_id and e["supports"])
            contradicting = sum(1 for e in self.evidence if e["hypothesis_id"] == hyp_id and not e["supports"])
            rows = []
            if supporting > 0:
                rows.append({"supports": True, "cnt": supporting})
            if contradicting > 0:
                rows.append({"supports": False, "cnt": contradicting})
            return rows

        if "hypothesis_prediction" in query_lower:
            hyp_id = params.get("hypothesis_id")
            return [p for p in self.predictions if p["hypothesis_id"] == hyp_id]

        if "from hypothesis h" in query_lower:
            results = list(self.hypotheses)
            if "structure_id" in params:
                results = [h for h in results if h["structure_id"] == params["structure_id"]]
            if "status" in params:
                results = [h for h in results if h["status"] == params["status"]]
            return results

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
def hypothesis_set_strategy(draw):
    """Generate a set of hypotheses across multiple structures and statuses."""
    num_hypotheses = draw(st.integers(min_value=1, max_value=10))
    structures = draw(st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=3, max_size=8),
        min_size=1,
        max_size=3,
    ))
    statuses = [s.value for s in HypothesisStatus]

    hypotheses = []
    for _ in range(num_hypotheses):
        hypotheses.append({
            "hypothesis_id": f"hyp_{uuid.uuid4().hex[:12]}",
            "structure_id": draw(st.sampled_from(structures)),
            "status": draw(st.sampled_from(statuses)),
        })

    return hypotheses, structures, statuses


# ---------------------------------------------------------------------------
# Property 6: Hypothesis retrieval with filtering
# ---------------------------------------------------------------------------


class TestHypothesisRetrieval:
    """Property 6: Hypothesis retrieval with filtering.

    For any set of hypotheses stored for a structure, querying by
    structure_id should return all of them. Querying with a status
    filter should return exactly those hypotheses matching that status.
    """

    @settings(max_examples=100)
    @given(data=hypothesis_set_strategy())
    @pytest.mark.asyncio
    async def test_structure_filter_returns_all_matching(self, data):
        """Feature: hypothesis-engine, Property 6: Hypothesis retrieval with filtering

        For any set of hypotheses, querying by structure_id returns exactly
        those hypotheses belonging to that structure.

        **Validates: Requirements 5.1, 5.2**
        """
        hypotheses, structures, _ = data

        # Set up mock DB
        mock_db = HypothesisRetrievalDB()
        for h in hypotheses:
            mock_db.add_hypothesis(
                hypothesis_id=h["hypothesis_id"],
                structure_id=h["structure_id"],
                status=h["status"],
            )
            # Add a prediction so the summary works
            mock_db.add_prediction(
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                hypothesis_id=h["hypothesis_id"],
            )

        # For each structure, verify retrieval returns exactly the right set
        for structure_id in structures:
            result = await get_hypotheses(structure_id=structure_id, db=mock_db)

            assert result.success is True

            expected_ids = {
                h["hypothesis_id"]
                for h in hypotheses
                if h["structure_id"] == structure_id
            }
            returned_ids = {
                h["hypothesis_id"]
                for h in result.data["hypotheses"]
            }

            assert returned_ids == expected_ids, (
                f"For structure '{structure_id}': "
                f"expected {expected_ids}, got {returned_ids}"
            )

    @settings(max_examples=100)
    @given(data=hypothesis_set_strategy())
    @pytest.mark.asyncio
    async def test_status_filter_returns_only_matching(self, data):
        """Feature: hypothesis-engine, Property 6: Hypothesis retrieval with filtering

        For any set of hypotheses, querying with a status filter returns
        exactly those hypotheses matching that status.

        **Validates: Requirements 5.1, 5.2**
        """
        hypotheses, structures, statuses = data

        # Set up mock DB
        mock_db = HypothesisRetrievalDB()
        for h in hypotheses:
            mock_db.add_hypothesis(
                hypothesis_id=h["hypothesis_id"],
                structure_id=h["structure_id"],
                status=h["status"],
            )
            mock_db.add_prediction(
                prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
                hypothesis_id=h["hypothesis_id"],
            )

        # For each status, verify retrieval returns exactly the right set
        for status in statuses:
            result = await get_hypotheses(status=status, db=mock_db)

            assert result.success is True

            expected_ids = {
                h["hypothesis_id"]
                for h in hypotheses
                if h["status"] == status
            }
            returned_ids = {
                h["hypothesis_id"]
                for h in result.data["hypotheses"]
            }

            assert returned_ids == expected_ids, (
                f"For status '{status}': "
                f"expected {expected_ids}, got {returned_ids}"
            )
