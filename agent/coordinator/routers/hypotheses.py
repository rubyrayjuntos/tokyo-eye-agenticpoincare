"""Hypothesis Engine API router — REST endpoints for hypothesis lifecycle tools.

Exposes 5 endpoints for proposing, testing, querying, adding evidence to,
and evaluating hypotheses. Wraps the existing tool functions in
agent/tools/hypothesis/tools.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.coordinator.deps import get_db
from shared.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/hypotheses", tags=["hypotheses"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class PredictionInput(BaseModel):
    statement: str = Field(..., description="What this prediction claims")
    test_tool: str | None = Field(None, description="Tool to call for testing")
    test_params: dict[str, Any] | None = Field(None, description="Parameters for the test tool")
    threshold: str | None = Field(None, description="Threshold expression (e.g., '> 0.7')")


class ProposeHypothesisRequest(BaseModel):
    structure_id: str = Field(..., description="Structure this hypothesis is about")
    statement: str = Field(..., description="The scientific claim")
    predictions: list[PredictionInput] = Field(default_factory=list, description="Testable predictions")
    mechanism: str | None = Field(None, description="Proposed mechanism")


class AddEvidenceRequest(BaseModel):
    source_tool: str = Field(..., description="Tool or source that produced this evidence")
    supports: bool = Field(..., description="True if evidence supports the hypothesis")
    description: str = Field(..., description="Human-readable description")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Evidence strength weight")
    source_run_id: str | None = Field(None, description="Optional run that produced this evidence")


# ---------------------------------------------------------------------------
# GET /api/hypotheses
# ---------------------------------------------------------------------------


@router.get("")
async def list_hypotheses(
    structure_id: str | None = Query(None, description="Filter by structure"),
    status: str | None = Query(None, description="Filter by status"),
    db=Depends(get_db),
):
    """List hypotheses filtered by structure_id and/or status.

    Returns hypotheses with their current status, confidence, evidence
    counts, and prediction pass/fail summary.
    """
    from agent.tools.hypothesis.tools import get_hypotheses

    result = await get_hypotheses(structure_id=structure_id, status=status, db=db)

    if not result.success:
        return JSONResponse(status_code=400, content={"error": "query_failed", "message": result.message})

    return {
        "hypotheses": result.data.get("hypotheses", []),
        "count": result.data.get("count", 0),
        "filters": result.data.get("filters", {}),
    }


# ---------------------------------------------------------------------------
# POST /api/hypotheses
# ---------------------------------------------------------------------------


@router.post("")
async def propose_hypothesis(request: ProposeHypothesisRequest, db=Depends(get_db)):
    """Propose a new hypothesis with falsifiability guardrail.

    Requires at least one testable prediction. Returns the created
    hypothesis_id and initial state.
    """
    from agent.tools.hypothesis.tools import propose_hypothesis as _propose

    predictions_dicts = [p.model_dump() for p in request.predictions]

    result = await _propose(
        structure_id=request.structure_id,
        statement=request.statement,
        predictions=predictions_dicts,
        mechanism=request.mechanism,
        db=db,
    )

    if not result.success:
        # Falsifiability guardrail returns a specific message
        if "falsifiability" in (result.message or "").lower():
            return JSONResponse(
                status_code=422,
                content={"error": "falsifiability_violation", "message": result.message},
            )
        return JSONResponse(status_code=400, content={"error": "proposal_failed", "message": result.message})

    return {
        "hypothesis_id": result.data.get("hypothesis_id"),
        "structure_id": result.data.get("structure_id"),
        "status": result.data.get("status"),
        "confidence": result.data.get("confidence"),
        "prediction_count": result.data.get("prediction_count"),
        "prediction_ids": result.data.get("prediction_ids", []),
        "run_id": result.data.get("run_id"),
    }


# ---------------------------------------------------------------------------
# POST /api/hypotheses/{id}/test
# ---------------------------------------------------------------------------


@router.post("/{hypothesis_id}/test")
async def test_hypothesis(hypothesis_id: str, db=Depends(get_db)):
    """Execute predictions for a hypothesis and update confidence.

    Transitions status to 'gathering', executes each prediction by calling
    the referenced tool, evaluates thresholds, creates evidence records,
    recalculates confidence, and updates status.
    """
    from agent.tools.hypothesis.tools import test_hypothesis as _test

    result = await _test(hypothesis_id=hypothesis_id, tool_dispatcher=None, db=db)

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "not_found", "message": result.message})

    return {
        "hypothesis_id": result.data.get("hypothesis_id"),
        "status": result.data.get("status"),
        "confidence": result.data.get("confidence"),
        "predictions_tested": result.data.get("predictions_tested"),
        "predictions_passed": result.data.get("predictions_passed"),
        "predictions_failed": result.data.get("predictions_failed"),
        "predictions_untestable": result.data.get("predictions_untestable"),
        "evidence_created": result.data.get("evidence_created"),
        "results": result.data.get("results", []),
    }


# ---------------------------------------------------------------------------
# POST /api/hypotheses/{id}/evidence
# ---------------------------------------------------------------------------


@router.post("/{hypothesis_id}/evidence")
async def add_evidence(hypothesis_id: str, request: AddEvidenceRequest, db=Depends(get_db)):
    """Add evidence to an existing hypothesis and recalculate confidence.

    Stores the evidence record through the Normalizer, then recalculates
    confidence and updates hypothesis status.
    """
    from agent.tools.hypothesis.tools import add_evidence as _add_evidence

    result = await _add_evidence(
        hypothesis_id=hypothesis_id,
        source_tool=request.source_tool,
        supports=request.supports,
        description=request.description,
        strength=request.strength,
        source_run_id=request.source_run_id,
        db=db,
    )

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "not_found", "message": result.message})

    return {
        "hypothesis_id": result.data.get("hypothesis_id"),
        "evidence_id": result.data.get("evidence_id"),
        "supports": result.data.get("supports"),
        "strength": result.data.get("strength"),
        "new_confidence": result.data.get("new_confidence"),
        "new_status": result.data.get("new_status"),
        "total_evidence": result.data.get("total_evidence"),
        "run_id": result.data.get("run_id"),
    }


# ---------------------------------------------------------------------------
# POST /api/hypotheses/{id}/evaluate
# ---------------------------------------------------------------------------


@router.post("/{hypothesis_id}/evaluate")
async def evaluate_confidence(hypothesis_id: str, db=Depends(get_db)):
    """Recalculate confidence for a hypothesis, applying decay if stale.

    Reloads all evidence, recalculates confidence, applies time-based
    decay if the hypothesis has been in 'gathering' status without new
    evidence for more than 7 days, and updates status.
    """
    from agent.tools.hypothesis.tools import evaluate_confidence as _evaluate

    result = await _evaluate(hypothesis_id=hypothesis_id, db=db)

    if not result.success:
        return JSONResponse(status_code=404, content={"error": "not_found", "message": result.message})

    return {
        "hypothesis_id": result.data.get("hypothesis_id"),
        "confidence": result.data.get("confidence"),
        "status": result.data.get("status"),
        "evidence_count": result.data.get("evidence_count"),
        "decay_applied": result.data.get("decay_applied"),
        "days_stale": result.data.get("days_stale"),
    }
