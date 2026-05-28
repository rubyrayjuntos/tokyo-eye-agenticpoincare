"""Hypothesis engine tools for the agent coordinator.

Provides five tools following the ToolResult pattern:
- propose_hypothesis: Create a hypothesis with falsifiability guardrail
- test_hypothesis: Execute predictions via existing tools
- get_hypotheses: List/filter hypotheses
- add_evidence: Manual evidence addition
- evaluate_confidence: Recalculate and update status

All writes go through the Normalizer. Reads use ToolDB.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.tools.dtie.tools import ToolDB, ToolResult
from agent.tools.hypothesis.confidence import (
    apply_decay,
    calculate_confidence,
    determine_status,
)
from agent.tools.hypothesis.models import Evidence, HypothesisStatus, Prediction
from agent.tools.hypothesis.threshold import evaluate_threshold
from science.dtie.common.normalizer_payloads import (
    EvidencePayload,
    HypothesisPredictionPayload,
    HypothesisPayload,
    ProvenanceContext,
    RunType,
    SourceType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: propose_hypothesis
# ---------------------------------------------------------------------------


async def propose_hypothesis(
    structure_id: str,
    statement: str,
    predictions: list[dict[str, Any]],
    mechanism: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Create a new hypothesis with falsifiability guardrail.

    Validates that at least one prediction is provided (falsifiability),
    generates a unique hypothesis_id, sets defaults, and writes through
    the Normalizer.

    Args:
        structure_id: Structure this hypothesis is about.
        statement: The scientific claim.
        predictions: List of prediction dicts with keys:
            - statement (required): What this prediction claims
            - test_tool (optional): Tool to call for testing
            - test_params (optional): Parameters for the test tool
            - threshold (optional): Threshold expression
        mechanism: Optional proposed mechanism.
        db: Database adapter.

    Returns:
        ToolResult with hypothesis_id on success.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    # Falsifiability guardrail: at least one prediction required
    if not predictions:
        return ToolResult(
            success=False,
            message=(
                "Hypothesis rejected: falsifiability requirement not met. "
                "At least one testable prediction must be provided that could "
                "contradict the hypothesis."
            ),
        )

    # Generate IDs
    hypothesis_id = f"hyp_{uuid.uuid4().hex[:12]}"
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    # Build prediction payloads
    prediction_payloads: list[HypothesisPredictionPayload] = []
    for pred in predictions:
        pred_statement = pred.get("statement", "")
        if not pred_statement:
            return ToolResult(
                success=False,
                message="Each prediction must have a non-empty 'statement' field.",
            )
        prediction_payloads.append(
            HypothesisPredictionPayload(
                prediction_id=f"pred_{uuid.uuid4().hex[:12]}",
                statement=pred_statement,
                test_tool=pred.get("test_tool"),
                test_params=pred.get("test_params"),
                threshold=pred.get("threshold"),
            )
        )

    # Build normalizer payload
    payload = HypothesisPayload(
        provenance=ProvenanceContext(
            run_id=run_id,
            structure_id=structure_id,
            model_version="hypothesis_engine_v1",
            pipeline_name="hypothesis_engine",
            run_type=RunType.ANALYSIS,
            source_type=SourceType.DERIVED,
        ),
        hypothesis_id=hypothesis_id,
        structure_id=structure_id,
        statement=statement,
        mechanism=mechanism,
        predictions=prediction_payloads,
        status="proposed",
        confidence=0.5,
        created_by="agent",
    )

    # Write through normalizer
    from data.normalizer.core import Normalizer

    normalizer = Normalizer(db=db, caller_identity="hypothesis_tool")
    try:
        result = await normalizer.normalize_hypothesis(payload)
    except Exception as e:
        logger.error("Failed to write hypothesis: %s", e)
        return ToolResult(
            success=False,
            message=f"Failed to persist hypothesis: {e}",
        )

    return ToolResult(
        success=True,
        data={
            "hypothesis_id": hypothesis_id,
            "structure_id": structure_id,
            "status": "proposed",
            "confidence": 0.5,
            "prediction_count": len(prediction_payloads),
            "prediction_ids": [p.prediction_id for p in prediction_payloads],
        },
        message=(
            f"Hypothesis '{hypothesis_id}' created for {structure_id} "
            f"with {len(prediction_payloads)} prediction(s)"
        ),
    )



# ---------------------------------------------------------------------------
# Tool 2: test_hypothesis
# ---------------------------------------------------------------------------


async def test_hypothesis(
    hypothesis_id: str,
    tool_dispatcher: Any = None,
    db: Any = None,
) -> ToolResult:
    """Test a hypothesis by executing its predictions.

    Loads the hypothesis and predictions from DB, transitions status to
    'gathering', executes each prediction by calling the referenced tool,
    evaluates thresholds, creates evidence from results, recalculates
    confidence, and updates status.

    Args:
        hypothesis_id: The hypothesis to test.
        tool_dispatcher: Callable(tool_name, params) -> result for executing tools.
        db: Database adapter.

    Returns:
        ToolResult with test results summary.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Load hypothesis
    hyp_row = await tool_db.fetch_one(
        "SELECT * FROM hypothesis WHERE hypothesis_id = :hypothesis_id",
        {"hypothesis_id": hypothesis_id},
    )
    if not hyp_row:
        return ToolResult(
            success=False,
            message=f"Hypothesis '{hypothesis_id}' not found",
        )

    # Load predictions
    predictions = await tool_db.fetch_all(
        "SELECT * FROM hypothesis_prediction WHERE hypothesis_id = :hypothesis_id",
        {"hypothesis_id": hypothesis_id},
    )
    if not predictions:
        return ToolResult(
            success=False,
            message=f"No predictions found for hypothesis '{hypothesis_id}'",
        )

    # Transition status to gathering
    structure_id = hyp_row["structure_id"]
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    await db.execute(
        "UPDATE hypothesis SET status = :status, updated_at = :updated_at WHERE hypothesis_id = :hypothesis_id",
        {
            "status": HypothesisStatus.GATHERING.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis_id": hypothesis_id,
        },
    )

    # Execute each prediction
    evidence_records: list[Evidence] = []
    prediction_results: list[dict[str, Any]] = []

    for pred in predictions:
        test_tool = pred.get("test_tool")
        test_params = pred.get("test_params") or {}
        threshold = pred.get("threshold")
        pred_id = pred["prediction_id"]

        # If no tool specified, mark as untestable
        if not test_tool:
            prediction_results.append({
                "prediction_id": pred_id,
                "status": "untestable",
                "reason": "No test_tool specified",
            })
            await db.execute(
                "UPDATE hypothesis_prediction SET result = :result, passed = NULL, tested_at = :tested_at WHERE prediction_id = :prediction_id",
                {
                    "result": "untestable: no test_tool specified",
                    "tested_at": datetime.now(timezone.utc).isoformat(),
                    "prediction_id": pred_id,
                },
            )
            continue

        # Call the referenced tool
        tool_result_value = None
        tool_error = None
        try:
            if tool_dispatcher is not None:
                tool_response = await tool_dispatcher(test_tool, test_params)
                if hasattr(tool_response, "data"):
                    tool_result_value = tool_response.data
                elif isinstance(tool_response, dict):
                    tool_result_value = tool_response
                else:
                    tool_result_value = str(tool_response)
            else:
                tool_error = "No tool_dispatcher provided"
        except Exception as e:
            tool_error = str(e)

        if tool_error:
            prediction_results.append({
                "prediction_id": pred_id,
                "status": "untestable",
                "reason": f"Tool call failed: {tool_error}",
            })
            await db.execute(
                "UPDATE hypothesis_prediction SET result = :result, passed = NULL, tested_at = :tested_at WHERE prediction_id = :prediction_id",
                {
                    "result": f"untestable: {tool_error}",
                    "tested_at": datetime.now(timezone.utc).isoformat(),
                    "prediction_id": pred_id,
                },
            )
            continue

        # Evaluate threshold if provided
        passed = None
        if threshold and tool_result_value is not None:
            try:
                # Extract numeric value from result if it's a dict
                eval_value = tool_result_value
                if isinstance(eval_value, dict):
                    # Try common keys for numeric results
                    for key in ("value", "result", "score", "count", "metric"):
                        if key in eval_value:
                            eval_value = eval_value[key]
                            break
                passed = evaluate_threshold(threshold, eval_value)
            except (ValueError, TypeError) as e:
                prediction_results.append({
                    "prediction_id": pred_id,
                    "status": "untestable",
                    "reason": f"Threshold evaluation failed: {e}",
                })
                await db.execute(
                    "UPDATE hypothesis_prediction SET result = :result, passed = NULL, tested_at = :tested_at WHERE prediction_id = :prediction_id",
                    {
                        "result": f"untestable: threshold eval failed: {e}",
                        "tested_at": datetime.now(timezone.utc).isoformat(),
                        "prediction_id": pred_id,
                    },
                )
                continue

        # Record prediction result
        result_str = str(tool_result_value)[:500]  # Truncate for storage
        await db.execute(
            "UPDATE hypothesis_prediction SET result = :result, passed = :passed, tested_at = :tested_at WHERE prediction_id = :prediction_id",
            {
                "result": result_str,
                "passed": passed,
                "tested_at": datetime.now(timezone.utc).isoformat(),
                "prediction_id": pred_id,
            },
        )

        prediction_results.append({
            "prediction_id": pred_id,
            "status": "tested",
            "passed": passed,
            "result_preview": result_str[:100],
        })

        # Create evidence from this prediction result
        if passed is not None:
            ev = Evidence(
                source_tool=test_tool,
                source_run_id=run_id,
                supports=passed,
                strength=0.5,
                description=f"Prediction '{pred.get('statement', '')}' {'passed' if passed else 'failed'}",
            )
            evidence_records.append(ev)

    # Write evidence through normalizer
    from data.normalizer.core import Normalizer

    normalizer = Normalizer(db=db, caller_identity="hypothesis_tool")

    for ev in evidence_records:
        ev_payload = EvidencePayload(
            provenance=ProvenanceContext(
                run_id=run_id,
                structure_id=structure_id,
                model_version="hypothesis_engine_v1",
                pipeline_name="hypothesis_engine",
                run_type=RunType.ANALYSIS,
                source_type=SourceType.DERIVED,
            ),
            hypothesis_id=hypothesis_id,
            evidence_id=ev.evidence_id,
            source_tool=ev.source_tool,
            source_run_id=ev.source_run_id,
            supports=ev.supports,
            strength=ev.strength,
            description=ev.description,
        )
        try:
            await normalizer.normalize_evidence(ev_payload)
        except Exception as e:
            logger.warning("Failed to write evidence %s: %s", ev.evidence_id, e)

    # Recalculate confidence from all evidence
    all_evidence_rows = await tool_db.fetch_all(
        "SELECT supports, strength FROM hypothesis_evidence WHERE hypothesis_id = :hypothesis_id",
        {"hypothesis_id": hypothesis_id},
    )
    all_evidence = [
        Evidence(
            source_tool="db",
            supports=row["supports"],
            strength=row["strength"],
            description="",
        )
        for row in all_evidence_rows
    ]

    new_confidence = calculate_confidence(all_evidence)
    new_status = determine_status(
        new_confidence, len(all_evidence), HypothesisStatus.GATHERING
    )

    # Update hypothesis
    await db.execute(
        "UPDATE hypothesis SET confidence = :confidence, status = :status, updated_at = :updated_at WHERE hypothesis_id = :hypothesis_id",
        {
            "confidence": new_confidence,
            "status": new_status.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis_id": hypothesis_id,
        },
    )

    return ToolResult(
        success=True,
        data={
            "hypothesis_id": hypothesis_id,
            "status": new_status.value,
            "confidence": new_confidence,
            "predictions_tested": len(prediction_results),
            "predictions_passed": sum(1 for p in prediction_results if p.get("passed") is True),
            "predictions_failed": sum(1 for p in prediction_results if p.get("passed") is False),
            "predictions_untestable": sum(1 for p in prediction_results if p.get("status") == "untestable"),
            "evidence_created": len(evidence_records),
            "results": prediction_results,
        },
        message=(
            f"Tested hypothesis '{hypothesis_id}': "
            f"confidence={new_confidence:.2f}, status={new_status.value}"
        ),
    )



# ---------------------------------------------------------------------------
# Tool 3: get_hypotheses
# ---------------------------------------------------------------------------


async def get_hypotheses(
    structure_id: str | None = None,
    status: str | None = None,
    db: Any = None,
) -> ToolResult:
    """List and filter hypotheses by structure or status.

    Returns hypotheses with their current status, confidence, evidence
    counts, and prediction pass/fail summary.

    Args:
        structure_id: Filter by structure (optional).
        status: Filter by status (optional).
        db: Database adapter.

    Returns:
        ToolResult with list of hypotheses.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    if not structure_id and not status:
        return ToolResult(
            success=False,
            message="Provide at least one filter: structure_id or status",
        )

    # Validate status if provided
    if status:
        valid_statuses = {s.value for s in HypothesisStatus}
        if status not in valid_statuses:
            return ToolResult(
                success=False,
                message=f"Invalid status '{status}'. Valid: {sorted(valid_statuses)}",
            )

    tool_db = ToolDB(db)

    # Build query
    conditions: list[str] = []
    params: dict[str, Any] = {}

    if structure_id:
        conditions.append("h.structure_id = :structure_id")
        params["structure_id"] = structure_id

    if status:
        conditions.append("h.status = :status")
        params["status"] = status

    where_clause = " AND ".join(conditions)

    hypotheses = await tool_db.fetch_all(
        f"""
        SELECT h.hypothesis_id, h.structure_id, h.statement, h.mechanism,
               h.status, h.confidence, h.created_by, h.created_at, h.updated_at
        FROM hypothesis h
        WHERE {where_clause}
        ORDER BY h.updated_at DESC
        """,
        params,
    )

    # Enrich with evidence counts and prediction summaries
    results: list[dict[str, Any]] = []
    for hyp in hypotheses:
        hyp_id = hyp["hypothesis_id"]

        # Evidence counts
        evidence_rows = await tool_db.fetch_all(
            """
            SELECT supports, COUNT(*) as cnt
            FROM hypothesis_evidence
            WHERE hypothesis_id = :hypothesis_id
            GROUP BY supports
            """,
            {"hypothesis_id": hyp_id},
        )
        supporting_count = 0
        contradicting_count = 0
        for row in evidence_rows:
            if row["supports"]:
                supporting_count = row["cnt"]
            else:
                contradicting_count = row["cnt"]

        # Prediction summary
        pred_rows = await tool_db.fetch_all(
            """
            SELECT prediction_id, statement, passed, tested_at
            FROM hypothesis_prediction
            WHERE hypothesis_id = :hypothesis_id
            """,
            {"hypothesis_id": hyp_id},
        )
        predictions_passed = sum(1 for p in pred_rows if p.get("passed") is True)
        predictions_failed = sum(1 for p in pred_rows if p.get("passed") is False)
        predictions_untested = sum(1 for p in pred_rows if p.get("passed") is None)

        results.append({
            **hyp,
            "evidence_supporting": supporting_count,
            "evidence_contradicting": contradicting_count,
            "evidence_total": supporting_count + contradicting_count,
            "predictions": pred_rows,
            "predictions_passed": predictions_passed,
            "predictions_failed": predictions_failed,
            "predictions_untested": predictions_untested,
            "predictions_total": len(pred_rows),
        })

    return ToolResult(
        success=True,
        data={
            "hypotheses": results,
            "count": len(results),
            "filters": {"structure_id": structure_id, "status": status},
        },
        message=f"Found {len(results)} hypothesis(es)",
    )



# ---------------------------------------------------------------------------
# Tool 4: add_evidence
# ---------------------------------------------------------------------------


async def add_evidence(
    hypothesis_id: str,
    source_tool: str,
    supports: bool,
    description: str,
    strength: float = 0.5,
    source_run_id: str | None = None,
    db: Any = None,
) -> ToolResult:
    """Add evidence to an existing hypothesis and recalculate confidence.

    Stores the evidence record through the Normalizer, then recalculates
    confidence and updates hypothesis status.

    Args:
        hypothesis_id: Hypothesis to add evidence to.
        source_tool: Tool or source that produced this evidence.
        supports: True if evidence supports the hypothesis.
        description: Human-readable description.
        strength: Evidence strength weight (0.0 to 1.0).
        source_run_id: Optional run that produced this evidence.
        db: Database adapter.

    Returns:
        ToolResult with updated confidence and status.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Verify hypothesis exists
    hyp_row = await tool_db.fetch_one(
        "SELECT hypothesis_id, structure_id, status FROM hypothesis WHERE hypothesis_id = :hypothesis_id",
        {"hypothesis_id": hypothesis_id},
    )
    if not hyp_row:
        return ToolResult(
            success=False,
            message=f"Hypothesis '{hypothesis_id}' not found",
        )

    structure_id = hyp_row["structure_id"]
    current_status = HypothesisStatus(hyp_row["status"])

    # Generate IDs
    evidence_id = f"ev_{uuid.uuid4().hex[:12]}"
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    # Write evidence through normalizer
    from data.normalizer.core import Normalizer

    normalizer = Normalizer(db=db, caller_identity="hypothesis_tool")

    ev_payload = EvidencePayload(
        provenance=ProvenanceContext(
            run_id=run_id,
            structure_id=structure_id,
            model_version="hypothesis_engine_v1",
            pipeline_name="hypothesis_engine",
            run_type=RunType.ANALYSIS,
            source_type=SourceType.DERIVED,
        ),
        hypothesis_id=hypothesis_id,
        evidence_id=evidence_id,
        source_tool=source_tool,
        source_run_id=source_run_id,
        supports=supports,
        strength=strength,
        description=description,
    )

    try:
        await normalizer.normalize_evidence(ev_payload)
    except Exception as e:
        logger.error("Failed to write evidence: %s", e)
        return ToolResult(
            success=False,
            message=f"Failed to persist evidence: {e}",
        )

    # Recalculate confidence from all evidence
    all_evidence_rows = await tool_db.fetch_all(
        "SELECT supports, strength FROM hypothesis_evidence WHERE hypothesis_id = :hypothesis_id",
        {"hypothesis_id": hypothesis_id},
    )
    all_evidence = [
        Evidence(
            source_tool="db",
            supports=row["supports"],
            strength=row["strength"],
            description="",
        )
        for row in all_evidence_rows
    ]

    new_confidence = calculate_confidence(all_evidence)
    new_status = determine_status(new_confidence, len(all_evidence), current_status)

    # Update hypothesis
    await db.execute(
        "UPDATE hypothesis SET confidence = :confidence, status = :status, updated_at = :updated_at WHERE hypothesis_id = :hypothesis_id",
        {
            "confidence": new_confidence,
            "status": new_status.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis_id": hypothesis_id,
        },
    )

    return ToolResult(
        success=True,
        data={
            "hypothesis_id": hypothesis_id,
            "evidence_id": evidence_id,
            "supports": supports,
            "strength": strength,
            "new_confidence": new_confidence,
            "new_status": new_status.value,
            "total_evidence": len(all_evidence),
        },
        message=(
            f"Evidence added to '{hypothesis_id}': "
            f"{'supports' if supports else 'contradicts'} (strength={strength:.2f}). "
            f"Confidence now {new_confidence:.2f}, status={new_status.value}"
        ),
    )



# ---------------------------------------------------------------------------
# Tool 5: evaluate_confidence
# ---------------------------------------------------------------------------


async def evaluate_confidence(
    hypothesis_id: str,
    db: Any = None,
) -> ToolResult:
    """Recalculate confidence for a hypothesis, applying decay if stale.

    Reloads all evidence, recalculates confidence, applies time-based
    decay if the hypothesis has been in 'gathering' status without new
    evidence for more than 7 days, and updates status.

    Args:
        hypothesis_id: Hypothesis to evaluate.
        db: Database adapter.

    Returns:
        ToolResult with updated confidence and status.
    """
    if db is None:
        return ToolResult(success=False, message="No database connection provided")

    tool_db = ToolDB(db)

    # Load hypothesis
    hyp_row = await tool_db.fetch_one(
        "SELECT * FROM hypothesis WHERE hypothesis_id = :hypothesis_id",
        {"hypothesis_id": hypothesis_id},
    )
    if not hyp_row:
        return ToolResult(
            success=False,
            message=f"Hypothesis '{hypothesis_id}' not found",
        )

    current_status = HypothesisStatus(hyp_row["status"])

    # Load all evidence
    all_evidence_rows = await tool_db.fetch_all(
        "SELECT supports, strength, gathered_at FROM hypothesis_evidence WHERE hypothesis_id = :hypothesis_id ORDER BY gathered_at DESC",
        {"hypothesis_id": hypothesis_id},
    )

    all_evidence = [
        Evidence(
            source_tool="db",
            supports=row["supports"],
            strength=row["strength"],
            description="",
        )
        for row in all_evidence_rows
    ]

    # Calculate base confidence
    new_confidence = calculate_confidence(all_evidence)

    # Apply decay if in gathering status and stale
    decay_applied = False
    days_stale = 0
    if current_status == HypothesisStatus.GATHERING and all_evidence_rows:
        # Find the most recent evidence timestamp
        latest_evidence_at = all_evidence_rows[0].get("gathered_at")
        if latest_evidence_at:
            if isinstance(latest_evidence_at, str):
                latest_evidence_at = datetime.fromisoformat(latest_evidence_at)
            if latest_evidence_at.tzinfo is None:
                latest_evidence_at = latest_evidence_at.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_stale = (now - latest_evidence_at).days
            if days_stale > 7:
                # Apply decay for days beyond the 7-day grace period
                decay_days = days_stale - 7
                new_confidence = apply_decay(new_confidence, decay_days)
                decay_applied = True

    # Determine new status
    new_status = determine_status(new_confidence, len(all_evidence), current_status)

    # Update hypothesis
    await db.execute(
        "UPDATE hypothesis SET confidence = :confidence, status = :status, updated_at = :updated_at WHERE hypothesis_id = :hypothesis_id",
        {
            "confidence": new_confidence,
            "status": new_status.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis_id": hypothesis_id,
        },
    )

    return ToolResult(
        success=True,
        data={
            "hypothesis_id": hypothesis_id,
            "confidence": new_confidence,
            "status": new_status.value,
            "evidence_count": len(all_evidence),
            "decay_applied": decay_applied,
            "days_stale": days_stale,
        },
        message=(
            f"Confidence for '{hypothesis_id}': {new_confidence:.2f} "
            f"(status={new_status.value}, evidence={len(all_evidence)}"
            f"{', decay applied' if decay_applied else ''})"
        ),
    )
