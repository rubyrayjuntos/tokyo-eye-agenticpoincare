"""Contradiction detection for the hypothesis engine.

After new pipeline results are written for a structure, this module checks
active hypotheses (status 'supported' or 'gathering') and re-evaluates
their predictions against the new data. If a previously-passing prediction
now fails, contradicting evidence is added automatically.

Requirements: 6.1, 6.2
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.tools.hypothesis.confidence import calculate_confidence, determine_status
from agent.tools.hypothesis.models import Evidence, HypothesisStatus
from agent.tools.hypothesis.threshold import evaluate_threshold
from science.dtie.common.normalizer_payloads import (
    EvidencePayload,
    ProvenanceContext,
    RunType,
    SourceType,
)

logger = logging.getLogger(__name__)


async def check_contradictions(
    structure_id: str,
    tool_dispatcher: Any = None,
    db: Any = None,
) -> list[dict[str, Any]]:
    """Check active hypotheses for contradictions after new pipeline results.

    This function:
    1. Finds all active hypotheses (status 'supported' or 'gathering') for the structure
    2. For each hypothesis, finds predictions that previously passed
    3. Re-evaluates those predictions using the tool_dispatcher
    4. If a previously-passing prediction now fails, adds contradicting evidence
    5. Recalculates confidence and updates status

    Args:
        structure_id: The structure that received new pipeline results.
        tool_dispatcher: Callable(tool_name, params) -> result for executing tools.
        db: Database adapter.

    Returns:
        List of contradiction records, each containing:
        - hypothesis_id: The affected hypothesis
        - prediction_id: The prediction that flipped
        - description: Human-readable description of the contradiction
        - new_confidence: Updated confidence after adding contradicting evidence
        - new_status: Updated status
    """
    if db is None:
        return []

    if tool_dispatcher is None:
        logger.warning("No tool_dispatcher provided for contradiction check")
        return []

    from agent.tools.dtie.tools import ToolDB

    tool_db = ToolDB(db)

    # 1. Find active hypotheses for this structure
    active_hypotheses = await tool_db.fetch_all(
        """
        SELECT hypothesis_id, status, confidence
        FROM hypothesis
        WHERE structure_id = :structure_id
          AND status IN ('supported', 'gathering')
        """,
        {"structure_id": structure_id},
    )

    if not active_hypotheses:
        return []

    contradictions: list[dict[str, Any]] = []
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    for hyp in active_hypotheses:
        hypothesis_id = hyp["hypothesis_id"]

        # 2. Find predictions that previously passed
        passed_predictions = await tool_db.fetch_all(
            """
            SELECT prediction_id, statement, test_tool, test_params, threshold
            FROM hypothesis_prediction
            WHERE hypothesis_id = :hypothesis_id
              AND passed = TRUE
            """,
            {"hypothesis_id": hypothesis_id},
        )

        if not passed_predictions:
            continue

        # 3. Re-evaluate each previously-passing prediction
        for pred in passed_predictions:
            test_tool = pred.get("test_tool")
            test_params = pred.get("test_params") or {}
            threshold = pred.get("threshold")
            pred_id = pred["prediction_id"]

            # Skip if no tool or threshold to evaluate
            if not test_tool or not threshold:
                continue

            # Call the tool with current data
            try:
                tool_response = await tool_dispatcher(test_tool, test_params)
                if hasattr(tool_response, "data"):
                    tool_result_value = tool_response.data
                elif isinstance(tool_response, dict):
                    tool_result_value = tool_response
                else:
                    tool_result_value = str(tool_response)
            except Exception as e:
                logger.warning(
                    "Tool call failed during contradiction check for %s: %s",
                    pred_id, e,
                )
                continue

            # Evaluate threshold
            try:
                eval_value = tool_result_value
                if isinstance(eval_value, dict):
                    for key in ("value", "result", "score", "count", "metric"):
                        if key in eval_value:
                            eval_value = eval_value[key]
                            break
                now_passes = evaluate_threshold(threshold, eval_value)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Threshold evaluation failed during contradiction check for %s: %s",
                    pred_id, e,
                )
                continue

            # 4. If previously passed but now fails → contradiction detected
            if not now_passes:
                description = (
                    f"Contradiction detected: prediction '{pred.get('statement', '')}' "
                    f"previously passed but now fails against new data "
                    f"(threshold: {threshold}, result: {eval_value})"
                )

                # Update prediction status
                await db.execute(
                    "UPDATE hypothesis_prediction SET passed = :passed, result = :result, tested_at = :tested_at WHERE prediction_id = :prediction_id",
                    {
                        "passed": False,
                        "result": f"contradiction: was passing, now fails ({eval_value})",
                        "tested_at": datetime.now(timezone.utc).isoformat(),
                        "prediction_id": pred_id,
                    },
                )

                # Add contradicting evidence through normalizer
                evidence_id = f"ev_{uuid.uuid4().hex[:12]}"

                from data.normalizer.core import Normalizer

                normalizer = Normalizer(db=db, caller_identity="contradiction_detector")

                ev_payload = EvidencePayload(
                    provenance=ProvenanceContext(
                        run_id=run_id,
                        structure_id=structure_id,
                        model_version="hypothesis_engine_v1",
                        pipeline_name="contradiction_detection",
                        run_type=RunType.ANALYSIS,
                        source_type=SourceType.DERIVED,
                    ),
                    hypothesis_id=hypothesis_id,
                    evidence_id=evidence_id,
                    source_tool=test_tool,
                    source_run_id=run_id,
                    supports=False,
                    strength=0.7,
                    description=description,
                )

                try:
                    await normalizer.normalize_evidence(ev_payload)
                except Exception as e:
                    logger.error(
                        "Failed to write contradiction evidence for %s: %s",
                        hypothesis_id, e,
                    )
                    continue

                # 5. Recalculate confidence
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
                current_status = HypothesisStatus(hyp["status"])
                new_status = determine_status(
                    new_confidence, len(all_evidence), current_status
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

                contradictions.append({
                    "hypothesis_id": hypothesis_id,
                    "prediction_id": pred_id,
                    "prediction_statement": pred.get("statement", ""),
                    "description": description,
                    "evidence_id": evidence_id,
                    "new_confidence": new_confidence,
                    "new_status": new_status.value,
                })

    if contradictions:
        logger.info(
            "Contradiction check for structure %s: found %d contradiction(s)",
            structure_id, len(contradictions),
        )

    return contradictions
