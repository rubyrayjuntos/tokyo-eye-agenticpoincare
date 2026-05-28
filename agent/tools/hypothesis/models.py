"""Hypothesis engine data models.

Pydantic models for the hypothesis lifecycle: Hypothesis, Prediction, Evidence.
Includes JSON serialization/deserialization for API responses and storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis."""

    PROPOSED = "proposed"
    GATHERING = "gathering"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


class Prediction(BaseModel):
    """A testable claim derived from a hypothesis."""

    prediction_id: str = Field(default_factory=lambda: f"pred_{uuid.uuid4().hex[:12]}")
    statement: str
    test_tool: str | None = None
    test_params: dict | None = None
    threshold: str | None = None
    result: str | None = None
    passed: bool | None = None
    tested_at: datetime | None = None


class Evidence(BaseModel):
    """A record of data supporting or contradicting a hypothesis."""

    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    source_tool: str
    source_run_id: str | None = None
    supports: bool
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    description: str
    gathered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Hypothesis(BaseModel):
    """A structured scientific claim about a protein structure."""

    hypothesis_id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:12]}")
    structure_id: str
    statement: str
    mechanism: str | None = None
    predictions: list[Prediction] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    created_by: str = "agent"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json(self) -> str:
        """Serialize to pretty-printed JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> Hypothesis:
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)
