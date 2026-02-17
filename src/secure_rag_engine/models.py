from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Set

from pydantic import BaseModel, Field, validator
from typing_extensions import Literal

ConfidenceLevel = Literal["low", "medium", "high"]
ActionType = Literal["notify_admin", "create_ticket", "escalate", "none"]
ActionPriority = Literal["low", "medium", "high"]
ContextSufficiency = Literal["insufficient", "partial", "sufficient"]

ALLOWED_ACTIONS = set(["notify_admin", "create_ticket", "escalate"])  # type: Set[str]


class RetrievedChunk(BaseModel):
    id: str
    text: str
    source: Optional[str] = None
    score: Optional[float] = None

    class Config:
        extra = "forbid"


class AiRequest(BaseModel):
    tenant_id: str
    conversation_id: str
    user_id: str
    user_question: str
    retrieved_chunks: List[RetrievedChunk]
    allowed_actions: List[Literal["notify_admin", "create_ticket", "escalate"]] = Field(
        default_factory=lambda: ["notify_admin", "create_ticket", "escalate"]
    )

    class Config:
        extra = "forbid"

    @validator("allowed_actions")
    def validate_allowed_actions(cls, value):
        unique = list(dict.fromkeys(value))
        if not unique:
            raise ValueError("allowed_actions must contain at least one action")
        for action in unique:
            if action not in ALLOWED_ACTIONS:
                raise ValueError("Unsupported action '{}'".format(action))
        return unique


class ActionPayload(BaseModel):
    priority: ActionPriority
    reason: str

    class Config:
        extra = "forbid"


class AiResponse(BaseModel):
    answer: str
    confidence: ConfidenceLevel
    action_required: bool
    action_type: ActionType
    action_payload: ActionPayload

    class Config:
        extra = "forbid"


class ClassificationResult(BaseModel):
    intent: str
    is_complaint: bool
    has_legal_threat: bool
    has_urgent_risk: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    context_sufficiency: ContextSufficiency
    classification_confidence: ConfidenceLevel

    class Config:
        extra = "forbid"


class ActionDecision(BaseModel):
    action_required: bool
    action_type: ActionType
    priority: ActionPriority
    reason: str

    class Config:
        extra = "forbid"


class ActionTask(BaseModel):
    tenant_id: str
    conversation_id: str
    user_id: str
    action_type: Literal["notify_admin", "create_ticket", "escalate"]
    priority: ActionPriority
    reason: str
    trace_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"


def fallback_response_from_decision(decision):
    # type: (ActionDecision) -> AiResponse
    return AiResponse(
        answer="INSUFFICIENT_DATA",
        confidence="low",
        action_required=decision.action_required,
        action_type=decision.action_type,
        action_payload=ActionPayload(priority=decision.priority, reason=decision.reason),
    )
