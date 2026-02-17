from __future__ import annotations

import re

from secure_rag_engine.models import ActionDecision, ClassificationResult

_LEGAL_THREAT_PATTERNS = [
    r"\blawyer\b",
    r"\battorney\b",
    r"\bsue\b",
    r"\blegal action\b",
    r"\bcourt\b",
    r"\bregulator\b",
]

_URGENT_RISK_PATTERNS = [
    r"\burgent\b",
    r"\bemergency\b",
    r"\brisk\b",
    r"\bunsafe\b",
    r"\bsecurity incident\b",
    r"\bbreach\b",
]

_COMPLAINT_PATTERNS = [
    r"\bcomplaint\b",
    r"\bnot working\b",
    r"\bterrible\b",
    r"\bfrustrated\b",
    r"\bissue\b",
    r"\bbroken\b",
]


def augment_with_heuristics(classification, user_question):
    # type: (ClassificationResult, str) -> ClassificationResult
    """Merge LLM classification with conservative lexical heuristics."""

    lower = user_question.lower()
    complaint_hit = any(re.search(pattern, lower) for pattern in _COMPLAINT_PATTERNS)
    legal_hit = any(re.search(pattern, lower) for pattern in _LEGAL_THREAT_PATTERNS)
    urgent_hit = any(re.search(pattern, lower) for pattern in _URGENT_RISK_PATTERNS)

    return classification.copy(
        update={
            "is_complaint": classification.is_complaint or complaint_hit,
            "has_legal_threat": classification.has_legal_threat or legal_hit,
            "has_urgent_risk": classification.has_urgent_risk or urgent_hit,
            "risk_score": max(
                classification.risk_score,
                0.9 if legal_hit else 0.0,
                0.85 if urgent_hit else 0.0,
                0.6 if complaint_hit else 0.0,
            ),
        }
    )


def decide_action(classification, allowed_actions):
    # type: (ClassificationResult, list[str]) -> ActionDecision
    """Deterministic policy with fixed precedence and allowed-actions fallback."""

    allowed = set(allowed_actions)

    if classification.has_legal_threat or classification.has_urgent_risk:
        action = _first_allowed(["escalate", "create_ticket", "notify_admin"], allowed)
        if action == "none":
            return ActionDecision(
                action_required=False,
                action_type="none",
                priority="low",
                reason="Risk signal detected but no allowed action configured.",
            )
        return ActionDecision(
            action_required=True,
            action_type=action,
            priority="high",
            reason="Legal threat or urgent risk detected.",
        )

    if classification.is_complaint:
        action = _first_allowed(["create_ticket", "notify_admin"], allowed)
        if action == "none":
            return ActionDecision(
                action_required=False,
                action_type="none",
                priority="low",
                reason="Complaint detected but no allowed action configured.",
            )
        return ActionDecision(
            action_required=True,
            action_type=action,
            priority="medium",
            reason="Complaint detected.",
        )

    return ActionDecision(
        action_required=False,
        action_type="none",
        priority="low",
        reason="No action required.",
    )


def _first_allowed(order, allowed):
    # type: (list[str], set[str]) -> str
    for action in order:
        if action in allowed:
            return action
    return "none"
