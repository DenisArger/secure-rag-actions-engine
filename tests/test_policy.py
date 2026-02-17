from secure_rag_engine.models import ClassificationResult
from secure_rag_engine.policy import decide_action


def _classification(complaint=False, legal=False, urgent=False):
    return ClassificationResult(
        intent="qa",
        is_complaint=complaint,
        has_legal_threat=legal,
        has_urgent_risk=urgent,
        risk_score=0.1,
        context_sufficiency="partial",
        classification_confidence="medium",
    )


def test_policy_precedence_escalate_over_complaint():
    decision = decide_action(
        _classification(complaint=True, legal=True),
        ["notify_admin", "create_ticket", "escalate"],
    )
    assert decision.action_type == "escalate"
    assert decision.priority == "high"


def test_policy_no_action_when_no_signals():
    decision = decide_action(_classification(), ["notify_admin", "create_ticket", "escalate"])
    assert decision.action_type == "none"
    assert decision.action_required is False
