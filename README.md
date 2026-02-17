# Secure RAG Engine

Production blueprint for a secure 2-stage RAG backend with deterministic action policy.

## What is implemented
- Stage 1 classification (`intent`, risk/compliance signals, context sufficiency)
- Deterministic policy engine for action routing (`escalate > create_ticket > notify_admin > none` fallback by allowed actions)
- Stage 2 strict grounded response with JSON schema enforcement
- Validation retries and safe fallback response
- Prompt-injection hardening and trust-boundary framing
- Async action dispatch interface (+ in-memory/noop/celery adapters)
- Observability fields in structured logs (trace, latency, retries, token usage)
- Phase 2 interfaces scaffold (prompt versioning, A/B, semantic cache, cost router)

## Install

```bash
pip install -e .[dev]
```

## Run tests

```bash
pytest
```

## Minimal usage

```python
from secure_rag_engine import EngineConfig, InMemoryActionDispatcher, SecureRagOrchestrator
from secure_rag_engine.llm import LLMProvider, LLMResult
from secure_rag_engine.models import AiRequest

class DummyProvider(LLMProvider):
    def generate_json(self, **kwargs):
        if kwargs["stage"] == "stage1_classification":
            return LLMResult(content={
                "intent": "qa",
                "is_complaint": False,
                "has_legal_threat": False,
                "has_urgent_risk": False,
                "risk_score": 0.1,
                "context_sufficiency": "sufficient",
                "classification_confidence": "high",
            })
        return LLMResult(content={
            "answer": "Example answer from context.",
            "confidence": "high",
            "action_required": False,
            "action_type": "none",
            "action_payload": {"priority": "low", "reason": "No action required."},
        })

orchestrator = SecureRagOrchestrator(
    llm_provider=DummyProvider(),
    action_dispatcher=InMemoryActionDispatcher(),
    config=EngineConfig(),
)

response = orchestrator.process(
    AiRequest(
        tenant_id="t1",
        conversation_id="c1",
        user_id="u1",
        user_question="What is our refund window?",
        retrieved_chunks=[{"id": "1", "text": "Refunds are allowed within 30 days.", "score": 0.91}],
        allowed_actions=["notify_admin", "create_ticket", "escalate"],
    )
)
print(response.model_dump())
```
