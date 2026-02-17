from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Union
from uuid import uuid4

from pydantic import ValidationError

from secure_rag_engine.config import EngineConfig
from secure_rag_engine.dispatcher import ActionDispatcher, NoopActionDispatcher
from secure_rag_engine.llm import LLMProvider, LLMResult, ProviderServerError, ProviderTimeoutError
from secure_rag_engine.models import (
    ActionPayload,
    ActionTask,
    AiRequest,
    AiResponse,
    ClassificationResult,
    fallback_response_from_decision,
)
from secure_rag_engine.policy import augment_with_heuristics, decide_action
from secure_rag_engine.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    RESPONSE_SYSTEM_PROMPT,
    build_classification_user_prompt,
    build_repair_user_prompt,
    build_response_user_prompt,
)
from secure_rag_engine.sanitization import apply_request_guards, context_has_conflicts


@dataclass
class _RunMetrics:
    stage1_ms: int = 0
    stage2_ms: int = 0
    total_ms: int = 0
    validation_failures: int = 0
    stage2_validation_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SecureRagOrchestrator:
    def __init__(
        self,
        *,
        llm_provider,  # type: LLMProvider
        action_dispatcher=None,  # type: ActionDispatcher
        config=None,  # type: EngineConfig
        logger=None,  # type: logging.Logger
    ):
        # type: (...) -> None
        self._llm_provider = llm_provider
        self._dispatcher = action_dispatcher or NoopActionDispatcher()
        self._config = config or EngineConfig()
        self._logger = logger or logging.getLogger("secure_rag_engine")

    def process(self, request):
        # type: (AiRequest) -> AiResponse
        trace_id = uuid4().hex
        metrics = _RunMetrics()
        start_total = perf_counter()

        guarded_question, guarded_chunks, _context_chars = apply_request_guards(
            user_question=request.user_question,
            retrieved_chunks=request.retrieved_chunks,
            max_question_chars=self._config.max_question_chars,
            top_k_chunks=self._config.top_k_chunks,
            max_context_chars=self._config.max_context_chars,
        )
        guarded_request = request.copy(
            update={"user_question": guarded_question, "retrieved_chunks": guarded_chunks}
        )

        stage1_start = perf_counter()
        classification = self._run_stage1(guarded_request, trace_id, metrics)
        classification = augment_with_heuristics(classification, guarded_request.user_question)
        metrics.stage1_ms = int((perf_counter() - stage1_start) * 1000)

        action_decision = decide_action(classification, guarded_request.allowed_actions)

        stage2_start = perf_counter()
        response = self._run_stage2_with_validation_retry(
            guarded_request=guarded_request,
            action_decision=action_decision,
            trace_id=trace_id,
            metrics=metrics,
        )
        metrics.stage2_ms = int((perf_counter() - stage2_start) * 1000)

        response = self._finalize_response(
            response=response,
            classification=classification,
            action_decision=action_decision,
            chunks=guarded_request.retrieved_chunks,
        )

        if response.action_required and response.action_type != "none":
            task = ActionTask(
                tenant_id=guarded_request.tenant_id,
                conversation_id=guarded_request.conversation_id,
                user_id=guarded_request.user_id,
                action_type=response.action_type,
                priority=response.action_payload.priority,
                reason=response.action_payload.reason,
                trace_id=trace_id,
            )
            self._dispatcher.dispatch(task)

        metrics.total_ms = int((perf_counter() - start_total) * 1000)
        self._logger.info(
            "[secure-rag] trace_id=%s tenant_id=%s stage1_ms=%s stage2_ms=%s total_ms=%s "
            "validation_failures=%s stage2_validation_retries=%s prompt_tokens=%s completion_tokens=%s action_type=%s",
            trace_id,
            guarded_request.tenant_id,
            metrics.stage1_ms,
            metrics.stage2_ms,
            metrics.total_ms,
            metrics.validation_failures,
            metrics.stage2_validation_retries,
            metrics.prompt_tokens,
            metrics.completion_tokens,
            response.action_type,
        )

        return response

    def _run_stage1(self, request, trace_id, metrics):
        # type: (AiRequest, str, _RunMetrics) -> ClassificationResult
        user_prompt = build_classification_user_prompt(request.user_question)

        try:
            result = self._call_provider_with_retry(
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                timeout_seconds=self._config.stage1_timeout_seconds,
                temperature=self._config.stage1_temperature,
                stage="stage1_classification",
                trace_id=trace_id,
            )
            self._accumulate_usage(metrics, result)
            payload = _coerce_json_dict(result.content)
            return ClassificationResult.parse_obj(payload)
        except Exception as exc:  # noqa: BLE001 - degrade safely
            self._logger.warning(
                "[secure-rag] trace_id=%s stage1_fallback reason=%s",
                trace_id,
                str(exc),
            )
            return ClassificationResult(
                intent="unknown",
                is_complaint=False,
                has_legal_threat=False,
                has_urgent_risk=False,
                risk_score=0.0,
                context_sufficiency="partial",
                classification_confidence="low",
            )

    def _run_stage2_with_validation_retry(self, *, guarded_request, action_decision, trace_id, metrics):
        # type: (AiRequest, Any, str, _RunMetrics) -> AiResponse
        base_prompt = build_response_user_prompt(
            user_question=guarded_request.user_question,
            retrieved_chunks=guarded_request.retrieved_chunks,
            action_decision=action_decision,
            allowed_actions=guarded_request.allowed_actions,
        )

        current_prompt = base_prompt
        last_output = ""

        for attempt in range(self._config.validation_retries + 1):
            try:
                result = self._call_provider_with_retry(
                    system_prompt=RESPONSE_SYSTEM_PROMPT,
                    user_prompt=current_prompt,
                    timeout_seconds=self._config.stage2_timeout_seconds,
                    temperature=self._config.stage2_temperature,
                    stage="stage2_response",
                    trace_id=trace_id,
                )
                self._accumulate_usage(metrics, result)
                last_output = _render_content(result.content)

                payload = _coerce_json_dict(result.content)
                return AiResponse.parse_obj(payload)
            except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
                metrics.validation_failures += 1
                if attempt >= self._config.validation_retries:
                    break
                metrics.stage2_validation_retries += 1
                current_prompt = build_repair_user_prompt(
                    invalid_output=last_output,
                    error_message=str(exc),
                    base_prompt=base_prompt,
                )
            except (ProviderTimeoutError, ProviderServerError) as exc:
                # Provider retries already exhausted in _call_provider_with_retry.
                self._logger.warning(
                    "[secure-rag] trace_id=%s stage2_provider_failure reason=%s",
                    trace_id,
                    str(exc),
                )
                break

        return fallback_response_from_decision(action_decision)

    def _call_provider_with_retry(
        self,
        *,
        system_prompt,  # type: str
        user_prompt,  # type: str
        timeout_seconds,  # type: float
        temperature,  # type: float
        stage,  # type: str
        trace_id,  # type: str
    ):
        # type: (...) -> LLMResult
        for attempt in range(self._config.provider_retries + 1):
            try:
                return self._llm_provider.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout_seconds=timeout_seconds,
                    temperature=temperature,
                    stage=stage,
                    trace_id=trace_id,
                )
            except (ProviderTimeoutError, ProviderServerError):
                if attempt >= self._config.provider_retries:
                    raise
                self._logger.warning(
                    "[secure-rag] trace_id=%s stage=%s provider_retry=%s",
                    trace_id,
                    stage,
                    attempt + 1,
                )

        raise RuntimeError("Unreachable provider retry state")

    def _finalize_response(self, *, response, classification, action_decision, chunks):
        # type: (AiResponse, ClassificationResult, Any, List[Any]) -> AiResponse
        answer = response.answer.strip() or "INSUFFICIENT_DATA"
        confidence = self._normalize_confidence(
            original=response.confidence,
            answer=answer,
            context_sufficiency=classification.context_sufficiency,
            has_conflicts=context_has_conflicts(chunks),
        )

        return response.copy(
            update={
                "answer": answer,
                "confidence": confidence,
                "action_required": action_decision.action_required,
                "action_type": action_decision.action_type,
                "action_payload": ActionPayload(
                    priority=action_decision.priority,
                    reason=action_decision.reason,
                ),
            }
        )

    @staticmethod
    def _normalize_confidence(*, original, answer, context_sufficiency, has_conflicts):
        # type: (str, str, str, bool) -> str
        if answer == "INSUFFICIENT_DATA" or context_sufficiency == "insufficient":
            return "low"
        if context_sufficiency == "sufficient" and not has_conflicts:
            return "high"
        if original in {"low", "medium", "high"}:
            return "medium"
        return "medium"

    @staticmethod
    def _accumulate_usage(metrics, result):
        # type: (_RunMetrics, LLMResult) -> None
        if result.prompt_tokens:
            metrics.prompt_tokens += result.prompt_tokens
        if result.completion_tokens:
            metrics.completion_tokens += result.completion_tokens


def _coerce_json_dict(content):
    # type: (Union[str, Dict[str, Any]]) -> Dict[str, Any]
    if isinstance(content, dict):
        return content
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise TypeError("LLM output must decode to a JSON object")
    return parsed


def _render_content(content):
    # type: (Union[str, Dict[str, Any]]) -> str
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True)
