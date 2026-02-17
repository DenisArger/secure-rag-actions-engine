from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union
from typing_extensions import Protocol


class LLMProviderError(RuntimeError):
    """Base LLM provider error."""


class ProviderTimeoutError(LLMProviderError):
    """Raised when provider request times out."""


class ProviderServerError(LLMProviderError):
    """Raised when provider returns a retriable server-side error."""


@dataclass(frozen=True)
class LLMResult:
    content: Union[str, Dict[str, Any]]
    prompt_tokens: int = None
    completion_tokens: int = None
    model: str = None


class LLMProvider(Protocol):
    def generate_json(
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
        """Return model output expected to be JSON (string or already parsed dict)."""
