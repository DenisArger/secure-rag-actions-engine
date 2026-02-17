from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Protocol


@dataclass(frozen=True)
class PromptVersion:
    prompt_key: str
    version: str
    status: str
    created_by: str
    body: str


class PromptVersionStore(Protocol):
    def get_active(self, prompt_key):
        # type: (str) -> PromptVersion
        """Return active prompt version for key."""


class AbAllocator(Protocol):
    def resolve_variant(self, tenant_id, experiment_key):
        # type: (str, str) -> str
        """Return variant label for tenant cohort routing."""


class SemanticCache(Protocol):
    def get(self, tenant_id, question):  # noqa: ANN201 - interface placeholder
        # type: (str, str)
        """Return cached response by semantic similarity with tenant isolation."""

    def put(self, tenant_id, question, response):  # noqa: ANN001
        # type: (str, str, object) -> None
        """Store response with TTL and tenant boundary."""


class CostRouter(Protocol):
    def choose_model(self, risk_score, context_sufficiency):
        # type: (float, str) -> str
        """Select model tier using risk + sufficiency signals."""
