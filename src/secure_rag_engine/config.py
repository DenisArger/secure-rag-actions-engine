from __future__ import annotations

from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    top_k_chunks: int = Field(default=6, ge=1, le=100)
    max_context_chars: int = Field(default=12000, ge=1000)
    max_question_chars: int = Field(default=2000, ge=50)

    stage1_timeout_seconds: float = Field(default=4.0, gt=0)
    stage2_timeout_seconds: float = Field(default=8.0, gt=0)

    stage1_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stage2_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    validation_retries: int = Field(default=2, ge=0, le=5)
    provider_retries: int = Field(default=1, ge=0, le=5)

    # Phase 2 feature flags (off by default for MVP)
    prompt_versioning_enabled: bool = False
    ab_testing_enabled: bool = False
    semantic_cache_enabled: bool = False
    cost_router_enabled: bool = False

    class Config:
        extra = "forbid"
