from __future__ import annotations

from typing import Dict, List

from secure_rag_engine.models import RetrievedChunk


def sanitize_text(value):
    # type: (str) -> str
    """Strip control characters while keeping basic whitespace."""

    cleaned = []
    for ch in value:
        code = ord(ch)
        if ch in {"\n", "\r", "\t"} or code >= 32:
            cleaned.append(ch)
    return "".join(cleaned).strip()


def apply_request_guards(
    *,
    user_question,  # type: str
    retrieved_chunks,  # type: List[RetrievedChunk]
    max_question_chars,  # type: int
    top_k_chunks,  # type: int
    max_context_chars,  # type: int
):
    # type: (...) -> tuple[str, List[RetrievedChunk], int]
    """Sanitize user input and enforce prompt-size limits."""

    safe_question = sanitize_text(user_question)[:max_question_chars]

    ordered = sorted(
        retrieved_chunks,
        key=lambda chunk: (chunk.score is not None, chunk.score if chunk.score is not None else -1e12),
        reverse=True,
    )
    limited = ordered[:top_k_chunks]

    context_chars = 0
    guarded_chunks = []  # type: List[RetrievedChunk]
    for chunk in limited:
        safe_text = sanitize_text(chunk.text)
        remaining = max_context_chars - context_chars
        if remaining <= 0:
            break
        clipped_text = safe_text[:remaining]
        if not clipped_text:
            continue
        guarded_chunks.append(
            chunk.copy(
                update={
                    "text": clipped_text,
                    "source": sanitize_text(chunk.source) if chunk.source else None,
                }
            )
        )
        context_chars += len(clipped_text)

    return safe_question, guarded_chunks, context_chars


def context_has_conflicts(chunks):
    # type: (List[RetrievedChunk]) -> bool
    """Conservative conflict check: same chunk id with different payload."""

    seen = {}  # type: Dict[str, str]
    for chunk in chunks:
        if chunk.id in seen and seen[chunk.id] != chunk.text:
            return True
        seen[chunk.id] = chunk.text
    return False
