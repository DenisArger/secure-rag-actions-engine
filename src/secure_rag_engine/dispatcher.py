from __future__ import annotations

from typing_extensions import Protocol

from secure_rag_engine.models import ActionTask


class ActionDispatcher(Protocol):
    def dispatch(self, task):
        # type: (ActionTask) -> None
        """Enqueue action task for async execution."""


class NoopActionDispatcher:
    def dispatch(self, task):
        # type: (ActionTask) -> None
        del task


class InMemoryActionDispatcher:
    """Useful for tests and local development."""

    def __init__(self):
        self.tasks = []  # type: list[ActionTask]

    def dispatch(self, task):
        # type: (ActionTask) -> None
        self.tasks.append(task)


class CeleryActionDispatcher:
    """Adapter around a Celery task callable accepting keyword payload."""

    def __init__(self, celery_task):
        self._celery_task = celery_task

    def dispatch(self, task):
        # type: (ActionTask) -> None
        self._celery_task.delay(**task.dict())
