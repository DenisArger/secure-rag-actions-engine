# secure-rag-actions-engine

## English
## Problem
RAG systems in production need deterministic safety controls and action governance, not only free-form LLM responses.
## Solution
This project provides a secure 2-stage RAG orchestration blueprint with policy-driven action routing, validation, and safe fallback handling.
## Tech Stack
- Python
- Pydantic
- pytest
- Structured policy/action orchestration
## Architecture
```text
src/secure_rag_engine/
tests/
pyproject.toml
```
```mermaid
flowchart TD
  A[User request + context] --> B[Stage 1 classification]
  B --> C[Policy engine]
  C --> D[Action routing]
  C --> E[Stage 2 grounded answer]
  E --> F[Validation + safe fallback]
```
## Features
- Two-stage RAG control flow
- Deterministic action policy (`escalate/create_ticket/notify_admin/none`)
- Injection hardening and trust boundaries
- Validation retries and fallback response
- Dispatcher interfaces for action backends
## How to Run
```bash
pip install -e .[dev]
pytest
```

## Русский
## Проблема
Продакшен RAG-системам нужны детерминированные механизмы безопасности и маршрутизации действий, а не только свободный ответ LLM.
## Решение
Проект предоставляет безопасный 2-stage RAG blueprint с policy-driven маршрутизацией действий, валидацией и fallback-логикой.
## Стек
- Python
- Pydantic
- pytest
- Оркестрация policy/action
## Архитектура
```text
src/secure_rag_engine/
tests/
pyproject.toml
```
```mermaid
flowchart TD
  A[Запрос + контекст] --> B[Stage 1 классификация]
  B --> C[Policy engine]
  C --> D[Маршрутизация действий]
  C --> E[Stage 2 grounded answer]
  E --> F[Валидация + safe fallback]
```
## Возможности
- Двухэтапный RAG-пайплайн
- Детерминированная policy маршрутизации действий
- Защита от prompt injection
- Повторы валидации и безопасный fallback
- Интерфейсы dispatch для action backend
## Как запустить
```bash
pip install -e .[dev]
pytest
```
