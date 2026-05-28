---
inclusion: fileMatch
fileMatchPattern: "**/*.py"
---

# Coding Standards

## Python Style
- Python 3.11+, use `from __future__ import annotations`
- Type hints on all function signatures
- Pydantic for data validation, dataclasses for internal state
- `ruff` for linting (line-length=100, select E/F/I/UP)
- `mypy --strict` for type checking

## Async Patterns
- CPU-bound work (torch, numpy, scipy) → `asyncio.to_thread`
- Model loading → protected by `asyncio.Lock`
- DB connections → from pool via `async with get_connection()`
- LLM calls → wrapped with `asyncio.wait_for(timeout=...)`

## Database
- Named params use `:name` style (converted to `%(name)s` by DBAdapter)
- Never interpolate user input into SQL — use allowlists for column names
- All writes through Normalizer, all reads through governed views or fact tables
- Idempotent upserts via `ON CONFLICT` on natural keys

## Error Handling
- Custom exceptions with context (run_id, structure_id)
- Never expose raw exceptions to users in production
- Audit logging is fire-and-forget (must not block primary path)
- GPU OOM → clear cache, raise clear error

## Testing
- Unit tests: `pytest tests/ -m "not integration"`
- Integration tests: `pytest tests/ -m integration` (requires DB)
- Use `QueryValidatingMockDB` from conftest for new tool tests
- Mark integration tests with `@pytest.mark.integration`
