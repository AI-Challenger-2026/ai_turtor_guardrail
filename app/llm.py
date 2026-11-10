"""Async chat client for the organizer-provided model (OPENAI_* environment only).

- One shared AsyncOpenAI client, no SDK retries (retries are explicit and budgeted here).
- Every call is logged as metadata only: kind, status, latency, attempt, error class.
"""
import asyncio
import time
from typing import Callable, List, Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from . import config, logs

T = TypeVar("T", bound=BaseModel)

# Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
_client = AsyncOpenAI(timeout=config.REQUEST_DEADLINE_S, max_retries=0)
# Embeddings use only the EMBEDDING_* settings; no client at all if they are missing.
_embed_client = (AsyncOpenAI(api_key=config.EMBED_API_KEY, base_url=config.EMBED_BASE_URL,
                             timeout=config.EMBED_STARTUP_TIMEOUT_S, max_retries=0)
                 if config.EMBED_ENABLED else None)


class EmptyCompletion(Exception):
    """The model returned no text."""


class TruncatedCompletion(EmptyCompletion):
    """finish_reason == "length" with no text (or, in JSON mode, any text): the token budget ran out."""


def _error_kind(exc: Exception) -> str:
    """Exception class name; for validation errors also the failing field names (never values)."""
    if isinstance(exc, ValidationError):
        fields = sorted({str(e["loc"][0]) if e.get("loc") else "-" for e in exc.errors()})
        return f"ValidationError[{','.join(fields)}]"
    return type(exc).__name__


async def _call(kind: str, messages: List[dict], *, effort: str, max_tokens: int, timeout: float,
                attempt: int, json_mode: bool, parse: Callable[[str], object]):
    t0 = time.perf_counter()
    status, error = "ok", None
    try:
        if timeout <= 0:
            raise asyncio.TimeoutError
        kwargs = dict(model=config.CHAT_MODEL, messages=messages, max_completion_tokens=max_tokens,
                      extra_body={"reasoning_effort": effort}, timeout=timeout)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await _client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        truncated = getattr(choice, "finish_reason", None) == "length"
        if not text or (json_mode and truncated):
            # A cut-off JSON object is reported as truncated rather than as a validation error.
            raise TruncatedCompletion if truncated else EmptyCompletion
        try:
            return parse(text)
        except Exception:
            status = "invalid"
            raise
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as exc:
        if status == "ok":
            status = "error"
        error = _error_kind(exc)
        raise
    finally:
        logs.llm_line(kind=kind, status=status, latency_ms=round((time.perf_counter() - t0) * 1000),
                      attempt=attempt, error=error)


async def chat_text(kind: str, messages: List[dict], *, effort: str, max_tokens: int,
                    timeout: float) -> Optional[str]:
    """One attempt; returns the stripped text or None on any failure."""
    try:
        return await _call(kind, messages, effort=effort, max_tokens=max_tokens, timeout=timeout,
                           attempt=1, json_mode=False, parse=lambda s: s)
    except Exception:
        return None


async def embed(kind: str, texts: List[str], *, timeout: float, attempt: int = 1) -> Optional[List[List[float]]]:
    """Embed `texts` (one request). Returns one vector per text, or None on any failure."""
    if _embed_client is None or not texts:
        return None
    t0 = time.perf_counter()
    status, error = "ok", None
    try:
        if timeout <= 0:
            raise asyncio.TimeoutError
        resp = await asyncio.wait_for(
            _embed_client.embeddings.create(model=config.EMBED_MODEL, input=texts, timeout=timeout),
            timeout=timeout)
        data = sorted(resp.data, key=lambda d: d.index)
        if len(data) != len(texts) or any(not d.embedding for d in data):
            raise EmptyCompletion
        return [list(d.embedding) for d in data]
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    except Exception as exc:
        status, error = "error", _error_kind(exc)
        return None
    finally:
        logs.llm_line(kind=kind, status=status, latency_ms=round((time.perf_counter() - t0) * 1000),
                      attempt=attempt, error=error)


async def chat_json(kind: str, messages: List[dict], schema: Type[T], *, effort: str, max_tokens: int,
                    timeout: Callable[[], float], retries: int = 1) -> Optional[T]:
    """JSON-mode call validated against `schema`; retried on any failure. None if all attempts fail.

    `timeout` is a callable so each attempt gets a fresh, budget-capped value.
    """
    for attempt in range(1, retries + 2):
        try:
            return await _call(kind, messages, effort=effort, max_tokens=max_tokens, timeout=timeout(),
                               attempt=attempt, json_mode=True, parse=schema.model_validate_json)
        except Exception:
            continue
    return None
