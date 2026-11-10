"""Async chat client for the organizer-provided model (OPENAI_* environment only).

- One shared AsyncOpenAI client, no SDK retries (retries are explicit and budgeted here).
- Every call is logged as metadata only: kind, status, latency, attempt, error class.
"""
import asyncio
import time
from typing import Callable, List, Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from . import config, logs

T = TypeVar("T", bound=BaseModel)

# Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
_client = AsyncOpenAI(timeout=config.REQUEST_DEADLINE_S, max_retries=0)


class EmptyCompletion(Exception):
    """The model returned no text."""


class TruncatedCompletion(EmptyCompletion):
    """No text and finish_reason == "length": the token budget was used up (often by reasoning)."""


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
        if not text:
            raise TruncatedCompletion if getattr(choice, "finish_reason", None) == "length" else EmptyCompletion
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
        error = type(exc).__name__
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
