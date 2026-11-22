"""Calls to the competition model (OPENAI_* environment). Prompts and completions are never logged."""
import asyncio
import json
import os
import time
from typing import Optional

from openai import AsyncOpenAI

from . import logs

MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-5-nano")
# No new attempt with less than this many seconds left.
MIN_ATTEMPT = 1.5

_client = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(timeout=25, max_retries=0)
    return _client


class DeadlineExceeded(Exception):
    pass


async def _once(kwargs, timeout: float) -> str:
    r = await asyncio.wait_for(client().chat.completions.create(**kwargs), timeout=timeout)
    return (r.choices[0].message.content or "").strip()


def _valid(text: str, json_mode: bool) -> str:
    if not text:
        raise ValueError("empty completion")
    if json_mode:
        json.loads(text)
    return text


# If the first call is slow, send the same request once more and use the first success.
async def _hedged(kwargs, t: float, hedge_after: Optional[float], json_mode: bool,
                  kind: str = "llm", rnd: int = 1) -> str:
    async def attempt(tt, hedge=0):
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            text = await _once(kwargs, tt)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as e:
            status, err = "error", type(e).__name__
            raise
        else:
            try:
                return _valid(text, json_mode)
            except Exception as e:
                status, err = "invalid", type(e).__name__
                raise
        finally:
            logs.llm_line(kind=kind, status=status, latency_ms=round((time.perf_counter() - t0) * 1000),
                          attempt=rnd, hedge=hedge, error=err)

    made = []

    def spawn(tt, hedge=0):
        task = asyncio.create_task(attempt(tt, hedge))
        made.append(task)
        return task

    try:
        first = spawn(t)
        if not hedge_after or t - hedge_after < MIN_ATTEMPT:
            return await first
        done, _ = await asyncio.wait({first}, timeout=hedge_after)
        if first in done and first.exception() is None:
            return first.result()
        tasks = {first} if first not in done else set()
        tasks.add(spawn(t - hedge_after, hedge=1))
        last: BaseException = first.exception() if first in done else RuntimeError("hedge")
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                if d.exception() is None:
                    return d.result()
                last = d.exception()
        raise last
    finally:
        # Cancel calls still running and collect the ones that failed.
        for x in made:
            if not x.done():
                x.cancel()
            elif not x.cancelled():
                x.exception()


async def chat(system: str, user: str, *, effort: str = "minimal", max_tokens: int = 2000,
               timeout: float = 20, json_mode: bool = False, retries: int = 1,
               deadline: Optional[float] = None, hedge_after: Optional[float] = None,
               kind: str = "llm") -> str:
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        # The API rejects non-streaming calls above 2000 tokens.
        max_completion_tokens=min(max_tokens, 2000),
        extra_body={"reasoning_effort": effort},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    last: Exception = RuntimeError("no attempt")
    # Retries stay within the request deadline.
    for rnd in range(1, retries + 2):
        t = timeout
        if deadline is not None:
            t = min(timeout, deadline - time.time())
            if t < MIN_ATTEMPT:
                raise DeadlineExceeded() from last
        try:
            return await _hedged(kwargs, t, hedge_after, json_mode, kind, rnd)
        except Exception as e:
            last = e
    raise last


async def chat_json(system: str, user: str, **kw) -> dict:
    out = json.loads(await chat(system, user, json_mode=True, **kw))
    if not isinstance(out, dict):
        raise ValueError("not a JSON object")
    return out
