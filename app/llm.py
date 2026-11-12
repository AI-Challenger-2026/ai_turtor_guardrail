"""운영진 제공 LLM(gpt-5-nano) 호출 헬퍼. 요청·응답 본문은 로그에 남기지 않는다.

시간 예산: 채점기는 요청당 30초에서 끊는다. 모든 호출은 절대 시각 `deadline`을 받아,
남은 시간 안에서만 시도·재시도한다(재시도가 30초를 넘기는 일을 막음).
"""
import asyncio
import json
import os
import time
from typing import Optional

from openai import AsyncOpenAI

from . import logs

MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-5-nano")
MIN_ATTEMPT = 1.5  # 남은 시간이 이보다 짧으면 새로 시도하지 않는다(초)

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


async def _hedged(kwargs, t: float, hedge_after: Optional[float], json_mode: bool,
                  kind: str = "llm", rnd: int = 1) -> str:
    """헤지 요청: hedge_after초 안에 안 끝나면 같은 요청을 하나 더 보내 먼저 성공한 쪽을 쓴다.
    API가 가끔 응답 없이 멈추는 경우(실제 채점에서 26초 대체 발생)의 꼬리 지연을 줄인다."""
    async def attempt(tt, hedge=0):
        # One metadata line per API call (status and error class only; never the completion).
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            text = await _once(kwargs, tt)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as e:  # noqa: BLE001
            status, err = "error", type(e).__name__
            raise
        else:
            try:
                return _valid(text, json_mode)
            except Exception as e:  # noqa: BLE001
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
        # Port fix: cancel calls still running (also when the caller itself is cancelled, which
        # final-v3 left running) and collect the exception of any that failed. In final-v3 a failed
        # call finishing together with a successful one was never collected, and asyncio printed
        # its traceback with the exception message.
        for x in made:
            if not x.done():
                x.cancel()
            elif not x.cancelled():
                x.exception()


async def chat(system: str, user: str, *, effort: str = "minimal", max_tokens: int = 2000,
               timeout: float = 20, json_mode: bool = False, retries: int = 1,
               deadline: Optional[float] = None, hedge_after: Optional[float] = None,
               kind: str = "llm") -> str:
    """LLM을 호출해 텍스트(json_mode면 파싱 가능한 JSON 텍스트)를 돌려준다.
    빈 응답·JSON 파싱 실패·오류는 deadline 안에서 retries만큼 다시 시도한다."""
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=min(max_tokens, 2000),  # 운영 API: 비스트리밍은 2000 초과 시 400
        extra_body={"reasoning_effort": effort},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    last: Exception = RuntimeError("no attempt")
    for rnd in range(1, retries + 2):
        t = timeout
        if deadline is not None:
            t = min(timeout, deadline - time.time())
            if t < MIN_ATTEMPT:
                raise DeadlineExceeded() from last
        try:
            return await _hedged(kwargs, t, hedge_after, json_mode, kind, rnd)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


async def chat_json(system: str, user: str, **kw) -> dict:
    out = json.loads(await chat(system, user, json_mode=True, **kw))
    if not isinstance(out, dict):
        raise ValueError("not a JSON object")
    return out
