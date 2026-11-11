"""운영진 제공 LLM(gpt-5-nano) 호출 헬퍼. 요청·응답 본문은 로그에 남기지 않는다.

시간 예산: 채점기는 요청당 30초에서 끊는다. 모든 호출은 절대 시각 `deadline`을 받아,
남은 시간 안에서만 시도·재시도한다(재시도가 30초를 넘기는 일을 막음).
"""
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

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


async def _hedged(kwargs, t: float, hedge_after: Optional[float], json_mode: bool) -> str:
    """헤지 요청: hedge_after초 안에 안 끝나면 같은 요청을 하나 더 보내 먼저 성공한 쪽을 쓴다.
    API가 가끔 응답 없이 멈추는 경우(실제 채점에서 26초 대체 발생)의 꼬리 지연을 줄인다."""
    async def attempt(tt):
        return _valid(await _once(kwargs, tt), json_mode)

    first = asyncio.create_task(attempt(t))
    if not hedge_after or t - hedge_after < MIN_ATTEMPT:
        return await first
    done, _ = await asyncio.wait({first}, timeout=hedge_after)
    if first in done and first.exception() is None:
        return first.result()
    tasks = {first} if first not in done else set()
    tasks.add(asyncio.create_task(attempt(t - hedge_after)))
    last: BaseException = first.exception() if first in done else RuntimeError("hedge")
    try:
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                if d.exception() is None:
                    return d.result()
                last = d.exception()
        raise last
    finally:
        for x in tasks:
            x.cancel()


async def chat(system: str, user: str, *, effort: str = "minimal", max_tokens: int = 2000,
               timeout: float = 20, json_mode: bool = False, retries: int = 1,
               deadline: Optional[float] = None, hedge_after: Optional[float] = None) -> str:
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
    for _ in range(retries + 1):
        t = timeout
        if deadline is not None:
            t = min(timeout, deadline - time.time())
            if t < MIN_ATTEMPT:
                raise DeadlineExceeded() from last
        try:
            return await _hedged(kwargs, t, hedge_after, json_mode)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


async def chat_json(system: str, user: str, **kw) -> dict:
    out = json.loads(await chat(system, user, json_mode=True, **kw))
    if not isinstance(out, dict):
        raise ValueError("not a JSON object")
    return out


LOG_FILE = os.environ.get("LOG_FILE", str(Path(__file__).resolve().parents[1] / "logs" / "server.log"))


def log(event: str, **fields):
    """시간·길이·상태·오류 종류만 남긴다(SPEC §8: 요청·답변 본문과 문항 내용에서 나온 값은 남기지 않음).
    화면과 파일(logs/server.log, git 제외)에 같이 쓴다 — 채점 뒤 오류·대체 횟수를 세기 위함."""
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{time.strftime('%m-%d %H:%M:%S')} {event} {parts}"
    print(line, flush=True)
    try:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass
