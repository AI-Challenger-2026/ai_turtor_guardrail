"""운영진 제공 LLM(gpt-5-nano) 호출 헬퍼. 요청·응답 본문은 로그에 남기지 않는다."""
import asyncio
import json
import os
import time

from openai import AsyncOpenAI

MODEL = os.environ.get("OPENAI_MODEL", "openai/gpt-5-nano")

_client = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(timeout=25, max_retries=0)
    return _client


async def chat(system: str, user: str, *, effort: str = "minimal", max_tokens: int = 2000,
               timeout: float = 20, json_mode: bool = False, retries: int = 1) -> str:
    """LLM을 한 번 호출해 텍스트를 돌려준다. 실패하면 retries만큼 다시 시도하고, 끝내 실패하면 예외."""
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=max_tokens,
        extra_body={"reasoning_effort": effort},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    last = None
    for _ in range(retries + 1):
        try:
            r = await asyncio.wait_for(client().chat.completions.create(**kwargs), timeout=timeout)
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
            last = RuntimeError("empty completion")
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


async def chat_json(system: str, user: str, **kw) -> dict:
    text = await chat(system, user, json_mode=True, **kw)
    return json.loads(text)


def log(event: str, **fields):
    """본문 없이 시간·길이·상태·오류 종류만 남긴다(SPEC §8)."""
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{time.strftime('%H:%M:%S')} {event} {parts}", flush=True)
