"""LLM 제공자 공통 인터페이스 (provider interface).

generate(system, messages, temperature, max_tokens, json_mode) → LLMResult
- result.output 은 json_mode=False 이면 텍스트(str), True 이면 dict 이다.
- 재시도(지수 백오프, exponential backoff), JSON 파싱 실패 시 1회 재요청, 토큰 사용량 기록을 여기서 공통 처리한다.
- 각 구현체는 _complete() 하나만 구현하면 된다.
"""

from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.errors import LLMError, LLMJSONError
from app.logging_utils import get_logger

log = get_logger("llm")

JSON_SUFFIX = "\n\n[출력 형식] 설명이나 코드 블록 없이 유효한 JSON 객체 하나만 출력한다."
JSON_RETRY_MESSAGE = "직전 응답은 유효한 JSON 이 아니었습니다. 다른 설명 없이 요청한 형식의 JSON 객체 하나만 다시 출력하세요."


@dataclass
class RawCompletion:
    """구현체가 반환하는 원시 결과 (raw completion from a provider)."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    model: str | None = None
    temperature_applied: bool = True
    usage_estimated: bool = False


@dataclass
class LLMResult:
    text: str
    data: dict[str, Any] | None
    provider: str
    model: str
    role: str
    task: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    stop_reason: str | None = None
    temperature_applied: bool = True
    attempts: int = 1

    @property
    def output(self) -> str | dict[str, Any]:
        return self.data if self.data is not None else self.text


# ---------------------------------------------------------------------------
# 사용량 기록 (usage recording) — 한 턴 안의 모든 LLM 호출을 contextvar 로 모은다.
# ---------------------------------------------------------------------------

_CALLS: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_calls", default=None)


@contextmanager
def usage_scope() -> Iterator[list[dict[str, Any]]]:
    """with usage_scope() as calls: 블록 안의 LLM 호출 기록을 calls 리스트로 수집한다."""
    calls: list[dict[str, Any]] = []
    token = _CALLS.set(calls)
    try:
        yield calls
    finally:
        _CALLS.reset(token)


def _record_call(record: dict[str, Any]) -> None:
    calls = _CALLS.get()
    if calls is not None:
        calls.append(record)


def estimate_cost_usd(pricing: dict[str, dict[str, float]], model: str, input_tokens: int, output_tokens: int) -> float:
    """가격표(USD / 1M tokens)로 예상 비용 계산. 가격표에 없으면 0 (unknown)."""
    price = pricing.get(model)
    if not price:
        return 0.0
    return round(
        input_tokens / 1_000_000 * float(price.get("input", 0.0))
        + output_tokens / 1_000_000 * float(price.get("output", 0.0)),
        6,
    )


# ---------------------------------------------------------------------------
# JSON 추출 (JSON extraction)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """모델 출력에서 JSON 객체를 꺼낸다. 코드 펜스나 앞뒤 설명이 섞여 있어도 첫 객체를 찾는다."""
    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("JSON 객체를 찾지 못했습니다 (no JSON object found)")


# ---------------------------------------------------------------------------
# 공통 제공자 (base provider)
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 16.0

    def delay(self, attempt: int) -> float:
        base = min(self.base_delay * (2**attempt), self.max_delay)
        return base * (1.0 + random.random() * 0.25)


@dataclass
class ProviderOptions:
    role: str = "generator"
    timeout_s: float = 60.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)


class LLMProvider(ABC):
    """모든 LLM 제공자의 부모 클래스 (base class for LLM providers)."""

    name: str = "base"

    def __init__(self, model: str, options: ProviderOptions | None = None) -> None:
        self.model = model
        self.options = options or ProviderOptions()

    @property
    def role(self) -> str:
        return self.options.role

    @abstractmethod
    def _complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        task: str | None,
    ) -> RawCompletion:
        """제공자별 API 호출 1회. 실패 시 app.errors.LLMError 계열로 변환해 던진다."""

    def _complete_with_retry(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        task: str | None,
    ) -> tuple[RawCompletion, int]:
        policy = self.options.retry
        attempt = 0
        while True:
            try:
                return self._complete(system, messages, temperature, max_tokens, json_mode, task), attempt + 1
            except LLMError as err:
                if err.retryable and attempt < policy.max_retries:
                    wait = policy.delay(attempt)
                    log.warning(
                        "LLM 재시도 (retry) provider=%s task=%s attempt=%d wait=%.2fs code=%s",
                        self.name, task, attempt + 1, wait, err.code,
                    )
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
        task: str | None = None,
    ) -> LLMResult:
        """LLM 을 호출한다 (공통 진입점).

        Args:
            system: 시스템 프롬프트.
            messages: [{"role": "user"|"assistant", "content": str}, ...] — 마지막은 user.
            temperature: 판정용 호출은 0 으로 고정한다 (모델이 지원하지 않으면 제공자가 생략하고 기록).
            max_tokens: 최대 출력 토큰.
            json_mode: True 이면 JSON 을 파싱해 result.data 로 돌려준다. 실패 시 1회 재요청.
            task: 호출 목적 태그 (예: "tutor_answer", "input_classify") — 로그와 Mock 분기에 사용.
        """
        started = time.perf_counter()
        sys_prompt = system + JSON_SUFFIX if json_mode else system
        total_in = total_out = 0
        attempts = 0
        raw: RawCompletion | None = None
        data: dict[str, Any] | None = None
        try:
            raw, n = self._complete_with_retry(sys_prompt, messages, temperature, max_tokens, json_mode, task)
            attempts += n
            total_in += raw.input_tokens
            total_out += raw.output_tokens
            if json_mode:
                try:
                    data = extract_json(raw.text)
                except ValueError:
                    log.warning("JSON 파싱 실패 → 1회 재요청 (JSON retry) provider=%s task=%s", self.name, task)
                    retry_messages = list(messages) + [
                        {"role": "assistant", "content": raw.text or "(빈 응답)"},
                        {"role": "user", "content": JSON_RETRY_MESSAGE},
                    ]
                    raw, n = self._complete_with_retry(sys_prompt, retry_messages, temperature, max_tokens, json_mode, task)
                    attempts += n
                    total_in += raw.input_tokens
                    total_out += raw.output_tokens
                    try:
                        data = extract_json(raw.text)
                    except ValueError as exc:
                        raise LLMJSONError(detail=f"JSON parse failed twice (task={task}): {raw.text[:200]!r}") from exc
        except LLMError as err:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _record_call(
                {
                    "task": task, "role": self.role, "provider": self.name, "model": self.model,
                    "input_tokens": total_in, "output_tokens": total_out, "latency_ms": latency_ms,
                    "estimated_cost_usd": estimate_cost_usd(self.options.pricing, self.model, total_in, total_out),
                    "ok": False, "error": err.code, "temperature_applied": None, "attempts": max(attempts, 1),
                }
            )
            raise

        assert raw is not None
        latency_ms = int((time.perf_counter() - started) * 1000)
        served_model = raw.model or self.model
        result = LLMResult(
            text=raw.text,
            data=data,
            provider=self.name,
            model=served_model,
            role=self.role,
            task=task,
            input_tokens=total_in,
            output_tokens=total_out,
            latency_ms=latency_ms,
            stop_reason=raw.stop_reason,
            temperature_applied=raw.temperature_applied,
            attempts=attempts,
        )
        _record_call(
            {
                "task": task, "role": self.role, "provider": self.name, "model": served_model,
                "input_tokens": total_in, "output_tokens": total_out, "latency_ms": latency_ms,
                "estimated_cost_usd": estimate_cost_usd(self.options.pricing, self.model, total_in, total_out),
                "ok": True, "error": None, "temperature_applied": raw.temperature_applied,
                "usage_estimated": raw.usage_estimated, "attempts": attempts,
            }
        )
        return result


class UnavailableProvider(LLMProvider):
    """설정 오류로 만들 수 없는 제공자의 자리표시 (placeholder that raises the stored error on use).

    서버 시작 시 설정 오류가 있어도 서버가 죽지 않도록, 호출 시점에 오류를 돌려준다.
    """

    name = "unavailable"

    def __init__(self, configured_provider: str, model: str, error: LLMError, options: ProviderOptions | None = None) -> None:
        super().__init__(model, options)
        self.configured_provider = configured_provider
        self.error = error

    def _complete(self, system, messages, temperature, max_tokens, json_mode, task) -> RawCompletion:  # noqa: ANN001
        raise self.error
