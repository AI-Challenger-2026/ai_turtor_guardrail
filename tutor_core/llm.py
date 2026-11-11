"""공용 API 연결, 구조화 응답 검증과 요청 전체 시간 예산을 관리한다."""

import asyncio
import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, TypeVar

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import NON_STREAM_MAX_COMPLETION_TOKENS, Settings
from .schemas import StructuredProbe


T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class ProviderFailure(RuntimeError):
    """상위 호출자가 안전하게 집계할 수 있는 고정 오류 코드만 보존한다."""

    def __init__(self, reason: str, status_code: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


@dataclass
class RequestBudget:
    expires_at: float
    max_chat_calls: int = 4
    chat_calls: int = 0

    @classmethod
    def start(cls, seconds: float, max_chat_calls: int = 4) -> "RequestBudget":
        return cls(time.monotonic() + seconds, max_chat_calls)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def claim_chat(self) -> None:
        if self.chat_calls >= self.max_chat_calls:
            raise ProviderFailure("call_budget_exhausted")
        self.chat_calls += 1

    async def run(self, operation: Callable[[], Awaitable[T]], seconds: Optional[float] = None) -> T:
        timeout = self.remaining()
        if seconds is not None:
            timeout = min(timeout, seconds)
        if timeout <= 0:
            raise ProviderFailure("deadline_exceeded")
        try:
            return await asyncio.wait_for(operation(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ProviderFailure("deadline_exceeded") from None


def silence_transport_logs() -> None:
    # SDK의 디버그 설정과 무관하게 인증 헤더·요청·응답이 로그로 전파되지 않게 한다.
    for name in ("openai", "httpx", "httpcore", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        logger.setLevel(logging.CRITICAL + 1)
        for child_name, child in list(logging.Logger.manager.loggerDict.items()):
            if child_name.startswith(name + ".") and isinstance(child, logging.Logger):
                child.handlers = [logging.NullHandler()]
                child.propagate = False
                child.setLevel(logging.CRITICAL + 1)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(schema)
    return schema


@dataclass
class ApiClients:
    settings: Settings = field(repr=False)
    counters: Counter = field(default_factory=Counter, init=False, repr=False)
    structured_mode: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        silence_transport_logs()
        self._chat_slots = asyncio.Semaphore(self.settings.chat_concurrency)
        self._embedding_slots = asyncio.Semaphore(self.settings.embedding_concurrency)
        self._chat = self._make_client(
            self.settings.chat_base_url, self.settings.chat_api_key, self.settings.chat_concurrency,
        )
        self._embedding = self._make_client(
            self.settings.embedding_base_url, self.settings.embedding_api_key, self.settings.embedding_concurrency,
        )

    @staticmethod
    def _make_client(base_url: str, api_key: str, concurrency: int) -> AsyncOpenAI:
        transport = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            timeout=httpx.Timeout(25.0, connect=5.0),
            follow_redirects=False,
            trust_env=False,
        )
        return AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=25.0, max_retries=0,
            http_client=transport, organization="", project="",
        )

    async def close(self) -> None:
        await asyncio.gather(self._chat.close(), self._embedding.close())

    async def initialize(self) -> None:
        budget = RequestBudget.start(self.settings.startup_timeout, max_chat_calls=2)
        prompt = (
            "연결 검사용 JSON을 반환하세요. status는 ok, values는 [1, 2], "
            "details.verified는 true로 작성하세요."
        )
        try:
            result = await self._structured(
                [{"role": "system", "content": prompt}], StructuredProbe, budget, 20.0, 1200, "json_schema",
            )
            mode = "json_schema"
        except ProviderFailure as exc:
            # 형식 지원에 관한 400·422만 대체 형식으로 확인하며 인증·장애 오류는 숨기지 않는다.
            if exc.reason != "upstream_status" or exc.status_code not in (400, 422):
                raise
            result = await self._structured(
                [{"role": "system", "content": prompt}], StructuredProbe, budget, 20.0, 1200, "json_object",
            )
            mode = "json_object"
        if result.values != [1, 2] or not result.details.verified:
            raise ProviderFailure("probe_mismatch")
        self.structured_mode = mode

    async def _invoke(self, operation: Callable[[], Awaitable[T]], budget: RequestBudget, seconds: float) -> T:
        try:
            return await budget.run(operation, seconds)
        except ProviderFailure as exc:
            self.counters[exc.reason] += 1
            raise
        except APITimeoutError:
            self.counters["upstream_timeout"] += 1
            raise ProviderFailure("upstream_timeout") from None
        except APIStatusError as exc:
            self.counters["upstream_status"] += 1
            if 100 <= exc.status_code <= 599:
                self.counters["upstream_http_" + str(exc.status_code)] += 1
            # 오류 원문은 저장하지 않고 알려진 매개변수 이름과의 일치 여부만 집계한다.
            body = exc.body if isinstance(exc.body, dict) else {}
            detail = body.get("error", body)
            if isinstance(detail, dict):
                parameter = detail.get("param")
                message = detail.get("message")
                for name in ("response_format", "max_completion_tokens", "max_tokens", "reasoning_effort", "messages", "stream", "model"):
                    if parameter == name or (isinstance(message, str) and name in message):
                        self.counters["upstream_parameter_" + name] += 1
            raise ProviderFailure("upstream_status", exc.status_code) from None
        except APIConnectionError:
            self.counters["upstream_connection"] += 1
            raise ProviderFailure("upstream_connection") from None
        except Exception:
            self.counters["upstream_error"] += 1
            raise ProviderFailure("upstream_error") from None

    async def complete_text(
        self, messages: list[dict[str, str]], budget: RequestBudget, *,
        seconds: float, max_tokens: int, response_format: Optional[dict[str, Any]] = None,
        reasoning_effort: str = "low",
    ) -> str:
        async def call() -> Any:
            async with self._chat_slots:
                budget.claim_chat()
                kwargs: dict[str, Any] = {}
                if response_format is not None:
                    kwargs["response_format"] = response_format
                self.counters["chat_attempts"] += 1
                return await self._chat.chat.completions.create(
                    model=self.settings.chat_model, messages=messages,
                    max_completion_tokens=min(max_tokens, NON_STREAM_MAX_COMPLETION_TOKENS), extra_body={"reasoning_effort": reasoning_effort},
                    stream=False, timeout=max(0.001, min(seconds, budget.remaining())), **kwargs,
                )

        result = await self._invoke(call, budget, seconds)
        if not result.choices:
            raise ProviderFailure("empty_response")
        choice = result.choices[0]
        if getattr(choice.message, "refusal", None):
            raise ProviderFailure("model_refusal")
        if choice.finish_reason != "stop":
            raise ProviderFailure("incomplete_response")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ProviderFailure("empty_response")
        self.counters["chat_successes"] += 1
        return content.strip()

    async def _structured(
        self, messages: list[dict[str, str]], model: type[M], budget: RequestBudget,
        seconds: float, max_tokens: int, mode: str, reasoning_effort: str = "low",
    ) -> M:
        schema = strict_schema(model)
        if mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": model.__name__, "strict": True, "schema": schema},
            }
        else:
            response_format = {"type": "json_object"}
        instructions = "반드시 다음 스키마에 맞는 JSON 객체 하나로만 답하세요.\n" + json.dumps(schema, ensure_ascii=False)
        # 호환 프록시마다 여러 시스템 메시지를 다루는 방식이 달라 하나로 합쳐 보낸다.
        system = "\n\n".join([message["content"] for message in messages if message["role"] == "system"] + [instructions])
        conversation = [message for message in messages if message["role"] != "system"]
        content = await self.complete_text(
            [{"role": "system", "content": system}, *conversation], budget,
            seconds=seconds, max_tokens=max_tokens, response_format=response_format, reasoning_effort=reasoning_effort,
        )
        try:
            return model.model_validate_json(content, strict=True)
        except ValidationError:
            self.counters["invalid_structured_response"] += 1
            raise ProviderFailure("invalid_structured_response") from None

    async def complete_json(
        self, messages: list[dict[str, str]], model: type[M], budget: RequestBudget, *,
        seconds: float, max_tokens: int, reasoning_effort: str = "low",
    ) -> M:
        if self.structured_mode is None:
            raise ProviderFailure("provider_not_ready")
        return await self._structured(messages, model, budget, seconds, max_tokens, self.structured_mode, reasoning_effort)

    async def embed(
        self, texts: list[str], budget: RequestBudget, *, seconds: Optional[float] = None,
    ) -> list[list[float]]:
        timeout = self.settings.embedding_timeout if seconds is None else seconds

        async def call() -> Any:
            async with self._embedding_slots:
                self.counters["embedding_attempts"] += 1
                return await self._embedding.embeddings.create(
                    model=self.settings.embedding_model, input=texts, encoding_format="float",
                    timeout=max(0.001, min(timeout, budget.remaining())),
                )

        result = await self._invoke(call, budget, timeout)
        rows = sorted(result.data, key=lambda row: row.index)
        if [row.index for row in rows] != list(range(len(texts))):
            raise ProviderFailure("invalid_embedding")
        vectors = [row.embedding for row in rows]
        if any(
            len(vector) != 1536
            or not all(math.isfinite(value) for value in vector)
            or not any(vector)
            for vector in vectors
        ):
            raise ProviderFailure("invalid_embedding")
        self.counters["embedding_successes"] += 1
        return vectors
