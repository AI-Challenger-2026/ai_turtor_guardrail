"""Elice ML API provider."""

from __future__ import annotations

from typing import Any

import httpx

from app.errors import (
    LLMAPIError,
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
)
from app.llm.base import LLMProvider, ProviderOptions, RawCompletion


class EliceProvider(LLMProvider):
    name = "elice"

    def __init__(self, model: str, options: ProviderOptions | None = None, *, api_key: str = "", api_base: str = "") -> None:
        super().__init__(model, options)
        self.api_base = api_base.rstrip("/")
        self._api_key = api_key

    @staticmethod
    def _text(data: Any) -> str:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    content = message["content"]
                    if isinstance(content, list):
                        return "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
                    return str(content)
                if choice.get("text") is not None:
                    return str(choice["text"])
        for key in ("text", "output", "response", "answer"):
            if isinstance(data.get(key), str):
                return data[key]
        return ""

    @staticmethod
    def _usage(data: Any) -> tuple[int, int]:
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        if not isinstance(usage, dict):
            return 0, 0
        return int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0), int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)

    def _complete(self, system, messages, temperature, max_tokens, json_mode, task) -> RawCompletion:  # noqa: ANN001
        if not self.api_base:
            raise LLMAPIError("ELICE_API_BASE가 설정되지 않았습니다.", detail="ELICE_API_BASE is empty")
        if not self._api_key:
            raise LLMAuthError("ELICE_API_KEY가 설정되지 않았습니다.", detail="ELICE_API_KEY is empty")

        request_messages: list[dict[str, str]] = []
        if system:
            request_messages.append({"role": "system", "content": system})
        request_messages.extend(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            response = httpx.post(self.api_base, json=payload, headers=headers, timeout=self.options.timeout_s)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(detail=f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(detail=f"{type(exc).__name__}: {exc}") from exc

        detail = response.text[:500]
        if response.status_code in (401, 403):
            raise LLMAuthError(detail=detail)
        if response.status_code == 429:
            raise LLMRateLimitError(detail=detail)
        if response.status_code >= 500:
            raise LLMAPIError(retryable=True, detail=detail)
        if response.status_code >= 400:
            raise LLMAPIError(detail=detail)
        try:
            data: Any = response.json()
        except ValueError as exc:
            raise LLMAPIError("Elice API가 JSON 응답을 반환하지 않았습니다.", detail=detail) from exc
        finish_reason = None
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
            finish_reason = data["choices"][0].get("finish_reason")
        if finish_reason in {"content_filter", "refusal"}:
            raise LLMRefusalError(
                category="content_filter" if finish_reason == "content_filter" else "model_refusal",
                detail=f"Elice finish_reason={finish_reason}",
            )
        text = self._text(data)
        if not text:
            raise LLMAPIError("Elice API 응답에서 답변을 찾지 못했습니다.", detail=detail)
        input_tokens, output_tokens = self._usage(data)
        return RawCompletion(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=finish_reason,
            model=(data.get("model") if isinstance(data, dict) else None) or self.model,
            temperature_applied=True,
        )
