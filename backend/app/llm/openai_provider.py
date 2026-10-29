"""OpenAI 제공자 (OpenAI SDK provider).

구현은 되어 있지만 OPENAI_API_KEY 가 있을 때만 실제 동작을 확인할 수 있다.
GENERATOR_PROVIDER=openai 또는 JUDGE_PROVIDER=openai 일 때 OPENAI_MODEL 에 지정한 모델을 사용한다.
"""

from __future__ import annotations

from typing import Any

from app.errors import (
    LLMAPIError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.llm.base import LLMProvider, ProviderOptions, RawCompletion


def map_openai_error(exc: Exception) -> LLMError:
    import openai

    detail = f"{type(exc).__name__}: {str(exc)[:300]}"
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMAuthError(detail=detail)
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(detail=detail)
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError(detail=detail)
    if isinstance(exc, openai.APIConnectionError):
        return LLMConnectionError(detail=detail)
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        if status >= 500:
            return LLMAPIError(retryable=True, detail=detail)
        return LLMAPIError("AI 서비스가 요청을 거부했습니다. 모델명 등 서버 설정을 확인해 주세요.", detail=detail)
    if isinstance(exc, openai.OpenAIError) and "api_key" in str(exc).lower():
        return LLMAuthError("API 키가 설정되지 않았습니다. backend/.env 의 OPENAI_API_KEY 를 확인해 주세요.", detail=detail)
    return LLMAPIError(detail=detail)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, options: ProviderOptions | None = None, *, api_key: str | None = None) -> None:
        super().__init__(model, options)
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        # 키가 없으면 SDK 생성자에서 예외가 나므로 호출 시점에 생성한다 (lazy init).
        if self._client is None:
            import openai

            try:
                self._client = openai.OpenAI(api_key=self._api_key or None, timeout=self.options.timeout_s, max_retries=0)
            except Exception as exc:
                raise map_openai_error(exc) from exc
        return self._client

    def _complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        task: str | None,
    ) -> RawCompletion:
        client = self._get_client()
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**params)
        except Exception as exc:
            raise map_openai_error(exc) from exc

        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return RawCompletion(
            text=choice.message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            stop_reason=choice.finish_reason,
            model=getattr(response, "model", None) or self.model,
            temperature_applied=True,
        )
