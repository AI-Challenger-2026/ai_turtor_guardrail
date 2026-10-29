"""Anthropic Claude 제공자 (Anthropic SDK provider).

모델별 API 차이 처리 (per-model API differences):
- Claude 5.x 계열(Sonnet 5.5, Haiku 5.5, Opus 5.5 등)은 temperature 같은 sampling 파라미터를 받지 않는다(400).
  → 허용 모델(4.6 이하 일부)에만 temperature 를 보내고, 생략한 경우 temperature_applied=False 로 기록한다.
- effort(output_config.effort)는 지원 모델에만 보낸다. 사고(thinking) 블록은 무시하고 text 블록만 사용한다.
- stop_reason == "refusal" 이면 LLMRefusalError 로 변환한다.
- ANTHROPIC_REFUSAL_FALLBACK=default 이면 지원 모델(Opus 5.x, Sonnet 5.5, Fable 5.x)에서
  서버측 refusal fallback(beta: server-side-fallback-2026-07-01, fallbacks="default")을 켠다.
- SDK 자체 재시도는 끄고(max_retries=0) 공통 계층의 지수 백오프를 사용한다.
"""

from __future__ import annotations

from typing import Any

from app.errors import (
    LLMAPIError,
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
)
from app.llm.base import LLMProvider, ProviderOptions, RawCompletion

# temperature 를 받는 모델 접두어 (models that still accept sampling params)
_SAMPLING_OK_PREFIXES = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-3",
)
# effort 를 받지 않는 모델 접두어 (models that reject output_config.effort)
_NO_EFFORT_PREFIXES = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-3",
)
# 서버측 fallbacks="default" 를 지원하는 모델 접두어
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def accepts_sampling(model: str) -> bool:
    return model.startswith(_SAMPLING_OK_PREFIXES)


def accepts_effort(model: str) -> bool:
    return not model.startswith(_NO_EFFORT_PREFIXES)


def supports_default_fallback(model: str) -> bool:
    return model.startswith(_FALLBACK_PREFIXES)


def map_anthropic_error(exc: Exception) -> LLMError:
    """anthropic SDK 예외를 앱 예외로 변환 (most specific first)."""
    import anthropic

    detail = f"{type(exc).__name__}: {str(exc)[:300]}"
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMAuthError(detail=detail)
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError(detail=detail)
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError(detail=detail)
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMConnectionError(detail=detail)
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        if status >= 500:
            return LLMAPIError(retryable=True, detail=detail)
        if status in (400, 404, 422):
            return LLMAPIError(
                "AI 서비스가 요청을 거부했습니다. 모델명 등 서버 설정을 확인해 주세요.", detail=detail
            )
        return LLMAPIError(detail=detail)
    credentials_error = getattr(anthropic, "CredentialsError", None)
    if credentials_error is not None and isinstance(exc, credentials_error):
        return LLMAuthError(detail=detail)
    if isinstance(exc, TypeError) and "authentication" in str(exc).lower():
        # API key 가 없을 때 SDK 가 던지는 TypeError ("Could not resolve authentication method")
        return LLMAuthError("API 키가 설정되지 않았습니다. backend/.env 의 ANTHROPIC_API_KEY 를 확인해 주세요.", detail=detail)
    return LLMAPIError(detail=detail)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        options: ProviderOptions | None = None,
        *,
        api_key: str | None = None,
        effort: str | None = None,
        refusal_fallback: str = "default",
    ) -> None:
        super().__init__(model, options)
        import anthropic

        client_kwargs: dict[str, Any] = {"timeout": self.options.timeout_s, "max_retries": 0}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**client_kwargs)
        self.effort = effort
        self.use_fallback = refusal_fallback == "default" and supports_default_fallback(model)

    def _complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        task: str | None,
    ) -> RawCompletion:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        temperature_applied = False
        if accepts_sampling(self.model):
            # SDK 1.x 의 타입 시그니처에서 temperature 가 빠져 있어 extra_body 로 전달한다.
            params["extra_body"] = {"temperature": temperature}
            temperature_applied = True
        if self.effort and accepts_effort(self.model):
            params["output_config"] = {"effort": self.effort}

        try:
            if self.use_fallback:
                response = self._client.beta.messages.create(
                    betas=[_FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = self._client.messages.create(**params)
        except Exception as exc:  # SDK 예외 → 앱 예외 (map to app errors)
            raise map_anthropic_error(exc) from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(
                category=getattr(details, "category", None),
                detail=f"model refusal: {getattr(details, 'explanation', None)}",
            )

        text = "".join(
            getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        return RawCompletion(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            stop_reason=response.stop_reason,
            model=getattr(response, "model", None) or self.model,
            temperature_applied=temperature_applied,
        )
