"""역할별 LLM 제공자 생성 (provider factory).

역할(role):
- generator: 튜터 답변 생성 (GENERATOR_PROVIDER / GENERATOR_MODEL)
- judge: 가드레일 판정, 질의 재작성, 품질 채점 (JUDGE_PROVIDER / JUDGE_MODEL)
MOCK_MODE=true 이면 모든 역할이 MockProvider 로 간다.
"""

from __future__ import annotations

from typing import Literal

from app.errors import LLMConfigError, LLMError
from app.llm.base import LLMProvider, ProviderOptions, RetryPolicy, UnavailableProvider
from app.llm.mock_provider import MockProvider
from app.settings import Settings

Role = Literal["generator", "judge"]
SUPPORTED_PROVIDERS = ("anthropic", "openai", "elice", "mock")


def _options(settings: Settings, role: Role) -> ProviderOptions:
    return ProviderOptions(
        role=role,
        timeout_s=settings.llm.timeout_seconds,
        retry=RetryPolicy(
            max_retries=settings.llm.max_retries,
            base_delay=settings.llm.retry_base_delay,
            max_delay=settings.llm.retry_max_delay,
        ),
        pricing=settings.pricing_usd_per_mtok,
    )


def resolve_model(settings: Settings, role: Role) -> tuple[str, str]:
    """(provider, model) — MOCK_MODE 이면 ("mock", "mock-<role>")."""
    if settings.mock_mode:
        return "mock", f"mock-{role}"
    cfg = settings.generator if role == "generator" else settings.judge
    if cfg.provider == "openai":
        return "openai", settings.openai_model
    return cfg.provider, cfg.model


def build_provider(settings: Settings, role: Role) -> LLMProvider:
    """설정에 맞는 제공자를 만든다. 설정 오류가 있으면 UnavailableProvider 를 돌려준다 (서버는 죽지 않음)."""
    options = _options(settings, role)
    provider_name, model = resolve_model(settings, role)
    try:
        if provider_name == "mock":
            return MockProvider(model=model, options=options)
        if provider_name == "anthropic":
            from app.llm.anthropic_provider import AnthropicProvider

            cfg = settings.generator if role == "generator" else settings.judge
            return AnthropicProvider(
                model,
                options,
                api_key=settings.anthropic_api_key.get_secret_value() or None,
                effort=cfg.effort,
                refusal_fallback=settings.anthropic_refusal_fallback,
            )
        if provider_name == "openai":
            if not model:
                raise LLMConfigError(
                    "OPENAI_MODEL 이 비어 있습니다. backend/.env 에 사용할 GPT 모델명을 입력해 주세요.",
                    detail="OPENAI_MODEL is empty",
                )
            from app.llm.openai_provider import OpenAIProvider

            return OpenAIProvider(model, options, api_key=settings.openai_api_key.get_secret_value() or None)
        if provider_name == "elice":
            from app.llm.elice_provider import EliceProvider

            return EliceProvider(
                model, options, api_key=settings.elice_api_key.get_secret_value(), api_base=settings.elice_api_base
            )
        raise LLMConfigError(
            f"알 수 없는 LLM 제공자입니다: {provider_name} (지원: {', '.join(SUPPORTED_PROVIDERS)})",
            detail=f"unknown provider {provider_name}",
        )
    except LLMError as err:
        return UnavailableProvider(provider_name, model, err, options)
    except Exception as exc:  # SDK 생성 실패 등 (e.g. SDK import error)
        return UnavailableProvider(provider_name, model, LLMConfigError(detail=f"{type(exc).__name__}: {exc}"), options)
