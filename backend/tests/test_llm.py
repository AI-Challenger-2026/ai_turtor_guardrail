from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import LLMAuthError, LLMJSONError, LLMNotImplementedError, LLMRateLimitError
from app.llm.anthropic_provider import accepts_sampling, map_anthropic_error, supports_default_fallback
from app.llm.base import LLMProvider, ProviderOptions, RawCompletion, RetryPolicy, extract_json, usage_scope
from app.llm.elice_provider import EliceProvider
from app.llm.factory import build_provider
from app.llm.mock_provider import MockProvider


class ScriptedProvider(LLMProvider):
    """미리 정한 순서대로 결과/예외를 돌려주는 테스트용 제공자."""

    name = "scripted"

    def __init__(self, script, **kw):
        super().__init__("scripted-model", ProviderOptions(retry=RetryPolicy(max_retries=2, base_delay=0, max_delay=0)))
        self.script = list(script)
        self.calls = 0

    def _complete(self, system, messages, temperature, max_tokens, json_mode, task):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return RawCompletion(text=item, input_tokens=10, output_tokens=5)


def test_mock_text_has_prefix_and_json_has_flag():
    p = MockProvider("mock-generator")
    r = p.generate("sys", [{"role": "user", "content": "안녕"}], task="tutor_answer")
    assert r.text.startswith("[MOCK]")
    assert r.provider == "mock"
    j = p.generate("sys", [{"role": "user", "content": "<question>x</question>"}], json_mode=True, task="input_classify")
    assert isinstance(j.output, dict) and j.data["_mock"] is True


def test_retry_with_backoff_then_success():
    p = ScriptedProvider([LLMRateLimitError(), LLMRateLimitError(), "ok"])
    r = p.generate("s", [{"role": "user", "content": "q"}])
    assert r.text == "ok" and p.calls == 3 and r.attempts == 3


def test_retry_gives_up_after_max_retries():
    p = ScriptedProvider([LLMRateLimitError()] * 3)
    with pytest.raises(LLMRateLimitError):
        p.generate("s", [{"role": "user", "content": "q"}])
    assert p.calls == 3


def test_non_retryable_error_is_not_retried():
    p = ScriptedProvider([LLMAuthError(), "never"])
    with pytest.raises(LLMAuthError):
        p.generate("s", [{"role": "user", "content": "q"}])
    assert p.calls == 1


def test_json_parse_failure_retries_once():
    p = ScriptedProvider(["이건 JSON 이 아님", '{"category": "in_scope"}'])
    r = p.generate("s", [{"role": "user", "content": "q"}], json_mode=True)
    assert r.data == {"category": "in_scope"} and p.calls == 2


def test_json_parse_failure_twice_raises():
    p = ScriptedProvider(["no json", "still no json"])
    with pytest.raises(LLMJSONError):
        p.generate("s", [{"role": "user", "content": "q"}], json_mode=True)
    assert p.calls == 2


def test_extract_json_handles_fences_and_prose():
    assert extract_json('결과:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('앞 설명 {"b": {"c": "}"}} 뒤 설명') == {"b": {"c": "}"}}


def test_usage_scope_records_calls():
    p = MockProvider("mock-judge", ProviderOptions(role="judge"))
    with usage_scope() as calls:
        p.generate("s", [{"role": "user", "content": "q"}], task="quality_judge", json_mode=True)
    assert len(calls) == 1 and calls[0]["task"] == "quality_judge" and calls[0]["role"] == "judge"
    assert calls[0]["ok"] is True


def test_elice_provider_requires_endpoint_and_key():
    p = EliceProvider("elice-model")
    with pytest.raises(Exception) as info:
        p.generate("s", [{"role": "user", "content": "q"}])
    assert "ELICE_API_BASE" in info.value.user_message


def test_factory_mock_mode_routes_everything_to_mock(settings):
    assert build_provider(settings, "generator").name == "mock"
    assert build_provider(settings, "judge").name == "mock"


def test_factory_real_mode_builds_anthropic(make_settings):
    s = make_settings(MOCK_MODE="false")
    gen = build_provider(s, "generator")
    assert gen.name == "anthropic" and gen.model == "claude-sonnet-5-5"
    assert gen.use_fallback is True  # Sonnet 5.5 는 서버측 refusal fallback 지원
    judge = build_provider(s, "judge")
    assert judge.name == "anthropic" and judge.use_fallback is False  # Haiku 5.5 는 미지원


def test_factory_openai_without_model_is_unavailable(make_settings):
    s = make_settings(MOCK_MODE="false", GENERATOR_PROVIDER="openai", OPENAI_MODEL="")
    p = build_provider(s, "generator")
    assert p.name == "unavailable"
    with pytest.raises(Exception) as info:
        p.generate("s", [{"role": "user", "content": "q"}])
    assert "OPENAI_MODEL" in info.value.user_message


def test_anthropic_model_capabilities():
    assert accepts_sampling("claude-sonnet-5-5") is False
    assert accepts_sampling("claude-haiku-5-5") is False
    assert accepts_sampling("claude-haiku-4-5") is True
    assert supports_default_fallback("claude-sonnet-5-5") is True
    assert supports_default_fallback("claude-haiku-5-5") is False


def test_anthropic_missing_key_maps_to_auth_error():
    err = map_anthropic_error(TypeError("Could not resolve authentication method. Expected one of api_key"))
    assert isinstance(err, LLMAuthError)


def test_anthropic_real_call_without_credentials_is_auth_error(monkeypatch):
    # 네트워크 요청 전에 SDK 가 인증 정보 부재로 실패해야 한다 (no network involved).
    if (Path.home() / ".config" / "anthropic").exists():
        pytest.skip("로컬 ant 프로필이 있어 실제 인증이 시도될 수 있음")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    from app.llm.anthropic_provider import AnthropicProvider

    p = AnthropicProvider("claude-haiku-5-5", ProviderOptions(retry=RetryPolicy(max_retries=0)))
    with pytest.raises(LLMAuthError):
        p.generate("s", [{"role": "user", "content": "hi"}], max_tokens=16)
