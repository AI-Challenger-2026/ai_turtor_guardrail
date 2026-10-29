from __future__ import annotations

from dotenv import dotenv_values

from app.settings import BACKEND_DIR, load_settings


def test_defaults_are_mock_and_general(settings):
    assert settings.mock_mode is True
    assert settings.answer_mode == "adaptive"
    assert settings.grounding_mode == "general"
    assert settings.generator.model == "claude-sonnet-5-5"
    assert settings.judge.model == "claude-haiku-5-5"
    assert settings.max_turns == 20


def test_overrides_and_cors_list(make_settings):
    s = make_settings(MAX_TURNS="5", CORS_ORIGINS="http://a.com, http://b.com", ANSWER_MODE="hint")
    assert s.max_turns == 5
    assert s.cors_origins == ["http://a.com", "http://b.com"]
    assert s.answer_mode == "hint"


def test_api_key_not_in_repr(make_settings):
    s = make_settings(ANTHROPIC_API_KEY="sk-ant-secret-value-123456")
    assert "sk-ant-secret-value-123456" not in repr(s)
    assert "sk-ant-secret-value-123456" not in s.model_dump_json()
    assert s.secret_values() == ["sk-ant-secret-value-123456"]


def test_env_example_parses_with_expected_values():
    values = dotenv_values(BACKEND_DIR / ".env.example", encoding="utf-8")
    assert values["MOCK_MODE"] == "true"
    assert values["GENERATOR_MODEL"] == "claude-sonnet-5-5"
    assert values["JUDGE_MODEL"] == "claude-haiku-5-5"
    assert values["OPENAI_MODEL"] == ""
    assert values["ANTHROPIC_API_KEY"] == ""
    s = load_settings(env_file=BACKEND_DIR / ".env.example", include_os_environ=False)
    assert s.mock_mode is True and s.guard_profile == "none"
