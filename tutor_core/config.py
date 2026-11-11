"""파일을 읽지 않고 프로세스 환경변수만 검증한다."""

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional
from urllib.parse import urlsplit


REQUIRED_ENV = (
    "TEAM_TOKEN",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
)

# 비스트리밍 요청은 기존 시작 코드의 토큰 설정을 호환성 기준으로 사용한다.
NON_STREAM_MAX_COMPLETION_TOKENS = 2000


class ConfigurationError(ValueError):
    """설정값 대신 잘못된 변수 이름만 전달하는 오류."""


def _provider_url(name: str, value: str) -> str:
    try:
        url = urlsplit(value)
        valid = (
            url.scheme == "https"
            and url.hostname == "mlapi.run"
            and url.port in (None, 443)
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
            and url.path.rstrip("/").endswith("/v1")
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError(f"운영진 API 주소 형식을 확인하세요: {name}")
    return value.rstrip("/")


@dataclass(frozen=True)
class Settings:
    team_token: str = field(repr=False)
    chat_base_url: str = field(repr=False)
    chat_api_key: str = field(repr=False)
    embedding_base_url: str = field(repr=False)
    embedding_api_key: str = field(repr=False)
    chat_model: str = "gpt-5-nano"
    embedding_model: str = "text-embedding-3-small"
    request_timeout: float = 27.0
    draft_timeout: float = 10.0
    baseline_timeout: float = 24.0
    input_policy_timeout: float = 24.0
    input_reasoning_effort: str = "low"
    output_timeout: float = 8.0
    tutor_verification_timeout: float = 11.0
    repair_timeout: float = 4.0
    embedding_timeout: float = 3.0
    startup_timeout: float = 45.0
    chat_concurrency: int = 128
    embedding_concurrency: int = 64
    max_chat_calls: int = 4

    @classmethod
    def from_environment(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        values = os.environ if env is None else env
        missing = [name for name in REQUIRED_ENV if not values.get(name, "").strip()]
        if missing:
            raise ConfigurationError("등록되지 않은 환경변수: " + ", ".join(missing))
        for name in REQUIRED_ENV:
            value = values[name]
            if value != value.strip() or any(ord(char) < 32 for char in value):
                raise ConfigurationError(f"공백·제어문자가 포함된 환경변수: {name}")
        token = values["TEAM_TOKEN"]
        if any(not 33 <= ord(char) <= 126 for char in token):
            raise ConfigurationError("TEAM_TOKEN에는 공백 없는 영문·숫자·기호만 사용하세요.")
        if token in (values["OPENAI_API_KEY"], values["EMBEDDING_API_KEY"]):
            raise ConfigurationError("TEAM_TOKEN은 API 키와 다른 값이어야 합니다.")
        expected = {"OPENAI_MODEL": "gpt-5-nano", "EMBEDDING_MODEL": "text-embedding-3-small"}
        for name, model in expected.items():
            if values[name] != model:
                raise ConfigurationError(f"대회 지정 모델을 사용하세요: {name}")
        return cls(
            team_token=token,
            chat_base_url=_provider_url("OPENAI_BASE_URL", values["OPENAI_BASE_URL"]),
            chat_api_key=values["OPENAI_API_KEY"],
            embedding_base_url=_provider_url("EMBEDDING_BASE_URL", values["EMBEDDING_BASE_URL"]),
            embedding_api_key=values["EMBEDDING_API_KEY"],
        )
