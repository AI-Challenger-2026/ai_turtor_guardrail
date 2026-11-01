"""환경변수와 설정 파일 로딩 (settings loader).

우선순위 (precedence): 명시적 override > OS 환경변수 > backend/.env > configs/settings.yaml > 코드 기본값.
- 비밀값(API key)은 환경변수에서만 읽고 SecretStr 로 감싸 repr/로그에 노출되지 않게 한다.
- 튜닝용 비밀 아닌 값(청크 크기, top_k, 가격표 등)은 configs/settings.yaml 에 둔다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr

BACKEND_DIR = Path(__file__).resolve().parent.parent

AnswerMode = Literal["direct", "hint", "adaptive"]
GroundingMode = Literal["general", "kb"]

ANSWER_MODES: tuple[str, ...] = ("direct", "hint", "adaptive")
GROUNDING_MODES: tuple[str, ...] = ("general", "kb")


class ProviderConfig(BaseModel):
    """역할별 LLM 제공자 설정 (per-role provider config)."""

    provider: str
    model: str
    effort: str | None = None


class LLMParams(BaseModel):
    temperature: float = 0.3
    max_tokens: int = 2048


class LLMSection(BaseModel):
    generator: LLMParams = LLMParams(temperature=0.3, max_tokens=2000)
    judge: LLMParams = LLMParams(temperature=0.0, max_tokens=2000)
    rewriter: LLMParams = LLMParams(temperature=0.0, max_tokens=512)
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_base_delay: float = 1.0
    retry_max_delay: float = 16.0


class KBSection(BaseModel):
    dir: str = "data/sample/kb"
    index_path: str = "data/index/kb_index.json"
    chunk_size: int = 600
    chunk_overlap: int = 100
    auto_build: bool = True


class DenseSection(BaseModel):
    enabled: bool = False
    model: str = "intfloat/multilingual-e5-small"
    min_score: float = 0.80
    cache: bool = True


class RetrievalSection(BaseModel):
    top_k: int = 4
    min_score: float = 0.25
    min_score_simple: float = 0.20
    tokenizer: Literal["auto", "kiwi", "simple"] = "auto"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    dense: DenseSection = DenseSection()


class RewriteSection(BaseModel):
    policy: Literal["auto", "always", "never"] = "auto"
    max_short_chars: int = 20
    history_turns: int = 3


class GuardsSection(BaseModel):
    default_on_error: Literal["allow", "block"] = "block"
    history_turns: int = 3
    enforce_rule_first: bool = True


class Settings(BaseModel):
    """애플리케이션 전체 설정 (application settings)."""

    backend_dir: Path = BACKEND_DIR
    mock_mode: bool = True

    generator: ProviderConfig = ProviderConfig(provider="anthropic", model="claude-sonnet-5-5", effort="low")
    judge: ProviderConfig = ProviderConfig(provider="anthropic", model="claude-haiku-5-5", effort="low")
    openai_model: str = ""

    anthropic_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    openai_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    elice_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    elice_api_base: str = ""
    anthropic_refusal_fallback: Literal["default", "off"] = "default"

    answer_mode: AnswerMode = "adaptive"
    grounding_mode: GroundingMode = "general"
    guard_profile: str = "default"

    max_message_chars: int = 2000
    max_turns: int = 20
    max_request_bytes: int = 524288
    cors_origins: list[str] = ["http://localhost:5173"]

    log_dir: Path = BACKEND_DIR / "logs"
    log_level: str = "INFO"
    log_turns: bool = True

    prompts_dir: Path = BACKEND_DIR / "prompts"
    configs_dir: Path = BACKEND_DIR / "configs"

    llm: LLMSection = LLMSection()
    kb: KBSection = KBSection()
    retrieval: RetrievalSection = RetrievalSection()
    query_rewrite: RewriteSection = RewriteSection()
    guards: GuardsSection = GuardsSection()
    pricing_usd_per_mtok: dict[str, dict[str, float]] = Field(default_factory=dict)

    # ---- 경로 헬퍼 (path helpers) ----
    def resolve_path(self, value: str | Path) -> Path:
        """backend/ 기준 상대경로를 절대경로로 바꾼다 (resolve relative to backend dir)."""
        p = Path(value)
        return p if p.is_absolute() else (self.backend_dir / p).resolve()

    @property
    def kb_dir(self) -> Path:
        return self.resolve_path(self.kb.dir)

    @property
    def kb_index_path(self) -> Path:
        return self.resolve_path(self.kb.index_path)

    def secret_values(self) -> list[str]:
        """로그 마스킹용 비밀값 목록 (values to redact from logs)."""
        values = [
            self.anthropic_api_key.get_secret_value(),
            self.openai_api_key.get_secret_value(),
            self.elice_api_key.get_secret_value(),
        ]
        return [v for v in values if v and len(v) >= 6]


# ---------------------------------------------------------------------------
# 로딩 함수 (loader)
# ---------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"불리언 값으로 해석할 수 없습니다 (invalid bool): {value!r}")


def _to_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value.strip())


def _to_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value.strip())


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일 형식 오류 (YAML root must be a mapping): {path}")
    return data


def load_settings(
    overrides: Mapping[str, str] | None = None,
    *,
    env_file: Path | None | Literal[False] = None,
    include_os_environ: bool = True,
    backend_dir: Path | None = None,
) -> Settings:
    """설정을 읽어 Settings 를 만든다.

    Args:
        overrides: 최우선 적용할 환경변수 형태의 값 (tests/CLI 에서 사용).
        env_file: .env 경로. None 이면 backend/.env, False 이면 읽지 않는다.
        include_os_environ: OS 환경변수를 반영할지 (테스트 격리용으로 False 가능).
        backend_dir: backend 루트 (기본: 이 파일 기준).
    """
    base_dir = (backend_dir or BACKEND_DIR).resolve()

    env: dict[str, str] = {}
    if env_file is not False:
        path = env_file or (base_dir / ".env")
        if path.exists():
            env.update({k: v for k, v in dotenv_values(path, encoding="utf-8").items() if v is not None})
    if include_os_environ:
        env.update(os.environ)
    if overrides:
        env.update({k: str(v) for k, v in overrides.items()})

    configs_dir = Path(env.get("CONFIGS_DIR") or (base_dir / "configs"))
    if not configs_dir.is_absolute():
        configs_dir = (base_dir / configs_dir).resolve()
    yaml_cfg = _load_yaml(Path(env.get("SETTINGS_FILE") or (configs_dir / "settings.yaml")))

    llm = LLMSection(**(yaml_cfg.get("llm") or {}))
    kb = KBSection(**(yaml_cfg.get("kb") or {}))
    retrieval = RetrievalSection(**(yaml_cfg.get("retrieval") or {}))
    rewrite = RewriteSection(**(yaml_cfg.get("query_rewrite") or {}))
    guards = GuardsSection(**(yaml_cfg.get("guards") or {}))
    pricing = yaml_cfg.get("pricing_usd_per_mtok") or {}

    # 환경변수 override (env overrides for tunables)
    llm.timeout_seconds = _to_float(env.get("LLM_TIMEOUT_SECONDS"), llm.timeout_seconds)
    llm.max_retries = _to_int(env.get("LLM_MAX_RETRIES"), llm.max_retries)
    if _clean(env.get("LLM_RETRY_BASE_DELAY")):
        llm.retry_base_delay = _to_float(env.get("LLM_RETRY_BASE_DELAY"), llm.retry_base_delay)
    if _clean(env.get("KB_DIR")):
        kb.dir = _clean(env.get("KB_DIR"))
    if _clean(env.get("KB_INDEX_PATH")):
        kb.index_path = _clean(env.get("KB_INDEX_PATH"))
    retrieval.top_k = _to_int(env.get("RETRIEVAL_TOP_K"), retrieval.top_k)
    retrieval.dense.enabled = _to_bool(env.get("HYBRID_SEARCH"), retrieval.dense.enabled)

    def _effort(name: str) -> str | None:
        raw = env.get(name)
        if raw is None:
            return "low"
        raw = raw.strip()
        return raw or None

    cors = [o.strip() for o in _clean(env.get("CORS_ORIGINS") or "http://localhost:5173").split(",") if o.strip()]

    log_dir = Path(_clean(env.get("LOG_DIR")) or "logs")
    prompts_dir = Path(_clean(env.get("PROMPTS_DIR")) or "prompts")

    answer_mode = _clean(env.get("ANSWER_MODE")) or "adaptive"
    grounding_mode = _clean(env.get("GROUNDING_MODE")) or "general"
    if answer_mode not in ANSWER_MODES:
        raise ValueError(f"ANSWER_MODE 는 {ANSWER_MODES} 중 하나여야 합니다: {answer_mode!r}")
    if grounding_mode not in GROUNDING_MODES:
        raise ValueError(f"GROUNDING_MODE 는 {GROUNDING_MODES} 중 하나여야 합니다: {grounding_mode!r}")

    settings = Settings(
        backend_dir=base_dir,
        mock_mode=_to_bool(env.get("MOCK_MODE"), True),
        generator=ProviderConfig(
            provider=(_clean(env.get("GENERATOR_PROVIDER")) or "anthropic").lower(),
            model=_clean(env.get("GENERATOR_MODEL")) or "claude-sonnet-5-5",
            effort=_effort("GENERATOR_EFFORT"),
        ),
        judge=ProviderConfig(
            provider=(_clean(env.get("JUDGE_PROVIDER")) or "anthropic").lower(),
            model=_clean(env.get("JUDGE_MODEL")) or "claude-haiku-5-5",
            effort=_effort("JUDGE_EFFORT"),
        ),
        openai_model=_clean(env.get("OPENAI_MODEL")),
        anthropic_api_key=SecretStr(_clean(env.get("ANTHROPIC_API_KEY"))),
        openai_api_key=SecretStr(_clean(env.get("OPENAI_API_KEY"))),
        elice_api_key=SecretStr(_clean(env.get("ELICE_API_KEY"))),
        elice_api_base=_clean(env.get("ELICE_API_BASE")),
        anthropic_refusal_fallback=(_clean(env.get("ANTHROPIC_REFUSAL_FALLBACK")) or "default").lower(),  # type: ignore[arg-type]
        answer_mode=answer_mode,  # type: ignore[arg-type]
        grounding_mode=grounding_mode,  # type: ignore[arg-type]
        guard_profile=_clean(env.get("GUARD_PROFILE")) or "none",
        max_message_chars=_to_int(env.get("MAX_MESSAGE_CHARS"), 2000),
        max_turns=_to_int(env.get("MAX_TURNS"), 20),
        max_request_bytes=_to_int(env.get("MAX_REQUEST_BYTES"), 524288),
        cors_origins=cors,
        log_dir=log_dir if log_dir.is_absolute() else (base_dir / log_dir).resolve(),
        log_level=(_clean(env.get("LOG_LEVEL")) or "INFO").upper(),
        log_turns=_to_bool(env.get("LOG_TURNS"), True),
        prompts_dir=prompts_dir if prompts_dir.is_absolute() else (base_dir / prompts_dir).resolve(),
        configs_dir=configs_dir,
        llm=llm,
        kb=kb,
        retrieval=retrieval,
        query_rewrite=rewrite,
        guards=guards,
        pricing_usd_per_mtok=pricing,
    )
    return settings
