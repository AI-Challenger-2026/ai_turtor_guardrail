"""가드 등록소 (guard registry).

- 기본 가드 모듈과 guardrails/members/ 아래 팀원 플러그인을 자동으로 import 해 등록한다.
- 프로필 YAML 의 `guard: <name>` 은 여기 등록된 이름으로 찾는다.
"""

from __future__ import annotations

import importlib
import pkgutil
from threading import Lock

from app.guardrails.base import Guard
from app.logging_utils import get_logger

log = get_logger("guardrails.registry")

_REGISTRY: dict[str, type[Guard]] = {}
_LOADED = False
_LOCK = Lock()

_BUILTIN_MODULES = (
    "app.guardrails.input_rule",
    "app.guardrails.input_llm",
    "app.guardrails.citation",
    "app.guardrails.groundedness",
    "app.guardrails.output_safety",
)


def register_guard(cls: type[Guard]) -> type[Guard]:
    """클래스 데코레이터: @register_guard 를 붙이면 프로필에서 name 으로 쓸 수 있다."""
    if not cls.name or cls.name == "base":
        raise ValueError(f"가드 클래스에 고유한 name 이 필요합니다: {cls.__name__}")
    if cls.stage not in ("input", "output"):
        raise ValueError(f"stage 는 input 또는 output 이어야 합니다: {cls.__name__}")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls and existing.__qualname__ != cls.__qualname__:
        raise ValueError(f"가드 이름이 중복됩니다: {cls.name} ({existing.__module__} vs {cls.__module__})")
    _REGISTRY[cls.name] = cls
    return cls


def _load_all() -> None:
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
        import app.guardrails.members as members_pkg

        for info in pkgutil.iter_modules(members_pkg.__path__, prefix="app.guardrails.members."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:  # 팀원 플러그인 하나의 오류가 서버 전체를 막지 않게 한다
                log.error("가드 플러그인 로딩 실패 (plugin import failed) %s: %s", info.name, exc)
        _LOADED = True


def get_guard_class(name: str) -> type[Guard]:
    _load_all()
    if name not in _REGISTRY:
        raise KeyError(f"등록되지 않은 가드입니다: {name} (등록된 가드: {', '.join(sorted(_REGISTRY))})")
    return _REGISTRY[name]


def registered_guards() -> dict[str, type[Guard]]:
    _load_all()
    return dict(_REGISTRY)
