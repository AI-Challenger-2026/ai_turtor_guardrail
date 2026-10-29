"""테스트 공통 설정 (shared fixtures). 모든 테스트는 MOCK_MODE 로 API 키 없이 돌아간다."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.settings import Settings, load_settings  # noqa: E402


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """OS 환경변수와 .env 를 무시하는 격리된 설정을 만든다 (hermetic settings)."""

    def _make(**overrides: str) -> Settings:
        env = {
            "MOCK_MODE": "true",
            "LOG_DIR": str(tmp_path / "logs"),
            "KB_INDEX_PATH": str(tmp_path / "index" / "kb_index.json"),
            "LLM_RETRY_BASE_DELAY": "0",
            "GUARD_PROFILE": "none",
            "GROUNDING_MODE": "general",
        }
        env.update({k: str(v) for k, v in overrides.items()})
        return load_settings(env, env_file=False, include_os_environ=False)

    return _make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()
