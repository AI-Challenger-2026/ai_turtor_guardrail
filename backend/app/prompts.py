"""프롬프트 파일 로더 (prompt store).

모든 시스템/판정 프롬프트는 backend/prompts/ 의 텍스트·YAML 파일로 관리한다.
치환 문법은 string.Template 의 ${name} 을 사용한다 — 프롬프트 안의 JSON 예시 중괄호({})와 충돌하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from threading import Lock
from typing import Any

import yaml


class PromptStore:
    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, Any] = {}
        self._lock = Lock()

    def _path(self, name: str) -> Path:
        path = (self.prompts_dir / name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"프롬프트 파일이 없습니다 (prompt not found): {path}")
        return path

    def text(self, name: str) -> str:
        """텍스트 프롬프트 원문 (raw prompt text)."""
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._path(name).read_text(encoding="utf-8")
            return self._cache[name]

    def render(self, name: str, **variables: Any) -> str:
        """${var} 치환 후 반환 (render with string.Template.safe_substitute)."""
        return Template(self.text(name)).safe_substitute({k: str(v) for k, v in variables.items()})

    def yaml(self, name: str) -> dict[str, Any]:
        """YAML 프롬프트 묶음 (e.g. refusal_templates.yaml)."""
        with self._lock:
            key = f"yaml::{name}"
            if key not in self._cache:
                with self._path(name).open("r", encoding="utf-8") as f:
                    self._cache[key] = yaml.safe_load(f) or {}
            return self._cache[key]


def render_template(template: str, **variables: Any) -> str:
    """임의 문자열 템플릿 치환 (render an inline template string)."""
    return Template(template).safe_substitute({k: str(v) for k, v in variables.items()})
