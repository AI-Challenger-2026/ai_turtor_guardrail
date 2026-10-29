"""로깅 유틸리티 (logging utilities): 콘솔 로거 설정, 턴 추적 로그(JSONL), 비밀값 마스킹.

- 턴 로그는 logs/turns_YYYY-MM-DD.jsonl 에 한 줄씩 기록한다 (one JSON object per turn).
- 기록 직전에 API key 값과 흔한 키 패턴을 마스킹한다 (redaction) — 키는 절대 로그에 남지 않는다.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

_KEY_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"(?i)(x-api-key|authorization)\s*[:=]\s*\S+"),
]
REDACTED = "***REDACTED***"


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    """문자열에서 비밀값과 키 패턴을 가린다 (mask secrets)."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def setup_logging(level: str = "INFO") -> None:
    """콘솔 로거 설정 (console logger). 여러 번 호출해도 핸들러가 중복되지 않는다."""
    root = logging.getLogger("tutor")
    root.setLevel(level.upper())
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"tutor.{name}")


class TurnLogger:
    """턴 단위 추적 로그 기록기 (per-turn JSONL trace writer)."""

    def __init__(self, log_dir: Path, secrets: Iterable[str] = (), enabled: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.secrets = [s for s in secrets if s]
        self.enabled = enabled
        self._lock = threading.Lock()

    def path_for_today(self) -> Path:
        return self.log_dir / f"turns_{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    def write(self, record: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        line = json.dumps(record, ensure_ascii=False, default=str)
        line = redact_text(line, self.secrets)
        path = self.path_for_today()
        with self._lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return path
