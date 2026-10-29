"""대화 이력 포맷 유틸리티 (history formatting) — 재작성기와 판정 프롬프트에서 공통으로 쓴다."""

from __future__ import annotations

from typing import Sequence

_LABEL = {"user": "학습자", "assistant": "튜터"}


def format_history(history: Sequence[dict[str, str]], max_turns: int = 3, max_chars: int = 400) -> str:
    """최근 max_turns 개 사용자 턴을 "학습자: ... / 튜터: ..." 줄로 만든다. 긴 발화는 잘라 비용을 줄인다."""
    if not history:
        return "(이전 대화 없음)"
    user_idx = [i for i, m in enumerate(history) if m["role"] == "user"]
    start = user_idx[-max_turns] if len(user_idx) > max_turns else 0
    lines = []
    for m in history[start:]:
        text = " ".join(m["content"].split())
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"{_LABEL.get(m['role'], m['role'])}: {text}")
    return "\n".join(lines)
