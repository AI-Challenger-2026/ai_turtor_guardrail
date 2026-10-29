"""가드 공통 인터페이스 (guard interface).

Guard:       name, stage("input" | "output"), kind("rule" | "llm"), check(ctx) → GuardResult
GuardResult: decision(allow | block | revise), category, reason, score(0~1), latency_ms  (app.schemas.GuardResult)
GuardContext: 원문 질문, 재작성 질문, 대화 이력, 검색 청크, 생성 답변(출력 단계에서만)

새 가드를 만들 때는 Guard 를 상속하고 @register_guard 를 붙인다 (app/guardrails/registry.py).
latency_ms 와 attempt 는 파이프라인이 채우므로 check() 에서는 신경 쓰지 않아도 된다.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from app.llm.base import LLMProvider
from app.prompts import PromptStore
from app.retrieval.types import RetrievedChunk
from app.schemas import GuardResult
from app.settings import LLMParams

Stage = Literal["input", "output"]
Kind = Literal["rule", "llm"]


@dataclass
class GuardDeps:
    """가드가 쓸 수 있는 공용 자원 (shared dependencies)."""

    judge: LLMProvider
    prompts: PromptStore
    judge_params: LLMParams
    configs_dir: Path
    history_turns: int = 3


@dataclass
class GuardContext:
    question: str
    rewritten_question: str
    history: list[dict[str, str]]
    grounding_mode: str = "general"
    scope_description: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    answer: str | None = None
    cited_indices: list[int] = field(default_factory=list)
    attempt: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


_TAG_LIKE = re.compile(r"</?\s*(question|rewritten_question|history|answer|sources?|scope)\b[^>]*>", re.IGNORECASE)


def sanitize_for_tags(text: str) -> str:
    """판정 프롬프트의 구분 태그를 사용자가 흉내 내지 못하게 지운다 (prompt-injection hardening)."""
    return _TAG_LIKE.sub("", text)


class Guard(ABC):
    name: ClassVar[str] = "base"
    stage: ClassVar[Stage] = "input"
    kind: ClassVar[Kind] = "rule"
    description: ClassVar[str] = ""

    def __init__(self, deps: GuardDeps, **params: Any) -> None:
        self.deps = deps
        self.params = params

    @abstractmethod
    def check(self, ctx: GuardContext) -> GuardResult:
        """판정 결과를 돌려준다. 예외를 던지면 파이프라인이 on_error 정책으로 처리한다."""

    # ---- 헬퍼 (helpers) ----
    def result(
        self,
        decision: Literal["allow", "block", "revise"],
        category: str = "none",
        reason: str = "",
        score: float = 0.0,
        **details: Any,
    ) -> GuardResult:
        return GuardResult(
            name=self.name,
            stage=self.stage,
            decision=decision,
            category=category,
            reason=reason,
            score=max(0.0, min(1.0, float(score))),
            details=details,
        )

    def not_applicable(self, reason: str) -> GuardResult:
        """이번 턴에 적용 대상이 아님 (예: general 모드의 근거 검증) — 통과로 기록하되 사유를 남긴다."""
        return self.result("allow", "not_applicable", reason, 0.0)
