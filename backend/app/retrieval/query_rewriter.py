"""멀티턴 질의 재작성 (multi-turn query rewriting).

"그럼 2번은요?", "왜요?" 같은 짧은 후속 질문을 대화 이력을 이용해 독립 질문으로 바꾼 뒤
검색과 입력 가드레일에 넘긴다. 답변 생성에는 원문 질문 + 대화 이력을 그대로 쓴다.

policy:
- auto   : 이력이 있고, 짧거나 지시어로 시작하는 질문만 LLM 으로 재작성 (비용 절약)
- always : 이력이 있으면 항상 재작성
- never  : 재작성하지 않음
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.errors import LLMJSONError
from app.history import format_history
from app.llm.base import LLMProvider
from app.prompts import PromptStore
from app.settings import LLMParams, RewriteSection
from app.tutor.modes import is_followup_like


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    applied: bool
    reason: str
    used_llm: bool = False


class QueryRewriter:
    def __init__(self, provider: LLMProvider, prompts: PromptStore, params: LLMParams, config: RewriteSection) -> None:
        self.provider = provider
        self.prompts = prompts
        self.params = params
        self.config = config

    def needs_rewrite(self, question: str, history: Sequence[dict[str, str]]) -> tuple[bool, str]:
        if not history:
            return False, "이전 대화 없음"
        if self.config.policy == "never":
            return False, "policy=never"
        if self.config.policy == "always":
            return True, "policy=always"
        if is_followup_like(question, len(history), self.config.max_short_chars):
            return True, "짧거나 지시어로 시작하는 후속 질문"
        return False, "독립적인 질문으로 판단"

    def rewrite(self, question: str, history: Sequence[dict[str, str]]) -> RewriteResult:
        needed, reason = self.needs_rewrite(question, history)
        if not needed:
            return RewriteResult(question, question, applied=False, reason=reason)
        user = (
            f"<history>\n{format_history(history, self.config.history_turns)}\n</history>\n"
            f"<question>{question}</question>"
        )
        try:
            result = self.provider.generate(
                system=self.prompts.text("query_rewrite.txt"),
                messages=[{"role": "user", "content": user}],
                temperature=self.params.temperature,
                max_tokens=self.params.max_tokens,
                json_mode=True,
                task="query_rewrite",
            )
        except LLMJSONError:
            return RewriteResult(question, question, applied=False, reason="재작성 결과 JSON 해석 실패 → 원문 사용", used_llm=True)
        rewritten = str((result.data or {}).get("standalone_question", "")).strip()
        if not rewritten:
            return RewriteResult(question, question, applied=False, reason="재작성 결과가 비어 원문 사용", used_llm=True)
        return RewriteResult(question, rewritten, applied=rewritten != question, reason=reason, used_llm=True)
