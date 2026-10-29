"""InputLLMGuard (LLM 판정, input) — 대화 맥락을 포함해 질문을 5개 카테고리로 분류한다.

카테고리: in_scope, out_of_scope, harmful_sensitive, jailbreak, disguised_attack
프롬프트: prompts/input_classifier.txt (temperature 0, JSON 출력)
params:
  block_categories: 차단할 카테고리 목록 (기본: in_scope 를 뺀 전부)
  min_score: 이 확신도 이상일 때만 차단 (기본 0.5)
"""

from __future__ import annotations

from app.guardrails.base import Guard, GuardContext, sanitize_for_tags
from app.guardrails.registry import register_guard
from app.history import format_history
from app.schemas import GuardResult

CATEGORIES = ("in_scope", "out_of_scope", "harmful_sensitive", "jailbreak", "disguised_attack")
DEFAULT_BLOCK = ["out_of_scope", "harmful_sensitive", "jailbreak", "disguised_attack"]


@register_guard
class InputLLMGuard(Guard):
    name = "input_llm"
    stage = "input"
    kind = "llm"
    description = "LLM 입력 분류기 (맥락 포함)"

    def check(self, ctx: GuardContext) -> GuardResult:
        block_categories = list(self.params.get("block_categories", DEFAULT_BLOCK))
        min_score = float(self.params.get("min_score", 0.5))
        system = self.deps.prompts.render(
            self.params.get("prompt", "input_classifier.txt"),
            scope=ctx.scope_description or "프로그래밍(파이썬) 학습",
        )
        user = (
            f"<history>\n{sanitize_for_tags(format_history(ctx.history, self.deps.history_turns))}\n</history>\n"
            f"<question>{sanitize_for_tags(ctx.question)}</question>\n"
            f"<rewritten_question>{sanitize_for_tags(ctx.rewritten_question)}</rewritten_question>"
        )
        res = self.deps.judge.generate(
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=self.deps.judge_params.max_tokens,
            json_mode=True,
            task="input_classify",
        )
        data = res.data or {}
        category = str(data.get("category", "")).strip()
        if category not in CATEGORIES:
            raise ValueError(f"알 수 없는 분류 결과: {category!r}")
        score = float(data.get("score", 0.0) or 0.0)
        reason = str(data.get("reason", ""))[:300]
        if category in block_categories and score >= min_score:
            return self.result("block", category, reason, score)
        return self.result("allow", category, reason, score)
