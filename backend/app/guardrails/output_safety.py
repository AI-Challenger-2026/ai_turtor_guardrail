"""OutputSafetyGuard (LLM 판정, output) — 미성년자를 포함할 수 있는 학습자에게 부적절한 내용인지 판정.

프롬프트: prompts/output_safety_judge.txt (temperature 0, JSON)
params:
  min_score: 이 확신도 이상으로 unsafe 이면 block (기본 0.5)
"""

from __future__ import annotations

from app.guardrails.base import Guard, GuardContext, sanitize_for_tags
from app.guardrails.registry import register_guard
from app.schemas import GuardResult


@register_guard
class OutputSafetyGuard(Guard):
    name = "output_safety"
    stage = "output"
    kind = "llm"
    description = "출력 부적절 내용 판정"

    def check(self, ctx: GuardContext) -> GuardResult:
        min_score = float(self.params.get("min_score", 0.5))
        user = f"<answer>\n{sanitize_for_tags(ctx.answer or '')}\n</answer>"
        res = self.deps.judge.generate(
            system=self.deps.prompts.text(self.params.get("prompt", "output_safety_judge.txt")),
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=self.deps.judge_params.max_tokens,
            json_mode=True,
            task="output_safety_judge",
        )
        data = res.data or {}
        if "safe" not in data:
            raise ValueError("판정 결과에 safe 필드가 없음")
        safe = bool(data.get("safe"))
        category = str(data.get("category", "safe" if safe else "other"))
        score = float(data.get("score", 0.0) or 0.0)
        reason = str(data.get("reason", ""))[:300]
        if not safe and score >= min_score:
            return self.result("block", category, reason, score)
        return self.result("allow", category, reason, score)
