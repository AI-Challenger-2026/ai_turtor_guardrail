"""GroundednessGuard (LLM 판정, output) — 답변을 주장 단위로 나눠 각 주장이 검색 청크로 뒷받침되는지 판정.

프롬프트: prompts/groundedness_judge.txt (temperature 0, JSON)
params:
  max_unsupported: 허용할 뒷받침되지 않는 주장 수 (기본 0) — 초과하면 revise
general 모드에서는 적용 대상이 아니다 (근거 청크가 없음).
"""

from __future__ import annotations

from app.guardrails.base import Guard, GuardContext, sanitize_for_tags
from app.guardrails.registry import register_guard
from app.schemas import GuardResult
from app.tutor.agent import render_sources


@register_guard
class GroundednessGuard(Guard):
    name = "groundedness"
    stage = "output"
    kind = "llm"
    description = "주장 단위 근거 충실도 판정"

    def check(self, ctx: GuardContext) -> GuardResult:
        if ctx.grounding_mode != "kb":
            return self.not_applicable("general 모드에는 근거 청크가 없음")
        if not ctx.chunks:
            return self.not_applicable("검색 청크 없음")
        max_unsupported = int(self.params.get("max_unsupported", 0))
        user = (
            f"<sources>\n{render_sources(ctx.chunks)}\n</sources>\n"
            f"<answer>\n{sanitize_for_tags(ctx.answer or '')}\n</answer>"
        )
        res = self.deps.judge.generate(
            system=self.deps.prompts.text(self.params.get("prompt", "groundedness_judge.txt")),
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=self.deps.judge_params.max_tokens,
            json_mode=True,
            task="groundedness_judge",
        )
        data = res.data or {}
        claims = data.get("claims") or []
        unsupported = [str(c) for c in (data.get("unsupported_claims") or [])]
        if not unsupported and isinstance(claims, list):
            unsupported = [str(c.get("claim", "")) for c in claims if isinstance(c, dict) and c.get("supported") is False]
        score = float(data.get("score", 1.0 if not unsupported else 0.0) or 0.0)
        details = {"unsupported_claims": unsupported, "num_claims": len(claims) if isinstance(claims, list) else 0}
        if len(unsupported) > max_unsupported:
            return self.result("revise", "unsupported_claims", f"뒷받침되지 않는 주장 {len(unsupported)}개", score, **details)
        return self.result("allow", "grounded", "모든 주장이 자료로 뒷받침됨" if not unsupported else "허용 범위 내", score, **details)
