"""정책 엔진 (policy engine) — 가드 판정을 실제 동작으로 바꾼다.

- 입력 block  → 카테고리별 교육적 거절 메시지 (prompts/refusal_templates.yaml)
- 출력 revise → 뒷받침되지 않는 주장을 빼라는 지시(prompts/revision_instruction.txt)를 붙여 재생성.
               재생성은 max_regenerations(기본 1)회까지만. 그래도 실패하면 안전한 fallback 답변.
- 출력 block  → 안전한 fallback 답변
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from app.guardrails.pipeline import StageResult
from app.prompts import PromptStore, render_template
from app.schemas import GuardResult
from app.tutor.agent import AgentAnswer

FinalAction = Literal["allow", "block", "revise", "fallback"]


@dataclass
class OutputOutcome:
    text: str
    answer: AgentAnswer | None
    status: Literal["answered", "revised", "fallback"]
    final_action: FinalAction
    regenerations: int
    results: list[GuardResult] = field(default_factory=list)
    raw_answers: list[str] = field(default_factory=list)


class PolicyEngine:
    def __init__(self, prompts: PromptStore, topics: Callable[[], str], max_regenerations: int = 1) -> None:
        self.prompts = prompts
        self.topics = topics
        self.max_regenerations = max_regenerations

    def _templates(self) -> dict:
        return self.prompts.yaml("refusal_templates.yaml")

    def refusal(self, category: str) -> str:
        refusals = self._templates().get("refusals", {})
        template = refusals.get(category) or refusals.get("default", "")
        return render_template(template, topics=self.topics()).strip()

    def fallback(self, kind: str) -> str:
        fallbacks = self._templates().get("fallbacks", {})
        template = fallbacks.get(kind) or fallbacks.get("unsupported", "")
        return render_template(template, topics=self.topics()).strip()

    def revision_instruction(self, results: list[GuardResult]) -> str:
        issues: list[str] = []
        for r in results:
            if r.decision != "revise":
                continue
            unsupported = r.details.get("unsupported_claims") or []
            problems = r.details.get("problems") or []
            for claim in unsupported:
                issues.append(f"- 교육 자료로 뒷받침되지 않는 주장: \"{claim}\"")
            for p in problems:
                issues.append(f"- {p}")
            if not unsupported and not problems:
                issues.append(f"- [{r.name}] {r.reason}")
        return self.prompts.render("revision_instruction.txt", issues="\n".join(issues) or "- (세부 사유 없음)")

    def run_output(
        self,
        generate: Callable[[str | None], AgentAnswer],
        check: Callable[[AgentAnswer, int], StageResult],
    ) -> OutputOutcome:
        """생성 → 출력 가드 → (필요 시) 재생성 루프. 재생성은 최대 max_regenerations 회."""
        answer = generate(None)
        raws = [answer.text]
        all_results: list[GuardResult] = []
        attempt = 0
        while True:
            stage = check(answer, attempt)
            all_results.extend(stage.results)
            if stage.decision == "allow":
                return OutputOutcome(
                    text=answer.text, answer=answer, status="answered" if attempt == 0 else "revised",
                    final_action="allow" if attempt == 0 else "revise", regenerations=attempt,
                    results=all_results, raw_answers=raws,
                )
            if stage.decision == "block":
                blocking = stage.blocking
                kind = "guard_error" if blocking is not None and blocking.category == "guard_error" else "unsafe"
                return OutputOutcome(
                    text=self.fallback(kind), answer=None, status="fallback", final_action="block",
                    regenerations=attempt, results=all_results, raw_answers=raws,
                )
            if attempt >= self.max_regenerations:
                return OutputOutcome(
                    text=self.fallback("unsupported"), answer=None, status="fallback", final_action="fallback",
                    regenerations=attempt, results=all_results, raw_answers=raws,
                )
            attempt += 1
            answer = generate(self.revision_instruction(stage.results))
            raws.append(answer.text)
