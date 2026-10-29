"""CitationGuard (규칙, output) — 출처 표기 검사.

- 인용한 번호가 실제 검색 청크 범위(1..N) 안에 있는지
- 출처 표기가 빠지지 않았는지 (min_citations 개 이상)
- "자료에서 찾을 수 없다"고 정직하게 답한 경우는 인용이 없어도 통과 (allow_not_found_without_citation)
general 모드에서는 적용 대상이 아니다.
"""

from __future__ import annotations

from app.guardrails.base import Guard, GuardContext
from app.guardrails.registry import register_guard
from app.schemas import GuardResult
from app.tutor.agent import parse_citation_indices

NOT_FOUND_MARKERS = ("찾을 수 없", "자료에 없", "자료에서는 다루지", "자료에 나와 있지 않")


@register_guard
class CitationGuard(Guard):
    name = "citation"
    stage = "output"
    kind = "rule"
    description = "인용 번호 유효성·누락 검사"

    def check(self, ctx: GuardContext) -> GuardResult:
        if ctx.grounding_mode != "kb":
            return self.not_applicable("general 모드에는 출처 표기가 없음")
        answer = ctx.answer or ""
        min_citations = int(self.params.get("min_citations", 1))
        allow_not_found = bool(self.params.get("allow_not_found_without_citation", True))
        cited = parse_citation_indices(answer)
        n = len(ctx.chunks)
        invalid = [i for i in cited if not 1 <= i <= n]
        valid = [i for i in cited if 1 <= i <= n]
        problems: list[str] = []
        if invalid:
            problems.append(f"검색된 자료에 없는 인용 번호 {invalid} 를 사용했다. 1~{n} 번만 쓸 수 있다.")
        if len(valid) < min_citations:
            if allow_not_found and any(m in answer for m in NOT_FOUND_MARKERS):
                if not invalid:
                    return self.result("allow", "not_found_answer", "자료에서 찾을 수 없다고 안내한 답변", 1.0, cited=cited)
            else:
                problems.append("자료 내용을 사용한 문장에 [번호] 출처 표기가 빠졌다.")
        score = (len(valid) / len(cited)) if cited else 0.0
        if problems:
            category = "invalid_citation" if invalid else "missing_citation"
            return self.result("revise", category, " ".join(problems), score, problems=problems, cited=cited, invalid=invalid)
        return self.result("allow", "ok", f"유효한 인용 {len(valid)}개", score, cited=cited)
