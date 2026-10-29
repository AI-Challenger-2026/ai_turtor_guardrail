"""member_a 의 가드 구현 (예시 포함).

새 가드 만드는 법 (5분 요약):
  1. 아래 예시처럼 Guard 를 상속하고 name / stage / kind 를 정한다. name 은 "member_a.<이름>".
  2. check(ctx) 에서 self.result("allow" | "block" | "revise", category, reason, score) 를 돌려준다.
  3. LLM 판정이 필요하면 self.deps.judge.generate(..., temperature=0.0, json_mode=True, task="...") 를 쓰고,
     프롬프트는 backend/prompts/ 에 텍스트 파일로 둔 뒤 self.deps.prompts.text("파일명") 으로 읽는다.
  4. configs/guards_member_a.yaml 의 input / output 목록에 `- guard: member_a.<이름>` 을 추가한다.
  5. 웹 화면 개발용 설정에서 가드 프로필 member_a 를 고르거나, python -m eval.run_eval --profile member_a 로 평가한다.
"""

from __future__ import annotations

from app.guardrails.base import Guard, GuardContext
from app.guardrails.registry import register_guard
from app.schemas import GuardResult


@register_guard
class KeywordBlockGuard(Guard):
    """예시 가드: params.keywords 에 있는 단어가 질문(또는 재작성 질문)에 있으면 차단한다."""

    name = "member_a.keyword_block"
    stage = "input"
    kind = "rule"
    description = "member_a 예시 — 키워드 차단"

    def check(self, ctx: GuardContext) -> GuardResult:
        keywords = [str(k).lower() for k in self.params.get("keywords", [])]
        category = str(self.params.get("category", "harmful_sensitive"))
        text = f"{ctx.question}\n{ctx.rewritten_question}".lower()
        for kw in keywords:
            if kw and kw in text:
                return self.result("block", category, f"금지 키워드 '{kw}' 포함", 1.0, keyword=kw)
        return self.result("allow", "none", "금지 키워드 없음", 0.0)
