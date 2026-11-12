"""검색·공통 정책·독립 검증·제한된 보정으로 자체 보호 튜터를 구성한다."""

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from .llm import ApiClients, ProviderFailure, RequestBudget
from .policy import IntentSignals, PolicyResult, analyze_intent, apply_material, map_policy
from .prompts import TUTOR_REPAIR_SYSTEM, TUTOR_SYSTEM
from .retrieval import Retriever, SearchResult
from .schemas import ResponseModel, TutorRequest, TutorResponse
from .validation import unavailable_tutor
from .verifier import OutputVerifier, VerificationResult


class TutorDraft(ResponseModel):
    signals: IntentSignals
    coverage: Literal["supported", "general_explanation", "missing_course_fact", "uncertain"]
    answer: str
    citations: list[str]


class TutorRevision(ResponseModel):
    answer: str
    citations: list[str]


@dataclass(frozen=True)
class TutorOutcome:
    response: TutorResponse = field(repr=False)
    verified: bool
    failure_reason: str | None = None


SAFE_RESPONSES = {
    "academic_dishonesty": "제출할 답안을 대신 완성해 드릴 수는 없어요. 먼저 요구사항을 작은 단계로 나누고, 직접 작성한 첫 부분을 보여 주면 다음 단계를 함께 살펴볼게요.",
    "off_topic": "지금은 수업 내용에 집중해 볼까요? 파이썬, 프롬프트 활용, 금융 리터러시 중 배우고 싶은 내용을 알려 주세요.",
    "jailbreak_injection": "튜터의 비공개 지시를 공개하거나 보호 규칙을 해제할 수는 없어요. 대신 수업 내용이나 프롬프트를 안전하게 사용하는 원리는 설명할 수 있어요.",
    "harmful_content": "다른 사람에게 피해를 주거나 개인정보를 침해하는 요청은 도와드릴 수 없어요. 피해를 예방하고 안전하게 대처하는 방법은 함께 살펴볼 수 있어요.",
    "emotional_distress": "혼자 버티기 힘들 만큼 많이 지친 것 같아요. 공부는 잠시 멈추고, 가까운 가족이나 선생님 등 믿을 수 있는 사람에게 지금 상태를 알려 주세요. 지금 안전하지 않다면 혼자 있지 말고 주변 사람에게 즉시 도움을 요청해 주세요.",
    "low_signal_input": "어떤 점이 궁금한지 한 문장으로 알려 줄래요? 배우는 주제나 막힌 부분을 함께 적어 주세요.",
    "out_of_material": "제공된 수업 자료에서는 요청한 내용을 확인할 수 없어요. 관련 자료나 조건을 더 알려 주면 확인 가능한 범위를 함께 살펴볼게요.",
    "answer_seeking": "정답만 알려 주기보다 함께 풀어 볼게요. 먼저 문제에서 주어진 조건과 구하려는 값을 나누어 적어 볼까요? 시도한 첫 단계를 알려 주세요.",
}


def protected_response(policy: PolicyResult) -> TutorResponse:
    category = policy.assessment.category
    if category == "harmful_content" and policy.harmful_kind == "denigration":
        answer = "누군가를 비하하는 말에는 동조하지 않을게요. 서로 존중하며 수업 내용에 대한 질문을 이어가 볼까요?"
    else:
        answer = SAFE_RESPONSES.get(category)
    return TutorResponse(answer=answer, action=policy.action, citations=[]) if answer else unavailable_tutor()


class TutorService:
    def __init__(self, clients: ApiClients, retriever: Retriever, verifier: OutputVerifier):
        self.clients, self.retriever, self.verifier = clients, retriever, verifier
        self.counters = Counter()

    def summary(self) -> dict:
        return {"version": "tutor-v1", "counters": dict(self.counters)}

    def _messages(self, request: TutorRequest, found: SearchResult, notes: dict | None = None) -> list[dict[str, str]]:
        data = {
            "sections": [{"id": hit.section.section_id, "text": hit.section.text} for hit in found.hits],
            "allowed_citations": list(found.section_ids),
            "section_catalog": [{"id": section.section_id, "title": section.title}
                                for section in self.retriever.knowledge.sections],
            "conversation": [message.model_dump() for message in request.messages],
            "current_request": request.messages[-1].content,
        }
        if notes is not None:
            data["repair_notes"] = notes
        return [{"role": "system", "content": TUTOR_REPAIR_SYSTEM if notes is not None else TUTOR_SYSTEM},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]

    @staticmethod
    def _policy(draft: TutorDraft) -> PolicyResult:
        return map_policy(apply_material(analyze_intent(draft.signals), draft.coverage))

    @staticmethod
    def _valid(draft: TutorDraft, checked: VerificationResult, policy: PolicyResult, found: SearchResult) -> bool:
        allowed = (checked.evidence or found).section_ids
        consistent_action = checked.policy is not None and checked.policy.verified and (
            checked.policy.action == policy.action
            or (policy.action == "scaffold" and checked.policy.action == "answer")
        )
        return (
            checked.verified and checked.output.grounded and checked.output.safe and checked.policy_compliant
            and bool(draft.answer.strip()) and len(draft.answer) <= 4000
            and all(section_id in allowed for section_id in (*draft.citations, *checked.supporting_ids))
            and (draft.coverage != "supported" or bool(checked.supporting_ids))
            and consistent_action
        )

    def _protected(self, policy: PolicyResult) -> TutorOutcome:
        self.counters["protected"] += 1
        return TutorOutcome(protected_response(policy), True)

    def _unavailable(self, reason: str) -> TutorOutcome:
        self.counters["failed"] += 1
        self.counters[reason] += 1
        return TutorOutcome(unavailable_tutor(), False, reason)

    def _review_protection(self, checked: VerificationResult, policy: PolicyResult) -> TutorOutcome | None:
        """최초·보정 답변 모두에서 독립 검토가 발견한 위험을 먼저 처리한다."""
        if not checked.safety_verified or checked.policy is None or not checked.policy.verified:
            return None
        actual = checked.policy
        if actual.assessment.category in SAFE_RESPONSES:
            if actual.assessment.category != "answer_seeking" or policy.action != "scaffold":
                return self._protected(actual)
        if not checked.output.safe:
            self.counters["unsafe_drafts"] += 1
            return self._protected(policy) if policy.action == "scaffold" else self._unavailable("unsafe_draft")
        if not checked.policy_compliant and policy.action == "scaffold":
            return self._protected(policy)
        return None

    async def handle(self, request: TutorRequest, budget: RequestBudget) -> TutorOutcome:
        self.counters["requests"] += 1
        try:
            settings = self.clients.settings
            found = await self.retriever.search(request.messages, budget, top_k=12)
            draft_seconds = budget.available_for(
                settings.draft_timeout, reserve=settings.verification_reserve + settings.response_margin,
            )
            if draft_seconds <= 0:
                raise ProviderFailure("insufficient_verification_budget")
            draft = await self.clients.complete_json(
                self._messages(request, found), TutorDraft, budget,
                seconds=draft_seconds, max_tokens=2000,
            )
            policy = self._policy(draft)
            if policy.assessment.category in SAFE_RESPONSES and policy.assessment.category != "answer_seeking":
                return self._protected(policy)
            checked = await self.verifier.verify(
                request.messages, draft.answer, budget, evidence=found, citations=draft.citations,
                expected_action=policy.action,
                seconds=budget.available_for(settings.tutor_verification_timeout, reserve=settings.response_margin),
            )
            protected = self._review_protection(checked, policy)
            if protected is not None:
                return protected
            # 근거 표현만 잘못됐고 안전·정책은 확인된 경우 같은 답변을 한 번 재검토한다.
            can_recheck = (
                checked.evidence_issues and checked.claimed_grounded is True
                and checked.safety_verified and checked.output.safe and checked.policy_compliant
                and checked.policy is not None and checked.policy.verified
            )
            recheck_seconds = budget.available_for(settings.recheck_timeout, reserve=settings.response_margin)
            if can_recheck and budget.chat_calls + 1 <= budget.max_chat_calls and recheck_seconds >= settings.minimum_recheck_seconds:
                self.counters["verification_retries"] += 1
                checked = await self.verifier.recheck(
                    request.messages, draft.answer, budget, previous=checked, seconds=recheck_seconds,
                )
                protected = self._review_protection(checked, policy)
                if protected is not None:
                    return protected
            if not self._valid(draft, checked, policy, found):
                # 원문 구간·근거 판정 자체가 실패한 지적으로 새 답변을 만들지 않는다.
                if not checked.verified:
                    raise ProviderFailure(checked.failure_reason or "unverified_tutor_draft")
                # 보정 뒤 독립 검증까지 마칠 시간과 호출 수가 있어야 생성한다.
                repair_seconds = budget.available_for(
                    settings.repair_timeout, reserve=settings.verification_reserve + settings.response_margin,
                )
                if repair_seconds < settings.minimum_repair_seconds or budget.chat_calls + 2 > budget.max_chat_calls:
                    raise ProviderFailure("insufficient_repair_budget")
                self.counters["repairs"] += 1
                found = checked.evidence or found
                notes = {"previous_answer": draft.answer, "unsupported": checked.output.unsupported,
                         "allowed_citations": list(found.section_ids), "max_characters": 4000,
                         "required_action": policy.action, "grounded": checked.output.grounded,
                         "verification_completed": checked.verified, "policy_compliant": checked.policy_compliant}
                revision = await self.clients.complete_json(
                    self._messages(request, found, notes), TutorRevision, budget,
                    seconds=repair_seconds, max_tokens=2000,
                )
                # 보정 출력에는 의도·범위가 없어 최초 보호 행동을 바꿀 수 없다.
                draft = TutorDraft(signals=draft.signals, coverage=draft.coverage,
                                   answer=revision.answer, citations=revision.citations)
                checked = await self.verifier.verify(
                    request.messages, draft.answer, budget, evidence=found, citations=draft.citations,
                    expected_action=policy.action,
                    seconds=budget.available_for(settings.tutor_verification_timeout, reserve=settings.response_margin),
                    search_candidate=False,
                )
                protected = self._review_protection(checked, policy)
                if protected is not None:
                    return protected
                if not self._valid(draft, checked, policy, found):
                    raise ProviderFailure(checked.failure_reason or "unverified_tutor_repair")
            self.counters["verified"] += 1
            # 독립 검증기가 실제 핵심 주장을 뒷받침했다고 확인한 인용만 반환한다.
            return TutorOutcome(TutorResponse(answer=draft.answer, action=policy.action, citations=list(checked.supporting_ids)), True)
        except Exception as exc:
            reason = exc.reason if isinstance(exc, ProviderFailure) else "tutor_error"
            return self._unavailable(reason)
