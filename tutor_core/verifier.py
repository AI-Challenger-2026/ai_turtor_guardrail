"""후보의 근거·안전을 검증하고 근거 없는 구간을 원문 위치로 복원한다."""

import asyncio
import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .llm import ApiClients, ProviderFailure, RequestBudget
from .policy import InputAnalysis, PolicyResult, map_policy, unavailable_policy
from .prompts import GROUNDING_RECHECK_SYSTEM, INPUT_POLICY_RULES, OUTPUT_SYSTEM
from .retrieval import Retriever, SearchResult, merge_evidence
from .schemas import Action, Category, Message, OutputAssessment, ResponseModel


EVIDENCE_ISSUES = frozenset({"invalid_supporting_ids", "invalid_unsupported_spans", "inconsistent_grounding"})


class GroundingReview(ResponseModel):
    grounded: bool
    unsupported: list[str]
    supporting_ids: list[str]


class SafetyReview(ResponseModel):
    safe: bool
    category: Category
    policy_compliant: bool
    input_analysis: InputAnalysis


class OutputReview(ResponseModel):
    grounding: GroundingReview
    safety: SafetyReview


@dataclass(frozen=True)
class VerificationResult:
    output: OutputAssessment = field(repr=False)
    grounding_verified: bool
    safety_verified: bool
    supporting_ids: tuple[str, ...] = ()
    policy: PolicyResult | None = None
    policy_compliant: bool = False
    failure_reason: str | None = None
    evidence: SearchResult | None = field(default=None, repr=False)
    evidence_issues: tuple[str, ...] = ()
    claimed_grounded: bool | None = None
    candidate_digest: str | None = field(default=None, repr=False)
    context_digest: str | None = field(default=None, repr=False)

    @property
    def verified(self) -> bool:
        return self.grounding_verified and self.safety_verified


def _normalized_positions(text: str) -> tuple[str, list[int]]:
    normalized, positions = [], []
    for index, char in enumerate(text):
        for expanded in unicodedata.normalize("NFKC", char):
            if not expanded.isspace() and not unicodedata.category(expanded).startswith("P"):
                normalized.append(expanded)
                positions.append(index)
    return "".join(normalized), positions


def recover_spans(candidate: str, quotations: Sequence[str]) -> tuple[list[str], bool]:
    """띄어쓰기·문장부호 정규화 결과를 원래 문자열의 구간으로 되돌린다."""
    normalized, positions = _normalized_positions(candidate)
    spans, valid = [], True
    for quote in quotations:
        if not quote or not quote.strip():
            valid = False
            continue
        start = candidate.find(quote)
        if start >= 0:
            spans.append((start, start + len(quote)))
            continue
        needle, _ = _normalized_positions(quote)
        match = normalized.find(needle) if needle else -1
        if match < 0:
            valid = False
            continue
        spans.append((positions[match], positions[match + len(needle) - 1] + 1))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [candidate[start:end] for start, end in merged], valid


def unavailable_verification(reason: str) -> VerificationResult:
    return VerificationResult(
        OutputAssessment(grounded=False, unsupported=[], safe=False, category="verification_unavailable"),
        False, False, policy=unavailable_policy(reason), failure_reason=reason,
    )


class OutputVerifier:
    def __init__(self, clients: ApiClients, retriever: Retriever):
        self.clients = clients
        self.retriever = retriever
        self.counters = Counter()

    def summary(self) -> dict:
        return {"version": "output-v1", "counters": dict(self.counters)}

    @staticmethod
    def _digests(messages, candidate):
        """안전 판정을 다른 질문·답변에 재사용하지 않도록 요청 안에서만 비교한다."""
        context = json.dumps([message.model_dump() for message in messages], ensure_ascii=False, separators=(",", ":"))
        return tuple(hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest() for value in (candidate, context))

    def _grounding(self, candidate, review, found):
        unsupported, spans_valid = recover_spans(candidate, review.unsupported)
        ids = tuple(dict.fromkeys(review.supporting_ids))
        ids_valid = all(section_id in found.section_ids for section_id in ids)
        consistent = not (review.grounded and unsupported)
        issues = tuple(code for valid, code in (
            (ids_valid, "invalid_supporting_ids"), (spans_valid, "invalid_unsupported_spans"),
            (consistent, "inconsistent_grounding"),
        ) if not valid)
        verified = spans_valid and ids_valid and consistent
        if not verified:
            self.counters["invalid_output_evidence"] += 1
            self.counters.update(issues)
        return unsupported, ids if ids_valid else (), verified, issues

    async def recheck(self, messages, candidate, budget, *, previous: VerificationResult, seconds: float):
        """같은 질문·답변의 안전 판정을 보존하고 근거 표현만 한 번 다시 검증한다."""
        digests = self._digests(messages, candidate)
        if digests != (previous.candidate_digest, previous.context_digest):
            return unavailable_verification("recheck_context_mismatch")
        if not (previous.evidence is not None and previous.safety_verified and previous.output.safe
                and previous.policy_compliant and previous.policy is not None and previous.policy.verified
                and previous.claimed_grounded is True and previous.evidence_issues):
            return previous
        self.counters["requests"] += 1
        self.counters["representation_rechecks"] += 1

        async def evaluate():
            found = previous.evidence
            data = {
                "conversation": [message.model_dump() for message in messages], "candidate_answer": candidate,
                "sections": [{"id": hit.section.section_id, "text": hit.section.text} for hit in found.hits],
                "allowed_supporting_ids": list(found.section_ids), "representation_repair_issues": list(previous.evidence_issues),
            }
            review = await self.clients.complete_json(
                [{"role": "system", "content": GROUNDING_RECHECK_SYSTEM},
                 {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
                GroundingReview, budget, seconds=seconds, max_tokens=2000,
            )
            unsupported, ids, verified, issues = self._grounding(candidate, review, found)
            return VerificationResult(
                OutputAssessment(grounded=review.grounded and verified, unsupported=unsupported,
                                 safe=previous.output.safe, category=previous.output.category),
                verified, previous.safety_verified, ids, previous.policy, previous.policy_compliant,
                None if verified else "invalid_output_evidence", found, issues, review.grounded, *digests,
            )

        try:
            result = await budget.run(evaluate, seconds)
            self.counters["verified" if result.verified else "failed"] += 1
            return result
        except Exception as exc:
            reason = exc.reason if isinstance(exc, ProviderFailure) else "output_verification_error"
            self.counters["failed"] += 1
            self.counters[reason] += 1
            return VerificationResult(
                OutputAssessment(grounded=False, unsupported=[], safe=previous.output.safe, category=previous.output.category),
                False, previous.safety_verified, policy=previous.policy, policy_compliant=previous.policy_compliant,
                failure_reason=reason, evidence=previous.evidence, candidate_digest=digests[0], context_digest=digests[1],
            )

    async def _evidence(
        self, messages: Sequence[Message], candidate: str, budget: RequestBudget,
        evidence: SearchResult | None, citations: Sequence[str] | None, search_candidate: bool,
    ) -> SearchResult:
        # 후보는 출력의 근거 검색에만 사용하고 입력 분류에는 전달하지 않는다.
        claim_query = [Message(role="user", content=candidate)]
        if evidence is None:
            tasks = [asyncio.create_task(self.retriever.search(query, budget, top_k=8))
                     for query in (messages, claim_query)]
            try:
                question, claims = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        elif search_candidate:
            question = evidence
            claims = await self.retriever.search(claim_query, budget, top_k=8)
        else:
            question = claims = evidence
        return self._merge_evidence(question, claims, citations)

    def _merge_evidence(self, question, claims, citations):
        return merge_evidence(self.retriever.knowledge, question, claims, citations or ())

    async def verify(
        self, messages: Sequence[Message], candidate: str, budget: RequestBudget, *,
        evidence: SearchResult | None = None, citations: Sequence[str] | None = None,
        expected_action: Action | None = None, seconds: float | None = None,
        search_candidate: bool = True,
        repair_issues: Sequence[str] = (),
    ) -> VerificationResult:
        self.counters["requests"] += 1
        if candidate == "":
            self.counters["verified"] += 1
            return VerificationResult(
                OutputAssessment(grounded=True, unsupported=[], safe=True, category="normal"),
                True, True, policy_compliant=True,
            )

        async def evaluate():
            found = await self._evidence(messages, candidate, budget, evidence, citations, search_candidate)
            data = {
                "conversation": [message.model_dump() for message in messages],
                "candidate_answer": candidate,
                "sections": [{"id": hit.section.section_id, "text": hit.section.text} for hit in found.hits],
                "allowed_supporting_ids": list(found.section_ids),
                "citations": list(citations) if citations is not None else None,
                "expected_action": expected_action,
            }
            system = INPUT_POLICY_RULES + "\n\n" + OUTPUT_SYSTEM
            if repair_issues:
                data["representation_repair_issues"] = [issue for issue in repair_issues if issue in EVIDENCE_ISSUES]
                system += (
                    "\n이번 호출은 같은 후보·근거의 검토 표현을 재검사한다. "
                    "supporting_ids는 allowed_supporting_ids에서만 선택하고 unsupported는 후보 원문의 실제 부분 문자열로 적는다. "
                    "grounded와 unsupported의 모순을 남기지 않는다. 근거·안전을 자동으로 참으로 바꾸지 말고 원문을 다시 확인한다."
                )
                self.counters["representation_rechecks"] += 1
            review = await self.clients.complete_json(
                [{"role": "system", "content": system},
                 {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
                OutputReview, budget, seconds=seconds or self.clients.settings.output_timeout, max_tokens=2000,
            )
            unsupported, ids, grounding_verified, issues = self._grounding(candidate, review.grounding, found)
            grounded = review.grounding.grounded and grounding_verified
            return VerificationResult(
                OutputAssessment(grounded=grounded, unsupported=unsupported, safe=review.safety.safe,
                                 category=review.safety.category),
                grounding_verified, True, ids, map_policy(review.safety.input_analysis),
                review.safety.policy_compliant,
                None if grounding_verified else "invalid_output_evidence",
                found,
                issues,
                review.grounding.grounded,
                *self._digests(messages, candidate),
            )

        try:
            # 검색과 연결 대기까지 포함해 이 분기의 시간을 제한한다.
            timeout = seconds if seconds is not None else self.clients.settings.output_timeout + self.clients.settings.embedding_timeout
            result = await budget.run(evaluate, timeout)
            self.counters["verified" if result.verified else "failed"] += 1
            return result
        except Exception as exc:
            reason = exc.reason if isinstance(exc, ProviderFailure) else "output_verification_error"
            self.counters["failed"] += 1
            self.counters[reason] += 1
            return unavailable_verification(reason)
