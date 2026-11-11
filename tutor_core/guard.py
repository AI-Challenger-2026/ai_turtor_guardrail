"""후보를 입력 분류에서 격리하고 입력·출력을 병렬 판정한다."""

import asyncio
from dataclasses import dataclass

from .llm import RequestBudget
from .policy import InputPolicy, PolicyResult, unavailable_policy
from .schemas import GuardRequest, GuardResponse, OutputAssessment
from .verifier import OutputVerifier, VerificationResult, unavailable_verification


@dataclass(frozen=True)
class GuardAssessment:
    response: GuardResponse
    policy: PolicyResult
    verification: VerificationResult | None = None


async def assess(
    request: GuardRequest, policy: InputPolicy, budget: RequestBudget,
    verifier: OutputVerifier | None = None,
) -> GuardAssessment:
    async def classify():
        try:
            return await policy.classify(request.messages, budget)
        except Exception:
            return unavailable_policy("input_policy_error")

    async def verify():
        if request.candidate_answer == "":
            return VerificationResult(
                OutputAssessment(grounded=True, unsupported=[], safe=True, category="normal"),
                True, True, policy_compliant=True,
            )
        if verifier is None:
            return unavailable_verification("output_verifier_unavailable")
        try:
            return await verifier.verify(request.messages, request.candidate_answer, budget)
        except Exception:
            return unavailable_verification("output_verification_error")

    if not request.has_candidate:
        result = await classify()
        return GuardAssessment(GuardResponse(input=result.assessment), result)
    result, checked = await asyncio.gather(classify(), verify())
    return GuardAssessment(GuardResponse(input=result.assessment, output=checked.output), result, checked)
