"""최종 응답의 크기와 장애 시 규격 응답을 보호한다."""

from starlette.responses import JSONResponse

from .schemas import GuardResponse, InputAssessment, OutputAssessment, ResponseModel, TutorResponse


MAX_RESPONSE_BYTES = 256 * 1024
UNAVAILABLE_ANSWER = "지금은 답변을 확인할 수 없어요. 잠시 후 다시 질문해 주세요."


class ResponseLimitError(ValueError):
    """응답을 규격 안에서 직렬화하지 못했음을 나타낸다."""


def unavailable_tutor() -> TutorResponse:
    return TutorResponse(answer=UNAVAILABLE_ANSWER, action="abstain", citations=[])


def unavailable_guard(has_candidate: bool) -> GuardResponse:
    output = None
    if has_candidate:
        output = OutputAssessment(
            grounded=False, unsupported=[], safe=False, category="verification_unavailable",
        )
    return GuardResponse(input=InputAssessment(decision="block", category="normal"), output=output)


def json_response(model: ResponseModel) -> JSONResponse:
    try:
        response = JSONResponse(model.model_dump(exclude_none=True), headers={"Cache-Control": "no-store"})
    except (UnicodeError, ValueError, TypeError):
        raise ResponseLimitError("response_serialization_failed") from None
    if len(response.body) > MAX_RESPONSE_BYTES:
        raise ResponseLimitError("response_too_large")
    return response
