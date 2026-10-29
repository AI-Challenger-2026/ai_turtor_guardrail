"""POST /api/chat — TutorService.handle() 호출."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.service import HandleOptions, TutorService

router = APIRouter(tags=["chat"])

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 413, 422, 429, 500, 501, 502, 503, 504)}


@router.post("/chat", response_model=ChatResponse, responses=_ERRORS)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    # 동기 함수(def)로 두면 FastAPI 가 스레드풀에서 실행한다 — LLM SDK 동기 호출이 이벤트 루프를 막지 않는다.
    service: TutorService = request.app.state.service
    result = service.handle(
        body.messages,
        HandleOptions(
            answer_mode=body.answer_mode,
            guard_profile=body.guard_profile,
            grounding_mode=body.grounding_mode,
            source="api",
        ),
    )
    return result.to_response()
