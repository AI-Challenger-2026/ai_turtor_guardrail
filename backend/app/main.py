"""FastAPI 진입점 (entry point): 앱 생성, CORS, 요청 크기 제한, 오류 응답 변환, 라우터 등록.

실행: uvicorn app.main:app --reload --port 8000   (backend/ 폴더에서)
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import __version__
from app.api import routes_chat, routes_config, routes_health
from app.errors import AppError
from app.logging_utils import get_logger, setup_logging
from app.service import TutorService
from app.settings import Settings, load_settings

log = get_logger("api")


class PayloadTooLargeError(AppError):
    code = "payload_too_large"
    http_status = 413
    user_message = "요청 크기가 너무 큽니다. 대화를 새로 시작하거나 메시지를 줄여 주세요."


def error_payload(code: str, message: str, trace_id: str | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "trace_id": trace_id}}


class BodySizeLimitMiddleware:
    """요청 본문 크기 제한 (ASGI middleware). Content-Length 와 실제 수신 바이트를 모두 검사한다."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._reject(send)
                        return
                except ValueError:
                    pass
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise PayloadTooLargeError()
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, send: Send) -> None:
        body = json.dumps(error_payload(PayloadTooLargeError.code, PayloadTooLargeError.user_message), ensure_ascii=False).encode("utf-8")
        await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json; charset=utf-8"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None, service: TutorService | None = None) -> FastAPI:
    settings = settings or (service.settings if service else load_settings())
    setup_logging(settings.log_level)
    service = service or TutorService(settings)

    app = FastAPI(title="AI Tutor + Guardrail API", version=__version__)
    app.state.service = service
    app.state.settings = settings

    # ---- 오류 응답 변환 (error handlers) ----
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        trace_id = getattr(exc, "trace_id", None)
        if exc.http_status >= 500:
            log.error("AppError %s (%s) trace=%s detail=%s", exc.code, exc.http_status, trace_id, exc.detail)
        else:
            log.info("AppError %s (%s) trace=%s detail=%s", exc.code, exc.http_status, trace_id, exc.detail)
        return JSONResponse(status_code=exc.http_status, content=error_payload(exc.code, exc.user_message, trace_id))

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        where = ""
        if errors:
            loc = [str(x) for x in errors[0].get("loc", []) if x != "body"]
            where = ".".join(loc)
        message = "요청 형식이 올바르지 않습니다." + (f" (확인할 항목: {where})" if where else "")
        return JSONResponse(status_code=422, content=error_payload("validation_error", message))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # FastAPI 는 본문 읽기 중 예외를 400 으로 감싼다 → 원인이 크기 초과이면 413 으로 되돌린다.
        if isinstance(exc.__cause__, PayloadTooLargeError):
            cause = exc.__cause__
            return JSONResponse(status_code=413, content=error_payload(cause.code, cause.user_message))
        messages = {
            400: "요청 본문을 해석할 수 없습니다.",
            404: "요청한 주소를 찾을 수 없습니다.",
            405: "허용되지 않은 요청 방식입니다.",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(f"http_{exc.status_code}", messages.get(exc.status_code, str(exc.detail))),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("처리되지 않은 오류 (unhandled error): %s", exc)
        return JSONResponse(
            status_code=500,
            content=error_payload("internal_error", "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", getattr(exc, "trace_id", None)),
        )

    # ---- 미들웨어 (middleware) — 나중에 추가한 것이 바깥쪽. CORS 를 가장 바깥에 두어 413 응답에도 CORS 헤더가 붙게 한다.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    # ---- 루트 안내 페이지 — 백엔드 주소를 브라우저로 열었을 때 채팅 화면 위치를 알려 준다 ----
    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def root() -> str:
        return (
            "<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>AI 튜터 API</title></head>"
            "<body style='font-family:sans-serif;max-width:640px;margin:48px auto;padding:0 16px;line-height:1.6'>"
            "<h1>AI 튜터 백엔드 API 서버</h1>"
            "<p>이 주소는 API 서버입니다. <b>채팅 화면은 프론트엔드 개발 서버</b>에서 열립니다 "
            "(기본 <a href='http://localhost:5173'>http://localhost:5173</a>, "
            "<code>frontend</code> 폴더에서 <code>npm run dev</code> 실행 필요).</p>"
            "<ul><li><a href='/api/health'>/api/health</a> — 서버 상태</li>"
            "<li><a href='/api/config'>/api/config</a> — 적용 중인 설정</li>"
            "<li><a href='/docs'>/docs</a> — API 문서</li></ul></body></html>"
        )

    # ---- 라우터 (routers) ----
    app.include_router(routes_health.router, prefix="/api")
    app.include_router(routes_config.router, prefix="/api")
    app.include_router(routes_chat.router, prefix="/api")
    return app


def _create_default_app() -> FastAPI:
    return create_app()


app = _create_default_app()
