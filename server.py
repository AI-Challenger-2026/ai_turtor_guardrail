"""대회 API의 인증·수명주기·시간 제한을 관리하는 진입점."""

import logging
import secrets
import time
from collections import Counter
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from tutor_core import guard, tutor
from tutor_core.config import ConfigurationError, Settings
from tutor_core.llm import ApiClients, ProviderFailure, RequestBudget, silence_transport_logs
from tutor_core.knowledge import KnowledgeError
from tutor_core.retrieval import prepare_retriever
from tutor_core.policy import InputPolicy
from tutor_core.verifier import OutputVerifier
from tutor_core.identity import IDENTITY_VERSION, runtime_fingerprint
from tutor_core.schemas import GuardRequest, TutorRequest
from tutor_core.validation import json_response, unavailable_guard, unavailable_tutor


LOGGER = logging.getLogger("tutor_server")
IMPLEMENTATION_STAGE = "IMP-001-7"


class BearerAuthentication:
    """본문을 파싱하기 전에 인증하고 원문 토큰을 기록하지 않는다."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
        settings = getattr(scope["app"].state, "settings", None)
        expected = b"" if settings is None else ("Bearer " + settings.team_token).encode("ascii")
        if settings is None or len(headers) != 1 or not secrets.compare_digest(headers[0], expected):
            response = JSONResponse(
                {"detail": "인증에 실패했습니다."}, status_code=401,
                headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class DeadlineRoute(APIRoute):
    """본문 파싱과 연결 대기를 포함해 요청 전체의 내부 마감을 적용한다."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            settings = request.app.state.settings
            budget = RequestBudget.start(settings.request_timeout, settings.max_chat_calls)
            request.state.budget = budget
            request.state.has_candidate = None
            start = time.monotonic()
            endpoint = self.path.lstrip("/")
            status = 200
            reason = "ok"
            try:
                response = await budget.run(lambda: original(request))
                status = response.status_code
                return response
            except (RequestValidationError, HTTPException) as exc:
                status = 422 if isinstance(exc, RequestValidationError) else exc.status_code
                reason = "invalid_request"
                raise
            except Exception as exc:
                reason = exc.reason if isinstance(exc, ProviderFailure) else "internal_error"
                request.app.state.counters["degraded_responses"] += 1
                request.app.state.counters[reason] += 1
                if self.path == "/tutor":
                    return json_response(unavailable_tutor())
                if self.path == "/guard":
                    # 파싱 중 시간 초과라 후보 유무도 모르면 필수 출력이 누락되지 않게 한다.
                    return json_response(unavailable_guard(request.state.has_candidate is not False))
                status = 503
                return JSONResponse({"detail": "상태를 확인할 수 없습니다."}, status_code=status)
            finally:
                request.app.state.counters["requests"] += 1
                LOGGER.info(
                    "endpoint=%s status=%d elapsed_ms=%d result=%s",
                    endpoint, status, int((time.monotonic() - start) * 1000), reason,
                )

        return handler


def create_app(
    settings: Optional[Settings] = None, client_factory: Callable[[Settings], ApiClients] = ApiClients,
    retrieval_factory: Callable = prepare_retriever,
    policy_factory: Callable = InputPolicy,
    verifier_factory: Callable = OutputVerifier,
    tutor_factory: Callable = tutor.TutorService,
) -> FastAPI:
    silence_transport_logs()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ready = False
        application.state.counters = Counter()
        application.state.instance_id = secrets.token_hex(12)
        clients = None
        try:
            resolved = settings if settings is not None else Settings.from_environment()
            application.state.settings = resolved
            startup_fingerprint = runtime_fingerprint()
            clients = client_factory(resolved)
            application.state.clients = clients
            await clients.initialize()
            application.state.retriever = await retrieval_factory(clients)
            application.state.input_policy = policy_factory(clients, application.state.retriever)
            application.state.output_verifier = verifier_factory(clients, application.state.retriever)
            application.state.tutor_service = tutor_factory(
                clients, application.state.retriever, application.state.output_verifier,
            )
            if runtime_fingerprint() != startup_fingerprint:
                raise ProviderFailure("startup_source_changed")
            application.state.runtime_sha256 = startup_fingerprint
            application.state.ready = True
        except ConfigurationError as exc:
            raise RuntimeError(str(exc)) from None
        except ProviderFailure as exc:
            if clients is not None:
                await clients.close()
            raise RuntimeError("API 준비 실패: " + exc.reason) from None
        except KnowledgeError as exc:
            if clients is not None:
                await clients.close()
            raise RuntimeError("KB 준비 실패: " + str(exc)) from None
        except Exception:
            if clients is not None:
                await clients.close()
            raise RuntimeError("서버 준비 실패: startup_error") from None
        try:
            yield
        finally:
            application.state.ready = False
            if clients is not None:
                await clients.close()

    application = FastAPI(
        lifespan=lifespan, redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None,
    )
    application.router.route_class = DeadlineRoute
    application.add_middleware(BearerAuthentication)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # 기본 상세 오류의 input·ctx에 요청 본문이 포함될 수 있어 고정 안내만 반환한다.
        return JSONResponse({"detail": "요청 형식이 규격과 맞지 않습니다."}, status_code=422)

    @application.post("/tutor")
    async def tutor_endpoint(payload: TutorRequest, request: Request):
        result = await request.app.state.tutor_service.handle(payload, request.state.budget)
        if not result.verified:
            request.app.state.counters["degraded_responses"] += 1
            request.app.state.counters[result.failure_reason or "unverified_tutor_draft"] += 1
        return json_response(result.response)

    @application.post("/guard")
    async def guard_endpoint(payload: GuardRequest, request: Request):
        request.state.has_candidate = payload.has_candidate
        result = await guard.assess(
            payload, request.app.state.input_policy, request.state.budget, request.app.state.output_verifier,
        )
        if not result.policy.verified:
            request.app.state.counters["degraded_responses"] += 1
            request.app.state.counters[result.policy.failure_reason] += 1
        if result.verification is not None and not result.verification.verified:
            request.app.state.counters["degraded_responses"] += 1
            request.app.state.counters[result.verification.failure_reason] += 1
        return json_response(result.response)

    @application.get("/healthz")
    async def health(request: Request):
        return JSONResponse({
            "ready": request.app.state.ready,
            "spec_version": "1.0",
            "implementation_stage": IMPLEMENTATION_STAGE,
            "instance_id": request.app.state.instance_id,
            "runtime_sha256": request.app.state.runtime_sha256,
            "runtime_identity_version": IDENTITY_VERSION,
            "policy_ready": request.app.state.ready,
            "input_policy_ready": True,
            "output_policy_ready": True,
            "tutor_policy_ready": request.app.state.ready,
            "tutor_policy": request.app.state.tutor_service.summary(),
            "output_policy": request.app.state.output_verifier.summary(),
            "input_policy": request.app.state.input_policy.summary(),
            "retrieval_ready": request.app.state.ready,
            "retrieval": request.app.state.retriever.summary(),
            "structured_output": request.app.state.clients.structured_mode,
            "counters": dict(request.app.state.counters),
            "provider_counters": dict(request.app.state.clients.counters),
        }, headers={"Cache-Control": "no-store"})

    return application


app = create_app()
