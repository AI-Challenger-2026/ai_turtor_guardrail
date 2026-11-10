"""FastAPI app serving POST /tutor and POST /guard.

Error policy: every authenticated request gets HTTP 200 with a spec-valid JSON body.
- 401 only for a missing/wrong token (also checked on the validation-error path, because
  FastAPI parses the JSON body before it runs dependencies).
- Request validation errors -> 200 fallback (exception handler below).
- Any other exception -> 200 fallback, caught inside the route or, as a last resort, by
  SafetyNet. Nothing is re-raised: Starlette's ServerErrorMiddleware re-raises to uvicorn,
  which would log a traceback including the exception message (which may contain input).
"""
import asyncio
import json
import secrets
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import config, generate, logs, retrieval, templates
from .deadline import Deadline
from .schemas import (GuardInput, GuardOutput, GuardRequest, GuardResponse, TutorRequest,
                      TutorResponse)

logs.setup()

TUTOR_PATH = "/tutor"
GUARD_PATH = "/guard"
_EXPECTED_AUTH = f"Bearer {config.TEAM_TOKEN}".encode()


def token_ok(value) -> bool:
    raw = value if isinstance(value, bytes) else str(value or "").encode("utf-8")
    return secrets.compare_digest(raw, _EXPECTED_AUTH)


def check_token(authorization: str = Header("")) -> None:
    if not token_ok(authorization):
        raise HTTPException(status_code=401, detail="invalid token")


def tutor_fallback(*, invalid_request: bool = False) -> dict:
    if invalid_request:
        return TutorResponse(answer=templates.CLARIFY, action="clarify").model_dump()
    return TutorResponse(answer=templates.FALLBACK, action="answer").model_dump()


def guard_fallback(*, with_output: bool = True) -> dict:
    # When we cannot tell whether a candidate was sent, include "output": it is ignored
    # if no candidate was sent, and required if one was.
    output = GuardOutput(grounded=True, unsupported=[], safe=True) if with_output else None
    return GuardResponse(input=GuardInput(decision="allow", category="normal"), output=output).wire()


def _log_info(request: Request) -> dict:
    # The per-request metadata dict created by SafetyNet (a throwaway dict without it).
    return request.scope.get("state", {}).get("log", {})


def _stats(req) -> dict:
    return {"input_chars": req.input_chars(), "n_msgs": len(req.messages)}


class SafetyNet:
    """Outermost app middleware (inside Starlette's ServerErrorMiddleware).

    Logs one metadata line per request and converts any escaped exception into a 200
    fallback (401 if the token is wrong) without re-raising it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.perf_counter()
        info = {"body_bytes": 0}
        scope.setdefault("state", {})["log"] = info
        sent = {"started": False, "status": None}

        async def receive_counted():
            message = await receive()
            if message["type"] == "http.request":
                info["body_bytes"] += len(message.get("body", b""))
            return message

        async def send_tracked(message):
            if message["type"] == "http.response.start":
                sent["started"] = True
                sent["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive_counted, send_tracked)
        except Exception as exc:
            info["outcome"] = "fallback_escaped"
            info["error"] = type(exc).__name__
            if not sent["started"]:
                await self._send_fallback(scope, send, sent)
        finally:
            try:
                logs.request_line(
                    endpoint=logs.endpoint_label(scope.get("path", "")), status=sent["status"],
                    latency_ms=round((time.perf_counter() - t0) * 1000), body_bytes=info["body_bytes"],
                    input_chars=info.get("input_chars"), n_msgs=info.get("n_msgs"),
                    outcome=info.get("outcome"), error=info.get("error"))
            except Exception:
                pass

    @staticmethod
    async def _send_fallback(scope, send, sent) -> None:
        headers = dict(scope.get("headers") or [])
        if not token_ok(headers.get(b"authorization", b"")):
            status, body = 401, {"detail": "invalid token"}
        else:
            status = 200
            body = guard_fallback() if scope.get("path") == GUARD_PATH else tutor_fallback()
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})
            sent["status"] = status
        except Exception:
            pass  # client already gone


app = FastAPI(redirect_slashes=False, dependencies=[Depends(check_token)],
              docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SafetyNet)


@app.exception_handler(RequestValidationError)
async def on_invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    if not token_ok(request.headers.get("authorization", "")):
        return JSONResponse({"detail": "invalid token"}, status_code=401)
    _log_info(request).update(outcome="fallback_invalid", error=type(exc).__name__)
    if request.scope.get("path") == GUARD_PATH:
        return JSONResponse(guard_fallback())
    return JSONResponse(tutor_fallback(invalid_request=True))


@app.post(TUTOR_PATH)
async def tutor(req: TutorRequest, request: Request) -> JSONResponse:
    info = _log_info(request)
    info.update(_stats(req))
    deadline = Deadline(config.REQUEST_DEADLINE_S)
    try:
        body, outcome = await asyncio.wait_for(_tutor(req, deadline), timeout=config.REQUEST_DEADLINE_S)
        info["outcome"] = outcome
    except asyncio.TimeoutError:
        body = tutor_fallback()
        info.update(outcome="fallback_deadline", error="TimeoutError")
    except Exception as exc:
        body = tutor_fallback()
        info.update(outcome="fallback_exception", error=type(exc).__name__)
    return JSONResponse(body)


async def _tutor(req: TutorRequest, deadline: Deadline):
    hits = await retrieval.retrieve(req.messages)
    gen = await generate.generate_tutor(req.messages, None, hits, deadline=deadline)
    if gen is None:
        return tutor_fallback(), "fallback_generate"
    return TutorResponse(answer=gen.answer, action="answer", citations=gen.citations).model_dump(), "ok"


@app.post(GUARD_PATH)
async def guard(req: GuardRequest, request: Request) -> JSONResponse:
    info = _log_info(request)
    info.update(_stats(req))
    with_output = req.candidate_answer is not None
    deadline = Deadline(config.REQUEST_DEADLINE_S)
    try:
        body, outcome = await asyncio.wait_for(_guard(req, deadline), timeout=config.REQUEST_DEADLINE_S)
        info["outcome"] = outcome
    except asyncio.TimeoutError:
        body = guard_fallback(with_output=with_output)
        info.update(outcome="fallback_deadline", error="TimeoutError")
    except Exception as exc:
        body = guard_fallback(with_output=with_output)
        info.update(outcome="fallback_exception", error=type(exc).__name__)
    return JSONResponse(body)


async def _guard(req: GuardRequest, deadline: Deadline):
    # Step 2: the starter server's constant verdict. Step 3 adds the input classifier.
    output = GuardOutput(grounded=True, unsupported=[], safe=True) if req.candidate_answer is not None else None
    return GuardResponse(input=GuardInput(decision="allow", category="normal"), output=output).wire(), "ok"
