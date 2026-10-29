from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.errors import LLMAuthError, LLMRateLimitError, LLMTimeoutError
from app.llm.base import LLMProvider, RawCompletion
from app.main import create_app
from app.service import TutorService


class FailingProvider(LLMProvider):
    name = "failing"

    def __init__(self, error: Exception):
        super().__init__("failing-model")
        self.error = error

    def _complete(self, system, messages, temperature, max_tokens, json_mode, task) -> RawCompletion:
        raise self.error


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


def _assert_error_shape(resp, status: int, code: str | None = None):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    if code:
        assert body["error"]["code"] == code
    return body["error"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["mock_mode"] is True


def test_config_shape(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    for key in ("mock_mode", "generator", "judge", "answer_mode", "grounding_mode", "kb", "retrieval", "guards", "limits"):
        assert key in body
    assert body["generator"]["provider"] == "mock"
    assert body["generator"]["configured_model"] == "claude-sonnet-5-5"
    statuses = {"enabled", "disabled", "not_implemented"}
    assert body["kb"]["status"] in statuses
    for module in body["retrieval"].values():
        assert module["status"] in statuses
    assert body["guards"]["status"] in statuses
    assert "ANTHROPIC_API_KEY" not in r.text


def test_chat_response_shape(client):
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "변수가 뭐야?"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    expected = {"answer", "citations", "grounding", "grounded", "status", "blocked", "block_category", "guard_trace",
                "provider", "model", "answer_mode", "latency_ms", "usage", "meta", "trace_id"}
    assert expected <= set(body)
    assert body["provider"] == "mock" and body["answer"].startswith("[MOCK]")
    assert body["citations"] == [] and body["guard_trace"] is None and body["grounding"] == "general"
    assert "trace" not in body  # 내부 추적 정보는 API 응답에 싣지 않는다


def test_validation_error_format(client):
    _assert_error_shape(client.post("/api/chat", json={"messages": []}), 422, "validation_error")
    _assert_error_shape(client.post("/api/chat", json={"messages": [{"role": "system", "content": "x"}]}), 422)
    _assert_error_shape(client.post("/api/chat", content=b"{not json", headers={"content-type": "application/json"}), 422)


def test_message_too_long_is_400(make_settings):
    c = TestClient(create_app(make_settings(MAX_MESSAGE_CHARS="5")))
    err = _assert_error_shape(c.post("/api/chat", json={"messages": [{"role": "user", "content": "123456"}]}), 400, "message_too_long")
    assert "5자" in err["message"]


def test_body_too_large_is_413(make_settings):
    c = TestClient(create_app(make_settings(MAX_REQUEST_BYTES="200")))
    big = {"messages": [{"role": "user", "content": "가" * 300}]}
    _assert_error_shape(c.post("/api/chat", json=big), 413, "payload_too_large")


def test_unknown_route_is_json_404(client):
    _assert_error_shape(client.get("/api/nope"), 404)


def test_root_points_to_frontend(client):
    r = client.get("/")
    assert r.status_code == 200 and "localhost:5173" in r.text and "/api/health" in r.text


@pytest.mark.parametrize(
    "error,status,code",
    [
        (LLMAuthError(), 502, "llm_auth_error"),
        (LLMRateLimitError(), 429, "llm_rate_limited"),
        (LLMTimeoutError(), 504, "llm_timeout"),
    ],
)
def test_llm_errors_map_to_korean_messages(settings, error, status, code):
    svc = TutorService(settings)
    svc.generator = FailingProvider(error)
    svc.generator.options.retry.max_retries = 0
    svc.agent.provider = svc.generator
    c = TestClient(create_app(settings, service=svc))
    err = _assert_error_shape(c.post("/api/chat", json={"messages": [{"role": "user", "content": "q"}]}), status, code)
    assert err["trace_id"]
    assert any("가" <= ch <= "힣" for ch in err["message"])  # 한국어 메시지
    # 오류 후에도 서버는 계속 응답한다
    assert c.get("/api/health").status_code == 200


def test_unexpected_exception_returns_500_json(settings):
    svc = TutorService(settings)
    svc.generator = FailingProvider(RuntimeError("boom"))
    svc.agent.provider = svc.generator
    c = TestClient(create_app(settings, service=svc), raise_server_exceptions=False)
    _assert_error_shape(c.post("/api/chat", json={"messages": [{"role": "user", "content": "q"}]}), 500, "internal_error")
    assert c.get("/api/health").status_code == 200


def test_cors_header_for_allowed_origin(client):
    r = client.options(
        "/api/chat",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
