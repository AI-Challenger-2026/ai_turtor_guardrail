"""Offline tests for the app skeleton. No network: the OpenAI client is replaced by a fake.

Run: python3 -m unittest discover -s tests -v
"""
import asyncio
import json
import logging
import os
import re
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Must be set before importing the app. The base URL is unreachable on purpose so an
# accidental real call fails fast instead of reaching the API.
os.environ.update(TEAM_TOKEN="test-token", OPENAI_MODEL="test-model", OPENAI_API_KEY="test-key",
                  OPENAI_BASE_URL="http://127.0.0.1:9/v1")
# No embedding settings: retrieval runs on BM25 alone unless a test installs a fake client.
for _name in ("EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY"):
    os.environ.pop(_name, None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))

import httpx  # noqa: E402
import scoring  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app import config, generate, llm, main, prompts, templates  # noqa: E402
from app.main import SafetyNet, app  # noqa: E402

MARKER = "MARKER-7f3a9c"
AUTH = {"Authorization": "Bearer test-token"}


def chat(*turns):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": t} for i, t in enumerate(turns)]


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))  # includes traceback text if exc_info is set


NORMAL_JSON = '{"category": "normal", "harm_kind": "none"}'


def gen_json(answer, found=True, citations=()):
    return json.dumps({"found": found, "citations": list(citations), "answer": answer}, ensure_ascii=False)


DEFAULT_GEN = gen_json(f"기본 답변 {MARKER}")
DEFAULTS = {"classify": NORMAL_JSON, "generate": DEFAULT_GEN, "ground": '{"grounded": true, "unsupported": []}',
            "found": '{"found": true}', "translate": "한국어 검색어"}


def call_kind(call) -> str:
    """Which component made a chat call, told apart by its system prompt."""
    system = call["messages"][0]["content"]
    if system == prompts.CLASSIFIER_SYSTEM:
        return "classify"
    if system == prompts.GROUND_SYSTEM:
        return "ground"
    if system == prompts.FOUND_SYSTEM:
        return "found"
    if system.startswith(prompts.TRANSLATE_SYSTEM.split("{", 1)[0]):
        return "translate"
    return "generate"


def is_classify(call) -> bool:
    return call_kind(call) == "classify"


class FakeCompletions:
    """Scripted stand-in for client.chat.completions.

    Each component has its own script (`scripts[kind]`), falling back to DEFAULTS[kind]:
    `json_script` is the classifier's and `script` the generator's (any call that is not the
    classifier, the guard judges or the translator).
    """

    def __init__(self):
        self.scripts = {kind: [] for kind in DEFAULTS}
        self.calls = []

    @property
    def script(self):
        return self.scripts["generate"]

    @script.setter
    def script(self, value):
        self.scripts["generate"] = list(value)

    @property
    def json_script(self):
        return self.scripts["classify"]

    @json_script.setter
    def json_script(self, value):
        self.scripts["classify"] = list(value)

    def calls_of(self, kind):
        return [c for c in self.calls if call_kind(c) == kind]

    def gen_calls(self):
        return self.calls_of("generate")

    def json_calls(self):
        return self.calls_of("classify")

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        kind_of_call = call_kind(kwargs)
        default = DEFAULTS[kind_of_call]
        queue = self.scripts[kind_of_call]
        kind, value = queue.pop(0) if queue else ("text", default)
        if kind == "raise":
            raise value
        if kind == "sleep":
            await asyncio.sleep(value)
            return self._resp(default)
        if kind == "truncated":
            return self._resp(value, finish_reason="length")
        return self._resp(value)

    @staticmethod
    def _resp(text, finish_reason="stop"):
        choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])


class Base(unittest.TestCase):
    def setUp(self):
        self.fake = FakeCompletions()
        # One classifier call per request unless a test turns voting on (tests/test_classify.py VotingTests).
        self.patches = [mock.patch.object(llm, "_client", SimpleNamespace(chat=SimpleNamespace(completions=self.fake))),
                        mock.patch.object(config, "CLASSIFY_VOTES", 1)]
        for p in self.patches:
            p.start()
        self.cap = Capture()
        self.loggers = [logging.getLogger(), logging.getLogger("app")]
        self.saved_root_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.DEBUG)
        for lg in self.loggers:
            lg.addHandler(self.cap)
        self.client = TestClient(app, raise_server_exceptions=True, follow_redirects=False)

    def tearDown(self):
        for lg in self.loggers:
            lg.removeHandler(self.cap)
        logging.getLogger().setLevel(self.saved_root_level)
        for p in self.patches:
            p.stop()
        leaked = [line for line in self.cap.lines if MARKER in line]
        self.assertEqual(leaked, [], "request/answer content leaked into logs")

    def post(self, path, body, headers=AUTH):
        if isinstance(body, (bytes, str)):
            return self.client.post(path, content=body, headers={**headers, "Content-Type": "application/json"})
        return self.client.post(path, json=body, headers=headers)

    def request_lines(self):
        return [line for line in self.cap.lines if "event=request" in line]


class AuthTests(Base):
    def test_missing_and_wrong_token(self):
        body = {"messages": chat(f"질문 {MARKER}")}
        for path in ("/tutor", "/guard"):
            for headers in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "test-token"}):
                self.assertEqual(self.post(path, body, headers).status_code, 401)
        self.assertEqual(self.fake.calls, [])

    def test_wrong_token_beats_validation_errors(self):
        bad = {"Authorization": "Bearer nope"}
        for path in ("/tutor", "/guard"):
            self.assertEqual(self.post(path, "{not json " + MARKER, bad).status_code, 401)
            self.assertEqual(self.post(path, {"nothing": MARKER}, bad).status_code, 401)
            self.assertEqual(self.post(path, {"messages": [{"role": "system", "content": MARKER}]}, bad).status_code, 401)
        self.assertEqual(self.fake.calls, [])


class TutorTests(Base):
    def test_answer_path(self):
        self.fake.script = [("text", gen_json(f"  답변입니다 {MARKER}  ", citations=["FIN-02-04"]))]
        msgs = chat("학습자료의 내용을 요약해 주세요.", "네.", f"다람쥐 꿈모으기 정기예금 최소 가입 금액 {MARKER}")
        r = self.post("/tutor", {"spec_version": "1.0", "session_id": "s_1", "messages": msgs})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data, {"answer": f"답변입니다 {MARKER}", "action": "answer", "citations": ["FIN-02-04"]})
        self.assertIsNotNone(scoring.parse_tutor(data))
        self.assertEqual(len(self.fake.json_calls()), 1)
        call = self.fake.gen_calls()[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual([m["role"] for m in call["messages"]], ["system", "user"])
        self.assertIn("=== 섹션 FIN-02-04 ===", call["messages"][1]["content"])
        self.assertIn(MARKER, call["messages"][1]["content"])
        self.assertEqual(call["extra_body"], {"reasoning_effort": "low"})
        self.assertEqual(call["max_completion_tokens"], 2000)
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertGreater(call["timeout"], 20)

    def test_extra_fields_are_ignored(self):
        msgs = [{"role": "user", "content": f"질문 {MARKER}", "name": "x"}]
        r = self.post("/tutor", {"spec_version": 1.0, "session_id": 123, "extra": [1], "messages": msgs})
        self.assertEqual(r.json()["action"], "answer")
        self.assertIn(f"질문 {MARKER}", self.fake.gen_calls()[0]["messages"][1]["content"])

    def test_invalid_requests_fall_back_to_clarify(self):
        bodies = [
            "{not json " + MARKER, "", "[]", "null",
            {}, {"messages": []}, {"messages": "hi " + MARKER},
            {"messages": [{"role": "system", "content": MARKER}]},
            {"messages": [{"role": "user", "content": 1}]},
            {"messages": [{"role": "user", "content": None}]},
            {"messages": [{"role": "User", "content": MARKER}]},
            {"messages": [{"role": "user"}]},
            {"messages": [{"role": "assistant", "content": MARKER}]},
        ]
        for body in bodies:
            with self.subTest(body=body):
                r = self.post("/tutor", body)
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json(), {"answer": templates.CLARIFY, "action": "clarify", "citations": []})
        self.assertEqual(self.fake.calls, [])
        self.assertTrue(all("outcome=fallback_invalid" in line for line in self.request_lines()))

    def test_llm_error_falls_back(self):
        self.fake.script = [("raise", RuntimeError(f"boom {MARKER}"))] * 2
        r = self.post("/tutor", {"messages": chat(f"질문 {MARKER}")})
        self.assertEqual(r.json(), {"answer": templates.FALLBACK, "action": "answer", "citations": []})
        self.assertTrue(any("event=llm" in l and "error=RuntimeError" in l for l in self.cap.lines))
        self.assertTrue(any("outcome=fallback_generate" in l for l in self.request_lines()))

    def test_empty_completion_falls_back(self):
        self.fake.script = [("text", "   ")] * 2
        r = self.post("/tutor", {"messages": chat("질문")})
        self.assertEqual(r.json()["answer"], templates.FALLBACK)
        self.assertTrue(any("error=EmptyCompletion" in l for l in self.cap.lines))

    def test_token_limit_is_logged_as_truncated(self):
        self.fake.script = [("truncated", '{"found": true, "answer": "가'), ("truncated", "")]
        r = self.post("/tutor", {"messages": chat("질문")})
        self.assertEqual(r.json()["answer"], templates.FALLBACK)
        self.assertTrue(any("error=TruncatedCompletion" in l for l in self.cap.lines))

    def test_answer_is_capped_at_4000_chars(self):
        self.fake.script = [("text", gen_json("가" * 5000))]
        data = self.post("/tutor", {"messages": chat("질문")}).json()
        self.assertEqual(len(data["answer"]), 4000)
        self.assertIsNotNone(scoring.parse_tutor(data))

    def test_request_deadline(self):
        self.fake.script = [("sleep", 5.0)]
        with mock.patch.object(config, "REQUEST_DEADLINE_S", 1.0):
            t0 = time.perf_counter()
            r = self.post("/tutor", {"messages": chat(f"질문 {MARKER}")})
            elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.5)
        self.assertEqual(r.json(), {"answer": templates.FALLBACK, "action": "answer", "citations": []})
        self.assertTrue(any("outcome=fallback_deadline" in l for l in self.request_lines()))
        self.assertTrue(any("event=llm" in l and "status=cancelled" in l for l in self.cap.lines))

    def test_exception_inside_route_is_contained(self):
        with mock.patch.object(main.retrieval, "retrieve", side_effect=ValueError(MARKER)):
            r = self.post("/tutor", {"messages": chat(f"질문 {MARKER}")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], templates.FALLBACK)
        self.assertTrue(any("outcome=fallback_exception error=ValueError" in l for l in self.request_lines()))

    def test_escaped_exception_is_not_reraised(self):
        # _stats runs outside the route's try block, so this exercises SafetyNet. With
        # raise_server_exceptions=True a re-raise would fail the test.
        with mock.patch.object(main, "_stats", side_effect=RuntimeError(MARKER)):
            r = self.post("/tutor", {"messages": chat(f"질문 {MARKER}")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"answer": templates.FALLBACK, "action": "answer", "citations": []})
        self.assertTrue(any("outcome=fallback_escaped error=RuntimeError" in l for l in self.request_lines()))


class GuardTests(Base):
    def test_without_candidate_has_no_output(self):
        data = self.post("/guard", {"messages": chat(f"질문 {MARKER}")}).json()
        self.assertEqual(data, {"input": {"decision": "allow", "category": "normal"}})
        self.assertIsNotNone(scoring.parse_guard(data, False))

    def test_with_candidate_has_output(self):
        for cand in (f"답변 {MARKER}", ""):
            with self.subTest(cand=cand):
                data = self.post("/guard", {"messages": chat("질문"), "candidate_answer": cand}).json()
                self.assertEqual(data["output"], {"grounded": True, "unsupported": [], "safe": True, "category": "normal"})
                self.assertIsNotNone(scoring.parse_guard(data, True))

    def test_invalid_request_falls_back_with_output(self):
        for body in ("{bad " + MARKER, {"messages": []}, {"messages": chat("q"), "candidate_answer": 5}):
            with self.subTest(body=body):
                r = self.post("/guard", body)
                self.assertEqual(r.status_code, 200)
                self.assertIsNotNone(scoring.parse_guard(r.json(), True))
                self.assertEqual(r.json()["input"], {"decision": "allow", "category": "normal"})

    def test_escaped_exception_keeps_output(self):
        with mock.patch.object(main, "_stats", side_effect=RuntimeError(MARKER)):
            r = self.post("/guard", {"messages": chat("q"), "candidate_answer": "a"})
        self.assertEqual(r.status_code, 200)
        self.assertIsNotNone(scoring.parse_guard(r.json(), True))

    def test_guard_calls_the_classifier_and_the_grounding_judge(self):
        self.post("/guard", {"messages": chat("q"), "candidate_answer": "a"})
        self.assertEqual(len(self.fake.json_calls()), 1)
        self.assertEqual(len(self.fake.calls_of("ground")), 1)
        self.assertEqual(self.fake.gen_calls(), [])


class RoutingTests(Base):
    def test_no_redirects_and_no_extra_routes(self):
        r = self.post("/tutor/", {"messages": chat("q")})
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("location", r.headers)
        self.assertEqual(self.client.get("/tutor", headers=AUTH).status_code, 405)
        for path in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_unknown_path_is_not_logged_verbatim(self):
        self.assertEqual(self.post(f"/{MARKER}", {"messages": chat("q")}).status_code, 404)
        self.assertTrue(any("endpoint=other status=404" in l for l in self.request_lines()))


class LogTests(Base):
    def test_one_metadata_line_per_request(self):
        msgs = chat("가나다", "라마", "바사아자")
        self.post("/tutor", {"messages": msgs})
        self.post("/guard", {"messages": msgs, "candidate_answer": "12345"})
        lines = self.request_lines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"event=request endpoint=/tutor status=200 latency_ms=\d+ body_bytes=\d+ "
                                   r"input_chars=9 n_msgs=3 outcome=ok$")
        self.assertRegex(lines[1], r"endpoint=/guard status=200 .* input_chars=14 n_msgs=3 outcome=ok$")
        self.assertTrue(any(re.search(r"event=llm kind=generate status=ok latency_ms=\d+ attempt=1$", l)
                            for l in self.cap.lines))

    def test_third_party_debug_loggers_are_quiet(self):
        for name in ("openai", "httpx", "httpcore"):
            self.assertEqual(logging.getLogger(name).level, logging.WARNING)

    def test_uvicorn_access_log_is_off(self):
        # It would log the client address and raw path of every request.
        self.assertTrue(logging.getLogger("uvicorn.access").disabled)


class SafetyNetUnitTests(unittest.TestCase):
    def _run(self, path, headers):
        async def boom(scope, receive, send):
            raise RuntimeError(MARKER)
        sent = []

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        scope = {"type": "http", "path": path, "headers": headers}
        asyncio.run(SafetyNet(boom)(scope, receive, send))
        return sent[0]["status"], json.loads(sent[1]["body"])

    def test_bad_token_gets_401(self):
        status, _ = self._run("/tutor", [])
        self.assertEqual(status, 401)

    def test_good_token_gets_fallback(self):
        status, body = self._run("/guard", [(b"authorization", b"Bearer test-token")])
        self.assertEqual(status, 200)
        self.assertIsNotNone(scoring.parse_guard(body, True))


class Verdict(BaseModel):
    ok: bool


class ChatJsonTests(Base):
    # These calls carry no classifier system prompt, so the fake serves them from `script`.
    def run_json(self, timeout=lambda: 5.0):
        return asyncio.run(llm.chat_json("classify", [{"role": "user", "content": MARKER}], Verdict,
                                         effort="minimal", max_tokens=50, timeout=timeout))

    def test_valid(self):
        self.fake.script = [("text", '{"ok": true}')]
        self.assertEqual(self.run_json(), Verdict(ok=True))
        self.assertEqual(self.fake.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(self.fake.calls[0]["extra_body"], {"reasoning_effort": "minimal"})

    def test_retry_once_then_succeed(self):
        self.fake.script = [("text", f'{{"ok": "maybe {MARKER}"}}'), ("text", '{"ok": false}')]
        self.assertEqual(self.run_json(), Verdict(ok=False))
        self.assertEqual(len(self.fake.calls), 2)
        self.assertTrue(any("status=invalid" in l and "attempt=1" in l for l in self.cap.lines))

    def test_gives_up_after_one_retry(self):
        self.fake.script = [("raise", RuntimeError(MARKER)), ("text", "not json"), ("text", '{"ok": true}')]
        self.assertIsNone(self.run_json())
        self.assertEqual(len(self.fake.calls), 2)

    def test_truncated_json_is_reported_and_retried(self):
        self.fake.script = [("truncated", '{"ok": tr'), ("text", '{"ok": true}')]
        self.assertEqual(self.run_json(), Verdict(ok=True))
        self.assertTrue(any("error=TruncatedCompletion" in l and "attempt=1" in l for l in self.cap.lines))

    def test_no_call_when_budget_is_spent(self):
        self.assertIsNone(self.run_json(timeout=lambda: 0.0))
        self.assertEqual(self.fake.calls, [])


class ConcurrencyTests(Base):
    def test_64_concurrent_requests_do_not_serialize(self):
        self.fake.script = [("sleep", 0.5)] * 64

        async def go():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                body = {"messages": chat(f"질문 {MARKER}")}
                return await asyncio.gather(*(c.post("/tutor", json=body, headers=AUTH) for _ in range(64)))
        t0 = time.perf_counter()
        rs = asyncio.run(go())
        elapsed = time.perf_counter() - t0
        self.assertTrue(all(r.status_code == 200 and r.json()["action"] == "answer" for r in rs))
        self.assertLess(elapsed, 3.0)


if __name__ == "__main__":
    unittest.main()
