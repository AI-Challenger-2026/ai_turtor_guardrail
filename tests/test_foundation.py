"""외부 통신 없이 계약과 장애 처리를 검사하며 실제 API 성공으로 간주하지 않는다."""

import asyncio
import io
import json
import logging
import time
import unittest
from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from server import create_app
from tutor_core.config import ConfigurationError, Settings
from tutor_core.llm import ApiClients, ProviderFailure, RequestBudget
from tutor_core.policy import InputAnalysis, map_policy
from tutor_core.schemas import GuardRequest, OutputAssessment, StructuredProbe, TutorRequest, TutorResponse
from tutor_core.validation import ResponseLimitError, json_response, unavailable_guard
from tutor_core.tutor import TutorOutcome


TEST_ENV = {
    "TEAM_TOKEN": "test-only-bearer",
    "OPENAI_BASE_URL": "https://mlapi.run/test-chat/v1",
    "OPENAI_MODEL": "gpt-5-nano",
    "OPENAI_API_KEY": "test-only-chat-key",
    "EMBEDDING_BASE_URL": "https://mlapi.run/test-embedding/v1",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "EMBEDDING_API_KEY": "test-only-embedding-key",
}
SETTINGS = Settings.from_environment(TEST_ENV)
PAYLOAD = {"spec_version": "1.0", "session_id": "test-session", "messages": [{"role": "user", "content": "학습 질문"}]}


class ConfigurationTests(unittest.TestCase):
    def test_missing_names_without_values(self):
        with self.assertRaises(ConfigurationError) as caught:
            Settings.from_environment({"TEAM_TOKEN": "must-not-appear"})
        self.assertIn("OPENAI_API_KEY", str(caught.exception))
        self.assertNotIn("must-not-appear", str(caught.exception))

    def test_settings_repr_hides_credentials_and_addresses(self):
        representation = repr(SETTINGS)
        for key in ("TEAM_TOKEN", "OPENAI_API_KEY", "EMBEDDING_API_KEY", "OPENAI_BASE_URL", "EMBEDDING_BASE_URL"):
            self.assertNotIn(TEST_ENV[key], representation)

    def test_environment_loader_does_not_open_files(self):
        with patch("builtins.open", side_effect=AssertionError("파일 접근 금지")):
            self.assertEqual(Settings.from_environment(TEST_ENV).chat_model, "gpt-5-nano")

    def test_only_organizer_https_without_credentials_or_redirect_target(self):
        for url in (
            "http://mlapi.run/test/v1", "https://api.openai.com/v1", "https://mlapi.run.invalid/test/v1",
            "https://name:password@mlapi.run/test/v1", "https://mlapi.run/test/v1?key=hidden",
            "https://mlapi.run:bad/test/v1", "https://mlapi.run/test/v1#hidden",
        ):
            with self.subTest(url=url), self.assertRaises(ConfigurationError) as caught:
                Settings.from_environment({**TEST_ENV, "OPENAI_BASE_URL": url})
            self.assertNotIn(url, str(caught.exception))

    def test_invalid_token_and_model_values_are_not_echoed(self):
        for name, value in (("TEAM_TOKEN", "공백 토큰"), ("OPENAI_MODEL", "other-model"), ("OPENAI_API_KEY", "leading\nvalue")):
            with self.subTest(name=name), self.assertRaises(ConfigurationError) as caught:
                Settings.from_environment({**TEST_ENV, name: value})
            self.assertNotIn(value, str(caught.exception))

    def test_token_must_differ_from_api_key(self):
        with self.assertRaises(ConfigurationError):
            Settings.from_environment({**TEST_ENV, "TEAM_TOKEN": TEST_ENV["OPENAI_API_KEY"]})


class SchemaTests(unittest.TestCase):
    def test_missing_empty_and_null_candidate_are_distinct(self):
        self.assertFalse(GuardRequest.model_validate(PAYLOAD).has_candidate)
        self.assertTrue(GuardRequest.model_validate({**PAYLOAD, "candidate_answer": ""}).has_candidate)
        with self.assertRaises(ValidationError):
            GuardRequest.model_validate({**PAYLOAD, "candidate_answer": None})

    def test_invalid_roles_conversation_and_types(self):
        cases = [
            {**PAYLOAD, "messages": []},
            {**PAYLOAD, "messages": [{"role": "system", "content": "지시"}]},
            {**PAYLOAD, "messages": [{"role": "assistant", "content": "답변"}]},
            {**PAYLOAD, "messages": [{"role": "user", "content": 7}]},
            {**PAYLOAD, "spec_version": 1.0},
            {"messages": PAYLOAD["messages"]},
        ]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                TutorRequest.model_validate(payload)

    def test_unicode_limit_counts_codepoints(self):
        result = TutorResponse(answer="😀" * 4000, action="answer", citations=[])
        self.assertEqual(len(result.answer), 4000)
        with self.assertRaises(ValidationError):
            TutorResponse(answer="😀" * 4001, action="answer", citations=[])

    def test_boolean_fields_do_not_coerce_strings_or_numbers(self):
        for invalid in ("false", "true", 0, 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                OutputAssessment(grounded=invalid, unsupported=[], safe=True, category="normal")

    def test_response_bytes_and_invalid_unicode_are_rejected(self):
        for answer, citations in (("한글", ["가" * 100000]), ("\ud800", [])):
            with self.subTest(citations=bool(citations)), self.assertRaises((ResponseLimitError, ValidationError)):
                json_response(TutorResponse(answer=answer, action="answer", citations=citations))

    def test_unavailable_guard_never_invents_spans(self):
        result = unavailable_guard(True)
        self.assertEqual(result.input.decision, "block")
        self.assertFalse(result.output.grounded)
        self.assertFalse(result.output.safe)
        self.assertEqual(result.output.unsupported, [])
        self.assertEqual(result.output.category, "verification_unavailable")


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_cancels_operation(self):
        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()

        start = time.monotonic()
        with self.assertRaises(ProviderFailure) as caught:
            await RequestBudget.start(0.03).run(slow)
        self.assertEqual(caught.exception.reason, "deadline_exceeded")
        self.assertTrue(cancelled.is_set())
        self.assertLess(time.monotonic() - start, 1)

    async def test_expired_budget_does_not_construct_coroutine(self):
        called = False

        def operation():
            nonlocal called
            called = True
            return asyncio.sleep(0)

        with self.assertRaises(ProviderFailure):
            await RequestBudget.start(-1).run(operation)
        self.assertFalse(called)

    async def test_call_budget_is_shared_across_parallel_branches(self):
        budget = RequestBudget.start(1, 4)

        async def claim():
            budget.claim_chat()
            await asyncio.sleep(0)

        results = await asyncio.gather(*(claim() for _ in range(6)), return_exceptions=True)
        self.assertEqual(sum(isinstance(result, ProviderFailure) for result in results), 2)
        self.assertEqual(budget.chat_calls, 4)


def completion(content, finish="stop", refusal=None):
    return httpx.Response(200, json={
        "id": "test-completion", "object": "chat.completion", "created": 0, "model": "gpt-5-nano",
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content, "refusal": refusal}}],
    })


class TransportFaultTests(unittest.IsolatedAsyncioTestCase):
    """SDK를 실제 호출하지 않고 전송 경계에만 정해진 오류를 주입한다."""

    async def asyncSetUp(self):
        self.requests = []
        self.reply = lambda request: completion('{"status":"ok","values":[1,2],"details":{"verified":true}}')

        def transport(request):
            self.requests.append(json.loads(request.content))
            return self.reply(request)

        def factory(base_url, api_key, concurrency):
            return AsyncOpenAI(
                base_url=base_url, api_key=api_key, max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport), follow_redirects=False, trust_env=False),
            )

        with patch.object(ApiClients, "_make_client", side_effect=factory):
            self.clients = ApiClients(SETTINGS)

    async def asyncTearDown(self):
        await self.clients.close()

    async def test_strict_probe_validates_nested_schema(self):
        await self.clients.initialize()
        self.assertEqual(self.clients.structured_mode, "json_schema")
        schema = self.requests[0]["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["$defs"]["ProbeDetails"]["additionalProperties"])
        self.assertEqual(self.requests[0]["reasoning_effort"], "low")
        self.assertFalse(self.requests[0]["stream"])

    async def test_non_stream_token_limit_applies_to_all_chat_paths(self):
        await self.clients.complete_text([], RequestBudget.start(1), seconds=1, max_tokens=3000)
        self.assertEqual(self.requests[-1]["max_completion_tokens"], 2000)
        await self.clients.complete_text([], RequestBudget.start(1), seconds=1, max_tokens=1200)
        self.assertEqual(self.requests[-1]["max_completion_tokens"], 1200)

    async def test_reasoning_effort_is_selected_per_call(self):
        self.clients.structured_mode = "json_schema"
        await self.clients.complete_json([], StructuredProbe, RequestBudget.start(1), seconds=1, max_tokens=1200, reasoning_effort="medium")
        self.assertEqual(self.requests[-1]["reasoning_effort"], "medium")

    async def test_structured_call_merges_policy_and_schema_into_one_system_message(self):
        self.clients.structured_mode = "json_schema"
        await self.clients.complete_json(
            [{"role": "system", "content": "정책 지시 첫째"}, {"role": "system", "content": "정책 지시 둘째"},
             {"role": "user", "content": "분류할 데이터"}],
            StructuredProbe, RequestBudget.start(1), seconds=1, max_tokens=1200,
        )
        messages = self.requests[-1]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("정책 지시 첫째", messages[0]["content"])
        self.assertIn("정책 지시 둘째", messages[0]["content"])
        self.assertIn("JSON", messages[0]["content"])

    async def test_unsupported_schema_uses_validated_json_object(self):
        def reply(request):
            if len(self.requests) == 1:
                return httpx.Response(400, json={"error": {"message": "test-body-must-not-be-logged"}})
            return completion('{"status":"ok","values":[1,2],"details":{"verified":true}}')

        self.reply = reply
        await self.clients.initialize()
        self.assertEqual(self.clients.structured_mode, "json_object")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[1]["response_format"]["type"], "json_object")

    async def test_authentication_and_transient_errors_are_not_retried(self):
        for code in (401, 403, 429, 500):
            self.requests.clear()
            self.reply = lambda request: httpx.Response(code, json={"error": {"message": "private-error-body"}})
            with self.subTest(code=code), self.assertRaises(ProviderFailure) as caught:
                await self.clients.initialize()
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(caught.exception.status_code, code)
            self.assertGreaterEqual(self.clients.counters["upstream_http_" + str(code)], 1)
            self.assertNotIn("private-error-body", str(caught.exception))

    async def test_error_parameter_is_recorded_without_raw_message(self):
        self.reply = lambda request: httpx.Response(400, json={"error": {
            "message": "private-marker response_format", "param": "response_format",
        }})
        with self.assertRaises(ProviderFailure):
            await self.clients.complete_text([], RequestBudget.start(1), seconds=1, max_tokens=10)
        self.assertEqual(self.clients.counters["upstream_parameter_response_format"], 1)
        self.assertNotIn("private-marker", str(self.clients.counters))

    async def test_redirect_is_not_followed(self):
        self.reply = lambda request: httpx.Response(307, headers={"Location": "https://invalid.example/"})
        with self.assertRaises(ProviderFailure):
            await self.clients.initialize()
        self.assertEqual(len(self.requests), 1)

    async def test_refusal_truncation_and_malformed_json_are_distinct_failures(self):
        cases = (
            (completion("{}", "length"), "incomplete_response"),
            (completion(None, refusal="거절"), "model_refusal"),
            (completion("not-json"), "invalid_structured_response"),
            (completion('{"status":"ok","values":[true],"details":{"verified":true}}'), "invalid_structured_response"),
        )
        for response, reason in cases:
            self.reply = lambda request: response
            with self.subTest(reason=reason), self.assertRaises(ProviderFailure) as caught:
                await self.clients.initialize()
            self.assertEqual(caught.exception.reason, reason)

    async def test_semaphore_wait_is_included_in_deadline(self):
        self.clients._chat_slots = asyncio.Semaphore(0)
        with self.assertRaises(ProviderFailure) as caught:
            await self.clients.complete_text([], RequestBudget.start(0.02), seconds=1, max_tokens=10)
        self.assertEqual(caught.exception.reason, "deadline_exceeded")
        self.assertEqual(self.requests, [])

    async def test_embedding_reorders_indices_and_rejects_missing_vectors(self):
        self.reply = lambda request: httpx.Response(200, json={
            "object": "list", "model": "text-embedding-3-small", "usage": {"prompt_tokens": 1, "total_tokens": 1},
            "data": [
                {"object": "embedding", "index": 1, "embedding": [2.0] * 1536},
                {"object": "embedding", "index": 0, "embedding": [1.0] * 1536},
            ],
        })
        vectors = await self.clients.embed(["첫째", "둘째"], RequestBudget.start(1))
        self.assertEqual([row[0] for row in vectors], [1.0, 2.0])
        with self.assertRaises(ProviderFailure):
            await self.clients.embed(["하나"], RequestBudget.start(1))


class ProtocolDouble:
    """실제 API 성공을 흉내 내지 않고 HTTP 계약과 장애만 검사하는 대역."""

    def __init__(self, settings):
        self.settings = settings
        self.counters = Counter()
        self.structured_mode = "json_schema"
        self.closed = False
        self.error = None
        self.delay = 0

    async def initialize(self):
        pass

    async def close(self):
        self.closed = True

    async def complete_text(self, messages, budget, **kwargs):
        self.counters["chat_attempts"] += 1
        if self.error:
            raise self.error
        await asyncio.sleep(self.delay)
        return messages[-1]["content"]


class ProtocolPolicyDouble:
    """프로토콜 검사에서 실제 입력 분류 API를 호출하지 않는 대역."""

    def __init__(self, clients, retriever):
        pass

    async def classify(self, messages, budget):
        return map_policy(InputAnalysis(category="normal", harmful_kind="none", material_status="supported"))

    def summary(self):
        return {"version": "input-v1", "counters": {}}


class ProtocolTutorDouble:
    """HTTP 시간 제한·로그·요청 격리를 검사하는 서비스 경계 대역."""

    def __init__(self, clients, retriever, verifier):
        self.clients = clients

    async def handle(self, request, budget):
        answer = await self.clients.complete_text(
            [message.model_dump() for message in request.messages], budget,
            seconds=self.clients.settings.draft_timeout, max_tokens=2000,
        )
        return TutorOutcome(TutorResponse(answer=answer, action="answer", citations=[]), True)

    def summary(self):
        return {"version": "tutor-v1", "counters": {}}


class ContractFaultTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = replace(SETTINGS, request_timeout=0.1)
        async def prepare_protocol_search(clients):
            class SearchDouble:
                def summary(self):
                    return {"sections": 0, "dense_ready": False}
            return SearchDouble()

        self.application = create_app(
            self.settings, ProtocolDouble, prepare_protocol_search, ProtocolPolicyDouble, tutor_factory=ProtocolTutorDouble,
        )
        self.lifespan = self.application.router.lifespan_context(self.application)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.application), base_url="http://test",
            headers={"Authorization": "Bearer " + self.settings.team_token}, follow_redirects=False,
        )
        self.provider = self.application.state.clients

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.assertTrue(self.provider.closed)

    async def test_authentication_happens_before_body_parsing(self):
        response = await self.client.post("/tutor", content=b"invalid-json", headers={"Authorization": "wrong"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.provider.counters["chat_attempts"], 0)

    async def test_duplicate_authentication_headers_are_rejected(self):
        response = await self.client.post("/tutor", json=PAYLOAD, headers=[
            ("Authorization", "Bearer " + self.settings.team_token),
            ("Authorization", "Bearer " + self.settings.team_token),
        ])
        self.assertEqual(response.status_code, 401)

    async def test_no_redirect_and_no_open_documentation(self):
        for path in ("/tutor/", "/guard/", "/docs", "/openapi.json"):
            response = await self.client.post(path, json=PAYLOAD)
            self.assertEqual(response.status_code, 404)
            self.assertNotIn("location", response.headers)

    async def test_guard_candidate_presence_contract(self):
        absent = await self.client.post("/guard", json=PAYLOAD)
        empty = await self.client.post("/guard", json={**PAYLOAD, "candidate_answer": ""})
        self.assertNotIn("output", absent.json())
        self.assertTrue(empty.json()["output"]["grounded"])
        self.assertTrue(empty.json()["output"]["safe"])

    async def test_validation_does_not_echo_input(self):
        response = await self.client.post("/tutor", json={**PAYLOAD, "messages": [{"role": "hidden-marker", "content": "private-marker"}]})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("hidden-marker", response.text)
        self.assertNotIn("private-marker", response.text)

    async def test_api_failure_uses_json_200_and_metadata_only(self):
        self.provider.error = ValueError("private-upstream-body")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("tutor_server")
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            response = await self.client.post("/tutor", json=PAYLOAD)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "abstain")
        self.assertNotIn("private-upstream-body", response.text + stream.getvalue())
        self.assertNotIn(PAYLOAD["messages"][0]["content"], stream.getvalue())
        self.assertNotIn(self.settings.team_token, stream.getvalue())
        health = (await self.client.get("/healthz")).json()
        self.assertEqual(health["counters"]["degraded_responses"], 1)
        self.assertTrue(health["policy_ready"])

    async def test_route_deadline_bounds_uncooperative_service(self):
        self.provider.delay = 10
        start = time.monotonic()
        response = await self.client.post("/tutor", json=PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "abstain")
        self.assertLess(time.monotonic() - start, 1)

    async def test_parallel_requests_do_not_share_conversation(self):
        async def one(number):
            payload = {**PAYLOAD, "messages": [{"role": "user", "content": str(number)}]}
            return (await self.client.post("/tutor", json=payload)).json()["answer"]

        self.assertEqual(await asyncio.gather(*(one(n) for n in range(8))), [str(n) for n in range(8)])


if __name__ == "__main__":
    unittest.main()
