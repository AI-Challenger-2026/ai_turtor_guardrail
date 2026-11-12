"""평가 기준 일치·진단 격리·본문 비노출을 외부 API 없이 검사한다."""

import asyncio
import io
import json
import unittest
from contextlib import nullcontext, redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from harness.score_local import score_local
from tools.diagnostic_server import DiagnosticApplication, ObservedApiClients, ObservedInputPolicy, ObservedOutputVerifier, ObservedRetriever, ObservedTutorService
from tools.evaluate import EvaluationFailure, evaluate_public, interactive_environment, main, run_diagnose
from tools.evaluation_trace import CURRENT_TRACE, HEADER, MAX_EVENTS, RequestTrace, decode_trace, observed
from tools.release_checks import observe_public, public_metrics, score_public_cases
from tutor_core.llm import ApiClients, ProviderFailure, RequestBudget
from tutor_core.tutor import TutorDraft, TutorRevision
from server import create_app
from tests.test_foundation import PAYLOAD, SETTINGS
from tests.test_tutor import PipelineClient, SearchDouble, draft
from tests.test_release import successful_public_result


def public_item(endpoint="tutor", **extra):
    item = {"id": "unit-" + endpoint, "endpoint": endpoint,
            "messages": [{"role": "user", "content": "저장 금지 질문 표식"}],
            "labels": {"category": "normal", "action": "answer", "decision": "allow", "evidence": ["PY-01-01"],
                       "gold_facts": ["저장 금지 정답 표식"]}}
    item.update(extra)
    return item


class OriginalJudgeTests(unittest.TestCase):
    def test_original_factory_receives_original_settings_without_extra_probe(self):
        def score(items, client, judge, kb, workers, **kwargs):
            self.assertEqual(judge("facts", {"answer": "저장 금지 답변 표식"}), {"score": .5})
            self.assertEqual(judge("complied", {}), {"complied": False})
            return {"score": {"judge_errors": 0, "invalid": 0}, "endpoint_errors": {}, "deadline_violations": 0}

        health = {"instance_id": "same", "counters": {}}
        original = lambda kind, payload: {"score": .5} if kind == "facts" else {"complied": False}
        with patch("tools.evaluate.make_judge", return_value=original) as factory, \
             patch("tools.evaluate.read_health", return_value=health), \
             patch("tools.evaluate.load_kb_dir", return_value={}), \
             patch("tools.release_checks.score_public_cases", side_effect=score):
            result = evaluate_public(None, SETTINGS, 1)
        factory.assert_called_once_with(SETTINGS.chat_base_url, SETTINGS.chat_model, SETTINGS.chat_api_key, timeout=60, retries=2)
        self.assertEqual(result["judge"]["calls"], 2)
        self.assertTrue(result["judge"]["stream"])
        self.assertEqual(result["judge"]["max_completion_tokens"], 4000)
        self.assertIsNone(result["judge"]["reasoning_effort_override"])
        self.assertNotIn("저장 금지", json.dumps(result, ensure_ascii=False))

    def test_original_scoring_and_aggregate_are_preserved(self):
        items = [public_item(), public_item(id="second")]
        response = {"answer": "저장 금지 답변 표식", "action": "answer", "citations": ["PY-01-01"]}
        judge = lambda kind, payload: {"score": .5, "ok": False, "complied": False}
        reference = score_local(items, lambda endpoint, payload: response, judge, {}, workers=1)
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))) as client:
            measured = score_public_cases(items, client, judge, {}, 1)
        for key in ("total", "tutor", "guard", "judge_errors", "invalid", "per_category", "items"):
            self.assertEqual(measured["score"][key], reference[key])
        self.assertEqual(measured["metrics"]["tutor_facts_score"], .5)
        self.assertEqual(measured["metrics"]["tutor_faithfulness_rate"], 0)
        self.assertNotIn("저장 금지", json.dumps(measured, ensure_ascii=False))

    def test_unobserved_judge_fields_remain_unobserved(self):
        item = public_item()
        observation = observe_public(item, None, {"judge_error": True})
        metrics = public_metrics([observation])
        self.assertIsNone(metrics["tutor_facts_score"])
        self.assertEqual(metrics["tutor_judge_coverage"]["facts"], {"expected": 1, "eligible": 0, "observed": 0})

    def test_trace_is_linked_to_public_id_without_saving_bodies(self):
        trace = RequestTrace()
        trace.record("search", trace.started, section_ids=["PY-01-01"])
        trace.record("tutor", trace.started, "failed", verified=False, failure="deadline_exceeded")
        body = {"answer": "저장 금지 답변 표식", "action": "abstain", "citations": []}
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=body, headers={HEADER: trace.header().decode("ascii")}))) as client:
            result = score_public_cases([public_item()], client, lambda kind, payload: {"score": 0, "ok": True}, {}, 1, diagnostics=True)
        row = result["diagnostics"]["items"][0]
        self.assertEqual(row["id"], "unit-tutor")
        self.assertEqual(row["initial_search_reference_recall"], 1)
        self.assertIn("action_mismatch", row["flags"])
        self.assertIn("stage_failure", row["flags"])
        self.assertNotIn("initial_search_reference_missing", row["flags"])
        self.assertNotIn("저장 금지", json.dumps(result, ensure_ascii=False))


class TraceBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_observer_preserves_result_and_rethrows_same_failure(self):
        trace = RequestTrace()
        token = CURRENT_TRACE.set(trace)
        result = SimpleNamespace(verified=True, answer="본문 표식")
        failure = ProviderFailure("외부 예외 원문 표식")
        try:
            async def success():
                return result

            async def failed():
                raise failure

            self.assertIs(await observed("tutor", success), result)
            with self.assertRaises(ProviderFailure) as caught:
                await observed("tutor", failed)
            self.assertIs(caught.exception, failure)
        finally:
            CURRENT_TRACE.reset(token)
        value = trace.header().decode("ascii")
        self.assertNotIn("표식", value)
        self.assertEqual(decode_trace(value)["events"][-1]["failure"], "unknown_failure")

    async def test_cancellation_is_preserved(self):
        trace = RequestTrace()
        token = CURRENT_TRACE.set(trace)
        try:
            async def cancelled():
                raise asyncio.CancelledError

            with self.assertRaises(asyncio.CancelledError):
                await observed("output_review", cancelled)
        finally:
            CURRENT_TRACE.reset(token)
        self.assertEqual(trace.events[0]["status"], "cancelled")

    async def test_parallel_requests_do_not_share_trace(self):
        async def application(scope, receive, send):
            sid = "PY-01-01" if scope["query_string"] == b"one" else "PY-01-02"

            async def search():
                await asyncio.sleep(.001)
                return SimpleNamespace(section_ids=(sid,))

            await observed("search", search)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"unchanged"})

        async with httpx.AsyncClient(base_url="http://test", transport=httpx.ASGITransport(app=DiagnosticApplication(application))) as client:
            one, two = await asyncio.gather(client.post("/tutor?one"), client.post("/tutor?two"))
        self.assertEqual(one.content, b"unchanged")
        self.assertEqual(two.content, b"unchanged")
        self.assertEqual(decode_trace(one.headers[HEADER])["events"][0]["section_ids"], ["PY-01-01"])
        self.assertEqual(decode_trace(two.headers[HEADER])["events"][0]["section_ids"], ["PY-01-02"])
        self.assertIsNone(CURRENT_TRACE.get())

    async def test_overflow_and_unknown_fields_do_not_appear_complete(self):
        trace = RequestTrace()
        for _ in range(MAX_EVENTS + 1):
            trace.record("search", trace.started, body="본문 표식", token="키 표식")
        value = trace.header().decode("ascii")
        self.assertFalse(decode_trace(value)["complete"])
        self.assertNotIn("표식", value)
        for invalid in (None, "null", "[]", "invalid", '{"version":"wrong"}'):
            self.assertFalse(decode_trace(invalid)["complete"])

    async def test_client_wrapper_preserves_api_arguments_and_draft_order(self):
        clients = object.__new__(ObservedApiClients)
        trace = RequestTrace()
        token = CURRENT_TRACE.set(trace)
        budget = RequestBudget.start(1)
        result = draft()
        try:
            with patch.object(ApiClients, "complete_json", new_callable=AsyncMock, return_value=result) as original:
                self.assertIs(await clients.complete_json([], TutorDraft, budget, seconds=.1, max_tokens=2000), result)
                await clients.complete_json([], TutorDraft, budget, seconds=.1, max_tokens=2000)
                original.assert_awaited_with([], TutorDraft, budget, seconds=.1, max_tokens=2000)
        finally:
            CURRENT_TRACE.reset(token)
        self.assertEqual([event["stage"] for event in trace.events], ["tutor_draft", "tutor_repair"])

    async def test_compact_revision_and_evidence_issue_metadata_are_observed(self):
        clients = object.__new__(ObservedApiClients)
        trace = RequestTrace()
        token = CURRENT_TRACE.set(trace)
        result = TutorRevision(answer="본문 표식", citations=[])
        try:
            with patch.object(ApiClients, "complete_json", new_callable=AsyncMock, return_value=result):
                self.assertIs(await clients.complete_json([], TutorRevision, RequestBudget.start(1), seconds=.1, max_tokens=2000), result)
            trace.record("output_recheck", trace.started, "failed", evidence_issues=["invalid_supporting_ids", "본문 표식"])
        finally:
            CURRENT_TRACE.reset(token)
        decoded = decode_trace(trace.header().decode("ascii"))
        self.assertTrue(decoded["complete"])
        by_stage = {event["stage"]: event for event in decoded["events"]}
        self.assertIn("tutor_repair", by_stage)
        self.assertEqual(by_stage["output_recheck"]["evidence_issues"], ["invalid_supporting_ids"])
        self.assertNotIn("본문 표식", json.dumps(decoded, ensure_ascii=False))

    async def test_instrumented_routes_preserve_body_and_provider_calls(self):
        async def search(clients):
            return SearchDouble()

        async def measured_search(clients):
            return ObservedRetriever(SearchDouble())

        original = create_app(SETTINGS, PipelineClient, search)
        measured = create_app(SETTINGS, PipelineClient, measured_search, ObservedInputPolicy, ObservedOutputVerifier, ObservedTutorService)
        async with original.router.lifespan_context(original), measured.router.lifespan_context(measured):
            async with httpx.AsyncClient(base_url="http://test", transport=httpx.ASGITransport(app=original),
                                          headers={"Authorization": "Bearer " + SETTINGS.team_token}) as first, \
                 httpx.AsyncClient(base_url="http://test", transport=httpx.ASGITransport(app=DiagnosticApplication(measured)),
                                   headers={"Authorization": "Bearer " + SETTINGS.team_token}) as second:
                for endpoint, payload in (("/tutor", PAYLOAD), ("/guard", PAYLOAD), ("/guard", {**PAYLOAD, "candidate_answer": ""})):
                    before = await first.post(endpoint, json=payload)
                    after = await second.post(endpoint, json=payload)
                    self.assertEqual(before.status_code, after.status_code)
                    self.assertEqual(before.content, after.content)
                    self.assertNotIn(HEADER, before.headers)
                    self.assertTrue(decode_trace(after.headers[HEADER])["complete"])
                health = await second.get("/healthz")
                self.assertNotIn(HEADER, health.headers)
        self.assertEqual(original.state.clients.calls, measured.state.clients.calls)


class DiagnosisCompletionTests(unittest.TestCase):
    def test_followup_measurement_keeps_approved_work_item(self):
        saved = []

        def completed(args, report, settings):
            report.update(passed=True, measurement_complete=True)

        with patch("tools.evaluate.Settings.from_environment", return_value=SETTINGS), \
             patch("tools.evaluate.source_fingerprint", return_value="a" * 64), \
             patch("tools.evaluate.commit_id", return_value="b" * 40), \
             patch("tools.evaluate.run_diagnose", side_effect=completed), \
             patch("tools.evaluate.write_report", side_effect=lambda report, path: saved.append(dict(report))), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(["diagnose", "--start-server", "--work-item", "IMP-002-2"]), 0)
        self.assertEqual(saved[-1]["stage"], "IMP-002-2")

    def measure(self, change=None):
        result = successful_public_result()
        result["score"].update(judge_errors=0, invalid=0)
        result["diagnostics"] = {"trace_complete_count": 400}
        result["metrics"]["tutor_judge_coverage"] = {"facts": {"expected": 150, "eligible": 150, "observed": 150}}
        result["final_server"] = {"instance_id": "same", "runtime_sha256": "b" * 64}
        if change:
            change(result)
        report = {"source_sha256": "a" * 64}
        with patch("tools.evaluate.managed_server", return_value=nullcontext()) as server, \
             patch("tools.evaluate.read_health", return_value=result["final_server"]), \
             patch("tools.evaluate.evaluate_public", return_value=result), \
             patch("tools.evaluate.source_fingerprint", return_value="a" * 64), \
             patch("tutor_core.identity.runtime_fingerprint", return_value="b" * 64), \
             patch("tools.evaluate.write_report"):
            run_diagnose(SimpleNamespace(port=8765, workers=1), report, SETTINGS)
        self.assertEqual(server.call_args.kwargs["application"], "tools.diagnostic_server:app")
        return report

    def test_measured_internal_failures_do_not_become_quality_success(self):
        report = self.measure(lambda result: result.update(operational_pass=False, degraded_responses=10))
        self.assertTrue(report["measurement_complete"])
        self.assertTrue(report["passed"])
        self.assertFalse(report["quality_target_met"])
        self.assertEqual(report["passed_scope"], "baseline_measurement_only")

    def test_missing_trace_or_judgment_prevents_measurement_completion(self):
        for change in (
            lambda result: result["diagnostics"].update(trace_complete_count=399),
            lambda result: result["score"].update(judge_errors=1),
            lambda result: result["metrics"]["tutor_judge_coverage"]["facts"].update(observed=149),
            lambda result: result["metrics"].update(unique_observed_count=399),
        ):
            with self.subTest(change=change):
                self.assertFalse(self.measure(change)["passed"])


class InteractiveBoundaryTests(unittest.TestCase):
    def test_noninteractive_input_is_rejected_before_key_prompt(self):
        with patch("tools.evaluate.sys.stdin.isatty", return_value=False), \
             patch("tools.evaluate.getpass.getpass", side_effect=AssertionError("키 입력 금지")), \
             self.assertRaises(EvaluationFailure):
            interactive_environment()

    def test_interactive_input_never_reads_files_or_echoes_keys(self):
        output = io.StringIO()
        with patch("tools.evaluate.os.environ", {}), patch("tools.evaluate.sys.stdin.isatty", return_value=True), \
             patch("tools.evaluate.getpass.getpass", return_value="test-only-input-key"), \
             patch("builtins.open", side_effect=AssertionError("파일 접근 금지")), redirect_stdout(output):
            interactive_environment()
        self.assertNotIn("test-only-input-key", output.getvalue())


if __name__ == "__main__":
    unittest.main()
