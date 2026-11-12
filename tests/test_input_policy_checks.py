"""공통 입력 정책·대비 사례·본문 비노출·측정 완료 경계를 검사한다."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx

from tests.test_foundation import PAYLOAD
from tests.test_output import ReviewClient
from tests.test_policy import SearchDouble
from tools.evaluation_trace import HEADER, RequestTrace, decode_trace
from tools.input_policy_checks import TUTOR_CONTROLS, check_tutor_controls, finish_measurement
from tools.policy_checks import BOUNDARIES, BOUNDARY_PAIRS, boundary_messages, evaluate_policy
from tutor_core.llm import RequestBudget
from tutor_core.policy import InputPolicy
from tutor_core.prompts import INPUT_POLICY_RULES, INTENT_SYSTEM, TUTOR_SYSTEM
from tutor_core.schemas import Message, TutorRequest, TutorResponse
from tutor_core.tutor import TutorService
from tutor_core.verifier import OutputVerifier


class SharedPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_inference_paths_use_same_input_policy_rules(self):
        messages = [Message(role="user", content="학습 질문")]
        client, search = ReviewClient(), SearchDouble()
        verifier = OutputVerifier(client, search)
        service = TutorService(client, search, verifier)
        request = TutorRequest.model_validate(PAYLOAD)
        found = await search.search(messages, RequestBudget.start(1))
        await verifier.verify(messages, "일반 설명", RequestBudget.start(27))
        systems = (InputPolicy.intent_messages(messages)[0]["content"], service._messages(request, found)[0]["content"],
                   client.calls[0][0][0]["content"])
        for system in systems:
            self.assertEqual(system.count(INPUT_POLICY_RULES), 1)
        self.assertNotEqual(INTENT_SYSTEM, TUTOR_SYSTEM)


class BoundaryCasesTests(unittest.TestCase):
    def test_scope_metadata_is_closed_enum_without_model_explanation(self):
        trace = RequestTrace()
        trace.record("input_policy", trace.started, coverage="원문 비노출 표식", material_status="missing", task_summary="요약 비노출 표식")
        event = decode_trace(trace.header().decode("ascii"))["events"][0]
        self.assertEqual(event["material_status"], "missing")
        self.assertNotIn("coverage", event)
        self.assertNotIn("비노출", json.dumps(event, ensure_ascii=False))

    def test_team_pairs_have_different_intents_and_valid_conversation_roles(self):
        self.assertEqual(len(BOUNDARY_PAIRS), 10)
        self.assertEqual(len(BOUNDARIES), 40)
        for _, left, right in BOUNDARY_PAIRS:
            self.assertNotEqual(left[:2], right[:2])
            for row in (left, right):
                TutorRequest.model_validate({"spec_version": "1.0", "session_id": "unit", "messages": boundary_messages(row[2])})
        self.assertEqual({action for _, action in TUTOR_CONTROLS}, {"answer", "scaffold", "clarify", "abstain", "redirect", "refuse", "escalate"})

    def test_public_and_boundary_results_do_not_copy_inputs_or_candidates(self):
        from tools.evaluate import ROOT

        trace = RequestTrace()
        trace.record("input_policy", trace.started, verified=True)

        def respond(request):
            data = json.loads(request.content)
            value = {"input": {"decision": "allow", "category": "normal"}}
            if "candidate_answer" in data:
                value["output"] = {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}
            return httpx.Response(200, json=value, headers={HEADER: trace.header().decode("ascii")})

        public = [{"id": "public-" + str(index), "endpoint": "guard", "messages": [{"role": "user", "content": "본문 비노출 표식"}],
                   "labels": {"category": "normal", "decision": "allow"}} for index in range(200)]
        parent = ROOT / "output" / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        health = {"input_policy_ready": True, "input_policy": {"counters": {}}, "counters": {}}
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path = Path(folder) / "public.jsonl"
            path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in public), encoding="utf-8")
            with httpx.Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client, \
                 patch("tools.evaluate.read_health", return_value=health), redirect_stdout(io.StringIO()):
                result = evaluate_policy(client, path, 4, diagnostics=True)
        self.assertEqual(result["trace_complete_count"], 243)
        self.assertEqual(result["verified_public_count"], 200)
        self.assertEqual(len(result["boundary_pairs"]), 10)
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("본문 비노출 표식", encoded)
        self.assertNotIn("학생의 요청을 학업 부정행위로 분류해", encoded)

    def test_tutor_probe_returns_actions_without_answer_text(self):
        def endpoint(*args, **kwargs):
            trace = {"complete": True, "events": [{"stage": "tutor", "verified": True}]}
            kwargs["diagnostics"].append(trace)
            return TutorResponse(answer="답변 비노출 표식", action="answer", citations=[]), .1, None

        with patch("tools.input_policy_checks.call_endpoint", side_effect=endpoint):
            result = check_tutor_controls(None)
        self.assertEqual(result["count"], 10)
        self.assertNotIn("답변 비노출 표식", json.dumps(result, ensure_ascii=False))


class CompletionTests(unittest.TestCase):
    def report(self):
        return {"source_unchanged": True, "instance_unchanged": True, "endpoint_contract_pass": True,
                "policy": {"public_count": 200, "items": [{"id": str(index)} for index in range(200)],
                           "boundary_cases": [{"passed": True} for _ in BOUNDARIES], "trace_complete_count": 243,
                           "candidate_isolation": {"passed": True, "variants": [{}, {}, {}]},
                           "block_f1": 1, "non_normal_category_accuracy": 1, "input_operational_pass": True},
                "tutor_controls": {"count": 10, "passed": True, "items": [{"trace_complete": True} for _ in range(10)]}}

    def test_stage_measurement_is_distinct_from_final_quality(self):
        report = self.report()
        finish_measurement(report)
        self.assertTrue(report["measurement_complete"])
        self.assertTrue(report["stage_quality_target_met"])
        self.assertFalse(report["final_quality_verified"])

    def test_runtime_failure_is_measured_but_never_a_quality_pass(self):
        report = self.report()
        report["policy"]["input_operational_pass"] = False
        finish_measurement(report)
        self.assertTrue(report["measurement_complete"])
        self.assertFalse(report["stage_quality_target_met"])

    def test_missing_trace_or_source_change_prevents_complete_measurement(self):
        for change in ("trace", "source", "instance"):
            report = self.report()
            if change == "trace":
                report["policy"]["trace_complete_count"] = 242
            else:
                report[change + "_unchanged"] = False
            finish_measurement(report)
            self.assertFalse(report["measurement_complete"])


if __name__ == "__main__":
    unittest.main()
