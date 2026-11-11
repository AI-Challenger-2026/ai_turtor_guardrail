"""공개 지표·유한 부하·터널 검사에서 범위와 본문 비노출을 확인한다."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from threading import Lock
from unittest.mock import patch

import httpx

from tools.evaluate import EvaluationFailure, read_health, repeated_public
from tools.fault_checks import evaluate_faults
from tools.release_checks import counter_delta, evaluate_load, latency_summary, observe_public, public_metrics, release_targets, score_public_cases
from tools.runbox import check_live_server, validate_tunnel_url
from tutor_core.identity import runtime_paths
from tutor_core.schemas import GuardResponse, InputAssessment, OutputAssessment, TutorResponse


class PublicMetricTests(unittest.TestCase):
    def test_observation_contains_no_answer_or_message_body(self):
        marker = "본문을 저장하면 안 되는 표식"
        item = {"id": "unit-tutor", "endpoint": "tutor", "labels": {"action": "answer", "evidence": ["PY-01-01"]}}
        result = observe_public(item, TutorResponse(answer=marker, action="answer", citations=["PY-01-01", "PY-01-02"]))
        self.assertNotIn(marker, json.dumps(result, ensure_ascii=False))
        self.assertAlmostEqual(result["citation_f1"], 2 / 3)
        self.assertTrue(result["action_correct"])

    def test_unsafe_and_ungrounded_are_measured_independently_from_public_labels(self):
        item = {"id": "unit-guard", "endpoint": "guard", "messages": [{"role": "user", "content": "질문"}],
                "candidate_answer": "근거 없는 내용", "labels": {"category": "normal", "decision": "allow",
                "grounded": False, "safe": False, "unsupported": ["근거 없는 내용"]}}
        parsed = GuardResponse(input=InputAssessment(decision="allow", category="normal"),
                               output=OutputAssessment(grounded=False, unsupported=["근거 없는 내용"], safe=False, category="harmful_content"))
        observed = observe_public(item, parsed)
        self.assertTrue(observed["grounded_correct"] and observed["safe_correct"])
        self.assertEqual(observed["span_f1"], 1)
        self.assertTrue(observed["substrings_valid"])
        self.assertEqual(public_metrics([observed])["grounded_accuracy"], 1)
        self.assertNotIn("근거 없는 내용", json.dumps(observed, ensure_ascii=False))

    def test_invalid_output_counts_as_incorrect_instead_of_being_excluded(self):
        item = {"id": "unit-guard", "endpoint": "guard", "messages": [], "candidate_answer": "설명",
                "labels": {"category": "harmful_content", "decision": "block", "grounded": True, "safe": True}}
        metrics = public_metrics([observe_public(item, None)])
        self.assertEqual(metrics["guard_block_f1"], 0)
        self.assertEqual(metrics["guard_non_normal_category_accuracy"], 0)
        self.assertEqual(metrics["safe_accuracy"], 0)

    def test_nearest_rank_percentiles_preserve_maximum(self):
        values = list(range(1, 101))
        self.assertEqual(latency_summary(values), {"p50": 50, "p95": 95, "p99": 99, "max": 100})

    def test_identical_public_messages_keep_separate_ids_without_body_in_report(self):
        marker = "공개 본문을 저장하지 않을 표식"
        items = [{"id": "unit-" + str(index), "endpoint": "tutor", "messages": [{"role": "user", "content": marker}],
                  "labels": {"category": "normal", "action": "answer"}} for index in range(2)]
        parsed = TutorResponse(answer=marker, action="answer", citations=[])
        with patch("tools.release_checks.call_endpoint", return_value=(parsed, .01, None)):
            result = score_public_cases(items, None, lambda kind, payload: {"ok": True}, {}, 2)
        self.assertEqual({row["id"] for row in result["score"]["items"]}, {"unit-0", "unit-1"})
        self.assertEqual(result["metrics"]["unique_observed_count"], 2)
        self.assertNotIn(marker, json.dumps(result, ensure_ascii=False))

    def test_public_empty_candidate_is_sent_and_scored_separately_from_missing(self):
        items = [{"id": "missing", "endpoint": "guard", "messages": [{"role": "user", "content": "질문"}],
                  "labels": {"category": "normal", "decision": "allow"}},
                 {"id": "empty", "endpoint": "guard", "messages": [{"role": "user", "content": "질문"}],
                  "candidate_answer": "", "labels": {"category": "normal", "decision": "allow", "grounded": True, "safe": True}}]
        presence = []

        def transport(request):
            payload = json.loads(request.content)
            has_candidate = "candidate_answer" in payload
            presence.append(has_candidate)
            body = {"input": {"decision": "allow", "category": "normal"}}
            if has_candidate:
                body["output"] = {"grounded": True, "safe": True, "unsupported": [], "category": "normal"}
            return httpx.Response(200, json=body)

        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
            result = score_public_cases(items, client, None, {}, 1)
        self.assertEqual(presence, [False, True])
        self.assertEqual(result["score"]["invalid"], 0)
        self.assertEqual(result["metrics"]["candidate_count"], 1)


def successful_public_result():
    return {"score": {"total": 95, "tutor": 95, "guard": 95}, "operational_pass": True,
            "judge": {"attack_compliance": 0}, "metrics": {
                "observed_count": 400, "unique_observed_count": 400, "tutor_count": 200, "guard_count": 200,
                "candidate_count": 100, "tutor_evidence_count": 100, "non_normal_count": 50,
                "tutor_action_accuracy": 1, "tutor_citation_f1": 1, "guard_block_f1": 1,
                "guard_non_normal_category_accuracy": 1, "grounded_accuracy": 1, "safe_accuracy": 1,
                "unsupported_span_f1": 1, "invalid_substrings": 0}}


class CompletionBoundaryTests(unittest.TestCase):
    def test_high_total_score_cannot_hide_missing_scope_or_metric_failure(self):
        result = successful_public_result()
        self.assertTrue(all(release_targets(result).values()))
        result["metrics"]["safe_accuracy"] = .94
        result["metrics"]["candidate_count"] = 0
        targets = release_targets(result)
        self.assertFalse(targets["safe_accuracy"])
        self.assertFalse(targets["complete_scope"])

    def test_single_public_run_never_satisfies_two_run_gate(self):
        report = {"source_sha256": "a" * 64}
        with patch("tools.evaluate.source_fingerprint", return_value="a" * 64), \
             patch("tools.evaluate.read_health", return_value={"instance_id": "b" * 24}), \
             patch("tools.evaluate.evaluate_public", return_value=successful_public_result()):
            passed = repeated_public(None, None, SimpleNamespace(repeat=1, workers=1), report, lambda: None)
        self.assertFalse(passed)
        self.assertTrue(report["release"]["measurement_complete"])
        self.assertEqual(report["release"]["stopped_reason"], "second_run_not_requested")

    def test_source_change_stops_second_public_run(self):
        report = {"source_sha256": "a" * 64}
        with patch("tools.evaluate.source_fingerprint", side_effect=["a" * 64, "c" * 64]), \
             patch("tools.evaluate.read_health", return_value={"instance_id": "b" * 24}), \
             patch("tools.evaluate.evaluate_public", return_value=successful_public_result()) as evaluate:
            passed = repeated_public(None, None, SimpleNamespace(repeat=2, workers=1), report, lambda: None)
        self.assertFalse(passed)
        self.assertEqual(evaluate.call_count, 1)
        self.assertFalse(report["release"]["measurement_complete"])

    def test_server_restart_or_counter_reset_cannot_appear_as_zero_failures(self):
        with self.assertRaises(EvaluationFailure):
            counter_delta({"instance_id": "a", "counters": {}}, {"instance_id": "b", "counters": {}})
        with self.assertRaises(EvaluationFailure):
            counter_delta({"counters": {"degraded_responses": 9}}, {"counters": {"degraded_responses": 0}})

    def test_fault_report_does_not_include_assertion_body(self):
        class FailingCase(unittest.TestCase):
            def runTest(self):
                self.fail("private-failure-marker")

        with patch("unittest.TestLoader.loadTestsFromNames", return_value=unittest.TestSuite([FailingCase()])):
            result = evaluate_faults(("unused",))
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"], 1)
        self.assertNotIn("private-failure-marker", json.dumps(result))
        self.assertFalse(result["api_attempted"])


class LoadBoundaryTests(unittest.TestCase):
    def public_root(self, root):
        for endpoint in ("tutor", "guard"):
            item = {"id": "unit-" + endpoint, "endpoint": endpoint, "messages": [{"role": "user", "content": "학습"}]}
            (root / (endpoint + ".jsonl")).write_text(json.dumps(item), encoding="utf-8")

    def test_full_run_is_finite_and_balanced_after_small_steps(self):
        calls = {"tutor": 0, "guard": 0}
        counter_lock = Lock()

        def endpoint(client, name, messages, **kwargs):
            with counter_lock:
                calls[name] += 1
            return object(), .01, None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.public_root(root)
            with patch("tools.evaluate.read_health", return_value={"counters": {}}), \
                 patch("tools.release_checks.call_endpoint", side_effect=endpoint):
                result = evaluate_load(None, root, full=True)
        self.assertEqual([row["workers"] for row in result["phases"]], [1, 8, 16, 32, 64, 64])
        self.assertEqual(result["phases"][-1]["request_count"], 600)
        self.assertEqual(calls["tutor"], calls["guard"])
        self.assertEqual(sum(calls.values()), 2 * (1 + 8 + 16 + 32 + 64) + 600)
        self.assertTrue(result["full_600_executed"])

    def test_failure_stops_escalation_before_high_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.public_root(root)
            with patch("tools.evaluate.read_health", side_effect=[{"counters": {}}, {"counters": {"degraded_responses": 1}}]), \
                 patch("tools.release_checks.call_endpoint", return_value=(object(), .01, None)) as call:
                result = evaluate_load(None, root, full=True)
        self.assertFalse(result["full_600_executed"])
        self.assertEqual(len(result["phases"]), 1)
        self.assertEqual(call.call_count, 2)

    def test_slow_64_step_prevents_additional_600_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.public_root(root)
            with patch("tools.evaluate.read_health", return_value={"counters": {}}), \
                 patch("tools.release_checks.call_endpoint", return_value=(object(), 26, None)) as call:
                result = evaluate_load(None, root, full=True, seed=7)
        self.assertFalse(result["full_600_executed"])
        self.assertEqual(result["stopped_reason"], "latency_target_missed")
        self.assertEqual(result["seed"], 7)
        self.assertEqual(call.call_count, 242)


class RunboxBoundaryTests(unittest.TestCase):
    def test_only_runbox_proxy_base_urls_can_receive_authentication(self):
        valid = "https://example.tunnel.elice.io/proxy/8000"
        self.assertEqual(validate_tunnel_url(valid), valid + "/")
        for value in ("http://example.tunnel.elice.io/proxy/8000", "https://example.org/proxy/8000",
                      "https://example.tunnel.elice.io.invalid/proxy/8000", "https://example.tunnel.elice.io/proxy/8080",
                      "https://name:pass@example.tunnel.elice.io/proxy/8000", valid + "?token=hidden",
                      valid + "/tutor", "https://example.tunnel.elice.io/proxy/99999"):
            with self.subTest(value=value), self.assertRaises(EvaluationFailure) as error:
                validate_tunnel_url(value)
            self.assertNotIn(value, str(error.exception))

    def test_probe_preserves_proxy_path_and_verifies_runtime_identity(self):
        observed_paths = []
        fingerprint = "a" * 64
        health = {"ready": True, "spec_version": "1.0", "implementation_stage": "IMP-001-7",
                  "structured_output": "json_schema", "policy_ready": True, "retrieval_ready": True,
                  "input_policy_ready": True, "output_policy_ready": True, "tutor_policy_ready": True,
                  "runtime_sha256": fingerprint, "runtime_identity_version": "runtime-text-v1", "instance_id": "b" * 24,
                  "counters": {}, "provider_counters": {"chat_attempts": 1, "chat_successes": 1}}

        def transport(request):
            observed_paths.append(request.url.path)
            if request.headers.get("Authorization") in ("", "Basic invalid-runbox-check"):
                return httpx.Response(401)
            if request.url.path.endswith("/healthz"):
                return httpx.Response(200, json=health)
            if request.url.path.endswith(("/tutor/", "/guard/")):
                return httpx.Response(404)
            health["provider_counters"]["chat_attempts"] += 1
            health["provider_counters"]["chat_successes"] += 1
            if request.url.path.endswith("/tutor"):
                return httpx.Response(200, json={"answer": "반복문을 설명해요.", "action": "answer", "citations": []})
            return httpx.Response(200, json={"input": {"decision": "allow", "category": "normal"},
                                           "output": {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}})

        with httpx.Client(base_url="https://example.tunnel.elice.io/proxy/8000/", trust_env=False, follow_redirects=False,
                          transport=httpx.MockTransport(transport)) as client:
            result = check_live_server(client, fingerprint)
        self.assertTrue(result["passed"])
        self.assertTrue(all(path.startswith("/proxy/8000/") for path in observed_paths))

    def test_runtime_manifest_never_includes_secret_or_local_test_artifacts(self):
        paths = runtime_paths()
        self.assertTrue(paths)
        self.assertTrue(all(path.suffix in (".py", ".txt", ".md", ".csv", ".html") for path in paths))
        self.assertFalse(any(path.name.startswith(".env") or path.name == "submission.json" or "output" in path.parts for path in paths))

    def test_health_component_metadata_uses_known_counters_only(self):
        value = {"ready": True, "spec_version": "1.0", "implementation_stage": "IMP-001-7", "structured_output": "json_schema",
                 "output_policy": {"version": "output-v1", "counters": {"verified": 3, "private-marker": 7}},
                 "tutor_policy": {"version": "tutor-v1", "counters": {"repairs": 2}, "answer": "private-marker"}}
        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=value))) as client:
            cleaned = read_health(client)
        self.assertNotIn("private-marker", json.dumps(cleaned))
        self.assertEqual(cleaned["output_policy"]["counters"], {"verified": 3})


if __name__ == "__main__":
    unittest.main()
