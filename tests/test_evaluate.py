"""검증 도구가 응답 본문을 보고서로 흘리지 않도록 경계를 검사한다."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from tools.evaluate import EvaluationFailure, bounded_json, main, read_health, write_report
from tests.test_foundation import SETTINGS


class EvaluationBoundaryTests(unittest.TestCase):
    def test_redirect_and_error_bodies_are_not_returned(self):
        for status in (301, 307, 401, 500):
            with self.subTest(status=status), self.assertRaises(EvaluationFailure) as caught:
                bounded_json(httpx.Response(status, text="private-body"))
            self.assertNotIn("private-body", str(caught.exception))

    def test_streaming_and_oversize_responses_are_rejected(self):
        responses = (
            httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text="data: {}"),
            httpx.Response(200, headers={"Content-Type": "application/json"}, content=b" " * (256 * 1024 + 1)),
        )
        for response in responses:
            with self.assertRaises(EvaluationFailure):
                bounded_json(response)

    def test_health_report_copies_only_known_metadata(self):
        payload = {
            "ready": True, "spec_version": "1.0", "implementation_stage": "IMP-001-1",
            "structured_output": "json_schema", "policy_ready": False, "retrieval_ready": False,
            "unexpected": "private-marker", "counters": {"requests": 7, "incomplete_response": 1, "private-marker": 1},
            "provider_counters": {"chat_attempts": 2, "upstream_status": "private-marker"},
        }
        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        ) as client:
            report = read_health(client)
        self.assertNotIn("private-marker", json.dumps(report))
        self.assertEqual(report["counters"], {"requests": 7, "incomplete_response": 1})
        self.assertEqual(report["provider_counters"], {"chat_attempts": 2})

    def test_old_or_unready_server_does_not_pass_health(self):
        with httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ready": False})),
        ) as client:
            with self.assertRaises(EvaluationFailure):
                read_health(client)


class ReportBoundaryTests(unittest.TestCase):
    def test_failed_atomic_replace_preserves_previous_report_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            write_report({"passed": False, "generation": 1}, target)
            with patch("tools.evaluate.os.replace", side_effect=OSError("private-error-marker")), self.assertRaises(OSError):
                write_report({"passed": True, "generation": 2}, target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"passed": False, "generation": 1})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_later_failure_clears_partial_success(self):
        saved = []

        def interrupted(args, report, settings):
            report["passed"] = True
            raise EvaluationFailure("smoke_operational_failure")

        with patch("tools.evaluate.Settings.from_environment", return_value=SETTINGS), \
             patch("tools.evaluate.source_fingerprint", return_value="a" * 64), \
             patch("tools.evaluate.commit_id", return_value="b" * 40), \
             patch("tools.evaluate.run_extended", side_effect=interrupted), \
             patch("tools.evaluate.write_report", side_effect=lambda report, path: saved.append(json.loads(json.dumps(report)))):
            code = main(["smoke"])
        self.assertEqual(code, 1)
        self.assertFalse(saved[-1]["passed"])
        self.assertEqual(saved[-1]["error"]["reason"], "smoke_operational_failure")

    def test_offline_fault_command_never_loads_api_environment(self):
        saved = []
        with patch("tools.evaluate.Settings.from_environment", side_effect=AssertionError("환경변수 접근 금지")) as settings, \
             patch("tools.evaluate.source_fingerprint", return_value="a" * 64), \
             patch("tools.evaluate.commit_id", return_value="b" * 40), \
             patch("tools.fault_checks.evaluate_faults", return_value={"passed": True, "tests_run": 1}), \
             patch("tools.evaluate.write_report", side_effect=lambda report, path: saved.append(dict(report))):
            code = main(["faults"])
        self.assertEqual(code, 0)
        settings.assert_not_called()
        self.assertEqual(saved[-1]["api_mode"], "offline")
        self.assertFalse(saved[-1]["api_attempted"])

    def test_final_source_change_invalidates_previously_successful_repeated_run(self):
        saved = []

        def completed(args, report, settings):
            report.update(passed=True, release={"runs": [], "two_runs_verified": True})

        with patch("tools.evaluate.Settings.from_environment", return_value=SETTINGS), \
             patch("tools.evaluate.source_fingerprint", side_effect=["a" * 64, "c" * 64]), \
             patch("tools.evaluate.commit_id", return_value="b" * 40), \
             patch("tools.evaluate.run_extended", side_effect=completed), \
             patch("tools.evaluate.write_report", side_effect=lambda report, path: saved.append(json.loads(json.dumps(report)))):
            code = main(["release"])
        self.assertEqual(code, 1)
        self.assertFalse(saved[-1]["source_unchanged"])
        self.assertFalse(saved[-1]["passed"])
        self.assertFalse(saved[-1]["release"]["two_runs_verified"])


if __name__ == "__main__":
    unittest.main()
