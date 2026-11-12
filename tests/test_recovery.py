"""제한된 공개 복구 비교의 표본·범위·계측 집계를 검사한다."""

import unittest
import io
import json
from contextlib import nullcontext, redirect_stdout
from unittest.mock import patch

from tools.evaluate import EvaluationFailure
from tools.recovery_checks import main, paired_summary, select_cases
from tests.test_foundation import SETTINGS


class RecoverySelectionTests(unittest.TestCase):
    def test_failures_old_attacks_and_controls_are_all_selected(self):
        public = [
            {"id": "failed", "endpoint": "tutor", "labels": {"category": "normal"}},
            {"id": "old-attack", "endpoint": "tutor", "labels": {"category": "harmful_content"}},
            {"id": "control", "endpoint": "tutor", "labels": {"category": "out_of_material"}},
            {"id": "guard", "endpoint": "guard", "labels": {"decision": "allow"}},
        ]
        previous = {"diagnostics": {"items": [{"id": "failed", "endpoint": "tutor", "flags": ["stage_failure"]}]}}
        baseline = {"diagnostics": {"items": [{"id": "old-attack", "endpoint": "tutor", "flags": ["attack_compliance"]}]}}
        self.assertEqual({row["id"] for row in select_cases(public, baseline, previous)}, {row["id"] for row in public})

    def test_unknown_ids_are_rejected_instead_of_loading_another_dataset(self):
        report = {"diagnostics": {"items": [{"id": "unknown", "endpoint": "tutor", "flags": ["stage_failure"]}]}}
        with self.assertRaises(EvaluationFailure):
            select_cases([], report, report)

    def test_nested_failure_events_count_as_one_failed_request(self):
        result = {"diagnostics": {"items": [{"id": "case", "endpoint": "tutor", "server_elapsed_seconds": 5,
            "flags": [], "events": [{"stage": "tutor", "verified": False}, {"stage": "output_verification", "verified": False}]}]}}
        summary = paired_summary(result, {"case"})
        self.assertEqual(summary["tutor_failures"], 1)
        self.assertEqual(summary["tutor_count"], 1)


class RecoveryCompletionTests(unittest.TestCase):
    def execute(self, *, invalid=0, changed=False):
        item = {"id": "case", "endpoint": "tutor", "labels": {"category": "normal"}}
        old_row = {"id": "case", "endpoint": "tutor", "server_elapsed_seconds": 5,
                   "flags": [], "events": [{"stage": "tutor", "verified": False}]}
        old = {"diagnostics": {"items": [old_row]}}
        measured = {"diagnostics": {"items": [{**old_row, "events": [{"stage": "tutor", "verified": True}]}],
                                    "trace_complete_count": 1},
                    "metrics": {"unique_observed_count": 1}, "score": {"judge_errors": 0, "invalid": invalid},
                    "endpoint_errors": {}, "deadline_violations": 0}
        corpus = ["\n".join(json.dumps({"endpoint": endpoint}) for _ in range(200)) for endpoint in ("tutor", "guard")]
        saved = []
        with patch("tools.recovery_checks.Settings.from_environment", return_value=SETTINGS), \
             patch("tools.recovery_checks.references", return_value=(old, old)), \
             patch("pathlib.Path.read_text", side_effect=corpus), \
             patch("tools.recovery_checks.select_cases", return_value=[item]), \
             patch("tools.recovery_checks.source_fingerprint", side_effect=["a" * 64, "a" * 64, ("c" if changed else "a") * 64]), \
             patch("tools.recovery_checks.commit_id", return_value="d" * 40), \
             patch("tools.recovery_checks.make_judge"), \
             patch("tools.recovery_checks.managed_server", return_value=nullcontext()), \
             patch("tools.recovery_checks.runtime_fingerprint", return_value="b" * 64), \
             patch("tools.recovery_checks.read_health", return_value={"instance_id": "same", "runtime_sha256": "b" * 64}), \
             patch("tools.recovery_checks.load_kb_dir", return_value={}), \
             patch("tools.recovery_checks.score_public_cases", return_value=measured), \
             patch("tools.recovery_checks.write_report", side_effect=lambda report, path: saved.append(json.loads(json.dumps(report)))), \
             redirect_stdout(io.StringIO()):
            code = main([])
        return code, saved[-1]

    def test_reduced_failures_with_complete_observations_allow_full_comparison(self):
        code, report = self.execute()
        self.assertEqual(code, 0)
        self.assertTrue(report["eligible_for_full_evaluation"])

    def test_endpoint_contract_error_does_not_allow_full_comparison(self):
        _, report = self.execute(invalid=1)
        self.assertFalse(report["eligible_for_full_evaluation"])

    def test_final_source_change_invalidates_previously_completed_measurement(self):
        code, report = self.execute(changed=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["measurement_complete"])
        self.assertFalse(report["eligible_for_full_evaluation"])


if __name__ == "__main__":
    unittest.main()
