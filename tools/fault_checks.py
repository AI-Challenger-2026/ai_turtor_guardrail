"""통제된 대역 검사만 실행하고 실패 본문 대신 검사 ID·종류·개수만 반환한다."""

import unittest
import io
from contextlib import redirect_stderr, redirect_stdout


FAULT_SUITES = (
    "tests.test_foundation", "tests.test_retrieval", "tests.test_policy", "tests.test_output", "tests.test_tutor",
    "tests.test_evaluate", "tests.test_release.PublicMetricTests", "tests.test_release.CompletionBoundaryTests",
    "tests.test_release.LoadBoundaryTests",
    "tests.test_release.RunboxBoundaryTests", "tests.test_runbox",
)


class MetadataResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.events = []

    def addFailure(self, test, err):
        self.events.append({"test": test.id(), "type": "assertion_failure"})
        super().addFailure(test, err)

    def addError(self, test, err):
        self.events.append({"test": test.id(), "type": "test_error"})
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.events.append({"test": test.id(), "type": "subtest_failure"})
        super().addSubTest(test, subtest, err)


def evaluate_faults(suite_names=FAULT_SUITES):
    result = MetadataResult()
    # 대역 검사의 출력이 실제 부하 실행이나 판정 본문처럼 표시되지 않게 한다.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        suite = unittest.TestLoader().loadTestsFromNames(suite_names)
        suite.run(result)
    return {"scope": "offline_fault_injection", "api_mode": "offline", "api_attempted": False,
            "test_suites": list(suite_names),
            "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
            "unexpected_successes": len(result.unexpectedSuccesses), "events": result.events,
            "passed": result.testsRun > 0 and result.wasSuccessful() and not result.skipped
            and not result.expectedFailures and not result.unexpectedSuccesses}
