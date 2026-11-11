"""eval/log_summary.py: counts by cause, run selection, and that log text is never printed."""
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("log_summary", ROOT / "eval/log_summary.py")
log_summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_summary)

MARKER = "LEAKED_TEXT_7f3a"


def request(t, endpoint="/tutor", latency=1000, outcome="ok", degraded=None, status=200):
    extra = f" degraded={degraded}" if degraded else ""
    return (f"2026-11-11 05:00:{t:02d},000 event=request endpoint={endpoint} status={status} "
            f"latency_ms={latency} body_bytes=10 input_chars=5 n_msgs=1 outcome={outcome}{extra}")


def call(t, kind="generate", status="ok", attempt=1, error=None, hedge=False):
    extra = (" hedge=1" if hedge else "") + (f" error={error}" if error else "")
    return f"2026-11-11 05:00:{t:02d},000 event=llm kind={kind} status={status} latency_ms=500 attempt={attempt}{extra}"


class CauseTests(unittest.TestCase):
    def report(self, lines):
        events, _ = log_summary.parse(lines)
        return log_summary.summarize(events)

    def test_fallbacks_are_counted_by_cause(self):
        r = self.report([
            request(1, outcome="fallback_deadline"),
            request(2, outcome="fallback_extractive", degraded="generate_timeout"),
            request(3, outcome="fallback_generate"),
            request(4, endpoint="/guard", degraded="classify,ground"),
            request(5, outcome="fallback_exception"),
            request(6),
            request(7, outcome="fallback_extractive", degraded="generate"),
            request(8, outcome="fallback_academic", degraded="generate"),
            request(9, degraded="generate"),  # speculative generation failed, fixed text sent
        ])
        self.assertEqual(r["fallback"], {"deadline": 2, "generate": 3, "judge": 1, "other": 1})
        self.assertEqual(r["degraded"], {"generate_timeout": 1, "classify": 1, "ground": 1, "generate": 3})
        self.assertEqual(r["requests"], {"/tutor": 8, "/guard": 1})

    def test_calls_are_counted_by_kind_and_status(self):
        r = self.report([
            call(1, "classify"), call(2, "classify", attempt=2), call(3, "ground", "cancelled"),
            call(4, "generate", "error", error="TimeoutError"), call(5, "generate", "invalid"),
            call(6, "found", hedge=True),
        ])["llm"]
        self.assertEqual(r["calls"], 6)
        self.assertEqual(r["by_kind"]["classify"], {"ok": 2})
        self.assertEqual((r["retries"], r["cancelled"], r["invalid"], r["hedges"]), (1, 1, 1, 1))
        self.assertEqual(r["api_errors"], {"TimeoutError": 1})

    def test_latency_uses_only_answered_requests(self):
        r = self.report([request(i, latency=1000 * (i + 1)) for i in range(20)] + [request(21, latency=99000, status=401)])
        self.assertEqual(r["latency_s"]["/tutor"], {"n": 20, "p50": 11.0, "p95": 20.0, "max": 20.0})

    def test_validation_error_field_names_are_allowed(self):
        events, audit = log_summary.parse([call(1, "generate", "invalid", error="ValidationError[answer]")])
        self.assertEqual((len(events), audit["bad_fields"]), (1, 0))
        self.assertEqual(log_summary.summarize(events)["llm"]["invalid"], 1)


class AuditTests(unittest.TestCase):
    def test_unexpected_lines_are_counted(self):
        _, audit = log_summary.parse([
            request(1),
            "INFO:     Application startup complete.",
            f"Traceback: {MARKER}",
            f"2026-11-11 05:00:02,000 event=request answer={MARKER}",
        ])
        self.assertEqual(dict(audit), {"event_lines": 1, "uvicorn_lines": 1, "other_lines": 1, "bad_fields": 1})

    def test_log_text_is_never_printed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "server.log"
            path.write_text("\n".join([request(1), f"body {MARKER}",
                                       f"2026-11-11 05:00:02,000 event=request answer={MARKER}"]), encoding="utf-8")
            for flags in ([], ["--json"], ["--row"], ["--last-run"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    log_summary.main([str(path), *flags])
                self.assertNotIn(MARKER, out.getvalue(), flags)


class RunSelectionTests(unittest.TestCase):
    def test_last_run_starts_after_the_last_gap(self):
        lines = [request(1), request(2), request(40), request(41)]
        events, _ = log_summary.parse(lines)
        start = log_summary.last_run_start(events, gap_s=30)
        self.assertEqual(start.second, 40)

    def test_no_requests(self):
        self.assertIsNone(log_summary.last_run_start([], gap_s=30))
        self.assertEqual(log_summary.summarize([])["requests"], {})


if __name__ == "__main__":
    unittest.main()
