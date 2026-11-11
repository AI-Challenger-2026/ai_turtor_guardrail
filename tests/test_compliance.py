"""Static tripwires for the competition rules (CLAUDE.md section 2). No network.

These do not prove compliance; they fail loudly if code drifts toward something the rules
forbid: printing or logging content, other network clients or hosts, caches, file writes,
data files in the server, public-set text in prompts, or answer tables.
"""
import json
import re
import unittest
from pathlib import Path

import tests.test_server  # noqa: F401  (sets the test environment first)

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SOURCES = {p.name: p.read_text(encoding="utf-8") for p in sorted(APP.glob("*.py"))}
SOURCES["server.py"] = (ROOT / "server.py").read_text(encoding="utf-8")


def code_only(src: str) -> str:
    """Source without comments and docstrings, for keyword checks."""
    src = re.sub(r'(?s)""".*?"""', '""', src)
    src = re.sub(r"#[^\n]*", "", src)
    return src


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


class OutputAndLoggingTests(unittest.TestCase):
    def test_no_print(self):
        for name, src in SOURCES.items():
            self.assertNotRegex(code_only(src), r"\bprint\s*\(", name)

    def test_only_logs_module_uses_logging(self):
        for name, src in SOURCES.items():
            if name != "logs.py":
                self.assertNotRegex(code_only(src), r"^\s*(import logging|from logging)", name)

    def test_log_fields_are_allowlisted(self):
        allowed = {"event", "endpoint", "status", "latency_ms", "body_bytes", "input_chars", "n_msgs",
                   "outcome", "degraded", "error", "kind", "attempt", "hedge", "kb_sections", "kb_chunks",
                   "kb_en_chunks"}
        fields = set(re.findall(r'"(\w+)":', SOURCES["logs.py"]))
        self.assertTrue(fields, "no log fields found")
        self.assertLessEqual(fields, allowed)

    def test_no_traceback_or_exception_text_logging(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotIn("exc_info", code, name)
            self.assertNotIn("traceback", code, name)
            self.assertNotRegex(code, r"str\(\s*(e|exc)\b", name)

    def test_errors_are_logged_by_class_name_only(self):
        for name, src in SOURCES.items():
            for m in re.finditer(r"error=([^,\n]+)", code_only(src)):
                value = m.group(1).strip()
                self.assertTrue(re.match(r"(err|error|None)\b", value) or value.startswith('info.get("error"')
                                or "__name__" in value, (name, value))


class ExternalCallTests(unittest.TestCase):
    def test_no_other_network_clients(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotRegex(code, r"^\s*(import|from)\s+(requests|urllib|httpx|aiohttp|socket|subprocess|http\.client)\b",
                                name)

    def test_no_hard_coded_hosts(self):
        for name, src in SOURCES.items():
            self.assertNotRegex(src, r"https?://", name)

    def test_one_chat_client_from_openai_env(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotRegex(code, r"\bOpenAI\(", name)  # no sync clients
            self.assertNotIn("embeddings.create", code, name)  # final-v3 calls no embedding API
            n = len(re.findall(r"AsyncOpenAI\(", code))
            self.assertEqual(n, 1 if name == "llm.py" else 0, name)
        # No key or base URL in code: the client reads OPENAI_API_KEY / OPENAI_BASE_URL itself.
        self.assertIn("AsyncOpenAI(timeout=25, max_retries=0)", SOURCES["llm.py"])


class StateTests(unittest.TestCase):
    def test_no_caches_or_memoization(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotRegex(code, r"lru_cache|functools\.cache|@cache\b|cachetools|memoize", name)

    def test_module_state_is_only_the_api_client(self):
        found = [(name, m.group(1)) for name, src in SOURCES.items()
                 for m in re.finditer(r"^\s*global\s+(\w+)", src, re.M)]
        self.assertEqual(found, [("llm.py", "_client")])

    def test_no_file_writes(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotRegex(code, r"write_text|write_bytes|\.dump\(|open\([^)]*['\"][wa]", name)

    def test_server_never_reads_public_items(self):
        for name, src in SOURCES.items():
            self.assertNotRegex(code_only(src), r"public[/\"']|jsonl", name)


class DataTests(unittest.TestCase):
    def test_app_package_holds_no_data_files(self):
        others = sorted(p.name for p in APP.iterdir() if p.is_file() and p.suffix != ".py")
        self.assertEqual(others, [])

    def test_no_public_text_in_prompts_or_code(self):
        # final-v3 was tuned on all 400 public items, so every item is checked, not only the dev split.
        blob = "\n".join(SOURCES.values())
        # Quick-question button phrases are published in SPEC.md section 2; they are not item text.
        spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
        for endpoint in ("tutor", "guard"):
            for item in read_jsonl(ROOT / f"public/{endpoint}.jsonl"):
                self.assertNotIn(item["id"], blob)
                for m in item["messages"]:
                    text = m["content"].strip()
                    if m["role"] == "user" and len(text) >= 20 and text not in spec:
                        self.assertNotIn(text, blob, item["id"])
                cand = (item.get("candidate_answer") or "").strip()
                if len(cand) >= 20:
                    self.assertNotIn(cand, blob, item["id"])

    def test_no_answer_table(self):
        # No reference answers or gold facts from the public set are written into the code.
        blob = "\n".join(SOURCES.values())
        for item in read_jsonl(ROOT / "public/tutor.jsonl"):
            labels = item["labels"]
            for text in [labels.get("reference_answer") or "", *(labels.get("gold_facts") or [])]:
                if len(text) >= 20:
                    self.assertNotIn(text, blob, item["id"])


if __name__ == "__main__":
    unittest.main()
