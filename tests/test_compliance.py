"""Static checks for the competition rules: no content in logs, no other network clients or
hosts, no caches or file writes, no data files in the server, no whole public utterance in the
code, no answer tables.
"""
import json
import re
import unittest
from pathlib import Path

import tests.test_server

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SOURCES = {p.name: p.read_text(encoding="utf-8") for p in sorted(APP.glob("*.py"))}
SOURCES["server.py"] = (ROOT / "server.py").read_text(encoding="utf-8")


def code_only(src: str) -> str:
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
            self.assertNotRegex(code, r"\bOpenAI\(", name)
            self.assertNotIn("embeddings.create", code, name)
            n = len(re.findall(r"AsyncOpenAI\(", code))
            self.assertEqual(n, 1 if name == "llm.py" else 0, name)
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
        blob = "\n".join(SOURCES.values())
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
        blob = "\n".join(SOURCES.values())
        for item in read_jsonl(ROOT / "public/tutor.jsonl"):
            labels = item["labels"]
            for text in [labels.get("reference_answer") or "", *(labels.get("gold_facts") or [])]:
                if len(text) >= 20:
                    self.assertNotIn(text, blob, item["id"])


if __name__ == "__main__":
    unittest.main()
