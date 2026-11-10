"""Static tripwires for the competition rules (CLAUDE.md section 2). No network.

These do not prove compliance; they fail loudly if code drifts toward something the rules
forbid: printing or logging content, other network clients or hosts, caches, file writes,
stored data beyond kb/ and public/ ids, dev-split text in prompts, or answer tables.
"""
import json
import re
import unittest
from pathlib import Path

import tests.test_app  # noqa: F401  (sets the test environment first)

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SOURCES = {p.name: p.read_text(encoding="utf-8") for p in sorted(APP.glob("*.py"))}
SERVER = (ROOT / "server.py").read_text(encoding="utf-8")


def code_only(src: str) -> str:
    """Source without comments and docstrings/strings, for keyword checks."""
    src = re.sub(r'(?s)""".*?"""', '""', src)
    src = re.sub(r"#[^\n]*", "", src)
    return src


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


class OutputAndLoggingTests(unittest.TestCase):
    def test_no_print(self):
        for name, src in {**SOURCES, "server.py": SERVER}.items():
            self.assertNotRegex(code_only(src), r"\bprint\s*\(", name)

    def test_only_logs_module_uses_logging(self):
        for name, src in SOURCES.items():
            if name != "logs.py":
                self.assertNotRegex(code_only(src), r"^\s*(import logging|from logging)", name)

    def test_log_fields_are_allowlisted(self):
        allowed = {"event", "endpoint", "status", "latency_ms", "body_bytes", "input_chars", "n_msgs",
                   "outcome", "degraded", "error", "kind", "attempt", "kb_sections", "kb_chunks", "embeddings"}
        fields = set(re.findall(r'"(\w+)":', SOURCES["logs.py"]))
        self.assertTrue(fields, "no log fields found")
        self.assertLessEqual(fields, allowed)

    def test_no_traceback_or_exception_text_logging(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotIn("exc_info", code, name)
            self.assertNotIn("traceback", code, name)
            self.assertNotRegex(code, r"str\(\s*exc", name)


class ExternalCallTests(unittest.TestCase):
    def test_no_other_network_clients(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotRegex(code, r"^\s*(import|from)\s+(requests|urllib|httpx|aiohttp|socket|subprocess|http\.client)\b",
                                name)

    def test_no_hard_coded_hosts(self):
        for name, src in {**SOURCES, "server.py": SERVER}.items():
            self.assertNotRegex(src, r"https?://", name)

    def test_openai_clients_only_in_llm_module(self):
        for name, src in SOURCES.items():
            n = len(re.findall(r"AsyncOpenAI\(", code_only(src)))
            self.assertEqual(n, 2 if name == "llm.py" else 0, name)
        llm_src = SOURCES["llm.py"]
        # Chat client: environment OPENAI_* only; embedding client: EMBEDDING_* settings only.
        self.assertIn("_client = AsyncOpenAI(timeout=config.REQUEST_DEADLINE_S, max_retries=0)", llm_src)
        self.assertIn("api_key=config.EMBED_API_KEY, base_url=config.EMBED_BASE_URL", llm_src)

    def test_embedding_settings_come_from_embedding_env(self):
        cfg = SOURCES["config.py"]
        for var in ("EMBEDDING_MODEL", "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY"):
            self.assertIn(f'os.environ.get("{var}"', cfg)


class StateTests(unittest.TestCase):
    def test_no_caches_or_memoization(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotRegex(code, r"lru_cache|functools\.cache|@cache\b|cachetools|memoize", name)

    def test_module_state_is_written_only_at_start_up(self):
        for name, src in SOURCES.items():
            for m in re.finditer(r"^\s*global\s+(\w+)", src, re.M):
                self.assertEqual((name, m.group(1)), ("retrieval.py", "_VECTORS"))
                fn = src[:m.start()].rsplit("\ndef ", 1)[-1].rsplit("\nasync def ", 1)[-1]
                self.assertTrue(fn.startswith("build_embeddings("), "global outside the start-up builder")

    def test_no_file_writes(self):
        for name, src in SOURCES.items():
            code = code_only(src)
            self.assertNotRegex(code, r"write_text|write_bytes|\bopen\(|\.dump\(", name)


class DataTests(unittest.TestCase):
    def test_app_package_holds_no_data_files(self):
        others = sorted(p.name for p in APP.iterdir() if p.is_file() and p.suffix != ".py")
        self.assertEqual(others, ["fewshot_ids.json"])

    def test_fewshot_file_holds_ids_only(self):
        data = json.loads((APP / "fewshot_ids.json").read_text(encoding="utf-8"))
        for shot in data["shots"]:
            self.assertLessEqual(set(shot), {"id", "harm_kind"})
            self.assertRegex(shot["id"], r"^pub-[0-9a-f]{8}$")

    def test_no_dev_text_in_prompts_or_code(self):
        blob = "\n".join(SOURCES.values())
        # Quick-question button phrases are published in SPEC.md section 2; they are not item text.
        spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
        split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
        dev = set(split["dev"])
        for endpoint in ("tutor", "guard"):
            for item in read_jsonl(ROOT / f"public/{endpoint}.jsonl"):
                if item["id"] not in dev:
                    continue
                self.assertNotIn(item["id"], blob)
                for m in item["messages"]:
                    text = m["content"].strip()
                    if m["role"] == "user" and len(text) >= 20 and text not in spec:
                        self.assertNotIn(text, blob, item["id"])

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
