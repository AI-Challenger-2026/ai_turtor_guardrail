"""자료·캐시·결합 검색의 무결성과 장애 경계를 외부 통신 없이 검사한다."""

import asyncio
import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from server import create_app
from tests.test_foundation import ProtocolDouble, SETTINGS
from tutor_core.embedding_cache import EmbeddingCache, normalize_vector
from tutor_core.knowledge import KnowledgeBase, KnowledgeError, KnowledgeSection, load_knowledge
from tutor_core.llm import ProviderFailure, RequestBudget
from tutor_core.retrieval import Retriever, SearchConfig, SearchQueries, prepare_retriever
from tutor_core.schemas import Message


ROOT = Path(__file__).resolve().parents[1]


def basis(index):
    return tuple(1.0 if slot == index else 0.0 for slot in range(1536))


def small_kb():
    sections = tuple(KnowledgeSection(f"PY-01-0{index + 1}", text, text, (text,), ())
                     for index, text in enumerate(("alpha 반복문", "beta 조건문", "gamma 함수")))
    return KnowledgeBase(sections, "a" * 64, 0, 0)


class WorkspaceTemporaryDirectory(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "output" / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, prefix="retrieval-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class KnowledgeTests(WorkspaceTemporaryDirectory):
    def test_distributed_data_has_all_sections_tables_and_duplicates(self):
        kb = load_knowledge(ROOT / "kb")
        self.assertEqual((len(kb.sections), kb.table_count, kb.duplicate_html_sections), (48, 23, 4))
        self.assertIn("3만 원", kb.by_id["FIN-01-01"].text)
        self.assertNotIn("(표:", kb.by_id["FIN-01-01"].text)

    def test_csv_is_inserted_at_original_location_and_changes_fingerprint(self):
        self.write("finance/FIN-01-01.md", "# FIN-01-01 제목\n앞\n(표: FIN-01-01-table1.csv)\n뒤")
        self.write("finance/FIN-01-01-table1.csv", "이름,금액\n하람,100\n")
        before = load_knowledge(self.root)
        self.assertLess(before.sections[0].text.index("앞"), before.sections[0].text.index("하람,100"))
        self.assertLess(before.sections[0].text.index("하람,100"), before.sections[0].text.index("뒤"))
        self.write("finance/FIN-01-01-table1.csv", "이름,금액\n하람,200\n")
        self.assertNotEqual(before.fingerprint, load_knowledge(self.root).fingerprint)

    def test_duplicate_ids_orphan_tables_and_missing_references_fail(self):
        cases = (
            {"a/PY-01-01.md": "# PY-01-01 하나", "b/PY-01-01.md": "# PY-01-01 둘"},
            {"a/PY-01-01.md": "# PY-01-01 제목", "a/PY-01-02-table1.csv": "값\n1"},
            {"a/PY-01-01.md": "# PY-01-01 제목\n(표: PY-01-01-table1.csv)"},
            {"a/PY-01-01.md": "# PY-01-01 제목", "a/PY-01-01-table1.csv": "열1,열2\n1"},
        )
        for number, files in enumerate(cases):
            base = self.root / str(number)
            for name, text in files.items():
                self.write(str(number) + "/" + name, text)
            with self.subTest(number=number), self.assertRaises(KnowledgeError):
                load_knowledge(base)

    def test_empty_files_and_heading_mismatch_fail(self):
        for number, text in enumerate(("", "# PY-01-02 잘못된 제목")):
            self.write(f"{number}/a/PY-01-01.md", text)
            with self.assertRaises(KnowledgeError):
                load_knowledge(self.root / str(number))

    def test_html_duplicate_is_compared_and_never_indexed_twice(self):
        self.write("a/PY-01-01.md", "# PY-01-01 제목\n## 금액\n값은 100원.")
        self.write("a/PY-01.html", "<h1>PY-01-01 제목</h1><h2>금액</h2><p>값은 100원.</p>")
        kb = load_knowledge(self.root)
        self.assertEqual((len(kb.sections), kb.duplicate_html_sections), (1, 1))
        self.write("a/PY-01.html", "<h1>PY-01-01 제목</h1><h2>금액</h2><p>값은 200원.</p>")
        with self.assertRaisesRegex(KnowledgeError, "html_content_mismatch"):
            load_knowledge(self.root)


class CacheTests(WorkspaceTemporaryDirectory):
    def test_roundtrip_and_atomic_write_leave_no_temporary_files(self):
        cache = EmbeddingCache(self.root / "cache" / "kb.json", "text-embedding-3-small")
        kb = small_kb()
        self.assertTrue(cache.save(kb, [basis(index) for index in range(3)]))
        self.assertEqual(cache.load(kb), tuple(basis(index) for index in range(3)))
        self.assertEqual(list(cache.path.parent.glob("*.tmp")), [])
        data = json.loads(cache.path.read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"version", "kb_fingerprint", "model", "dimensions", "preprocess_version", "section_ids", "vectors", "vector_sha256"})

    def test_each_identity_change_invalidates_cache(self):
        cache = EmbeddingCache(self.root / "kb.json", "text-embedding-3-small")
        kb = small_kb()
        cache.save(kb, [basis(index) for index in range(3)])
        self.assertIsNone(cache.load(replace(kb, fingerprint="b" * 64)))
        self.assertIsNone(EmbeddingCache(cache.path, "different-model").load(kb))
        self.assertIsNone(EmbeddingCache(cache.path, "text-embedding-3-small", 2).load(kb))
        original = json.loads(cache.path.read_text(encoding="utf-8"))
        for field, value in (("preprocess_version", "old"), ("version", 0), ("section_ids", list(reversed(original["section_ids"])))):
            cache.path.write_text(json.dumps({**original, field: value}), encoding="utf-8")
            self.assertIsNone(cache.load(kb))

    def test_corrupt_or_tampered_cache_is_not_used(self):
        cache = EmbeddingCache(self.root / "kb.json", "text-embedding-3-small")
        kb = small_kb()
        self.write("kb.json", "invalid-json")
        self.assertIsNone(cache.load(kb))
        cache.save(kb, [basis(index) for index in range(3)])
        data = json.loads(cache.path.read_text(encoding="utf-8"))
        data["vectors"][0][0] = 0.2
        cache.path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIsNone(cache.load(kb))

    def test_invalid_vectors_fail_before_use(self):
        for values in ([0.0] * 1536, [math.nan] * 1536, [True] * 1536, [1.0]):
            with self.assertRaises(ValueError):
                normalize_vector(values)


class EmbeddingDouble:
    """검색 결합과 장애 경계를 위한 결정적인 벡터 대역."""

    def __init__(self):
        self.settings = SimpleNamespace(embedding_model="text-embedding-3-small")
        self.counters = Counter()
        self.failure = None

    async def embed(self, texts, budget, **kwargs):
        self.counters["embedding_attempts"] += 1
        if self.failure:
            raise self.failure
        return [basis(1 if "semantic-two" in text else 2 if "semantic-three" in text else 0) for text in texts]


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = EmbeddingDouble()
        self.kb = small_kb()
        self.retriever = Retriever(self.kb, self.client, tuple(basis(index) for index in range(3)))

    async def test_semantic_search_can_find_section_without_lexical_overlap(self):
        messages = [Message(role="user", content="semantic-two")]
        result = await self.retriever.search(messages, RequestBudget.start(1), top_k=1)
        self.assertEqual(result.section_ids, ("PY-01-02",))
        self.assertEqual(result.mode, "hybrid")

    async def test_failure_preserves_lexical_result_and_records_only_reason(self):
        messages = [Message(role="user", content="반복문 alpha")]
        expected = self.retriever.lexical(messages).section_ids
        self.client.failure = ProviderFailure("upstream_timeout")
        result = await self.retriever.search(messages, RequestBudget.start(1))
        self.assertEqual(result.section_ids, expected)
        self.assertTrue(result.degraded)
        self.assertEqual(result.failure_reason, "upstream_timeout")
        self.assertEqual(self.retriever.counters["embedding_fallbacks"], 1)

    async def test_requests_are_independent_and_no_query_cache_is_created(self):
        async def one(text):
            return await self.retriever.search([Message(role="user", content=text)], RequestBudget.start(1), top_k=1)

        results = await asyncio.gather(one("semantic-two"), one("semantic-three"), one("semantic-two"))
        self.assertEqual([result.section_ids[0] for result in results], ["PY-01-02", "PY-01-03", "PY-01-02"])
        self.assertEqual(self.client.counters["embedding_attempts"], 3)
        self.assertNotIn("semantic-two", repr(self.retriever.counters))

    async def test_empty_query_makes_no_embedding_call_and_limits_are_checked(self):
        result = await self.retriever.search([Message(role="user", content=" ")], RequestBudget.start(1))
        self.assertEqual(result.hits, ())
        self.assertEqual(self.client.counters["embedding_attempts"], 0)
        with self.assertRaises(ValueError):
            self.retriever.lexical([Message(role="user", content="alpha")], top_k=13)

    async def test_history_is_only_a_query_clue_and_results_are_kb_sections(self):
        messages = [Message(role="user", content="alpha"), Message(role="assistant", content="과거의 틀린 설명"), Message(role="user", content="그 내용을 설명해줘")]
        query = SearchQueries.from_messages(messages)
        self.assertEqual(query.latest, messages[-1].content)
        self.assertIn("alpha", query.history)
        result = self.retriever.lexical(messages)
        self.assertTrue(all(hit.section is self.kb.by_id[hit.section.section_id] for hit in result.hits))

    async def test_ties_are_ordered_by_section_id(self):
        messages = [Message(role="user", content="semantic-two")]
        same = Retriever(self.kb, self.client, (basis(1), basis(1), basis(1)))
        first = await same.search(messages, RequestBudget.start(1))
        second = await same.search(messages, RequestBudget.start(1))
        self.assertEqual(first.section_ids, tuple(sorted(self.kb.by_id)))
        self.assertEqual(first.section_ids, second.section_ids)


class PreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        parent = ROOT / "output" / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, prefix="retrieval-preparation-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "kb" / "a").mkdir(parents=True)
        self.document = self.root / "kb" / "a" / "PY-01-01.md"
        self.document.write_text("# PY-01-01 제목\n본문", encoding="utf-8")

    async def test_startup_cache_hit_avoids_embeddings_and_kb_change_rebuilds(self):
        client = EmbeddingDouble()
        kwargs = {"kb_root": self.root / "kb", "cache_path": self.root / "cache" / "kb.json"}
        first = await prepare_retriever(client, **kwargs)
        self.assertEqual(first.cache_state, "built")
        self.assertEqual(client.counters["embedding_attempts"], 1)
        second = await prepare_retriever(client, **kwargs)
        self.assertEqual(second.cache_state, "hit")
        self.assertEqual(client.counters["embedding_attempts"], 1)
        before = hashlib.sha256(kwargs["cache_path"].read_bytes()).hexdigest()
        await second.search([Message(role="user", content="semantic-two")], RequestBudget.start(1))
        self.assertEqual(before, hashlib.sha256(kwargs["cache_path"].read_bytes()).hexdigest())
        self.document.write_text("# PY-01-01 제목\n변경 본문", encoding="utf-8")
        third = await prepare_retriever(client, **kwargs)
        self.assertEqual(third.cache_state, "built")
        self.assertNotEqual(first.knowledge.fingerprint, third.knowledge.fingerprint)

    async def test_failed_startup_embedding_does_not_leave_partial_cache(self):
        client = EmbeddingDouble()
        client.failure = ProviderFailure("upstream_timeout")
        path = self.root / "cache" / "kb.json"
        with self.assertRaises(ProviderFailure):
            await prepare_retriever(client, kb_root=self.root / "kb", cache_path=path)
        self.assertFalse(path.exists())

    async def test_server_does_not_become_ready_before_index_preparation(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def factory(clients):
            entered.set()
            await release.wait()
            return Retriever(small_kb())

        application = create_app(SETTINGS, ProtocolDouble, factory)

        async def lifetime():
            async with application.router.lifespan_context(application):
                self.assertTrue(application.state.ready)

        task = asyncio.create_task(lifetime())
        await entered.wait()
        self.assertFalse(application.state.ready)
        release.set()
        await task
        self.assertFalse(application.state.ready)
        self.assertTrue(application.state.clients.closed)

    async def test_invalid_kb_startup_closes_client(self):
        async def factory(clients):
            raise KnowledgeError("html_content_mismatch")

        application = create_app(SETTINGS, ProtocolDouble, factory)
        with self.assertRaisesRegex(RuntimeError, "html_content_mismatch"):
            async with application.router.lifespan_context(application):
                self.fail("자료 오류가 있는 서버가 준비 상태가 되면 안 됩니다.")
        self.assertTrue(application.state.clients.closed)


if __name__ == "__main__":
    unittest.main()
