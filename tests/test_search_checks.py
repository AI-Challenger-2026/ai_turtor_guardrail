"""검색 비교의 공정성·표본 분리·본문 비노출·완료 경계를 외부 API 없이 검사한다."""

import json
import unittest
from collections import Counter
from copy import deepcopy

from tests.test_retrieval import EmbeddingDouble, basis, small_kb
from tools.evaluation_trace import clean_event, result_metadata
from tools.search_baseline_server import LegacyRetriever, legacy_embedding_queries, legacy_evidence, legacy_queries
from tools.search_checks import (
    PROFILES, finalize_comparison, load_public, paired_search, profile_summary, retrieval_summary, search_cases, select_cases,
)
from tutor_core.llm import ProviderFailure, RequestBudget
from tutor_core.retrieval import Retriever, SearchHit, SearchQueries, SearchResult, merge_evidence
from tutor_core.schemas import Message


class PublicSelectionTests(unittest.TestCase):
    def test_subset_and_long_probes_are_deterministic_and_do_not_modify_public_data(self):
        public = load_public()
        before = deepcopy(public)
        selected = select_cases(public)
        self.assertEqual(Counter(item["endpoint"] for item in selected), {"tutor": 24, "guard": 6})
        self.assertEqual(len({item["id"] for item in selected}), 30)
        courses = Counter(item["labels"]["evidence"][0].split("-")[0] for item in selected if item["endpoint"] == "tutor")
        self.assertEqual(courses, {"PY": 8, "PR": 8, "FIN": 8})
        cases = search_cases(public)
        self.assertEqual(Counter(row[1] for row in cases), {"public": 119, "long_latest": 12, "long_history": 12})
        self.assertEqual(len({row[0] for row in cases}), 143)
        for _, kind, messages, _ in cases:
            if kind == "long_latest":
                self.assertGreater(len(messages[-1].content), 12000)
            elif kind == "long_history":
                self.assertGreater(len(messages[-2].content), 24000)
        self.assertEqual(public, before)
        self.assertEqual([item["id"] for item in selected], [item["id"] for item in select_cases(public)])


class PairedSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clients = EmbeddingDouble()
        self.retriever = Retriever(small_kb(), self.clients, tuple(basis(index) for index in range(3)))

    async def test_same_query_uses_one_embedding_and_has_identical_ranking(self):
        messages = [Message(role="user", content="semantic-two 검사 표식")]
        result = await paired_search(self.retriever, messages)
        self.assertEqual(self.clients.counters["embedding_attempts"], 1)
        self.assertEqual(result["unique_embedding_inputs"], 1)
        self.assertFalse(result["queries_changed"])
        self.assertTrue(result["rankings_identical"])
        self.assertEqual(result["current_search"]["ids"][0], "PY-01-02")
        self.assertNotIn("검사 표식", json.dumps(result, ensure_ascii=False))
        await paired_search(self.retriever, messages)
        self.assertEqual(self.clients.counters["embedding_attempts"], 2)

    async def test_long_tail_comparison_records_limits_without_saving_inputs(self):
        messages = [Message(role="user", content="배경 " * 10000 + "semantic-two")]
        result = await paired_search(self.retriever, messages)
        self.assertTrue(result["queries_changed"])
        self.assertEqual(result["unique_embedding_inputs"], 2)
        for name in PROFILES:
            self.assertLessEqual(result[name]["query_characters"][0], 12000)
            self.assertLessEqual(max(result[name]["embedding_bytes"]), 7500)
        self.assertNotIn("semantic-two", json.dumps(result, ensure_ascii=False))

    async def test_failed_embedding_does_not_count_as_complete_real_search(self):
        self.clients.failure = ProviderFailure("upstream_timeout")
        result = await paired_search(self.retriever, [Message(role="user", content="alpha")])
        for name in PROFILES:
            self.assertEqual(result[name]["mode"], "lexical")
            self.assertTrue(result[name]["degraded"])
            self.assertEqual(result[name]["failure_reason"], "upstream_timeout")

    async def test_baseline_subclass_replays_head_truncation_and_new_class_retains_tail(self):
        messages = [Message(role="user", content="무관한 배경 " * 4000 + "\nsemantic-two")]
        legacy = LegacyRetriever(small_kb(), self.clients, self.retriever.vectors)
        self.assertEqual(legacy._queries(messages), legacy_queries(messages))
        self.assertEqual(legacy._embedding_queries(legacy._queries(messages)), legacy_embedding_queries(legacy_queries(messages)))
        old = await legacy.search(messages, RequestBudget.start(1), top_k=1)
        current = await self.retriever.search(messages, RequestBudget.start(1), top_k=1)
        self.assertNotEqual(old.section_ids, current.section_ids)
        self.assertEqual(current.section_ids, ("PY-01-02",))


class EvidenceComparisonTests(unittest.TestCase):
    def test_legacy_and_new_profiles_keep_complete_sources_but_balance_differently(self):
        from tests.test_retrieval import EvidenceSelectionTests

        kb = EvidenceSelectionTests().knowledge()
        question = SearchResult(tuple(SearchHit(section, 1) for section in kb.sections[12:16]), "test")
        claims = SearchResult(tuple(SearchHit(section, 1) for section in kb.sections[16:20]), "test")
        citations = tuple(section.section_id for section in kb.sections[:12])
        old, new = (function(kb, question, claims, citations) for function in (legacy_evidence, merge_evidence))
        self.assertEqual(dict(old.source_coverage), {"citation": 12, "question": 0, "claim": 0})
        self.assertEqual(dict(new.source_coverage), {"citation": 4, "question": 4, "claim": 4})
        self.assertTrue(all(hit.section is kb.by_id[hit.section.section_id] for result in (old, new) for hit in result.hits))
        metadata = result_metadata(new)
        event = clean_event({"stage": "output_verification", "status": "ok", "started_seconds": 0,
                            "elapsed_seconds": 1, **metadata, "candidate_answer": "기록하면 안 되는 표식"})
        self.assertEqual(event["source_coverage"], {"citation": 4, "question": 4, "claim": 4})
        self.assertEqual(event["context_characters"], new.context_characters)
        self.assertNotIn("기록하면 안 되는 표식", json.dumps(event, ensure_ascii=False))

    def test_short_message_profiles_are_exactly_compatible(self):
        messages = [Message(role="user", content="alpha"), Message(role="assistant", content="이전 설명"), Message(role="user", content=" beta ")]
        self.assertEqual(legacy_queries(messages), SearchQueries.from_messages(messages))
        self.assertEqual(legacy_embedding_queries(legacy_queries(messages)), Retriever._embedding_queries(SearchQueries.from_messages(messages)))


class ComparisonSummaryTests(unittest.TestCase):
    def test_synthetic_probe_recall_is_not_pooled_into_public_denominator(self):
        rows = []
        for kind, ids in (("public", ["PY-01-01"]), ("long_latest", []), ("long_history", [])):
            profile = {"ids": ids, "degraded": False, "context_characters": 100, "ranking_elapsed_seconds": .01}
            rows.append({"kind": kind, "evidence": ["PY-01-01"], "queries_changed": False, "rankings_identical": True,
                         "embedding_elapsed_seconds": .1, **{name: profile for name in PROFILES}})
        result = retrieval_summary(rows)
        self.assertEqual(result["public"]["current_search"]["recall_at_k"]["8"], 1)
        self.assertEqual(result["long_latest"]["current_search"]["recall_at_k"]["8"], 0)

    def sample_report(self):
        measured = {"metrics": {"tutor_citation_f1": .5}, "server_latency_seconds": {"p95": 5},
                    "diagnostics": {"items": []}}
        return {"source_unchanged": True, "retrieval": {"measurement_complete": True},
                "profiles": {name: {"measurement_complete": True, "endpoint_contract_pass": True,
                                     "summary": profile_summary(measured)} for name in PROFILES}}

    def test_two_profile_measurements_are_never_a_final_quality_pass(self):
        report = self.sample_report()
        finalize_comparison(report)
        self.assertTrue(report["measurement_complete"])
        self.assertFalse(report["final_quality_verified"])
        self.assertEqual(report["selected_metric_deltas"]["tutor_citation_f1"], 0)

    def test_missing_profile_source_change_or_api_failure_rejects_completion(self):
        for change in ("source", "profile", "embedding", "error"):
            report = self.sample_report()
            if change == "source":
                report["source_unchanged"] = False
            elif change == "profile":
                report["profiles"].pop("current_search")
            elif change == "embedding":
                report["retrieval"]["measurement_complete"] = False
            else:
                report["error"] = {"type": "Interrupted"}
            finalize_comparison(report)
            self.assertFalse(report["measurement_complete"])
            self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
