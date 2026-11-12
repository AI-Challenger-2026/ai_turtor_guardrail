"""같은 현재 정책·예산에서 이전 검색·근거 배분만 재현하는 평가 대조군."""

from server import create_app
from tools.diagnostic_server import (
    DiagnosticApplication, ObservedApiClients, ObservedInputPolicy, ObservedOutputVerifier,
    ObservedRetriever, ObservedTutorService,
)
from tutor_core.retrieval import Retriever, SearchHit, SearchQueries, SearchResult, prepare_retriever


def legacy_queries(messages):
    return SearchQueries(messages[-1].content[:12000].strip(),
                         "\n".join(message.content for message in messages[:-1])[-24000:].strip())


def legacy_embedding_queries(queries):
    return [text.encode("utf-8", errors="replace")[:7500].decode("utf-8", errors="ignore").strip()
            for text in (queries.latest, queries.history) if text]


class LegacyRetriever(Retriever):
    _queries = staticmethod(legacy_queries)
    _embedding_queries = staticmethod(legacy_embedding_queries)


def legacy_evidence(knowledge, question, claims, citations=()):
    cited = tuple(SearchHit(knowledge.by_id[sid], 0.0) for sid in citations if sid in knowledge.by_id)
    sources = (("citation", cited), ("question", question.hits[:8]), ("claim", claims.hits))
    hits, seen = [], set()
    for _, source in sources:
        for hit in source:
            if hit.section.section_id not in seen and len(hits) < 12:
                hits.append(hit)
                seen.add(hit.section.section_id)
    coverage = tuple((name, len(seen & {hit.section.section_id for hit in source})) for name, source in sources)
    return SearchResult(tuple(hits), "verification", question.degraded or claims.degraded,
                        question.failure_reason or claims.failure_reason, source_coverage=coverage)


class LegacyVerifier(ObservedOutputVerifier):
    def _merge_evidence(self, question, claims, citations):
        return legacy_evidence(self.retriever.knowledge, question, claims, citations or ())


async def prepare_legacy(clients):
    return ObservedRetriever(await prepare_retriever(clients, retriever_class=LegacyRetriever))


app = DiagnosticApplication(create_app(
    client_factory=ObservedApiClients, retrieval_factory=prepare_legacy,
    policy_factory=ObservedInputPolicy, verifier_factory=LegacyVerifier, tutor_factory=ObservedTutorService,
))
