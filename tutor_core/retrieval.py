"""최신 발화와 대화 맥락으로 KB 섹션을 찾는 무상태 결합 검색."""

import asyncio
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .embedding_cache import EMBEDDING_DIMENSIONS, EmbeddingCache, normalize_vector
from .knowledge import KnowledgeBase, KnowledgeSection, load_knowledge
from .llm import ApiClients, ProviderFailure, RequestBudget
from .schemas import Message


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_ROOT = PROJECT_ROOT / "kb"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "output" / "cache" / "kb-embeddings.json"
WORDS = re.compile(r"[a-z_][a-z_0-9]*|[0-9]+|[\uac00-\ud7a3]+")
HANGUL = re.compile(r"[\uac00-\ud7a3]+")


@dataclass(frozen=True)
class SearchConfig:
    k1: float = 1.5
    b: float = 0.75
    rrf_constant: int = 60
    latest_weight: float = 3.0
    history_weight: float = 1.0
    dense_weight: float = 2.0
    top_k: int = 8
    expanded_top_k: int = 12
    startup_embedding_seconds: float = 120.0
    batch_size: int = 16


def word_tokens(text: str) -> list[str]:
    return WORDS.findall(unicodedata.normalize("NFKC", text).casefold())


def gram_tokens(text: str) -> list[str]:
    return [word[index:index + 2] for word in HANGUL.findall(unicodedata.normalize("NFKC", text)) for index in range(len(word) - 1)]


def bounded_excerpt(text: str, limit: int) -> str:
    """검색용 문자열의 앞·뒤를 함께 보존하고 문자 수 상한을 지킨다."""
    if len(text) <= limit:
        return text
    marker = "\n...\n"
    if limit <= len(marker):
        return text[-limit:] if limit > 0 else ""
    head = (limit - len(marker)) // 2
    return text[:head] + marker + text[-(limit - len(marker) - head):]


def _embedding_input(text: str) -> str:
    # UTF-8 바이트 한도 안에서 앞의 맥락과 뒤의 실제 요청을 모두 남긴다.
    data = text.encode("utf-8", errors="replace")
    if len(data) <= 7500:
        return data.decode("utf-8").strip()
    marker = "\n...\n"
    head = data[:2500].decode("utf-8", errors="ignore")
    tail = data[-(7500 - 2500 - len(marker)):].decode("utf-8", errors="ignore")
    return (head + marker + tail).strip()


@dataclass(frozen=True)
class SearchQueries:
    latest: str
    history: str

    @classmethod
    def from_messages(cls, messages: Sequence[Message]) -> "SearchQueries":
        if not messages or messages[-1].role != "user":
            raise ValueError("invalid_search_messages")
        latest = bounded_excerpt(messages[-1].content, 12000).strip()
        # 과거 assistant 발화는 대상 복원용 단서다. 반환 근거는 언제나 KB 원문이다.
        history = "\n".join(message.content for message in messages[:-1])
        if len(history) > 24000 or any(len(message.content) > 4000 for message in messages[:-1]):
            fragments, remaining = [], 24000
            for message in reversed(messages[:-1]):
                separator = 1 if fragments else 0
                allowance = min(4000, remaining - separator)
                if allowance <= 0:
                    break
                fragment = bounded_excerpt(message.content, allowance)
                fragments.append(fragment)
                remaining -= len(fragment) + separator
            history = "\n".join(reversed(fragments))
        history = history.strip()
        return cls(latest, history)


class _BM25:
    def __init__(self, documents: Sequence[Sequence[str]], config: SearchConfig):
        self.config = config
        self.count = len(documents)
        self.lengths = tuple(len(document) for document in documents)
        self.average = sum(self.lengths) / max(1, self.count) or 1.0
        postings = defaultdict(list)
        for index, document in enumerate(documents):
            for term, frequency in Counter(document).items():
                postings[term].append((index, frequency))
        self.postings = dict(postings)
        self.idf = {term: math.log(1 + (self.count - len(rows) + 0.5) / (len(rows) + 0.5)) for term, rows in postings.items()}

    def scores(self, tokens: Sequence[str]) -> dict[int, float]:
        scores = defaultdict(float)
        for term, query_frequency in Counter(tokens).items():
            for index, frequency in self.postings.get(term, ()):
                denominator = frequency + self.config.k1 * (1 - self.config.b + self.config.b * self.lengths[index] / self.average)
                scores[index] += self.idf[term] * frequency * (self.config.k1 + 1) / denominator * min(query_frequency, 3)
        return dict(scores)


@dataclass(frozen=True)
class SearchHit:
    section: KnowledgeSection
    score: float


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    mode: str
    degraded: bool = False
    failure_reason: str | None = None
    semantic_section_ids: tuple[str, ...] = ()
    source_coverage: tuple[tuple[str, int], ...] = ()

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(hit.section.section_id for hit in self.hits)

    @property
    def context_characters(self) -> int:
        return sum(len(hit.section.text) for hit in self.hits)


def merge_evidence(knowledge: KnowledgeBase, question: SearchResult, claims: SearchResult,
                   citations: Sequence[str] = (), *, limit: int = 12) -> SearchResult:
    """기존 근거 범위·순서를 보존하고 한도를 넘을 때 출처별 독점을 막는다."""
    if type(limit) is not int or not 1 <= limit <= 12:
        raise ValueError("invalid_evidence_limit")
    cited = tuple(SearchHit(knowledge.by_id[sid], 0.0) for sid in dict.fromkeys(citations) if sid in knowledge.by_id)
    sources = (("citation", cited), ("question", question.hits[:8]), ("claim", claims.hits))
    ordered = {}
    for _, source in sources:
        for hit in source:
            ordered.setdefault(hit.section.section_id, hit)
    if len(ordered) <= limit:
        # 공간이 충분하면 불필요한 근거 추가와 순서 변경을 피한다.
        coverage = tuple((name, len({hit.section.section_id for hit in source})) for name, source in sources)
        return SearchResult(tuple(ordered.values()), "verification", question.degraded or claims.degraded,
                            question.failure_reason or claims.failure_reason, source_coverage=coverage)
    positions = [0, 0, 0]
    hits, seen = [], set()
    while len(hits) < limit:
        added = False
        for index, (_, source) in enumerate(sources):
            while positions[index] < len(source) and source[positions[index]].section.section_id in seen:
                positions[index] += 1
            if positions[index] < len(source) and len(hits) < limit:
                hit = source[positions[index]]
                positions[index] += 1
                hits.append(hit)
                seen.add(hit.section.section_id)
                added = True
        if not added:
            break
    coverage = tuple((name, len(seen & {hit.section.section_id for hit in source})) for name, source in sources)
    return SearchResult(tuple(hits), "verification", question.degraded or claims.degraded,
                        question.failure_reason or claims.failure_reason, source_coverage=coverage)


@dataclass
class Retriever:
    knowledge: KnowledgeBase
    clients: ApiClients | None = field(default=None, repr=False)
    vectors: tuple[tuple[float, ...], ...] | None = field(default=None, repr=False)
    config: SearchConfig = field(default_factory=SearchConfig)
    cache_state: str = "not_used"
    counters: Counter = field(default_factory=Counter, init=False, repr=False)

    def __post_init__(self):
        texts = [section.search_text for section in self.knowledge.sections]
        self._words = _BM25([word_tokens(text) for text in texts], self.config)
        self._grams = _BM25([gram_tokens(text) for text in texts], self.config)
        if self.vectors is not None:
            if len(self.vectors) != len(self.knowledge.sections):
                raise ValueError("invalid_kb_embedding_count")
            self.vectors = tuple(normalize_vector(vector) for vector in self.vectors)

    def summary(self) -> dict:
        return {
            "sections": len(self.knowledge.sections), "tables": self.knowledge.table_count,
            "duplicate_html_sections": self.knowledge.duplicate_html_sections,
            "kb_fingerprint": self.knowledge.fingerprint,
            "dense_ready": self.vectors is not None, "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "cache_state": self.cache_state, "config": asdict(self.config), "counters": dict(self.counters),
        }

    @staticmethod
    def _queries(messages: Sequence[Message]) -> SearchQueries:
        return SearchQueries.from_messages(messages)

    @staticmethod
    def _embedding_queries(queries: SearchQueries) -> list[str]:
        return [_embedding_input(text) for text in (queries.latest, queries.history) if text]

    def _limit(self, top_k: int | None, expand: bool) -> int:
        count = top_k if top_k is not None else self.config.expanded_top_k if expand else self.config.top_k
        if type(count) is not int or not 1 <= count <= self.config.expanded_top_k:
            raise ValueError("invalid_search_top_k")
        return count

    def _ranking(self, scores: dict[int, float]) -> list[int]:
        return sorted(scores, key=lambda index: (-scores[index], self.knowledge.sections[index].section_id))

    def _fuse(self, queries: SearchQueries, dense: Sequence[tuple[float, ...]] = ()) -> tuple[dict[int, float], dict[int, float]]:
        fused = defaultdict(float)
        semantic = defaultdict(float)

        def add(ranking, weight, is_dense=False):
            for rank, index in enumerate(ranking, start=1):
                fused[index] += weight / (self.config.rrf_constant + rank)
                if is_dense:
                    semantic[index] += weight / (self.config.rrf_constant + rank)

        parts = [(queries.latest, self.config.latest_weight), (queries.history, self.config.history_weight)]
        dense_index = 0
        for text, weight in parts:
            if not text:
                continue
            add(self._ranking(self._words.scores(word_tokens(text))), weight)
            add(self._ranking(self._grams.scores(gram_tokens(text))), weight)
            if dense and self.vectors is not None:
                vector = dense[dense_index]
                similarities = {index: math.fsum(a * b for a, b in zip(vector, stored)) for index, stored in enumerate(self.vectors)}
                add(self._ranking(similarities), weight * self.config.dense_weight, True)
                dense_index += 1
        return dict(fused), dict(semantic)

    def _result(self, queries, limit, dense=(), failure_reason=None) -> SearchResult:
        scores, semantic = self._fuse(queries, dense)
        ranking = self._ranking(scores)[:limit]
        return SearchResult(
            tuple(SearchHit(self.knowledge.sections[index], scores[index]) for index in ranking),
            "hybrid" if dense else "lexical", failure_reason is not None, failure_reason,
            tuple(self.knowledge.sections[index].section_id for index in self._ranking(semantic)[:self.config.expanded_top_k]),
        )

    def lexical(self, messages: Sequence[Message], *, top_k: int | None = None, expand: bool = False) -> SearchResult:
        return self._result(self._queries(messages), self._limit(top_k, expand))

    async def search(
        self, messages: Sequence[Message], budget: RequestBudget, *, top_k: int | None = None, expand: bool = False,
    ) -> SearchResult:
        queries = self._queries(messages)
        limit = self._limit(top_k, expand)
        self.counters["queries"] += 1
        inputs = self._embedding_queries(queries)
        if not inputs or self.clients is None or self.vectors is None:
            return await asyncio.to_thread(self._result, queries, limit)
        try:
            raw = await self.clients.embed(inputs, budget)
            if len(raw) != len(inputs):
                raise ValueError("invalid_query_embedding_count")
            dense = tuple(normalize_vector(vector) for vector in raw)
        except (ProviderFailure, ValueError) as exc:
            reason = exc.reason if isinstance(exc, ProviderFailure) else "invalid_embedding"
            self.counters["embedding_fallbacks"] += 1
            return await asyncio.to_thread(self._result, queries, limit, (), reason)
        return await asyncio.to_thread(self._result, queries, limit, dense)


async def prepare_retriever(
    clients: ApiClients, *, kb_root: Path = DEFAULT_KB_ROOT, cache_path: Path = DEFAULT_CACHE_PATH,
    config: SearchConfig | None = None,
    retriever_class: type[Retriever] = Retriever,
) -> Retriever:
    resolved = config or SearchConfig()
    kb = await asyncio.to_thread(load_knowledge, kb_root)
    cache = EmbeddingCache(cache_path, clients.settings.embedding_model)
    vectors = await asyncio.to_thread(cache.load, kb)
    cache_state = "hit"
    if vectors is None:
        budget = RequestBudget.start(resolved.startup_embedding_seconds)
        vectors_list = []
        for offset in range(0, len(kb.sections), resolved.batch_size):
            inputs = [section.embedding_text for section in kb.sections[offset:offset + resolved.batch_size]]
            result = await clients.embed(inputs, budget, seconds=min(45.0, budget.remaining()))
            if len(result) != len(inputs):
                raise ProviderFailure("invalid_embedding")
            vectors_list.extend(normalize_vector(vector) for vector in result)
        vectors = tuple(vectors_list)
        saved = await asyncio.to_thread(cache.save, kb, vectors)
        cache_state = "built" if saved else "write_unavailable"
    return await asyncio.to_thread(retriever_class, kb, clients, vectors, resolved, cache_state)
