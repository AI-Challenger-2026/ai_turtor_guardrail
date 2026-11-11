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


def _embedding_input(text: str) -> str:
    # 토큰 하나는 적어도 한 바이트를 쓰므로 UTF-8 바이트 제한으로 입력 한도에 여유를 둔다.
    return text.encode("utf-8", errors="replace")[:7500].decode("utf-8", errors="ignore").strip()


@dataclass(frozen=True)
class SearchQueries:
    latest: str
    history: str

    @classmethod
    def from_messages(cls, messages: Sequence[Message]) -> "SearchQueries":
        if not messages or messages[-1].role != "user":
            raise ValueError("invalid_search_messages")
        latest = messages[-1].content[:12000].strip()
        # 과거 assistant 발화는 대상 복원용 단서다. 반환 근거는 언제나 KB 원문이다.
        history = "\n".join(message.content for message in messages[:-1])[-24000:].strip()
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

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(hit.section.section_id for hit in self.hits)


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
        return self._result(SearchQueries.from_messages(messages), self._limit(top_k, expand))

    async def search(
        self, messages: Sequence[Message], budget: RequestBudget, *, top_k: int | None = None, expand: bool = False,
    ) -> SearchResult:
        queries = SearchQueries.from_messages(messages)
        limit = self._limit(top_k, expand)
        self.counters["queries"] += 1
        inputs = [_embedding_input(text) for text in (queries.latest, queries.history) if text]
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
    return await asyncio.to_thread(Retriever, kb, clients, vectors, resolved, cache_state)
