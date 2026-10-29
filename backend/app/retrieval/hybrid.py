"""검색기 (retriever): BM25 기본 + 선택적 dense 를 RRF(Reciprocal Rank Fusion)로 결합.

- BM25 만: 정규화 BM25 점수 순으로 top_k, 임계값(min_score) 이상만 근거로 통과.
- 하이브리드: 두 순위를 RRF 로 합쳐 top_k. 통과 조건 = BM25 정규화 점수 ≥ min_score 또는 dense 유사도 ≥ dense.min_score.
- 통과한 청크가 없으면 호출자가 "근거 없음"으로 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.kb.index import KBIndex
from app.logging_utils import get_logger
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex, DenseUnavailable, Encoder
from app.retrieval.tokenizer import get_tokenizer
from app.retrieval.types import RetrievedChunk
from app.schemas import ModuleInfo
from app.settings import RetrievalSection

log = get_logger("retrieval")


@dataclass
class RetrievalResult:
    query: str
    candidates: list[RetrievedChunk] = field(default_factory=list)  # top_k 후보 (임계값 미달 포함)
    passed: list[RetrievedChunk] = field(default_factory=list)  # 근거로 쓸 청크
    threshold: float = 0.0
    mode: str = "bm25"

    @property
    def no_evidence(self) -> bool:
        return not self.passed


def index_text(unit: str, section: str, text: str) -> str:
    """검색용 텍스트 — 단원·섹션 제목을 함께 넣어 제목 단어도 매칭되게 한다."""
    return f"{unit}\n{section}\n{text}"


class Retriever:
    def __init__(
        self,
        index: KBIndex,
        config: RetrievalSection,
        *,
        dense_encoder: Encoder | None = None,
        dense_cache_path: Path | None = None,
    ) -> None:
        self.index = index
        self.config = config
        self.tokenizer = get_tokenizer(config.tokenizer)
        texts = [index_text(c.unit, c.section, c.text) for c in index.chunks]
        self.bm25 = BM25Index([self.tokenizer.tokenize(t) for t in texts], k1=config.bm25_k1, b=config.bm25_b)
        self.dense: DenseIndex | None = None
        self.dense_status = ModuleInfo(status="disabled", detail="설정에서 꺼짐 (retrieval.dense.enabled=false)")
        if config.dense.enabled:
            try:
                self.dense = DenseIndex(
                    texts,
                    config.dense.model,
                    encoder=dense_encoder,
                    cache_path=dense_cache_path if config.dense.cache else None,
                )
                self.dense_status = ModuleInfo(status="enabled", detail=config.dense.model)
            except DenseUnavailable as exc:
                log.warning("dense 검색 비활성화 (disabled): %s", exc)
                self.dense_status = ModuleInfo(status="disabled", detail=f"켜도록 설정됐지만 사용할 수 없음: {exc}")

    @property
    def threshold(self) -> float:
        """BM25 정규화 점수 임계값 — 토크나이저마다 점수 분포가 달라 따로 둔다."""
        return self.config.min_score if self.tokenizer.name == "kiwi" else self.config.min_score_simple

    # ---- 상태 (status for /api/config) ----
    def bm25_status(self) -> ModuleInfo:
        return ModuleInfo(status="enabled", detail=f"tokenizer={self.tokenizer.name}, chunks={len(self.index.chunks)}")

    def hybrid_status(self) -> ModuleInfo:
        if self.dense is not None:
            return ModuleInfo(status="enabled", detail=f"BM25 + dense RRF (k={self.config.rrf_k})")
        return ModuleInfo(status="disabled", detail="dense 검색이 꺼져 있어 BM25 단독 검색")

    # ---- 검색 (search) ----
    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self.config.top_k
        chunks = self.index.chunks
        if not chunks or not query.strip():
            return RetrievalResult(query=query, threshold=self.threshold)
        raw, norm = self.bm25.normalized_scores(self.tokenizer.tokenize(query))

        if self.dense is None:
            order = sorted(range(len(chunks)), key=lambda i: (-norm[i], i))[:k]
            candidates = [
                self._make(i, rank, score=norm[i], bm25=norm[i], dense=None, passed=norm[i] >= self.threshold)
                for rank, i in enumerate(order, start=1)
                if raw[i] > 0
            ]
            mode = "bm25"
        else:
            dense = self.dense.scores(query)
            bm25_rank = {i: r for r, i in enumerate(sorted(range(len(chunks)), key=lambda i: (-norm[i], i)), start=1)}
            dense_rank = {i: r for r, i in enumerate(sorted(range(len(chunks)), key=lambda i: (-dense[i], i)), start=1)}
            rrf = {i: 1.0 / (self.config.rrf_k + bm25_rank[i]) + 1.0 / (self.config.rrf_k + dense_rank[i]) for i in range(len(chunks))}
            order = sorted(range(len(chunks)), key=lambda i: (-rrf[i], i))[:k]
            candidates = [
                self._make(
                    i, rank, score=rrf[i], bm25=norm[i], dense=dense[i],
                    passed=norm[i] >= self.threshold or dense[i] >= self.config.dense.min_score,
                )
                for rank, i in enumerate(order, start=1)
            ]
            mode = "hybrid"
        passed = [c for c in candidates if c.passed_threshold]
        return RetrievalResult(query=query, candidates=candidates, passed=passed, threshold=self.threshold, mode=mode)

    def _make(self, i: int, rank: int, *, score: float, bm25: float | None, dense: float | None, passed: bool) -> RetrievedChunk:
        c = self.index.chunks[i]
        return RetrievedChunk(
            chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title, unit=c.unit, section=c.section, text=c.text,
            score=score, bm25_score=bm25, dense_score=dense, rank=rank, passed_threshold=passed,
        )
