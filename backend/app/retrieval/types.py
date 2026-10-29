"""검색 결과 공통 타입 (retrieval result types)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    unit: str
    section: str
    text: str
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    rank: int = 0
    passed_threshold: bool = True
