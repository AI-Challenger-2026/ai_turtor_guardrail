"""BM25 (Okapi BM25) 키워드 검색 — 외부 라이브러리 없이 구현.

정규화 점수 (normalized score, 0~1):
  score / Σ idf(t)   (t = 질의의 고유 토큰)
  - 문서 길이가 평균이고 각 질의어가 한 번씩 나오면 BM25 기여도가 idf 와 같다는 점을 이용한 근사 상한.
  - 교육 자료에 없는 질의어는 가장 큰 idf 로 계산되어 점수를 낮춘다 → 범위 밖 질문은 "근거 없음"이 되기 쉽다.
  - 1 을 넘으면 1 로 자른다. 임계값(min_score)은 configs/settings.yaml 에서 조정한다.
"""

from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(toks) for toks in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.tf: list[Counter[str]] = [Counter(toks) for toks in docs_tokens]
        df: Counter[str] = Counter()
        for counter in self.tf:
            df.update(counter.keys())
        self.df = df

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score_doc(self, q_terms: list[str], i: int) -> float:
        tf = self.tf[i]
        dl = self.doc_len[i]
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1
        total = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f:
                total += self.idf(t) * f * (self.k1 + 1) / (f + norm)
        return total

    def scores(self, query_tokens: list[str]) -> list[float]:
        q_terms = list(dict.fromkeys(query_tokens))
        return [self.score_doc(q_terms, i) for i in range(self.n_docs)]

    def normalizer(self, query_tokens: list[str]) -> float:
        return sum(self.idf(t) for t in dict.fromkeys(query_tokens))

    def normalized_scores(self, query_tokens: list[str]) -> tuple[list[float], list[float]]:
        """(raw_scores, normalized_scores)"""
        raw = self.scores(query_tokens)
        denom = self.normalizer(query_tokens)
        if denom <= 0:
            return raw, [0.0] * len(raw)
        return raw, [min(1.0, s / denom) for s in raw]
