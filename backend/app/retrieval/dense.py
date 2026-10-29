"""선택 기능: 다국어 임베딩 기반 dense 검색 (optional dense retrieval).

sentence-transformers 가 설치되어 있고 retrieval.dense.enabled=true(또는 HYBRID_SEARCH=true)일 때만 사용한다.
encoder 를 주입할 수 있어(테스트용 가짜 임베더) 무거운 모델 없이도 RRF 결합 로직을 검증할 수 있다.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

from app.logging_utils import get_logger

log = get_logger("retrieval.dense")

Encoder = Callable[[Sequence[str]], list[list[float]]]


class DenseUnavailable(RuntimeError):
    """dense 검색을 켤 수 없는 상태 (라이브러리 미설치, 모델 로딩 실패 등)."""


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def load_sentence_transformer(model_name: str) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DenseUnavailable("sentence-transformers 미설치 (pip install -r requirements-optional.txt)") from exc
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise DenseUnavailable(f"임베딩 모델 로딩 실패: {model_name} ({exc})") from exc

    def encode(texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in model.encode(list(texts), normalize_embeddings=True)]

    return encode


class DenseIndex:
    def __init__(
        self,
        texts: list[str],
        model_name: str,
        encoder: Encoder | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder or load_sentence_transformer(model_name)
        self._is_e5 = "e5" in model_name.lower()
        passages = [f"passage: {t}" if self._is_e5 else t for t in texts]
        self.vectors = self._load_or_encode(passages, cache_path)

    def _load_or_encode(self, passages: list[str], cache_path: Path | None) -> list[list[float]]:
        key = hashlib.sha1((self.model_name + "\x00" + "\x00".join(passages)).encode("utf-8")).hexdigest()
        if cache_path and cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if data.get("key") == key:
                    return data["vectors"]
            except (OSError, ValueError, KeyError):
                log.warning("dense 캐시를 읽지 못해 다시 계산합니다: %s", cache_path)
        vectors = [_normalize(v) for v in self._encoder(passages)]
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"key": key, "model": self.model_name, "vectors": vectors}), encoding="utf-8")
        return vectors

    def scores(self, query: str) -> list[float]:
        q = f"query: {query}" if self._is_e5 else query
        qv = _normalize(self._encoder([q])[0])
        return [sum(a * b for a, b in zip(qv, dv)) for dv in self.vectors]
