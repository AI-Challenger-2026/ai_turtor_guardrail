"""질의 정보를 저장하지 않는 KB 전용 임베딩 캐시."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

from .knowledge import KnowledgeBase, PREPROCESS_VERSION


CACHE_VERSION = 1
EMBEDDING_DIMENSIONS = 1536
MAX_CACHE_BYTES = 16 * 1024 * 1024


def normalize_vector(values: Sequence[float], dimensions: int = EMBEDDING_DIMENSIONS) -> tuple[float, ...]:
    if len(values) != dimensions or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("invalid_embedding_vector")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid_embedding_norm")
    return tuple(value / norm for value in values)


def _vector_digest(vectors) -> str:
    return hashlib.sha256(json.dumps(vectors, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path, model: str, dimensions: int = EMBEDDING_DIMENSIONS):
        self.path = Path(path)
        self.model = model
        self.dimensions = dimensions

    def _safe_path(self) -> bool:
        return not any(path.is_symlink() for path in (self.path, self.path.parent, self.path.parent.parent))

    def _metadata(self, kb: KnowledgeBase) -> dict:
        return {
            "version": CACHE_VERSION, "kb_fingerprint": kb.fingerprint, "model": self.model,
            "dimensions": self.dimensions, "preprocess_version": PREPROCESS_VERSION,
            "section_ids": [section.section_id for section in kb.sections],
        }

    def load(self, kb: KnowledgeBase) -> tuple[tuple[float, ...], ...] | None:
        if not self._safe_path():
            return None
        try:
            if self.path.stat().st_size > MAX_CACHE_BYTES:
                return None
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or any(data.get(key) != value for key, value in self._metadata(kb).items()):
                return None
            vectors = data.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(kb.sections):
                return None
            if _vector_digest(vectors) != data.get("vector_sha256"):
                return None
            return tuple(normalize_vector(vector, self.dimensions) for vector in vectors)
        except (OSError, ValueError, TypeError, OverflowError):
            return None

    def save(self, kb: KnowledgeBase, vectors: Sequence[Sequence[float]]) -> bool:
        if not self._safe_path() or len(vectors) != len(kb.sections):
            return False
        # 캐시에는 완성된 KB 벡터만 넣으며 부분 배치나 사용자 질의는 저장하지 않는다.
        normalized = [normalize_vector(vector, self.dimensions) for vector in vectors]
        data = {**self._metadata(kb), "vectors": normalized, "vector_sha256": _vector_digest(normalized)}
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_CACHE_BYTES:
            return False
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self._safe_path():
                return False
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
            os.replace(temporary, self.path)
            return True
        except OSError:
            return False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
