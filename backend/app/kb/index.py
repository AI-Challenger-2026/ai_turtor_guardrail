"""교육 자료 인덱스 파일 (KB index file) — 청크와 메타데이터를 JSON 파일 하나로 저장·로드한다.

BM25 통계는 로드할 때 메모리에서 계산하므로 파일에는 청크만 저장한다 (벡터 DB 없음).
fingerprint(원본 파일 경로·크기·수정 시각)로 원본이 바뀌었는지(stale) 판단한다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.kb.chunker import chunk_document
from app.kb.loader import load_directory, supported_extensions
from app.kb.models import Chunk

INDEX_VERSION = 1


def source_fingerprint(source_dir: Path) -> str:
    h = hashlib.sha1()
    exts = set(supported_extensions())
    for p in sorted(x for x in Path(source_dir).rglob("*") if x.is_file() and x.suffix.lower() in exts):
        st = p.stat()
        h.update(f"{p.relative_to(source_dir).as_posix()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8"))
    return h.hexdigest()


class KBIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        documents: list[dict[str, Any]],
        *,
        built_at: str,
        source_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        fingerprint: str,
    ) -> None:
        self.chunks = chunks
        self.documents = documents
        self.built_at = built_at
        self.source_dir = source_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fingerprint = fingerprint

    # ---- 생성 (build) ----
    @classmethod
    def build(cls, source_dir: Path, chunk_size: int, chunk_overlap: int, base_dir: Path | None = None) -> "KBIndex":
        source_dir = Path(source_dir)
        docs = load_directory(source_dir, base_dir=base_dir)
        chunks: list[Chunk] = []
        documents: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for doc in docs:
            if doc.doc_id in seen_ids:
                raise ValueError(f"doc_id 가 중복됩니다 (duplicate doc_id): {doc.doc_id}")
            seen_ids.add(doc.doc_id)
            doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
            chunks.extend(doc_chunks)
            documents.append(
                {"doc_id": doc.doc_id, "doc_title": doc.doc_title, "source_path": doc.source_path,
                 "units": doc.units, "num_chunks": len(doc_chunks)}
            )
        src = source_dir.as_posix()
        if base_dir is not None:
            try:
                src = source_dir.resolve().relative_to(Path(base_dir).resolve()).as_posix()
            except ValueError:
                pass
        return cls(
            chunks,
            documents,
            built_at=datetime.now().isoformat(timespec="seconds"),
            source_dir=src,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            fingerprint=source_fingerprint(source_dir),
        )

    # ---- 저장/로드 (persist) ----
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "built_at": self.built_at,
            "source_dir": self.source_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "fingerprint": self.fingerprint,
            "documents": self.documents,
            "chunks": [c.model_dump() for c in self.chunks],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "KBIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != INDEX_VERSION:
            raise ValueError(f"인덱스 버전이 다릅니다. python -m app.kb.build 로 다시 만드세요: {path}")
        return cls(
            [Chunk(**c) for c in data["chunks"]],
            data.get("documents", []),
            built_at=data.get("built_at", ""),
            source_dir=data.get("source_dir", ""),
            chunk_size=int(data.get("chunk_size", 0)),
            chunk_overlap=int(data.get("chunk_overlap", 0)),
            fingerprint=data.get("fingerprint", ""),
        )

    # ---- 조회 (queries) ----
    def units(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            if c.unit not in seen:
                seen.append(c.unit)
        return seen

    def is_stale(self, source_dir: Path) -> bool:
        try:
            return source_fingerprint(Path(source_dir)) != self.fingerprint
        except FileNotFoundError:
            return False
