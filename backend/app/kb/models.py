"""지식 베이스 데이터 모델 (knowledge-base data models).

loader 는 형식(.md/.txt/.json/.pdf)과 무관하게 Document(blocks=[Block(unit, section, text)]) 로 정규화하고,
chunker 는 Block 단위로 청크를 만든다. 공식 데이터 형식이 오면 loader 만 추가하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass
class Block:
    """구조 단위 텍스트 (heading 으로 나뉜 한 덩어리)."""

    unit: str
    section: str
    text: str


@dataclass
class Document:
    doc_id: str
    doc_title: str
    source_path: str
    blocks: list[Block] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def units(self) -> list[str]:
        seen: list[str] = []
        for b in self.blocks:
            if b.unit not in seen:
                seen.append(b.unit)
        return seen


class Chunk(BaseModel):
    """검색 단위 청크 (retrieval chunk) — 출처 표기에 필요한 메타데이터를 모두 가진다."""

    chunk_id: str
    doc_id: str
    doc_title: str
    unit: str
    section: str
    text: str
    order: int = 0
