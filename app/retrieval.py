"""Knowledge-base retrieval (placeholder until task 2).

Task 2 builds the index and embeddings at server start-up and fills in `retrieve`.
"""
from dataclasses import dataclass
from typing import List, Sequence

from .schemas import Message


@dataclass(frozen=True)
class SectionHit:
    section_id: str  # e.g. "FIN-02-01"
    score: float
    text: str
    kind: str  # e.g. "section" or "table"


async def retrieve(messages: Sequence[Message], k: int = 5) -> List[SectionHit]:
    return []
