"""Candidate-answer grounding judgment for /guard (placeholder until task 3)."""
from typing import List, Sequence, Tuple

from .retrieval import SectionHit
from .schemas import Message


async def judge_grounding(messages: Sequence[Message], candidate: str,
                          hits: Sequence[SectionHit]) -> Tuple[bool, List[str]]:
    """Return (grounded, unsupported spans copied verbatim from the candidate)."""
    return True, []
