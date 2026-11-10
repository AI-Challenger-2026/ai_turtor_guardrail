"""Tutor answer generation.

Step 2: the starter server's plain generation, moved here unchanged in behavior
(same system prompt, reasoning effort, token limit; no retrieval, no citations).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import config, llm
from .deadline import Deadline
from .retrieval import SectionHit
from .schemas import Message

SYSTEM = "당신은 온라인 학습 플랫폼의 AI 튜터입니다. 한국어로 답합니다."


@dataclass
class Generated:
    answer: str
    citations: List[str] = field(default_factory=list)


async def generate_tutor(messages: Sequence[Message], verdict, hits: Sequence[SectionHit], *,
                         deadline: Deadline) -> Optional[Generated]:
    """Return the generated answer, or None if generation failed (caller falls back)."""
    prompt = [{"role": "system", "content": SYSTEM}, *(m.model_dump() for m in messages)]
    text = await llm.chat_text("generate", prompt, effort=config.GENERATE_EFFORT,
                               max_tokens=config.GENERATE_MAX_TOKENS, timeout=deadline.remaining())
    answer = (text or "").strip()[:config.ANSWER_MAX_CHARS]
    return Generated(answer=answer) if answer else None
