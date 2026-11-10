"""Tutor answer generation for the answer path (normal, pii_disclosure).

Task 1: the starter server's plain generation (same system prompt, reasoning effort and
token limit; no retrieval, no citations), plus PII handling: the model is told not to
repeat the learner's disclosed details, and a fixed notice is appended.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import config, llm, templates
from .deadline import Deadline
from .retrieval import SectionHit
from .schemas import Message

SYSTEM = "당신은 온라인 학습 플랫폼의 AI 튜터입니다. 한국어로 답합니다."
PII_INSTRUCTION = (
    "학습자가 자기 이름이나 학교, 학년 같은 개인정보를 밝혔습니다. "
    "답변에서 그 이름, 학교, 학년을 다시 쓰거나 부르지 말고 질문 내용에만 답합니다."
)
_NOTICE_SEP = "\n\n"


@dataclass
class Generated:
    answer: str
    citations: List[str] = field(default_factory=list)


async def generate_tutor(messages: Sequence[Message], verdict, hits: Sequence[SectionHit], *,
                         deadline: Deadline) -> Optional[Generated]:
    """Return the generated answer, or None if generation failed (caller falls back)."""
    pii = verdict is not None and verdict.category == "pii_disclosure"
    system = f"{SYSTEM}\n{PII_INSTRUCTION}" if pii else SYSTEM
    prompt = [{"role": "system", "content": system}, *(m.model_dump() for m in messages)]
    text = await llm.chat_text("generate", prompt, effort=config.GENERATE_EFFORT,
                               max_tokens=config.GENERATE_MAX_TOKENS, timeout=deadline.remaining())
    answer = (text or "").strip()
    if not answer:
        return None
    if pii:
        room = config.ANSWER_MAX_CHARS - len(_NOTICE_SEP) - len(templates.PII_NOTICE)
        return Generated(answer=answer[:room].rstrip() + _NOTICE_SEP + templates.PII_NOTICE)
    return Generated(answer=answer[:config.ANSWER_MAX_CHARS])
