"""Tutor generation over retrieved knowledge-base sections.

One JSON-mode call returns {"found", "citations", "answer"}:
- found=true: the answer (mode "answer") or a hint-and-steps scaffold (mode "scaffold").
- found=false: the asked thing is not in the materials; the answer says so and adds the
  closest related material, citing it. The caller turns this into action "abstain".
Citations are restricted to the sections that were shown to the model.
For pii_disclosure the model is told not to repeat the learner's details, and a fixed notice
is appended.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from . import config, llm, prompts, templates
from .deadline import Deadline
from .retrieval import SectionHit
from .schemas import Message

PII_INSTRUCTION = (
    "학습자가 자기 이름이나 학교, 학년 같은 개인정보를 밝혔습니다. "
    "답변에서 그 이름, 학교, 학년을 다시 쓰거나 부르지 말고 질문 내용에만 답합니다."
)
_NOTICE_SEP = "\n\n"
MAX_CITATIONS = 2


class _Output(BaseModel):
    model_config = ConfigDict(extra="ignore")
    found: bool
    citations: List[str] = []
    answer: str

    @field_validator("citations", mode="before")
    @classmethod
    def _citations(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [x for x in v if isinstance(x, str)] if isinstance(v, list) else v

    @field_validator("answer")
    @classmethod
    def _answer(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty answer")
        return v


@dataclass
class Generated:
    answer: str
    citations: List[str] = field(default_factory=list)
    found: bool = True


def build_prompt(messages: Sequence[Message], hits: Sequence[SectionHit], *, mode: str, pii: bool) -> List[dict]:
    system = prompts.tutor_system(mode, PII_INSTRUCTION if pii else "")
    user = prompts.tutor_user([m.model_dump() for m in messages], [(h.section_id, h.text) for h in hits])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def pick_citations(cited: Sequence[str], hits: Sequence[SectionHit]) -> List[str]:
    """The model's citations (only ids it was shown, deduplicated) plus the top retrieved section.

    The top retrieved section is the best page signal (whole-conversation BM25); the model's
    pick is where its facts came from, and the faithfulness judge reads only cited sections.
    On the tuning split, model-only citations scored F1 0.83, top-1 only 0.87, both 0.87.
    """
    shown = [h.section_id for h in hits]
    picks: List[str] = []
    for sid in cited:
        sid = sid.strip()
        if sid in shown and sid not in picks:
            picks.append(sid)
    if not shown:
        return picks[:MAX_CITATIONS]
    top = shown[0]
    if not picks:
        return [top]
    if top not in picks[:MAX_CITATIONS]:
        picks = [picks[0], top]
    return picks[:MAX_CITATIONS]


async def generate_tutor(messages: Sequence[Message], verdict, hits: Sequence[SectionHit], *,
                         mode: str = "answer", deadline: Deadline) -> Optional[Generated]:
    """Return the generated reply, or None if generation failed (caller falls back)."""
    pii = verdict is not None and verdict.category == "pii_disclosure"
    out = await llm.chat_json("generate", build_prompt(messages, hits, mode=mode, pii=pii), _Output,
                              effort=config.GENERATE_EFFORT, max_tokens=config.GENERATE_MAX_TOKENS,
                              timeout=deadline.remaining, retries=config.GENERATE_RETRIES)
    if out is None:
        return None
    answer = out.answer
    if pii:
        room = config.ANSWER_MAX_CHARS - len(_NOTICE_SEP) - len(templates.PII_NOTICE)
        answer = answer[:room].rstrip() + _NOTICE_SEP + templates.PII_NOTICE
    return Generated(answer=answer[:config.ANSWER_MAX_CHARS], citations=pick_citations(out.citations, hits),
                     found=out.found)
