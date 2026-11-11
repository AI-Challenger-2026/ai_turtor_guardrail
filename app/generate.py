"""Tutor generation over retrieved knowledge-base sections.

One JSON-mode call returns {"found", "citations", "answer"}:
- found=true: the answer (mode "answer") or a hint-and-steps scaffold (mode "scaffold").
- found=false: the asked thing is not in the materials; the answer says so and adds the
  closest related material, citing it. The caller turns this into action "abstain".
Citations are restricted to the sections that were shown to the model.
For pii_disclosure the model is told not to repeat the learner's details, and a fixed notice
is appended.
A found answer also shows verbatim sentences from the top section (see excerpt), and those
sentences stand in for the answer when generation fails. Scaffolds never get them: they could
give away what the learner is meant to work out.
"""
import re
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
EXCERPT_SENTENCES = 2
EXCERPT_MAX_CHARS = 320
EXCERPT_MIN_CHARS = 12
# A sentence must share this share of its 2-grams with the question or answer to be shown.
EXCERPT_MIN_RELATED = 0.2
# A sentence with this share of its 3-grams already in the answer is not repeated.
EXCERPT_REPEAT_SHARE = 0.8


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


def _grams(text: str, n: int) -> set:
    s = "".join(text.split())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def excerpt(section_text: str, question: str, answer: str = "", *, min_related: float = EXCERPT_MIN_RELATED) -> str:
    """Sentences of a section related to the question and answer, verbatim, in section order.

    Text comes only from the knowledge base; the learner's words only rank the sentences.
    Sentences the answer already says, and sentences sharing too little with either, are left out.
    """
    sentences = [s.strip().replace("**", "") for line in section_text.splitlines()
                 if not line.lstrip().startswith(("#", "```"))
                 for s in re.split(r"(?<=[.!?요다])\s+", line.strip(" -*\t"))
                 if len("".join(s.split())) >= EXCERPT_MIN_CHARS]
    said = _grams(answer, 3)
    asked, answered = _grams(question, 2), _grams(answer, 2)
    scored = []
    for i, sentence in enumerate(sentences):
        three, two = _grams(sentence, 3), _grams(sentence, 2)
        if len(three & said) >= EXCERPT_REPEAT_SHARE * max(1, len(three)):
            continue
        if len(two & (asked | answered)) >= min_related * max(1, len(two)):
            # The question counts double: the answer's wording alone would favour what it already covers.
            scored.append((-(2 * len(two & asked) + len(two & answered)), i))
    return " ".join(sentences[i] for i in sorted(i for _, i in sorted(scored)[:EXCERPT_SENTENCES]))[:EXCERPT_MAX_CHARS]


def _last_turn(messages: Sequence[Message]) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "")


def _with_notice(answer: str, pii: bool) -> str:
    if pii:
        room = config.ANSWER_MAX_CHARS - len(_NOTICE_SEP) - len(templates.PII_NOTICE)
        answer = answer[:room].rstrip() + _NOTICE_SEP + templates.PII_NOTICE
    return answer[:config.ANSWER_MAX_CHARS]


def from_material(messages: Sequence[Message], verdict, hits: Sequence[SectionHit]) -> Optional[Generated]:
    """A reply made of the top section's sentences, for when generation failed; None without material."""
    if not hits:
        return None
    text = excerpt(hits[0].text, _last_turn(messages), min_related=0.0)
    if not text:
        return None
    pii = verdict is not None and verdict.category == "pii_disclosure"
    return Generated(answer=_with_notice(f"{templates.FALLBACK_FROM_MATERIAL} {text}", pii), citations=[hits[0].section_id])


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
    if out.found and mode == "answer" and hits:
        shown = excerpt(hits[0].text, _last_turn(messages), answer)
        if shown:
            answer = f"{answer}{_NOTICE_SEP}{templates.MATERIAL_EXCERPT_HEAD} {shown}"
    return Generated(answer=_with_notice(answer, pii), citations=pick_citations(out.citations, hits), found=out.found)
