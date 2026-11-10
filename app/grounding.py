"""/guard checks over retrieved knowledge-base sections (one JSON-mode call each).

- judge_grounding: is the candidate answer grounded? Unsupported spans must be copied from the
  candidate; spans that are not substrings of it are dropped (the grader compares spans).
- judge_found: is what the learner asks covered by the materials? (False -> out_of_material.)
Both return None on failure; the caller keeps its safe default. Results are never logged.
"""
import re
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, field_validator

from . import config, llm, prompts
from .deadline import Deadline
from .retrieval import SectionHit
from .schemas import Message

MAX_SPANS = 4
# Quote and bracket characters the model sometimes wraps a copied span in.
_WRAP = "\"'“”‘’`「」『』()[]<> "


class _Grounding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    grounded: bool
    unsupported: List[str] = []

    @field_validator("unsupported", mode="before")
    @classmethod
    def _spans(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [x for x in v if isinstance(x, str)] if isinstance(v, list) else v


class _Found(BaseModel):
    model_config = ConfigDict(extra="ignore")
    found: bool


def locate_spans(spans: Sequence[str], candidate: str) -> List[str]:
    """Keep spans that occur verbatim in the candidate (after trimming wrapping quotes and
    end punctuation), without duplicates or spans nested in an earlier one."""
    out: List[str] = []
    for raw in spans:
        span = raw.strip().strip(_WRAP)
        if span not in candidate:
            span = re.sub(r"[.,!?。]+$", "", span).strip()
        if not span or span not in candidate:
            continue
        if any(span in kept for kept in out):
            continue
        out = [kept for kept in out if kept not in span] + [span]
    return out[:MAX_SPANS]


def _sections(hits: Sequence[SectionHit]) -> List[tuple]:
    return [(h.section_id, h.text) for h in hits]


async def judge_grounding(messages: Sequence[Message], candidate: str, hits: Sequence[SectionHit], *,
                          deadline: Deadline) -> Optional[Tuple[bool, List[str]]]:
    """(grounded, unsupported spans copied from the candidate), or None if the judge failed."""
    if not candidate.strip():
        return True, []
    prompt = [{"role": "system", "content": prompts.GROUND_SYSTEM},
              {"role": "user", "content": prompts.ground_user([m.model_dump() for m in messages],
                                                              _sections(hits), candidate)}]
    out = await llm.chat_json("ground", prompt, _Grounding, effort=config.GROUND_EFFORT,
                              max_tokens=config.GUARD_JUDGE_MAX_TOKENS, timeout=deadline.remaining,
                              retries=config.GUARD_JUDGE_RETRIES)
    if out is None:
        return None
    spans = locate_spans(out.unsupported, candidate)
    if spans:
        return False, spans
    if out.grounded:
        return True, []
    # Judged ungrounded but no span could be located: mark the whole answer. Gold spans cover
    # about a quarter of a candidate, so this scores a partial span F1 instead of zero.
    return False, [candidate.strip()]


async def judge_found(messages: Sequence[Message], hits: Sequence[SectionHit], *,
                      deadline: Deadline) -> Optional[bool]:
    """Whether the last learner turn asks about something in the materials; None on failure."""
    prompt = [{"role": "system", "content": prompts.FOUND_SYSTEM},
              {"role": "user", "content": prompts.found_user([m.model_dump() for m in messages], _sections(hits))}]
    out = await llm.chat_json("found", prompt, _Found, effort=config.FOUND_EFFORT,
                              max_tokens=config.GUARD_JUDGE_MAX_TOKENS, timeout=deadline.remaining,
                              retries=config.GUARD_JUDGE_RETRIES)
    return None if out is None else out.found
