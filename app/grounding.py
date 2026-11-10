"""/guard checks over retrieved knowledge-base sections (one JSON-mode call each).

- judge_grounding: is the candidate answer grounded? Unsupported spans must be copied from the
  candidate; spans that are not substrings of it are dropped (the grader compares spans).
- judge_found: is what the learner asks covered by the materials? (False -> out_of_material.)
Both return None on failure; the caller keeps its safe default. Results are never logged.
"""
import asyncio
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


def _common(a: str, b: str) -> str:
    """Longest common substring of two short strings."""
    best_len, best_end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return a[best_end - best_len:best_end]


def agreed_spans(first: Sequence[str], second: Sequence[str], candidate: str) -> List[str]:
    """For each span of the first judge, the text it shares with the second judge's spans.

    Every span is a substring of the candidate, so the shared text is one too. Overlaps shorter
    than AGREED_SPAN_MIN_CHARS (after trimming) are dropped; no agreement at all -> first's spans."""
    out = []
    for span in first:
        shared = max((_common(span, other) for other in second), key=len, default="")
        shared = shared.strip().strip(_WRAP)
        if len(re.sub(r"\s", "", shared)) >= config.AGREED_SPAN_MIN_CHARS:
            out.append(shared)
    return locate_spans(out, candidate) or list(first)


async def judge_grounding(messages: Sequence[Message], candidate: str, hits: Sequence[SectionHit], *,
                          deadline: Deadline) -> Optional[Tuple[bool, List[str]]]:
    """(grounded, unsupported spans copied from the candidate), or None if every judge failed.

    config.GROUND_JUDGES independent calls at once. Once more than half of all judges agree, the
    rest are cancelled (with three judges, two that agree decide, so a slow third does not hold
    the request). Otherwise the judges that answered decide: the majority wins and a tie goes to
    the first. Spans are the text the first two flagging judges agree on (or the one flagging
    judge's spans). "First" is launch order, never finishing order."""
    if not candidate.strip():
        return True, []
    judges = [asyncio.ensure_future(_judge_once(messages, candidate, hits, deadline=deadline))
              for _ in range(config.GROUND_JUDGES)]
    outs: List[Tuple[bool, List[str]]] = []
    try:
        pending = set(judges)
        while pending:
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            outs = [j.result() for j in judges if j.done() and j.result() is not None]
            flags = sum(not grounded for grounded, _ in outs)
            if 2 * max(flags, len(outs) - flags) > len(judges):
                break
    finally:
        for j in judges:
            j.cancel()
    return _majority(outs, candidate)


def _majority(outs: Sequence[Tuple[bool, List[str]]], candidate: str) -> Optional[Tuple[bool, List[str]]]:
    """The verdict of the judges that answered, in launch order (see judge_grounding)."""
    if not outs:
        return None
    flagged = [spans for grounded, spans in outs if not grounded]
    if 2 * len(flagged) < len(outs) or (2 * len(flagged) == len(outs) and outs[0][0]):
        return True, []
    if len(flagged) == 1:
        return False, flagged[0]
    return False, agreed_spans(flagged[0], flagged[1], candidate)


async def _judge_once(messages: Sequence[Message], candidate: str, hits: Sequence[SectionHit], *,
                      deadline: Deadline) -> Optional[Tuple[bool, List[str]]]:
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
