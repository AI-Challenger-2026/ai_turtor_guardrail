"""Input classifier shared by /tutor and /guard.

1. Rule: a last user turn with no meaningful characters is low_signal_input (no LLM call).
2. LLM: category principles + contrast-pair few-shot, JSON mode, validated, one retry.
   Majority vote: two calls at once, a third only if they disagree (config.CLASSIFY_VOTES).
   All calls failing -> normal. out_of_material is not decided here (task 2) -> normal.
3. Rule: a normal verdict on a last turn where the learner introduces themselves -> pii_disclosure.
The verdict has no free-text field, and it is never logged.
"""
import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Literal, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from . import config, llm, policy, prompts
from .deadline import Deadline
from .fewshot import Shot, load_pool, load_shots
from .retrieval import BM25
from .schemas import Category, Message

HarmKind = Literal["request", "statement", "none"]
Source = Literal["rule", "llm", "fallback"]

_MEANINGFUL = re.compile(r"[가-힣A-Za-z0-9]")
LOW_SIGNAL_MAX_CHARS = 12

# A learner introducing themselves: school + grade + name, or "my name is". A grammatical form
# only, with no lists of names or schools.
_NAME_END = r"(?:이야|야|이에요|예요|입니다|이고|인데)"
_SCHOOL_GRADE_NAME = r"(?:초등학교|중학교|고등학교|[가-힣]{1,8}(?:초|중|고))\s*\d\s*학년(?:\s*\d+\s*반)?\s*[가-힣]{1,5}" + _NAME_END
_SELF_INTRO = re.compile(r"(?:^|\s)(?:나는|난|나|저는|전|저)\s+[^\n]{0,20}?" + _SCHOOL_GRADE_NAME
                         + r"|내\s*이름은\s*[가-힣]{1,5}" + _NAME_END)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: Category
    harm_kind: HarmKind = "none"

    @field_validator("harm_kind", mode="before")
    @classmethod
    def _tolerant_harm_kind(cls, value):
        # Only the category is essential; an off-spec harm_kind must not cost a retry.
        return value if value in ("request", "statement", "none") else "none"


@dataclass(frozen=True)
class Classification:
    verdict: Verdict
    source: Source


def is_low_signal(text: str) -> bool:
    """Raw text (no NFKC) without whitespace: empty, or 1-12 chars with no Hangul syllable, ASCII letter or digit."""
    s = "".join(text.split())
    return len(s) == 0 or (len(s) <= LOW_SIGNAL_MAX_CHARS and not _MEANINGFUL.search(s))


def states_own_identity(text: str) -> bool:
    """Whether the learner gives their own name (with school and grade, or as "my name is")."""
    return _SELF_INTRO.search(text) is not None


def normalize(verdict: Verdict) -> Verdict:
    category = "normal" if verdict.category == "out_of_material" else verdict.category
    harm_kind = verdict.harm_kind if category == "harmful_content" else "none"
    return Verdict(category=category, harm_kind=harm_kind)


def _verdict_json(category: str, harm_kind: str) -> str:
    return json.dumps({"category": category, "harm_kind": harm_kind}, ensure_ascii=False)


def build_shot_messages(shots: Sequence[Shot]) -> List[dict]:
    out = []
    for shot in shots:
        out.append({"role": "user", "content": prompts.render(shot.messages, **prompts.SHOT_LIMITS)})
        out.append({"role": "assistant", "content": _verdict_json(shot.category, shot.harm_kind)})
    return out


def similarity_text(messages: Sequence) -> str:
    """Learner turns with the last one counted twice (the last turn is what gets classified)."""
    turns = [m["content"] if isinstance(m, dict) else m.content for m in messages
             if (m["role"] if isinstance(m, dict) else m.role) == "user"]
    return "\n".join(turns + turns[-1:])


# Built once at start-up from public/ (fewshot split only); never from requests.
SHOTS = load_shots()
_PREFIX = [{"role": "system", "content": prompts.CLASSIFIER_SYSTEM}, *build_shot_messages(SHOTS)]
_STATIC_IDS = {s.id for s in SHOTS}
POOL = [s for s in load_pool() if s.id not in _STATIC_IDS]
_POOL_BY_ID = {s.id: s for s in POOL}
_POOL_INDEX = BM25({s.id: similarity_text(s.messages) for s in POOL})


def similar_shots(messages: Sequence[Message], k: int, exclude: frozenset = frozenset()) -> List[Shot]:
    """The k pool examples most similar to the conversation, least similar first.

    `exclude` is for evaluation only (leave-group-out on fewshot-split items); routes never pass it.
    """
    if k <= 0:
        return []
    scores = _POOL_INDEX.scores(similarity_text(messages))
    ranked = sorted((sid for sid, s in scores.items() if s > 0 and sid not in exclude), key=lambda sid: (-scores[sid], sid))
    return [_POOL_BY_ID[sid] for sid in reversed(ranked[:k])]


def build_prompt(messages: Sequence[Message], exclude: frozenset = frozenset()) -> List[dict]:
    convo = [m.model_dump() for m in messages]
    dynamic = build_shot_messages(similar_shots(messages, config.CLASSIFY_DYNAMIC_SHOTS, exclude))
    return [*_PREFIX, *dynamic, {"role": "user", "content": prompts.render(convo, **prompts.LIVE_LIMITS)}]


def _key(verdict: Verdict):
    return verdict.category, verdict.harm_kind


def vote(verdicts: Sequence[Verdict]) -> Verdict:
    """Majority over normalized verdicts, in call order; ties go to the earliest call.

    1. a (category, harm_kind) pair named by more than half of the calls;
    2. else a category named by more than half (harm_kind from its earliest call);
    3. else the block/allow decision named by more than half (its earliest verdict);
    4. else (two calls, one each way) the first call.
    """
    half = len(verdicts) / 2
    pair, n = Counter(_key(v) for v in verdicts).most_common(1)[0]
    if n > half:
        return next(v for v in verdicts if _key(v) == pair)
    category, n = Counter(v.category for v in verdicts).most_common(1)[0]
    if n > half:
        return next(v for v in verdicts if v.category == category)
    blocks = [v for v in verdicts if policy.decision(v.category) == "block"]
    allows = [v for v in verdicts if policy.decision(v.category) != "block"]
    if len(blocks) > half:
        return blocks[0]
    if len(allows) > half:
        return allows[0]
    return verdicts[0]


async def _ask(prompt: List[dict], deadline: Deadline):
    verdict = await llm.chat_json(
        "classify", prompt, Verdict, effort=config.CLASSIFY_EFFORT,
        max_tokens=config.CLASSIFY_MAX_TOKENS, retries=config.CLASSIFY_RETRIES,
        timeout=lambda: deadline.cap(config.CLASSIFY_TIMEOUT_S))
    return None if verdict is None else normalize(verdict)


async def classify(messages: Sequence[Message], deadline: Deadline, *, exclude: frozenset = frozenset()) -> Classification:
    last = next(m.content for m in reversed(messages) if m.role == "user")
    if is_low_signal(last):
        return Classification(Verdict(category="low_signal_input"), "rule")
    prompt = build_prompt(messages, exclude)
    first = 1 if config.CLASSIFY_VOTES == 1 else 2
    verdicts = [v for v in await asyncio.gather(*(_ask(prompt, deadline) for _ in range(first))) if v is not None]
    if config.CLASSIFY_VOTES >= 3 and len(verdicts) == 2 and _key(verdicts[0]) != _key(verdicts[1]):
        third = await _ask(prompt, deadline)
        if third is not None:
            verdicts.append(third)
    if not verdicts:
        return Classification(Verdict(category="normal"), "fallback")
    verdict = vote(verdicts)
    # The LLM often misses a self-introduction; only a normal verdict changes, so blocks stay.
    if verdict.category == "normal" and states_own_identity(last):
        verdict = Verdict(category="pii_disclosure")
    return Classification(verdict, "llm")
