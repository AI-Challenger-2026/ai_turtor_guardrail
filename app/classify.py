"""Input classifier shared by /tutor and /guard.

1. Rule: a last user turn with no meaningful characters is low_signal_input (no LLM call).
2. LLM: category principles + contrast-pair few-shot, JSON mode, validated, one retry.
   Any failure -> normal. out_of_material is not decided here (task 2) -> normal.
The verdict has no free-text field, and it is never logged.
"""
import json
import re
from dataclasses import dataclass
from typing import List, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from . import config, llm, prompts
from .deadline import Deadline
from .fewshot import Shot, load_shots
from .schemas import Category, Message

HarmKind = Literal["request", "statement", "none"]
Source = Literal["rule", "llm", "fallback"]

_MEANINGFUL = re.compile(r"[가-힣A-Za-z0-9]")
LOW_SIGNAL_MAX_CHARS = 12


class Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: Category
    harm_kind: HarmKind = "none"


@dataclass(frozen=True)
class Classification:
    verdict: Verdict
    source: Source


def is_low_signal(text: str) -> bool:
    """Raw text (no NFKC) without whitespace: empty, or 1-12 chars with no Hangul syllable, ASCII letter or digit."""
    s = "".join(text.split())
    return len(s) == 0 or (len(s) <= LOW_SIGNAL_MAX_CHARS and not _MEANINGFUL.search(s))


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


# Built once at start-up from static files (public set ids); never from requests.
SHOTS = load_shots()
_PREFIX = [{"role": "system", "content": prompts.CLASSIFIER_SYSTEM}, *build_shot_messages(SHOTS)]


def build_prompt(messages: Sequence[Message]) -> List[dict]:
    convo = [m.model_dump() for m in messages]
    return [*_PREFIX, {"role": "user", "content": prompts.render(convo, **prompts.LIVE_LIMITS)}]


async def classify(messages: Sequence[Message], deadline: Deadline) -> Classification:
    last = next(m.content for m in reversed(messages) if m.role == "user")
    if is_low_signal(last):
        return Classification(Verdict(category="low_signal_input"), "rule")
    verdict = await llm.chat_json(
        "classify", build_prompt(messages), Verdict, effort=config.CLASSIFY_EFFORT,
        max_tokens=config.CLASSIFY_MAX_TOKENS, retries=config.CLASSIFY_RETRIES,
        timeout=lambda: deadline.cap(config.CLASSIFY_TIMEOUT_S))
    if verdict is None:
        return Classification(Verdict(category="normal"), "fallback")
    return Classification(normalize(verdict), "llm")
