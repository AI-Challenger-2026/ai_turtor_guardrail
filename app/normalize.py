"""Input rules written by the team as code (no word lists or dictionaries).

- A last learner turn with no letter or digit at all (only jamo, symbols, emoji, or nothing) is
  low_signal_input. In the 400 public items, all 11 such turns are labeled low_signal_input and
  every low_signal_input item is such a turn; the classifier sometimes called them
  emotional_distress.
- Letter-spaced Latin words ("c o o l i n g") are joined before classification and search: in
  rewordings of dev items letter-spaced turns were misclassified 13% of the time against about 5%
  (mostly normal questions flagged as attacks), and BM25 cannot match spaced letters. Hangul is
  left alone: runs of one-syllable words are ordinary Korean ("좀 해 줘", "더 잘 알 수"), and
  without a vocabulary they cannot be told from spaced-out words ("규 칙").
- Numbers written with Korean words for the decimal point and percent ("3점7퍼센트") are written as
  the course material writes them ("3.7%") before search.
INPUT_RULES=0 turns all three off (final-v3 behaviour).
"""
import os
import re
from typing import Dict, List

INPUT_RULES = os.environ.get("INPUT_RULES", "1") == "1"

_WORD = re.compile(r"[가-힣A-Za-z0-9]")
# three or more single Latin letters separated by single spaces
_SPACED = re.compile(r"(?<![가-힣A-Za-z0-9])[A-Za-z](?: [A-Za-z]){2,}(?![가-힣A-Za-z0-9])")
_DECIMAL = re.compile(r"(\d+)점(\d+)")
_PERCENT_POINT = re.compile(r"퍼센트\s*포인트|%\s*포인트")
_PERCENT = re.compile(r"퍼센트|(?<=\d)\s*프로(?![가-힣])")


def last_user_turn(messages: List[Dict[str, str]]) -> str:
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    return users[-1] if users else ""


def has_words(text: str) -> bool:
    return bool(_WORD.search(text or ""))


def join_spaced(text: str) -> str:
    return _SPACED.sub(lambda m: m.group(0).replace(" ", ""), text or "")


def write_numbers(text: str) -> str:
    text = _DECIMAL.sub(r"\1.\2", text or "")
    text = _PERCENT_POINT.sub("%p", text)
    return _PERCENT.sub("%", text)


def for_classifier(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Learner turns with letter-spaced words joined; tutor turns untouched."""
    if not INPUT_RULES:
        return messages
    return [{**m, "content": join_spaced(m.get("content", ""))} if m.get("role") == "user" else m for m in messages]


def for_search(query: str) -> str:
    if not INPUT_RULES:
        return query
    return write_numbers(join_spaced(query))


def is_low_signal(messages: List[Dict[str, str]]) -> bool:
    return INPUT_RULES and not has_words(last_user_turn(messages))
