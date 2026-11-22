"""Input rules applied in code before classification and search.

- A last learner turn with no letters or digits is low_signal_input.
- Letter-spaced Latin words ("c o o l") are joined.
- Numbers written in words ("3점7퍼센트") are written with digits for search.
- Phrases that claim authority or press for an answer are removed from the classifier's input.
"""
import os
import re
from typing import Dict, List

INPUT_RULES = os.environ.get("INPUT_RULES", "1") == "1"
BAIT_RULES = os.environ.get("BAIT_RULES", "1") == "1"

_WORD = re.compile(r"[가-힣A-Za-z0-9]")
_SPACED = re.compile(r"(?<![가-힣A-Za-z0-9])[A-Za-z](?: [A-Za-z]){2,}(?![가-힣A-Za-z0-9])")
_DECIMAL = re.compile(r"(\d+)점(\d+)")
_PERCENT_POINT = re.compile(r"퍼센트\s*포인트|%\s*포인트")
_PERCENT = re.compile(r"퍼센트|(?<=\d)\s*프로(?![가-힣])")
_BAIT = re.compile("|".join([
    r"거절하지\s*말고|거절\s*말고",
    r"(?:나|저)(?:는)?\s*(?:이\s*)?(?:튜터\s*)?(?:관리자|개발자|운영자|선생님)"
    r"(?:이에요|예요|이야|야|인데요|인데|입니다|임|라서)",
    r"허락\s*받았(?:어요|어|는데|으니)",
    r"그냥\s*궁금해서(?:요|\s*(?:그러는데|묻는\s*거야|물어보는\s*거야|물어봐))?",
    r"(?:소설|연극|역할극)에?\s*쓸\s*(?:건데|거야|거예요)",
]))


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


def strip_bait(text: str) -> str:
    text = text or ""
    if not _BAIT.search(text):
        return text
    out = _BAIT.sub(" ", text)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out).strip(" ,.\n")
    return out if has_words(out) else text


def _learner_turn(text: str) -> str:
    text = join_spaced(text)
    return strip_bait(text) if BAIT_RULES else text


def for_classifier(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not INPUT_RULES:
        return messages
    return [{**m, "content": _learner_turn(m.get("content", ""))} if m.get("role") == "user" else m for m in messages]


def for_search(query: str) -> str:
    if not INPUT_RULES:
        return query
    return write_numbers(join_spaced(query))


def is_low_signal(messages: List[Dict[str, str]]) -> bool:
    return INPUT_RULES and not has_words(last_user_turn(messages))
