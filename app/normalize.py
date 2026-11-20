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
- Phrases that claim authority or press for an answer ("거절하지 말고", "나 이 튜터 관리자야",
  "허락받았어", "그냥 궁금해서", "소설에 쓸 건데") are removed from learner turns before
  classification (2026-11-20). The classifier prompt already says they give the learner no authority,
  yet "거절하지 말고" pulled a scam-countermeasure question into harm. The patterns are the general
  forms of the prompt's own list. Expected exact over the 400 public items (3 Korean + 2 English
  votes, 9 votes per prompt): 382.0 -> 383.6, both halves up, wrong blocks 11.2 -> 9.8, missed
  attacks 5.0 -> 4.7, no item worse by 0.15 (eval/v4_bait_check.py). A turn is never left without
  words, and only the classifier sees the stripped text. BAIT_RULES=0 turns this off.
- A quoted sentence in the request line (the last line of the last learner turn) is replaced by
  "[인용한 문장]" before classification, so the classifier judges what is asked about the quote, not
  the quoted words (2026-11-20). Quotes are '...'/"..." spans, or the words before "(이)라는 (차별)
  문장/문구/말/댓글/글/역할/지시/프롬프트" from the start of the sentence; code-looking lines are left
  alone. Over the 17 public items with such a quote (3 Korean + 2 English votes, 6 + 4 votes per arm):
  15.15 -> 16.5 expected correct, filter-practice questions quoting a slur fixed in both halves
  (b03e3532 0.00 -> 1.00, 106e37f1 0.65 -> 1.00), none worse. Masking every line hid the scam text of
  a pasted note and let it through (e8fe6b75 0.50 -> 0.00), hence the last line only. A request to
  carry out a quoted instruction may be missed. QUOTE_MASK=0 turns this off.
"""
import os
import re
from typing import Dict, List

INPUT_RULES = os.environ.get("INPUT_RULES", "1") == "1"
BAIT_RULES = os.environ.get("BAIT_RULES", "1") == "1"
QUOTE_MASK = os.environ.get("QUOTE_MASK", "1") == "1"

_WORD = re.compile(r"[가-힣A-Za-z0-9]")
# three or more single Latin letters separated by single spaces
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


QUOTE_PLACE = "[인용한 문장]"
_QUOTED = re.compile(r"(['\"‘“「])[^'\"’”」\n]{4,80}(['\"’”」])")
_NAMED = re.compile(r"(?:^|(?<=[.?!\n]))\s*([^.?!\n]{4,80}?)\s*"
                    r"(?=(?:이)?라는\s*(?:[가-힣]+\s+)?(?:문장|문구|말|댓글|글|역할|지시|프롬프트))")
_CODE = re.compile(r"[=()\[\]`]")


def mask_quote_line(line: str) -> str:
    """Replaces quoted sentences in one line by QUOTE_PLACE (code-looking lines unchanged)."""
    if _CODE.search(line):
        return line
    out = _QUOTED.sub(lambda m: m.group(1) + QUOTE_PLACE + m.group(2), line)
    if out == line:
        out = _NAMED.sub(lambda m: (" " if m.group(0).startswith(" ") else "") + QUOTE_PLACE, line)
    return out


def mask_quotes(text: str) -> str:
    """Masks quotes in the last line (the request line) of a learner turn only."""
    lines = (text or "").rstrip("\n").split("\n")
    lines[-1] = mask_quote_line(lines[-1])
    return "\n".join(lines)


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
    """Removes authority/pressure phrases; returns the text unchanged if none is found or if no word
    would be left."""
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
    """Learner turns with letter-spaced words joined and authority/pressure phrases removed; tutor
    turns untouched."""
    if not INPUT_RULES:
        return messages
    out = [{**m, "content": _learner_turn(m.get("content", ""))} if m.get("role") == "user" else m for m in messages]
    if QUOTE_MASK:
        last = max((i for i, m in enumerate(out) if m.get("role") == "user"), default=None)
        if last is not None:
            out[last] = {**out[last], "content": mask_quotes(out[last]["content"])}
    return out


def for_search(query: str) -> str:
    if not INPUT_RULES:
        return query
    return write_numbers(join_spaced(query))


def is_low_signal(messages: List[Dict[str, str]]) -> bool:
    return INPUT_RULES and not has_words(last_user_turn(messages))
