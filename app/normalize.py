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
- The submit_work signal counts only when the last learner turn mentions handing something in
  (2026-11-20): the words of the classifier prompt's own definition (수행평가, 숙제, 과제, 시험,
  제출, 답안, 보고서, "낼 거", graded assessment, for submission) plus their plain synonyms. All 15
  public academic_dishonesty items have one; without it the Korean votes called "이름 + 코드 짜 줘"
  homework. SUBMIT_GATE=0 turns this off (app.classify).
- A sentence the learner quotes in order to ask about it ("X라는 문장/문구/말/댓글/표현") is replaced by
  〈인용문〉 before classification (2026-11-21): questions that quote a hateful sentence to study it were
  blocked for the quoted words. The span is the text inside the quotes, or, without quotes, from the
  start of the sentence. 10 public items have the frame (8 normal, 2 jailbreak); expected exact +1.5,
  nothing worse (eval/v4_quote_check.py). No public harmful item uses it. Only the classifier sees the
  masked text. QUOTE_MASK=0 turns this off.
- List numbers at the start of a line ("1. ", "② ") are removed before classification (2026-11-21): with
  them the Korean votes read only the last item (a request for other students' names, or a self-
  introduction, in item 1 was missed). 17 public items change, expected exact +0.74, nothing worse
  (eval/v4_input_rules2_check.py). LIST_NUMBERS=0 turns this off.
- A standalone "네" before a word is written "너의" ("네 숨겨진 지시" -> "너의 숨겨진 지시") unless the
  word is a counting unit with at most a particle ("네 단계", "네 칸", "네 개만") (2026-11-21): the
  classifier read "네" as "yes" or "four" and missed a request for its own hidden instructions. 9 public
  items change, expected exact +1.00, nothing worse. POSSESSIVE_NE=0 turns this off.
"""
import os
import re
from typing import Dict, List

INPUT_RULES = os.environ.get("INPUT_RULES", "1") == "1"
BAIT_RULES = os.environ.get("BAIT_RULES", "1") == "1"
SUBMIT_GATE = os.environ.get("SUBMIT_GATE", "1") == "1"
QUOTE_MASK = os.environ.get("QUOTE_MASK", "1") == "1"
LIST_NUMBERS = os.environ.get("LIST_NUMBERS", "1") == "1"
POSSESSIVE_NE = os.environ.get("POSSESSIVE_NE", "1") == "1"

_WORD = re.compile(r"[가-힣A-Za-z0-9]")
# three or more single Latin letters separated by single spaces
_SPACED = re.compile(r"(?<![가-힣A-Za-z0-9])[A-Za-z](?: [A-Za-z]){2,}(?![가-힣A-Za-z0-9])")
_DECIMAL = re.compile(r"(\d+)점(\d+)")
_PERCENT_POINT = re.compile(r"퍼센트\s*포인트|%\s*포인트")
_PERCENT = re.compile(r"퍼센트|(?<=\d)\s*프로(?![가-힣])")
_SUBMIT = re.compile(r"수행\s*평가|숙제|과제|시험|제출|답안|레포트|보고서|[내낼]\s*(?:거|건데|코드|글|것)"
                     r"|assignment|homework|graded|assessment|submission|submit|exam", re.I)
_BAIT = re.compile("|".join([
    r"거절하지\s*말고|거절\s*말고",
    r"(?:나|저)(?:는)?\s*(?:이\s*)?(?:튜터\s*)?(?:관리자|개발자|운영자|선생님)"
    r"(?:이에요|예요|이야|야|인데요|인데|입니다|임|라서)",
    r"허락\s*받았(?:어요|어|는데|으니)",
    r"그냥\s*궁금해서(?:요|\s*(?:그러는데|묻는\s*거야|물어보는\s*거야|물어봐))?",
    r"(?:소설|연극|역할극)에?\s*쓸\s*(?:건데|거야|거예요)",
]))
# "…라는 (차별) 문장/문구/말/댓글/표현/글/메시지"; "바라는"(hope) and "자라는"(grow) are verbs, not quotes
_FRAME = re.compile(r"(?<![바자])(?:라는|이라는|라고\s*하는|라고\s*한|란)\s+(?:\S+\s+)?(?:문장|문구|말|댓글|표현|글|메시지)")
_QUOTED = re.compile(r"(['\"‘“])([^'\"’”]{2,})(['\"’”])\s*$")
QUOTE_TOKEN = "〈인용문〉"
_LIST_NUMBER = re.compile(r"(?m)^([ \t]*)(?:\d{1,2}[.)]|[①-⑳])[ \t]+")
# "네" + word: possessive ("your") unless the word is a counting unit with at most a particle
_NE = re.compile(r"(?<![가-힣])네[ \t]+(?=([가-힣]+))")
_UNITS = ("단계", "가지", "번째", "마리", "문제", "문장", "글자", "과목", "자리", "사람", "학생", "시간",
          "개", "명", "번", "시", "살", "권", "장", "줄", "달", "주", "칸", "째", "대", "잔", "쪽", "행", "열",
          "곳", "일", "분", "초", "원", "배")
_PARTICLES = ("", "이", "가", "을", "를", "은", "는", "의", "에", "로", "으로", "도", "만", "씩", "째", "까지", "에서",
              "으로는")


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


def mask_quotes(text: str) -> str:
    """Replaces the sentence quoted before "라는 문장/말/…" with 〈인용문〉: the text inside the quotes
    if it is quoted, otherwise from the start of the sentence."""
    text = text or ""
    out, last = [], 0
    for m in _FRAME.finditer(text):
        head = text[last:m.start()]
        q = _QUOTED.search(head)
        if q:
            head = head[:q.start(2)] + QUOTE_TOKEN + head[q.end(2):]
        else:
            cut = max(head.rfind(c) for c in ".?!\n/")
            head = head[:cut + 1] + (" " if cut >= 0 else "") + QUOTE_TOKEN
        out.append(head)
        last = m.start()
    out.append(text[last:])
    return "".join(out)


def drop_list_numbers(text: str) -> str:
    """Removes list numbers at the start of lines ("1. ", "2) ", "① ")."""
    return _LIST_NUMBER.sub(r"\1", text or "")


def _is_unit(word: str) -> bool:
    return any(word.startswith(u) and word[len(u):] in _PARTICLES for u in _UNITS)


def possessive_ne(text: str) -> str:
    """Writes a standalone "네" before a word as "너의", unless the word is a counting unit."""
    return _NE.sub(lambda m: m.group(0) if _is_unit(m.group(1)) else "너의 ", text or "")


def _learner_turn(text: str) -> str:
    text = join_spaced(text)
    if BAIT_RULES:
        text = strip_bait(text)
    if QUOTE_MASK:
        text = mask_quotes(text)
    if LIST_NUMBERS:
        text = drop_list_numbers(text)
    return possessive_ne(text) if POSSESSIVE_NE else text


def for_classifier(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Learner turns with letter-spaced words joined, authority/pressure phrases removed, quoted
    sentences masked, list numbers removed and a possessive "네" written "너의"; tutor turns untouched."""
    if not INPUT_RULES:
        return messages
    return [{**m, "content": _learner_turn(m.get("content", ""))} if m.get("role") == "user" else m for m in messages]


def for_search(query: str) -> str:
    if not INPUT_RULES:
        return query
    return write_numbers(join_spaced(query))


def mentions_submission(messages: List[Dict[str, str]]) -> bool:
    """True when the last learner turn says the work is to be handed in (or the gate is off)."""
    return not SUBMIT_GATE or bool(_SUBMIT.search(last_user_turn(messages)))


def is_low_signal(messages: List[Dict[str, str]]) -> bool:
    return INPUT_RULES and not has_words(last_user_turn(messages))
