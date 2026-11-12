"""Verbatim material shown after a found answer (ported from our earlier server, 7d44c16).

A found answer in answer mode ends with up to two sentences of the cited (top-1) section, copied
verbatim, chosen by overlap with the question and the answer. The text comes only from the
knowledge base; the learner's words only rank the sentences. Sentences the answer already says,
and sentences that share too little with either, are left out. Scaffolds and abstentions get none:
a scaffold must not give the answer away, and an abstain cites nothing.
"""
import re

EXCERPT_HEAD = "수업 자료 원문:"
EXCERPT_SENTENCES = 2
EXCERPT_MAX_CHARS = 320
EXCERPT_MIN_CHARS = 12
EXCERPT_MIN_RELATED = 0.2   # share of a sentence's 2-grams found in the question or answer
EXCERPT_REPEAT_SHARE = 0.8  # share of a sentence's 3-grams already in the answer -> not repeated


def _grams(text: str, n: int) -> set:
    s = "".join(text.split())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def excerpt(section_text: str, question: str, answer: str = "", *, min_related: float = EXCERPT_MIN_RELATED) -> str:
    """Sentences of a section related to the question and answer, verbatim, in section order."""
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
    picked = sorted(i for _, i in sorted(scored)[:EXCERPT_SENTENCES])
    return " ".join(sentences[i] for i in picked)[:EXCERPT_MAX_CHARS]


def with_excerpt(answer: str, section_text: str, question: str) -> str:
    shown = excerpt(section_text, question, answer)
    return f"{answer}\n\n{EXCERPT_HEAD} {shown}" if shown else answer
