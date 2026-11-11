"""English side of the BM25 index, built at server start (added on top of final-v3).

Questions whose last user turn is written in English matched the Korean index badly: BM25 on Korean
syllable 2-grams finds nothing for English words, so only the earlier Korean turns and code tokens
counted. At start-up each KB chunk is translated into English with the competition model
(gpt-5-nano, the same OPENAI_* client as every other call), and English questions are ranked by
BM25 over "Korean chunk + its English translation". Korean questions keep the Korean-only index.

Public tutor items with evidence (eval/v3_bilingual.py): English last turn top-1 2/6 -> 4/6; Korean
items whose last turn was machine-translated 94/109 -> 100/109. The translation lives in memory
only; a chunk whose translation fails stays Korean-only.
"""
import asyncio
import re
import time
from typing import List, Optional

from .llm import chat

TRANSLATE_SYS = """Translate the Korean course material below into natural English for a search index.
Keep code, numbers, units, section IDs and English words exactly as they are.
For Korean names of banks, products, people, schools, places, tools and exercises, translate the name by
its meaning and add the romanized Korean in parentheses, e.g. a name meaning "blue bird bank" becomes
"Blue Bird Bank (Parangsae Eunhaeng)". Translate table cells too. Output only the translation."""

_HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
_LATIN = re.compile(r"[A-Za-z]")


def english_turn(messages) -> bool:
    """True when the last user turn is written in English: Latin letters and no Hangul at all."""
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    last = users[-1] if users else ""
    return bool(_LATIN.search(last)) and not _HANGUL.search(last)


async def translate_chunks(texts: List[str], concurrency: int = 24, budget: float = 150.0) -> List[Optional[str]]:
    """English translation of each text, None where the call failed or the start-up budget ran out."""
    sem = asyncio.Semaphore(concurrency)
    deadline = time.time() + budget

    async def one(text: str) -> Optional[str]:
        async with sem:
            try:
                out = await chat(TRANSLATE_SYS, text, effort="minimal", max_tokens=2000, timeout=60,
                                 deadline=deadline, kind="translate")
            except Exception:  # noqa: BLE001  (logged per call by app.llm)
                return None
            return out or None

    return list(await asyncio.gather(*[one(t) for t in texts]))
