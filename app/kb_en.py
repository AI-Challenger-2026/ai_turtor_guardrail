"""English side of the search index for English questions.

Each chunk is translated with the competition model at start-up and kept in memory only. A chunk
whose translation fails is indexed in Korean only.
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
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    last = users[-1] if users else ""
    return bool(_LATIN.search(last)) and not _HANGUL.search(last)


async def translate_chunks(texts: List[str], concurrency: int = 24, budget: float = 150.0) -> List[Optional[str]]:
    sem = asyncio.Semaphore(concurrency)
    deadline = time.time() + budget

    async def one(text: str) -> Optional[str]:
        async with sem:
            try:
                out = await chat(TRANSLATE_SYS, text, effort="minimal", max_tokens=2000, timeout=60,
                                 deadline=deadline, kind="translate")
            except Exception:
                return None
            return out or None

    return list(await asyncio.gather(*[one(t) for t in texts]))
