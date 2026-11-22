"""Fallback answer made of the top section's own sentences, used when generation fails."""
import re
from typing import Dict, List

from .kb import tokenize

_SKIP = re.compile(r"^\s*(#|\[표|\|)")


def _sentences(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        if not line.strip() or _SKIP.match(line) or "," in line and line.count(",") >= 2 and len(line) < 80:
            continue
        line = re.sub(r"^\s*(Q\.|A\.)\s*", "", line.strip())
        for s in re.split(r"(?<=[.?!])\s+", line):
            s = s.strip()
            if 8 <= len(s) <= 220:
                out.append(s)
    return out


def extractive_answer(messages: List[Dict[str, str]], section: Dict[str, str], max_sents: int = 2) -> str:
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    q = set(tokenize(users[-1] if users else ""))
    sents = _sentences(section.get("text", ""))
    if not sents:
        return ""
    scored = sorted(range(len(sents)), key=lambda i: -len(q & set(tokenize(sents[i]))))
    best = [i for i in scored[:max_sents] if len(q & set(tokenize(sents[i]))) > 0]
    if not best:
        best = list(range(min(max_sents, len(sents))))
    return " ".join(sents[i] for i in sorted(best))
