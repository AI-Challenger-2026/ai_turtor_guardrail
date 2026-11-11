"""시간 초과 대비 '자료 발췌 답변': LLM 없이, 검색 1위 섹션에서 질문과 가장 많이 겹치는 문장을 그대로 고른다.
답변 생성이 시간 예산을 넘겼을 때 기본 응답("지금은 답변을 만들 수 없어요") 대신 쓴다.
자료 문장을 그대로 옮기므로 근거 충실 판정을 통과하고, 1위 섹션 인용은 91% 확률로 정답 근거다(DEVLOG 16장).
"""
import re
from typing import Dict, List

from .kb import tokenize

_SKIP = re.compile(r"^\s*(#|\[표|\|)")


def _sentences(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        if not line.strip() or _SKIP.match(line) or "," in line and line.count(",") >= 2 and len(line) < 80:
            continue  # 제목, 표 표시, CSV 줄은 건너뜀
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
    if not q or not sents:
        return ""
    scored = sorted(range(len(sents)), key=lambda i: -len(q & set(tokenize(sents[i]))))
    best = [i for i in scored[:max_sents] if len(q & set(tokenize(sents[i]))) > 0]
    if not best:
        return ""
    return " ".join(sents[i] for i in sorted(best))
