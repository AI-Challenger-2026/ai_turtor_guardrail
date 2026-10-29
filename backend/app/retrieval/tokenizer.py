"""한국어 토큰화 (Korean tokenization).

- KiwiTokenizer: kiwipiepy 형태소 분석 — 내용어(명사, 동사·형용사 어간, 외국어, 숫자)만 남긴다.
- SimpleTokenizer: kiwipiepy 가 없을 때의 fallback — 공백/기호 분리 + 한글 문자 bigram.
설정 retrieval.tokenizer: auto(kiwi 우선) | kiwi | simple
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from app.logging_utils import get_logger

log = get_logger("retrieval.tokenizer")

# 질문에 흔하지만 내용과 무관한 말 (question filler words)
STOPWORDS = {
    "뭐", "뭐야", "뭔가요", "뭐예요", "뭐에요", "무엇", "무엇인가요", "어떻게", "왜", "언제", "어디", "알려", "알려줘",
    "설명", "설명해", "설명해줘", "해줘", "줘", "좀", "요", "거", "것", "수", "있", "없", "하", "되", "이", "그", "저",
    "나", "너", "제", "내", "말", "알", "모르", "궁금", "질문", "이거", "그거", "저거", "주", "보", "싶",
}

_KEEP_TAG_PREFIXES = ("NNG", "NNP", "NR", "VV", "VA", "XR", "SL", "SN", "SH")
_SIMPLE_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z_][A-Za-z0-9_]*|\d+")
_HANGUL_RE = re.compile(r"^[가-힣]+$")


class Tokenizer(Protocol):
    name: str

    def tokenize(self, text: str) -> list[str]: ...


class SimpleTokenizer:
    name = "simple"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for tok in _SIMPLE_TOKEN_RE.findall(text.lower()):
            if tok in STOPWORDS:
                continue
            if _HANGUL_RE.match(tok) and len(tok) >= 2:
                tokens.append(tok)
                tokens.extend(tok[i : i + 2] for i in range(len(tok) - 1))
            elif len(tok) >= 1:
                tokens.append(tok)
        return [t for t in tokens if t not in STOPWORDS]


class KiwiTokenizer:
    name = "kiwi"

    def __init__(self) -> None:
        from kiwipiepy import Kiwi  # type: ignore[import-not-found]

        self._kiwi = Kiwi()

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for tok in self._kiwi.tokenize(text):
            if not tok.tag.startswith(_KEEP_TAG_PREFIXES):
                continue
            form = tok.form.lower()
            if form in STOPWORDS:
                continue
            tokens.append(form)
        return tokens


@lru_cache(maxsize=4)
def get_tokenizer(preference: str = "auto") -> Tokenizer:
    """선호 토크나이저를 만든다. kiwi 를 쓸 수 없으면 simple 로 fallback 하고 경고를 남긴다."""
    if preference in ("auto", "kiwi"):
        try:
            return KiwiTokenizer()
        except Exception as exc:  # ImportError 또는 모델 로딩 실패
            log.warning("kiwipiepy 를 사용할 수 없어 simple 토크나이저로 대체 (fallback): %s", exc)
    return SimpleTokenizer()
