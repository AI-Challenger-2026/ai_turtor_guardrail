"""Knowledge-base retrieval: character-bigram BM25 plus one text-embedding-3-small pick.

- The KB is kb/<course>/<section>.md with its <section>-table<n>.csv files appended, joined
  the same way as the grader's loader, so a cited section id maps to the text the judge sees.
- The BM25 index is built when this module is imported (pure Python, no network).
- Embeddings are built once at server start-up by `build_embeddings()` (app lifespan). If that
  fails, or embeddings are not configured, retrieval runs on BM25 alone.
- Nothing is written per request: `retrieve` only reads the start-up index.
"""
import asyncio
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import config, llm, logs, prompts
from .schemas import Message

KB_ROOT = Path(__file__).resolve().parents[1] / "kb"


@dataclass(frozen=True)
class SectionHit:
    section_id: str  # e.g. "FIN-02-01"
    score: float
    text: str
    kind: str  # which ranking picked it: "bm25" or "vector"; text is always the whole section


# ---------------------------------------------------------------- knowledge base

def load_sections(root: Path = KB_ROOT) -> Dict[str, str]:
    """Section id -> markdown text with its CSV tables appended (same join as the grader)."""
    sections = {}
    for md in sorted(root.glob("*/*.md")):
        tables = sorted(md.parent.glob(f"{md.stem}-table*.csv"))
        sections[md.stem] = "\n\n".join([md.read_text(encoding="utf-8"),
                                         *(t.read_text(encoding="utf-8") for t in tables)])
    if not sections:
        raise RuntimeError(f"no knowledge-base sections under {root}")
    return sections


def section_chunks(section_id: str, text: str) -> List[str]:
    """Split a section at its level-2 headings; every chunk keeps the section title line."""
    title = text.split("\n", 1)[0].lstrip("# ").strip()
    parts = [p.strip() for p in re.split(r"\n(?=## )", text) if p.strip()]
    return [p if i == 0 else f"{title}\n{p}" for i, p in enumerate(parts)]


# ---------------------------------------------------------------- BM25

_WORD = re.compile(r"[가-힣]+|[a-z]+|\d+")


def tokens(text: str) -> List[str]:
    """Hangul runs -> character bigrams (a lone syllable stays a unigram); Latin words; digit runs."""
    out: List[str] = []
    for w in _WORD.findall(unicodedata.normalize("NFKC", text).lower()):
        if "가" <= w[0] <= "힣":
            out += [w] if len(w) == 1 else [w[i:i + 2] for i in range(len(w) - 1)]
        else:
            out.append(w)
    return out


class BM25:
    def __init__(self, docs: Dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.ids = list(docs)
        self.tf = [Counter(tokens(docs[i])) for i in self.ids]
        lengths = [sum(c.values()) for c in self.tf]
        avg = sum(lengths) / len(lengths)
        self.norm = [k1 * (1 - b + b * n / avg) for n in lengths]
        df = Counter(t for c in self.tf for t in c)
        n_docs = len(self.ids)
        self.idf = {t: math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.k1 = k1

    def scores(self, query: str) -> Dict[str, float]:
        q = Counter(tokens(query))
        out = {}
        for sid, tf, norm in zip(self.ids, self.tf, self.norm):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[sid] = s
        return out


# ---------------------------------------------------------------- start-up index

SECTIONS: Dict[str, str] = load_sections()
_BM25 = BM25(SECTIONS)
# (section id, chunk text) pairs that get embedded at start-up.
CHUNKS: List[Tuple[str, str]] = [(sid, c) for sid, text in SECTIONS.items() for c in section_chunks(sid, text)]
# Unit-normalized chunk vectors, aligned with CHUNKS; None until build_embeddings() succeeds.
_VECTORS: Optional[List[List[float]]] = None
_EMBED_BATCH = 64


def _unit(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def embed_texts(texts: List[str], *, kind: str, timeout: float, retries: int = 0) -> Optional[List[List[float]]]:
    """Embed in batches; every batch gets `retries` extra attempts. None if any batch fails."""
    out: List[List[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start:start + _EMBED_BATCH]
        for attempt in range(1, retries + 2):
            vecs = await llm.embed(kind, batch, timeout=timeout, attempt=attempt)
            if vecs is not None:
                break
        else:
            return None
        out += [_unit(v) for v in vecs]
    return out


async def build_embeddings() -> bool:
    """Embed every chunk once at start-up. Returns whether hybrid retrieval is on."""
    global _VECTORS
    t0 = time.perf_counter()
    vecs = None
    if config.EMBED_ENABLED:
        vecs = await embed_texts([c for _, c in CHUNKS], kind="embed_kb",
                                 timeout=config.EMBED_STARTUP_TIMEOUT_S, retries=config.EMBED_STARTUP_RETRIES)
    _VECTORS = vecs
    logs.startup_line(kb_sections=len(SECTIONS), kb_chunks=len(CHUNKS),
                      embeddings="on" if vecs else ("failed" if config.EMBED_ENABLED else "off"),
                      latency_ms=round((time.perf_counter() - t0) * 1000))
    return vecs is not None


def embeddings_ready() -> bool:
    return _VECTORS is not None


# ---------------------------------------------------------------- queries

def user_turns(messages: Sequence[Message]) -> List[str]:
    return [m.content for m in messages if m.role == "user"]


def query_text(messages: Sequence[Message], extra: str = "") -> str:
    # The whole conversation. Earlier turns, including the tutor's earlier answers (e.g. its
    # summary of the page), point at the page the learner is on, which is the cited section:
    # on the tuning split BM25 top-1 went from 0.83 (learner turns only) to 0.88.
    # `extra` is appended (the /guard candidate answer, whose claims name the topic).
    return "\n".join([m.content for m in messages] + ([extra] if extra else []))


# Embedding inputs are capped well under the model's input limit (long pastes keep head and tail).
EMBED_QUERY_MAX_CHARS = 4000


def embed_query_text(messages: Sequence[Message], extra: str = "") -> str:
    return prompts.clip(query_text(messages, extra), EMBED_QUERY_MAX_CHARS)


# ---------------------------------------------------------------- English questions

# Course terms for the translator, derived from the KB at import: section titles and the
# course's own names (particles stripped). BM25 cannot match "Squirrel Teens debit card"
# to "다람쥐 틴즈 체크카드", so English questions get a Korean rendering added to the query.
_NAME = re.compile(r"(?:다람쥐|다람|꿈틀|도토리)\s?[가-힣]+(?:\s(?:통장|적금|정기예금|예금|체크카드|포인트|문구점|중학교|코딩 동아리|동아리|토끼))*")
_PARTICLE = re.compile(r"(?:에게서|에게|에는|에서|으로|이고|이에요|예요|은|는|을|를|의|가|로|와|과|도|만)$")


def build_glossary(sections: Dict[str, str]) -> str:
    titles = [text.split("\n", 1)[0].lstrip("# ").split(" ", 1)[-1].strip() for text in sections.values()]
    names = sorted({_PARTICLE.sub("", m.group(0)) for text in sections.values() for m in _NAME.finditer(text)})
    return "\n".join(f"- {t}" for t in titles + names)


GLOSSARY = build_glossary(SECTIONS)
_LATIN = re.compile(r"[A-Za-z]")
_HANGUL = re.compile(r"[가-힣]")


def is_english(text: str) -> bool:
    latin, hangul = len(_LATIN.findall(text)), len(_HANGUL.findall(text))
    return latin >= 12 and latin > 2 * hangul


async def korean_query(text: str) -> Optional[str]:
    """Korean rendering of an English question with the course's terms; None on failure."""
    system = prompts.TRANSLATE_SYSTEM.format(glossary=GLOSSARY)
    out = await llm.chat_text("translate", [{"role": "system", "content": system},
                                            {"role": "user", "content": prompts.clip(text, 1000)}],
                              effort="minimal", max_tokens=config.TRANSLATE_MAX_TOKENS,
                              timeout=config.TRANSLATE_TIMEOUT_S)
    return out.strip() if out and out.strip() else None


def rank_bm25(query: str) -> List[Tuple[str, float]]:
    scores = _BM25.scores(query)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def rank_vector(qvec: Sequence[float], vectors: Sequence[Sequence[float]] = None) -> List[Tuple[str, float]]:
    """Sections ordered by their best chunk's cosine similarity to the (unit) query vector."""
    vectors = _VECTORS if vectors is None else vectors
    best: Dict[str, float] = {}
    for (sid, _), v in zip(CHUNKS, vectors):
        s = sum(map(float.__mul__, qvec, v))
        if s > best.get(sid, -2.0):
            best[sid] = s
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


# Candidate set handed to the generator: BM25's top sections plus the best embedding match
# not already among them. On the tuning split this kept BM25's top-1 accuracy (0.83) and
# raised top-5 recall from 0.93 (BM25 alone) to 0.97; rank fusion lowered top-1 to 0.80.
BM25_TAKE = 4
VECTOR_TAKE = 1


def combine(bm25: List[Tuple[str, float]], vector: List[Tuple[str, float]], k: int) -> List[Tuple[str, float, str]]:
    out = [(sid, s, "bm25") for sid, s in bm25[:min(BM25_TAKE, k)]]
    seen = {sid for sid, _, _ in out}
    extra = [(sid, s, "vector") for sid, s in vector if sid not in seen][:max(0, min(VECTOR_TAKE, k - len(out)))]
    out += extra
    seen |= {sid for sid, _, _ in extra}
    out += [(sid, s, "bm25") for sid, s in bm25 if sid not in seen][:k - len(out)]
    return out


async def _none():
    return None


async def retrieve(messages: Sequence[Message], k: int = config.RETRIEVE_K, extra: str = "") -> List[SectionHit]:
    """Top-k sections for the conversation (plus `extra` text). Never raises; degrades to BM25 alone.

    The query embedding and, for an English last turn, its Korean rendering run concurrently.
    """
    turns = user_turns(messages)
    embedding = (embed_texts([embed_query_text(messages, extra)], kind="embed_query",
                             timeout=config.EMBED_QUERY_TIMEOUT_S) if _VECTORS is not None else _none())
    korean = korean_query(turns[-1]) if turns and is_english(turns[-1]) else _none()
    qv, translated = await asyncio.gather(embedding, korean)
    vector = rank_vector(qv[0]) if qv else []
    query = query_text(messages, extra) + (f"\n{translated}" if translated else "")
    return [SectionHit(section_id=sid, score=round(s, 6), text=SECTIONS[sid], kind=kind)
            for sid, s, kind in combine(rank_bm25(query), vector, k)]
