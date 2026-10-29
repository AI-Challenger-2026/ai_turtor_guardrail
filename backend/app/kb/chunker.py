"""구조 인식 청킹 (structure-aware chunking).

1) loader 가 heading 기준으로 나눈 Block 을 기본 단위로 한다 (단원·섹션 메타데이터 보존).
2) Block 이 chunk_size 보다 길면 문단 기준으로 나누고, 이전 청크 끝부분을 overlap 만큼 겹친다.
3) 코드 블록(```)은 문단 하나로 취급해 중간에서 자르지 않는다. 그래도 너무 길면 줄 단위로 나눈다.
"""

from __future__ import annotations

import re

from app.kb.models import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_paragraphs(text: str) -> list[str]:
    """빈 줄 기준 문단 분리. 코드 펜스 안의 빈 줄에서는 나누지 않는다."""
    paragraphs: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if not in_fence and not line.strip():
            if buf:
                paragraphs.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf).strip())
    return [p for p in paragraphs if p]


def _split_long(paragraph: str, chunk_size: int) -> list[str]:
    """chunk_size 보다 긴 문단을 문장(코드는 줄) 단위로 나눈다. 그래도 길면 글자 수로 자른다."""
    is_code = paragraph.lstrip().startswith("```")
    units = paragraph.splitlines() if is_code else [s for s in _SENTENCE_SPLIT.split(paragraph) if s and s.strip()]
    joiner = "\n" if is_code else " "
    pieces: list[str] = []
    cur = ""
    for u in units:
        u = u.strip() if not is_code else u
        while len(u) > chunk_size:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(u[:chunk_size])
            u = u[chunk_size:]
        candidate = f"{cur}{joiner}{u}" if cur else u
        if len(candidate) <= chunk_size:
            cur = candidate
        else:
            pieces.append(cur)
            cur = u
    if cur:
        pieces.append(cur)
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """이전 청크의 끝부분 overlap 글자 (단어 중간에서 시작하지 않게 조정, 코드가 섞이면 생략)."""
    if overlap <= 0 or len(text) <= overlap:
        return "" if overlap <= 0 else text
    tail = text[-overlap:]
    if "```" in tail:
        return ""
    cut = re.search(r"[\s.]", tail)
    if cut and cut.end() < len(tail):
        tail = tail[cut.end():]
    return tail.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """텍스트 한 블록을 청크 문자열 리스트로 나눈다."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    pieces: list[str] = []
    for para in split_paragraphs(text):
        pieces.extend([para] if len(para) <= chunk_size else _split_long(para, chunk_size))

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        candidate = f"{cur}\n\n{piece}" if cur else piece
        if len(candidate) <= chunk_size or not cur:
            cur = candidate
            continue
        chunks.append(cur)
        tail = _overlap_tail(cur, overlap)
        cur = f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) + 2 <= chunk_size + overlap else piece
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc: Document, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    """Document 를 Chunk 리스트로 만든다. chunk_id 는 "<doc_id>#<순번 3자리>"."""
    if chunk_size <= 0:
        raise ValueError("chunk_size 는 0보다 커야 합니다")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap 은 0 이상 chunk_size 미만이어야 합니다")
    out: list[Chunk] = []
    for block in doc.blocks:
        for piece in chunk_text(block.text, chunk_size, overlap):
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(out):03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.doc_title,
                    unit=block.unit,
                    section=block.section,
                    text=piece,
                    order=len(out),
                )
            )
    return out
