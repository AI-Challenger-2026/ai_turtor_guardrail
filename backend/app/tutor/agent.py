"""튜터 에이전트 (tutor agent): 시스템 프롬프트 조립, 답변 생성, 인용 번호 파싱.

- general 모드: prompts/tutor_general.txt
- kb 모드: prompts/tutor_grounded.txt + 검색된 청크(<source id="n" ...>) — 청크 안에서만 답한다.
- 출력 가드가 revise 를 요청하면 revision 지시를 시스템 프롬프트 끝에 붙여 다시 생성한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from typing import Sequence

from app.llm.base import LLMProvider, LLMResult
from app.prompts import PromptStore
from app.retrieval.types import RetrievedChunk
from app.schemas import Citation
from app.settings import LLMParams
from app.tutor.modes import ModeContext

_FENCED_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# 인용 묶음: [1], [1][2], [1, 3]. a[1] 이나 m[1][2] 같은 인덱싱과 구분하기 위해
# 묶음 앞 글자가 단어 문자나 ']' 이면 인용으로 보지 않는다.
_CITE_RE = re.compile(r"(?<![\w\]])((?:\[\d{1,2}(?:\s*,\s*\d{1,2})*\])+)")
_NUM_RE = re.compile(r"\d{1,2}")


def strip_code(text: str) -> str:
    """코드 블록/인라인 코드를 지운다 (코드 속 a[1] 을 인용으로 오인하지 않도록)."""
    text = _FENCED_RE.sub(" ", text)
    return _INLINE_CODE_RE.sub(" ", text)


def parse_citation_indices(text: str) -> list[int]:
    """본문에서 [n] 인용 번호를 등장 순서대로 중복 없이 추출한다."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(strip_code(text)):
        for num in _NUM_RE.findall(m.group(1)):
            n = int(num)
            if n not in seen:
                seen.append(n)
    return seen


def render_sources(chunks: Sequence[RetrievedChunk]) -> str:
    """검색 청크를 프롬프트용 <source> 블록으로 렌더링한다 (번호는 1부터)."""
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        blocks.append(
            f'<source id="{i}" doc="{escape(ch.doc_title)}" unit="{escape(ch.unit)}" section="{escape(ch.section)}">\n'
            f"{ch.text}\n</source>"
        )
    return "\n\n".join(blocks)


def build_citations(text: str, chunks: Sequence[RetrievedChunk]) -> tuple[list[Citation], list[int], list[int]]:
    """(citations, cited_indices, invalid_indices) — 실제 청크에 대응하는 번호만 citations 로 만든다."""
    cited = parse_citation_indices(text)
    citations: list[Citation] = []
    invalid: list[int] = []
    for n in cited:
        if 1 <= n <= len(chunks):
            ch = chunks[n - 1]
            citations.append(
                Citation(
                    index=n,
                    doc_id=ch.doc_id,
                    doc_title=ch.doc_title,
                    unit=ch.unit,
                    section=ch.section,
                    chunk_id=ch.chunk_id,
                    text=ch.text,
                    score=round(ch.score, 4),
                )
            )
        else:
            invalid.append(n)
    citations.sort(key=lambda c: c.index)
    return citations, cited, invalid


@dataclass
class AgentAnswer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    llm_result: LLMResult | None = None
    system_prompt: str = ""


class TutorAgent:
    def __init__(self, provider: LLMProvider, prompts: PromptStore, params: LLMParams) -> None:
        self.provider = provider
        self.prompts = prompts
        self.params = params

    def build_system_prompt(
        self,
        *,
        grounding: str,
        mode: ModeContext,
        chunks: Sequence[RetrievedChunk] = (),
        revision: str | None = None,
    ) -> str:
        variables = {
            "answer_mode": mode.answer_mode,
            "question_type": f"{mode.question_type} ({mode.question_type_label})",
            "learner_level": f"{mode.learner_level} ({mode.learner_level_label})",
            "mode_instruction": mode.mode_instruction,
            "level_instruction": mode.level_instruction,
        }
        if grounding == "kb":
            variables["sources"] = render_sources(chunks)
            variables["num_sources"] = str(len(chunks))
            system = self.prompts.render("tutor_grounded.txt", **variables)
        else:
            system = self.prompts.render("tutor_general.txt", **variables)
        if revision:
            system += "\n\n# 수정 지시 (revision)\n" + revision.strip()
        return system

    def answer(
        self,
        *,
        history: Sequence[dict[str, str]],
        question: str,
        grounding: str,
        mode: ModeContext,
        chunks: Sequence[RetrievedChunk] = (),
        revision: str | None = None,
    ) -> AgentAnswer:
        system = self.build_system_prompt(grounding=grounding, mode=mode, chunks=chunks, revision=revision)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": question})
        result = self.provider.generate(
            system=system,
            messages=messages,
            temperature=self.params.temperature,
            max_tokens=self.params.max_tokens,
            json_mode=False,
            task="tutor_answer",
        )
        text = result.text.strip()
        if grounding == "kb":
            citations, cited, invalid = build_citations(text, chunks)
        else:
            citations, cited, invalid = [], [], []
        return AgentAnswer(
            text=text,
            citations=citations,
            cited_indices=cited,
            invalid_citations=invalid,
            llm_result=result,
            system_prompt=system,
        )
