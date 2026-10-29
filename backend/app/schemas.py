"""요청·응답 데이터 모델 (Pydantic request/response models).

frontend/src/types.ts 와 1:1로 맞춘다. 필드를 바꿀 때는 양쪽을 함께 수정한다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.settings import AnswerMode, GroundingMode

Role = Literal["user", "assistant"]
Decision = Literal["allow", "block", "revise"]
TurnStatus = Literal["answered", "revised", "blocked", "no_evidence", "fallback"]
FinalAction = Literal["allow", "block", "revise", "fallback"]
ModuleStatus = Literal["enabled", "disabled", "not_implemented"]


# ---------------------------------------------------------------------------
# 요청 (request)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    answer_mode: AnswerMode | None = None
    guard_profile: str | None = Field(default=None, max_length=64)
    grounding_mode: GroundingMode | None = None


# ---------------------------------------------------------------------------
# 응답 구성 요소 (response parts)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    index: int
    doc_id: str
    doc_title: str
    unit: str
    section: str
    chunk_id: str
    text: str
    score: float | None = None


class GuardResult(BaseModel):
    """가드 하나의 판정 결과 (single guard verdict)."""

    name: str
    stage: Literal["input", "output"]
    decision: Decision
    category: str = "none"
    reason: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: int = 0
    attempt: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class GuardTrace(BaseModel):
    profile: str
    input: list[GuardResult] = Field(default_factory=list)
    output: list[GuardResult] = Field(default_factory=list)
    regenerations: int = 0
    final_action: FinalAction = "allow"


class LLMCall(BaseModel):
    task: str | None = None
    role: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0
    ok: bool = True
    error: str | None = None
    temperature_applied: bool | None = None
    usage_estimated: bool = False
    attempts: int = 1


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    estimated_cost_usd: float = 0.0
    calls: list[LLMCall] = Field(default_factory=list)

    @classmethod
    def from_calls(cls, calls: list[dict[str, Any]]) -> "Usage":
        items = [LLMCall(**c) for c in calls]
        inp = sum(c.input_tokens for c in items)
        out = sum(c.output_tokens for c in items)
        return cls(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=inp + out,
            llm_calls=len(items),
            estimated_cost_usd=round(sum(c.estimated_cost_usd for c in items), 6),
            calls=items,
        )


class TurnMeta(BaseModel):
    rewritten_query: str | None = None
    rewrite_applied: bool = False
    question_type: str | None = None
    learner_level: str | None = None
    retrieved_count: int = 0


class RetrievedChunkInfo(BaseModel):
    """추적 로그용 검색 결과 (retrieval result for trace)."""

    rank: int
    chunk_id: str
    doc_id: str
    doc_title: str
    unit: str
    section: str
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    passed_threshold: bool = True
    text: str = ""


class RewriteInfo(BaseModel):
    original: str
    rewritten: str
    applied: bool = False
    reason: str = ""


class TurnTrace(BaseModel):
    """한 턴의 모든 중간 결과 (full per-turn trace) — 로그와 CLI --debug, 평가에 사용."""

    trace_id: str
    timestamp: str
    source: str = "api"
    question: str
    history_messages: int = 0
    truncated_messages: int = 0
    answer_mode: str
    grounding_mode: str
    guard_profile: str
    question_type: str | None = None
    learner_level: str | None = None
    rewrite: RewriteInfo | None = None
    retrieval_query: str | None = None
    retrieved: list[RetrievedChunkInfo] = Field(default_factory=list)
    retrieval_threshold: float | None = None
    raw_answers: list[str] = Field(default_factory=list)
    guard_trace: GuardTrace | None = None
    final_answer: str = ""
    status: str = ""
    notes: list[str] = Field(default_factory=list)
    error: str | None = None


class TurnResult(BaseModel):
    """TutorService.handle() 의 반환값. API 응답(ChatResponse)과 평가 스크립트가 같은 객체를 쓴다."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounding: GroundingMode
    grounded: bool | None = None
    status: TurnStatus = "answered"
    blocked: bool = False
    block_category: str | None = None
    guard_trace: GuardTrace | None = None
    provider: str
    model: str
    answer_mode: AnswerMode
    latency_ms: int = 0
    usage: Usage = Field(default_factory=Usage)
    meta: TurnMeta = Field(default_factory=TurnMeta)
    trace_id: str
    trace: TurnTrace

    def to_response(self) -> "ChatResponse":
        return ChatResponse(**self.model_dump(exclude={"trace"}))


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounding: GroundingMode
    grounded: bool | None
    status: TurnStatus
    blocked: bool
    block_category: str | None
    guard_trace: GuardTrace | None
    provider: str
    model: str
    answer_mode: AnswerMode
    latency_ms: int
    usage: Usage
    meta: TurnMeta
    trace_id: str


# ---------------------------------------------------------------------------
# 기타 엔드포인트 (health / config / error)
# ---------------------------------------------------------------------------


class ErrorBody(BaseModel):
    code: str
    message: str
    trace_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    mock_mode: bool


class ModuleInfo(BaseModel):
    status: ModuleStatus
    detail: str | None = None


class ProviderInfo(BaseModel):
    provider: str
    model: str
    configured_provider: str
    configured_model: str
    status: ModuleStatus
    detail: str | None = None


class KBInfo(BaseModel):
    status: ModuleStatus
    detail: str | None = None
    num_documents: int = 0
    num_chunks: int = 0
    units: list[str] = Field(default_factory=list)
    built_at: str | None = None
    stale: bool = False


class RetrievalInfo(BaseModel):
    bm25: ModuleInfo
    dense: ModuleInfo
    hybrid: ModuleInfo
    query_rewrite: ModuleInfo


class GuardInfo(BaseModel):
    name: str
    stage: Literal["input", "output"]
    kind: str


class GuardsInfo(BaseModel):
    status: ModuleStatus
    profile: str
    available_profiles: list[str]
    input: list[GuardInfo] = Field(default_factory=list)
    output: list[GuardInfo] = Field(default_factory=list)


class LimitsInfo(BaseModel):
    max_message_chars: int
    max_turns: int
    max_request_bytes: int


class ConfigResponse(BaseModel):
    mock_mode: bool
    generator: ProviderInfo
    judge: ProviderInfo
    answer_mode: AnswerMode
    answer_modes: list[AnswerMode]
    grounding_mode: GroundingMode
    grounding_modes: list[GroundingMode]
    kb: KBInfo
    retrieval: RetrievalInfo
    guards: GuardsInfo
    limits: LimitsInfo
