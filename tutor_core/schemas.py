"""대회 규격의 요청과 응답을 엄격한 타입으로 표현한다."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


Action = Literal["answer", "scaffold", "clarify", "abstain", "redirect", "refuse", "escalate"]
Category = Literal[
    "normal", "answer_seeking", "academic_dishonesty", "off_topic", "out_of_material",
    "jailbreak_injection", "pii_disclosure", "harmful_content", "emotional_distress", "low_signal_input",
]


class RequestModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class Message(RequestModel):
    role: Literal["user", "assistant"]
    content: str


class TutorRequest(RequestModel):
    spec_version: Literal["1.0"]
    session_id: str
    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conversation(self) -> "TutorRequest":
        if self.messages[0].role != "user" or self.messages[-1].role != "user":
            raise ValueError("첫 발화와 마지막 발화는 user여야 합니다.")
        return self


class GuardRequest(TutorRequest):
    candidate_answer: Optional[str] = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "GuardRequest":
        if "candidate_answer" in self.model_fields_set and self.candidate_answer is None:
            raise ValueError("후보 답변을 전달할 때는 문자열이어야 합니다.")
        return self

    @property
    def has_candidate(self) -> bool:
        return "candidate_answer" in self.model_fields_set


class ResponseModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class TutorResponse(ResponseModel):
    answer: str = Field(max_length=4000)
    action: Action
    citations: list[str]


class InputAssessment(ResponseModel):
    decision: Literal["allow", "block"]
    category: Category


class OutputAssessment(ResponseModel):
    grounded: bool
    unsupported: list[str]
    safe: bool
    category: str


class GuardResponse(ResponseModel):
    input: InputAssessment
    output: Optional[OutputAssessment] = None


class ProbeDetails(ResponseModel):
    verified: bool


class StructuredProbe(ResponseModel):
    status: Literal["ok"]
    values: list[int]
    details: ProbeDetails
