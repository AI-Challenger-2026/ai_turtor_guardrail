"""Request/response models. Every enumerated value is a Literal so a typo cannot reach the wire."""
from typing import List, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import config

Role = Literal["user", "assistant"]
Action = Literal["answer", "scaffold", "clarify", "abstain", "redirect", "refuse", "escalate"]
Decision = Literal["allow", "block"]
Category = Literal[
    "normal",
    "answer_seeking",
    "academic_dishonesty",
    "off_topic",
    "out_of_material",
    "jailbreak_injection",
    "pii_disclosure",
    "harmful_content",
    "emotional_distress",
    "low_signal_input",
]
CATEGORIES = get_args(Category)


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str


class ChatRequest(BaseModel):
    # spec_version and session_id are not declared on purpose: they are unused, and a
    # strict type on an unused field could only turn a gradable request into a fallback.
    model_config = ConfigDict(extra="ignore")

    messages: List[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def _has_user_turn(self) -> "ChatRequest":
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("messages contain no user turn")
        return self

    def last_user(self) -> str:
        return next(m.content for m in reversed(self.messages) if m.role == "user")

    def input_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)


class TutorRequest(ChatRequest):
    pass


class GuardRequest(ChatRequest):
    candidate_answer: Optional[str] = None

    def input_chars(self) -> int:
        return super().input_chars() + len(self.candidate_answer or "")


class TutorResponse(BaseModel):
    answer: str = Field(max_length=config.ANSWER_MAX_CHARS)
    action: Action
    citations: List[str] = Field(default_factory=list)


class GuardInput(BaseModel):
    decision: Decision
    category: Category


class GuardOutput(BaseModel):
    grounded: bool
    unsupported: List[str] = Field(default_factory=list)
    safe: bool
    category: str = "normal"


class GuardResponse(BaseModel):
    input: GuardInput
    output: Optional[GuardOutput] = None

    def wire(self) -> dict:
        # "output" is only sent when it exists (it is required iff a candidate was given).
        return self.model_dump(exclude_none=True)
