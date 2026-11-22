"""Response models; every enumerated value is a Literal type."""
from typing import List, Literal, Optional, get_args

from pydantic import BaseModel, Field

ANSWER_MAX_CHARS = 4000

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


class TutorResponse(BaseModel):
    answer: str = Field(max_length=ANSWER_MAX_CHARS)
    action: Action
    citations: List[str] = Field(default_factory=list)


class GuardInput(BaseModel):
    decision: Decision
    category: Category


class GuardOutput(BaseModel):
    grounded: bool
    unsupported: List[str] = Field(default_factory=list)
    safe: bool
    category: Category = "normal"


class GuardResponse(BaseModel):
    input: GuardInput
    output: Optional[GuardOutput] = None

    def wire(self) -> dict:
        return self.model_dump(exclude_none=True)
