"""Category -> guard decision and tutor route. The only place these mappings live."""
from dataclasses import dataclass
from typing import Optional

from . import templates
from .schemas import Action, Category, Decision

# Guard decision is fully determined by the category (verified on the public set).
BLOCK_CATEGORIES = frozenset({"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"})


def decision(category: Category) -> Decision:
    return "block" if category in BLOCK_CATEGORIES else "allow"


@dataclass(frozen=True)
class Route:
    action: Action
    template: Optional[str]  # None -> generation path


_ROUTES = {
    "normal": Route("answer", None),
    "pii_disclosure": Route("answer", None),  # generate.py adds the PII handling
    # Task 1 never produces out_of_material (the classifier maps it to normal);
    # task 2 decides it from the knowledge base and routes it to abstain.
    "out_of_material": Route("answer", None),
    "answer_seeking": Route("scaffold", templates.SCAFFOLD_ANSWER_SEEKING_TEMP),
    "academic_dishonesty": Route("scaffold", templates.SCAFFOLD_ACADEMIC),
    "jailbreak_injection": Route("refuse", templates.REFUSE_JAILBREAK),
    "off_topic": Route("redirect", templates.REDIRECT_OFF_TOPIC),
    "emotional_distress": Route("escalate", templates.ESCALATE),
    "low_signal_input": Route("clarify", templates.CLARIFY),
}
_HARMFUL_STATEMENT = Route("redirect", templates.REDIRECT_STATEMENT)
_HARMFUL_REQUEST = Route("refuse", templates.REFUSE_HARMFUL)


def tutor_route(category: Category, harm_kind: str) -> Route:
    if category == "harmful_content":
        # A discriminatory/insulting remark is redirected; anything else harmful is refused.
        return _HARMFUL_STATEMENT if harm_kind == "statement" else _HARMFUL_REQUEST
    return _ROUTES[category]
