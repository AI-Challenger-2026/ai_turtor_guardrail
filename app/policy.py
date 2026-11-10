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
    template: Optional[str]  # fixed reply; None -> generation path
    mode: str = "answer"  # generation mode: "answer" or "scaffold"
    fallback: Optional[str] = None  # fixed reply if generation fails (None -> FALLBACK, "answer")


# On the generation path the action becomes "abstain" when the generator finds that the
# asked thing is not in the knowledge base (out_of_material is decided there, not here).
_ROUTES = {
    "normal": Route("answer", None),
    "pii_disclosure": Route("answer", None),  # generate.py adds the PII handling
    # The classifier maps out_of_material to normal; kept for completeness.
    "out_of_material": Route("answer", None),
    "answer_seeking": Route("scaffold", None, mode="scaffold", fallback=templates.SCAFFOLD_ANSWER_SEEKING),
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
