"""입력 분류와 결정·튜터 행동 매핑을 공유하고 후보 답변을 격리한다."""

import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

from .llm import ApiClients, ProviderFailure, RequestBudget
from .prompts import INTENT_SYSTEM, MATERIAL_SYSTEM
from .retrieval import Retriever, SearchHit, SearchResult
from .schemas import Action, Category, InputAssessment, Message, ResponseModel


POLICY_VERSION = "input-v2"
HarmfulKind = Literal["none", "actionable", "denigration"]
MaterialStatus = Literal["supported", "missing", "unclear", "not_applicable"]
BLOCKED = frozenset({"academic_dishonesty", "off_topic", "jailbreak_injection", "harmful_content"})
ACTIONS: dict[str, Action] = {
    "normal": "answer", "answer_seeking": "scaffold", "academic_dishonesty": "scaffold",
    "off_topic": "redirect", "out_of_material": "abstain", "jailbreak_injection": "refuse",
    "pii_disclosure": "answer", "harmful_content": "refuse", "emotional_distress": "escalate",
    "low_signal_input": "clarify",
}


class InputAnalysis(ResponseModel):
    category: Category
    harmful_kind: HarmfulKind
    material_status: MaterialStatus


class IntentSignals(ResponseModel):
    task_summary: str
    personal_data: Literal["none", "self", "other", "fictional"]
    override_attempt: bool
    harmful_kind: HarmfulKind
    severe_distress: bool
    low_signal: bool
    task_kind: Literal["learning", "answer_only", "submission", "off_topic", "unclear"]


class MaterialAnalysis(ResponseModel):
    coverage: Literal["supported", "general_explanation", "missing_course_fact", "uncertain"]
    supporting_ids: list[str]


def analyze_intent(signals: IntentSignals) -> InputAnalysis:
    kind: HarmfulKind = "none"
    if signals.severe_distress:
        category = "emotional_distress"
    elif signals.override_attempt:
        category = "jailbreak_injection"
    elif signals.personal_data == "other" or signals.harmful_kind != "none":
        category = "harmful_content"
        kind = signals.harmful_kind if signals.harmful_kind != "none" else "actionable"
    elif signals.task_kind == "submission":
        category = "academic_dishonesty"
    elif signals.task_kind == "off_topic":
        category = "off_topic"
    elif signals.task_kind == "answer_only":
        category = "answer_seeking"
    elif signals.personal_data == "self":
        category = "pii_disclosure"
    elif signals.low_signal or signals.task_kind == "unclear":
        category = "low_signal_input"
    else:
        category = "normal"
    return InputAnalysis(category=category, harmful_kind=kind, material_status="unclear" if category == "normal" else "not_applicable")


@dataclass(frozen=True)
class PolicyResult:
    assessment: InputAssessment
    action: Action
    harmful_kind: HarmfulKind
    material_status: MaterialStatus
    verified: bool = True
    failure_reason: str | None = None


def map_policy(analysis: InputAnalysis) -> PolicyResult:
    category = analysis.category
    kind: HarmfulKind = analysis.harmful_kind if category == "harmful_content" else "none"
    action = "redirect" if category == "harmful_content" and kind == "denigration" else ACTIONS[category]
    return PolicyResult(
        InputAssessment(decision="block" if category in BLOCKED else "allow", category=category),
        action, kind, analysis.material_status,
    )


def unavailable_policy(reason: str) -> PolicyResult:
    return PolicyResult(InputAssessment(decision="block", category="normal"), "abstain", "none", "unclear", False, reason)


class InputPolicy:
    def __init__(self, clients: ApiClients, retriever: Retriever):
        self.clients = clients
        self.retriever = retriever
        self.counters = Counter()

    def summary(self) -> dict:
        return {"version": POLICY_VERSION, "intent_reasoning_effort": self.clients.settings.input_reasoning_effort,
                "timeout_seconds": self.clients.settings.input_policy_timeout, "counters": dict(self.counters)}

    def _material_messages(self, messages: Sequence[Message], found: SearchResult, count: int) -> list[dict[str, str]]:
        data = {
            "section_catalog": [{"id": section.section_id, "title": section.title} for section in self.retriever.knowledge.sections],
            "sections": [{"id": hit.section.section_id, "text": hit.section.text} for hit in found.hits[:count]],
            "allowed_supporting_ids": list(found.section_ids[:count]),
            "conversation": [{"role": message.role, "content": message.content} for message in messages],
            "current_request": messages[-1].content,
        }
        return [{"role": "system", "content": MATERIAL_SYSTEM},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]

    @staticmethod
    def intent_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
        data = {"conversation": [{"role": message.role, "content": message.content} for message in messages],
                "current_request": messages[-1].content}
        return [{"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]

    async def classify(self, messages: Sequence[Message], budget: RequestBudget) -> PolicyResult:
        self.counters["requests"] += 1
        expires = min(budget.expires_at, time.monotonic() + self.clients.settings.input_policy_timeout)

        def remaining():
            return max(0.0, expires - time.monotonic())

        async def classify_with_evidence():
            signals = await self.clients.complete_json(
                self.intent_messages(messages),
                IntentSignals, budget, seconds=remaining(), max_tokens=2000,
                reasoning_effort=self.clients.settings.input_reasoning_effort,
            )
            analysis = analyze_intent(signals)
            if analysis.category != "normal":
                return map_policy(analysis)
            found = await self.retriever.search(messages, budget, top_k=12)
            material = await self.clients.complete_json(
                self._material_messages(messages, found, 8), MaterialAnalysis, budget,
                seconds=remaining(), max_tokens=2000,
            )
            provided = set(found.section_ids[:8])
            invalid = any(section_id not in provided for section_id in material.supporting_ids)
            if remaining() >= 6.0 and (invalid or (material.coverage in ("missing_course_fact", "uncertain") and len(found.hits) > 8)):
                self.counters["evidence_expansions"] += 1
                hits = list(found.hits[:8])
                for section_id in material.supporting_ids:
                    if section_id in self.retriever.knowledge.by_id and section_id not in {hit.section.section_id for hit in hits} and len(hits) < 12:
                        hits.append(SearchHit(self.retriever.knowledge.by_id[section_id], 0.0))
                for hit in found.hits[8:]:
                    if hit.section.section_id not in {item.section.section_id for item in hits} and len(hits) < 12:
                        hits.append(hit)
                found = SearchResult(tuple(hits), found.mode, found.degraded, found.failure_reason)
                material = await self.clients.complete_json(
                    self._material_messages(messages, found, 12), MaterialAnalysis, budget,
                    seconds=remaining(), max_tokens=2000,
                )
            if any(section_id not in found.section_ids for section_id in material.supporting_ids):
                raise ProviderFailure("invalid_policy_evidence")
            missing = material.coverage == "missing_course_fact"
            analysis = InputAnalysis(
                category="out_of_material" if missing else "normal", harmful_kind="none",
                material_status="missing" if missing else "unclear" if material.coverage == "uncertain" else "supported",
            )
            return map_policy(analysis)

        try:
            result = await budget.run(classify_with_evidence, remaining())
            self.counters["verified"] += 1
            return result
        except Exception as exc:
            reason = exc.reason if isinstance(exc, ProviderFailure) else "input_policy_error"
            self.counters["failed"] += 1
            self.counters[reason] += 1
            return unavailable_policy(reason)
