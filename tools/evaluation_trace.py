"""본문 없이 공개 평가 요청의 처리 단계와 실패 종류만 전달한다."""

import asyncio
import json
import math
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field


HEADER = "x-tutor-diagnostics"
VERSION = "public-trace-v1"
MAX_HEADER_BYTES = 12000
MAX_EVENTS = 48
STAGES = frozenset({"search", "embedding", "input_intent", "material_check", "tutor_draft",
                    "tutor_repair", "output_review", "input_policy", "output_verification", "tutor",
                    "protection", "fallback", "output_recheck", "grounding_recheck"})
FAILURES = frozenset({"deadline_exceeded", "call_budget_exhausted", "upstream_status", "upstream_timeout",
                      "upstream_connection", "upstream_error", "internal_error", "invalid_structured_response",
                      "empty_response", "incomplete_response", "model_refusal", "invalid_embedding",
                      "provider_not_ready", "invalid_output_evidence", "output_verification_error",
                      "output_verifier_unavailable", "unverified_tutor_draft", "unverified_tutor_repair",
                      "unsafe_draft", "tutor_error", "input_policy_error", "invalid_policy_evidence",
                      "cancelled", "stage_error", "unknown_failure", "insufficient_repair_budget",
                      "insufficient_verification_budget", "recheck_context_mismatch"})
EVIDENCE_ISSUES = frozenset({"invalid_supporting_ids", "invalid_unsupported_spans", "inconsistent_grounding"})
CATEGORIES = frozenset({"normal", "answer_seeking", "academic_dishonesty", "off_topic", "out_of_material",
                        "jailbreak_injection", "pii_disclosure", "harmful_content", "emotional_distress", "low_signal_input"})
ACTIONS = frozenset({"answer", "scaffold", "clarify", "abstain", "redirect", "refuse", "escalate"})
SCOPE_FIELDS = {"coverage": ("supported", "general_explanation", "missing_course_fact", "uncertain"),
                "material_status": ("supported", "missing", "unclear", "not_applicable")}
BOOL_FIELDS = ("verified", "grounding_verified", "safety_verified", "grounded", "safe", "policy_compliant", "degraded", "claimed_grounded")
CURRENT_TRACE = ContextVar("public_evaluation_trace", default=None)


def clean_event(event):
    """허용한 메타데이터 항목만 복사한다."""
    if not isinstance(event, dict) or event.get("stage") not in STAGES or event.get("status") not in ("ok", "failed", "error", "cancelled"):
        return None
    result = {"stage": event["stage"], "status": event["status"]}
    for key in ("started_seconds", "elapsed_seconds"):
        value = event.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return None
        result[key] = round(value, 6)
    for key in BOOL_FIELDS:
        if type(event.get(key)) is bool:
            result[key] = event[key]
    if event.get("failure") is not None:
        result["failure"] = event["failure"] if event["failure"] in FAILURES else "unknown_failure"
    for key, allowed in (("category", CATEGORIES), ("action", ACTIONS)):
        if event.get(key) in allowed:
            result[key] = event[key]
    for key, allowed in SCOPE_FIELDS.items():
        if event.get(key) in allowed:
            result[key] = event[key]
    ids = event.get("section_ids")
    if isinstance(ids, list):
        result["section_ids"] = [sid for sid in ids[:12] if isinstance(sid, str) and re.fullmatch(r"[A-Z]{2,6}-[0-9]{2}-[0-9]{2}", sid)]
    issues = event.get("evidence_issues")
    if isinstance(issues, list):
        result["evidence_issues"] = [code for code in issues if isinstance(code, str) and code in EVIDENCE_ISSUES]
    coverage = event.get("source_coverage")
    if isinstance(coverage, dict):
        result["source_coverage"] = {name: count for name, count in coverage.items()
                                     if name in ("citation", "question", "claim") and type(count) is int and 0 <= count <= 12}
    size = event.get("context_characters")
    if type(size) is int and 0 <= size <= 1000000:
        result["context_characters"] = size
    return result


@dataclass
class RequestTrace:
    started: float = field(default_factory=time.monotonic)
    events: list = field(default_factory=list)
    truncated: bool = False
    drafts: int = 0

    def record(self, stage, started, status="ok", **metadata):
        event = clean_event({"stage": stage, "status": status, "started_seconds": max(0, started - self.started),
                             "elapsed_seconds": max(0, time.monotonic() - started), **metadata})
        if event is not None and len(self.events) < MAX_EVENTS:
            self.events.append(event)
        else:
            self.truncated = True

    def header(self):
        events = sorted(self.events, key=lambda event: event["started_seconds"])
        while True:
            value = json.dumps({"version": VERSION, "events": events, "truncated": self.truncated}, separators=(",", ":"))
            if len(value.encode("ascii")) <= MAX_HEADER_BYTES:
                return value.encode("ascii")
            events.pop()
            self.truncated = True


def decode_trace(value):
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_HEADER_BYTES:
        return {"complete": False, "events": []}
    try:
        parsed = json.loads(value)
        if parsed.get("version") != VERSION or not isinstance(parsed.get("events"), list):
            raise ValueError
        source = parsed["events"]
        events = [clean_event(event) for event in source[:MAX_EVENTS]]
        complete = bool(events) and len(source) <= MAX_EVENTS and all(event is not None for event in events) and parsed.get("truncated") is False
        return {"complete": complete, "events": [event for event in events if event is not None]}
    except (ValueError, TypeError, AttributeError):
        return {"complete": False, "events": []}


def result_metadata(result):
    metadata = {key: getattr(result, key) for key in BOOL_FIELDS if type(getattr(result, key, None)) is bool}
    for key, allowed in SCOPE_FIELDS.items():
        if getattr(result, key, None) in allowed:
            metadata[key] = getattr(result, key)
    reason = getattr(result, "failure_reason", None)
    if reason is not None:
        metadata["failure"] = reason if reason in FAILURES else "unknown_failure"
    assessment = getattr(result, "assessment", None)
    if assessment is not None:
        metadata["category"] = assessment.category
    response = getattr(result, "response", None)
    action = getattr(response, "action", getattr(result, "action", None))
    if action in ACTIONS:
        metadata["action"] = action
    output = getattr(result, "output", None)
    if output is not None:
        metadata.update(grounded=output.grounded, safe=output.safe)
    if hasattr(result, "section_ids"):
        metadata["section_ids"] = list(result.section_ids)
    if hasattr(result, "evidence_issues"):
        metadata["evidence_issues"] = list(result.evidence_issues)
    evidence = getattr(result, "evidence", None) or result
    if hasattr(evidence, "source_coverage"):
        metadata["source_coverage"] = dict(evidence.source_coverage)
        metadata["context_characters"] = evidence.context_characters
        metadata["section_ids"] = list(evidence.section_ids)
    return metadata


async def observed(stage, operation):
    trace = CURRENT_TRACE.get()
    if trace is None:
        return await operation()
    started = time.monotonic()
    try:
        result = await operation()
    except BaseException as exc:
        reason = getattr(exc, "reason", "cancelled" if isinstance(exc, asyncio.CancelledError) else "stage_error")
        trace.record(stage, started, "cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                     failure=reason if reason in FAILURES else "unknown_failure")
        raise
    metadata = result_metadata(result)
    trace.record(stage, started, "failed" if metadata.get("verified") is False or metadata.get("failure") else "ok", **metadata)
    return result
