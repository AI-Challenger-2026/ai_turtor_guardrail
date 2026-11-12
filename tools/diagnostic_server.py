"""원래 서버의 처리 결과를 유지하고 평가용 메타데이터 헤더만 추가한다."""

import time

from server import create_app
from tutor_core.llm import ApiClients
from tutor_core.policy import InputPolicy
from tutor_core.retrieval import prepare_retriever
from tutor_core.tutor import TutorService
from tutor_core.verifier import OutputVerifier
from tools.evaluation_trace import CURRENT_TRACE, HEADER, RequestTrace, observed


class ObservedApiClients(ApiClients):
    async def complete_json(self, messages, schema, budget, **kwargs):
        trace = CURRENT_TRACE.get()
        stage = {"IntentSignals": "input_intent", "MaterialAnalysis": "material_check", "OutputReview": "output_review",
                 "TutorRevision": "tutor_repair", "GroundingReview": "grounding_recheck"}.get(schema.__name__)
        if schema.__name__ == "TutorDraft" and trace is not None:
            trace.drafts += 1
            stage = "tutor_draft" if trace.drafts == 1 else "tutor_repair"
        operation = super().complete_json
        if stage is None:
            return await operation(messages, schema, budget, **kwargs)
        return await observed(stage, lambda: operation(messages, schema, budget, **kwargs))

    async def embed(self, texts, budget, **kwargs):
        operation = super().embed
        return await observed("embedding", lambda: operation(texts, budget, **kwargs))


class ObservedRetriever:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def search(self, *args, **kwargs):
        return await observed("search", lambda: self.original.search(*args, **kwargs))


async def observed_retriever(clients):
    return ObservedRetriever(await prepare_retriever(clients))


class ObservedInputPolicy(InputPolicy):
    async def classify(self, *args, **kwargs):
        operation = super().classify
        return await observed("input_policy", lambda: operation(*args, **kwargs))


class ObservedOutputVerifier(OutputVerifier):
    async def verify(self, *args, **kwargs):
        operation = super().verify
        stage = "output_recheck" if kwargs.get("repair_issues") else "output_verification"
        return await observed(stage, lambda: operation(*args, **kwargs))

    async def recheck(self, *args, **kwargs):
        operation = super().recheck
        return await observed("output_recheck", lambda: operation(*args, **kwargs))


class ObservedTutorService(TutorService):
    async def handle(self, *args, **kwargs):
        operation = super().handle
        return await observed("tutor", lambda: operation(*args, **kwargs))

    def _protected(self, policy):
        trace = CURRENT_TRACE.get()
        if trace is not None:
            trace.record("protection", time.monotonic(), category=policy.assessment.category, action=policy.action)
        return super()._protected(policy)

    def _unavailable(self, reason):
        trace = CURRENT_TRACE.get()
        if trace is not None:
            trace.record("fallback", time.monotonic(), "failed", failure=reason)
        return super()._unavailable(reason)


class DiagnosticApplication:
    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in ("/tutor", "/guard"):
            return await self.application(scope, receive, send)
        trace = RequestTrace()
        token = CURRENT_TRACE.set(trace)

        async def measured_send(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), (HEADER.encode("ascii"), trace.header())]}
            await send(message)

        try:
            await self.application(scope, receive, measured_send)
        finally:
            CURRENT_TRACE.reset(token)


app = DiagnosticApplication(create_app(
    client_factory=ObservedApiClients, retrieval_factory=observed_retriever,
    policy_factory=ObservedInputPolicy, verifier_factory=ObservedOutputVerifier, tutor_factory=ObservedTutorService,
))
