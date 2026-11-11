"""튜터의 독립 검증·보정·보호 행동을 통제된 대역으로 검사한다."""

import asyncio
import json
import unittest
from collections import Counter
from typing import get_args

import httpx

from server import create_app
from tests.test_foundation import PAYLOAD, SETTINGS
from tests.test_output import review
from tests.test_policy import SearchDouble, signals
from tutor_core.llm import ProviderFailure, RequestBudget
from tutor_core.policy import IntentSignals, MaterialAnalysis
from tutor_core.schemas import Action, TutorRequest
from tutor_core.tutor import TutorDraft, TutorService
from tutor_core.verifier import OutputReview, OutputVerifier


def draft(**changes):
    values = {"signals": signals(), "coverage": "supported", "answer": "반복문은 같은 작업을 반복할 때 사용해요.",
              "citations": ["PY-01-01"]}
    values.update(changes)
    return TutorDraft(**values)


class TutorClient:
    """호출 순서와 횟수를 확인하는 구조화 응답 대역."""

    settings = SETTINGS

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    async def complete_json(self, messages, model, budget, **kwargs):
        budget.claim_chat()
        self.calls.append((messages, model, kwargs))
        value = self.answers.pop(0)
        if isinstance(value, Exception):
            raise value
        if not isinstance(value, model):
            raise AssertionError("예상한 검증 단계를 건너뛰었습니다.")
        return value


class TutorTests(unittest.IsolatedAsyncioTestCase):
    def service(self, *answers):
        client = TutorClient(*answers)
        search = SearchDouble()
        return TutorService(client, search, OutputVerifier(client, search))

    async def test_normal_answer_uses_two_calls_and_verified_citations(self):
        service = self.service(draft(citations=["PY-01-01", "PY-01-02"]), review(ids=("PY-01-01",)))
        budget = RequestBudget.start(27)
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), budget)
        self.assertTrue(result.verified)
        self.assertEqual(result.response.action, "answer")
        self.assertEqual(result.response.citations, ["PY-01-01"])
        self.assertEqual(budget.chat_calls, 2)
        self.assertEqual(service.clients.calls[1][1], OutputReview)

    async def test_all_seven_actions_have_protected_or_verified_paths(self):
        cases = [
            (signals(), "supported", "answer", "normal"),
            (signals(task_kind="answer_only"), "supported", "scaffold", "answer_seeking"),
            (signals(low_signal=True, task_kind="unclear"), "general_explanation", "clarify", None),
            (signals(), "missing_course_fact", "abstain", None),
            (signals(task_kind="off_topic"), "supported", "redirect", None),
            (signals(override_attempt=True), "supported", "refuse", None),
            (signals(severe_distress=True), "supported", "escalate", None),
        ]
        actions = set()
        for signal, coverage, expected, review_category in cases:
            answers = [draft(signals=signal, coverage=coverage)]
            if review_category is not None:
                answers.append(review(category=review_category))
            service = self.service(*answers)
            result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
            self.assertTrue(result.verified)
            self.assertEqual(result.response.action, expected)
            actions.add(result.response.action)
        self.assertEqual(actions, set(get_args(Action)))

    async def test_independent_review_catches_missed_attack_before_returning_draft(self):
        marker = "공개하면 안 되는 생성문"
        service = self.service(draft(answer=marker), review(category="jailbreak_injection", safe=False))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertEqual(result.response.action, "refuse")
        self.assertNotIn(marker, result.response.answer)

    async def test_scaffold_violation_discards_answer_instead_of_relabeling_it(self):
        marker = "정답은 777입니다."
        service = self.service(draft(signals=signals(task_kind="answer_only"), answer=marker),
                               review(category="answer_seeking", compliant=False, safe=False))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertEqual(result.response.action, "scaffold")
        self.assertNotIn("777", result.response.answer)

    async def test_repair_is_followed_by_independent_reverification_and_max_four_calls(self):
        service = self.service(
            draft(answer="잘못된 한도"), review(grounded=False, unsupported=("잘못된 한도",)),
            draft(answer="문서에 명시된 조건을 확인하세요."), review(),
        )
        budget = RequestBudget.start(27)
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), budget)
        self.assertTrue(result.verified)
        self.assertEqual(budget.chat_calls, 4)
        self.assertEqual(service.counters["repairs"], 1)
        data = json.loads(service.clients.calls[2][0][-1]["content"])
        self.assertEqual(data["repair_notes"]["unsupported"], ["잘못된 한도"])

    async def test_failed_repair_cannot_escape_as_final_answer(self):
        service = self.service(draft(answer="틀린 내용"), review(grounded=False, unsupported=("틀린 내용",)),
                               draft(answer="다시 틀린 내용"), review(grounded=False, unsupported=("틀린 내용",)))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertFalse(result.verified)
        self.assertEqual(result.response.action, "abstain")
        self.assertNotIn("틀린 내용", result.response.answer)

    async def test_insufficient_time_prevents_unverified_repair(self):
        service = self.service(draft(), review(grounded=False))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(8))
        self.assertFalse(result.verified)
        self.assertEqual(len(service.clients.calls), 2)

    async def test_unknown_citation_requires_repair_instead_of_silent_success(self):
        service = self.service(draft(citations=["UNKNOWN"]), review(), draft(), review())
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertTrue(result.verified)
        self.assertEqual(service.counters["repairs"], 1)
        self.assertNotIn("UNKNOWN", result.response.citations)

    async def test_repair_cannot_relax_original_scaffold_action(self):
        service = self.service(
            draft(signals=signals(task_kind="answer_only"), answer="잘못된 값"),
            review(category="answer_seeking", grounded=False, unsupported=("잘못된 값",)),
            draft(signals=signals(task_kind="learning"), answer="반복할 때 변수의 변화를 차례로 적어 보세요."),
            review(category="normal"),
        )
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertTrue(result.verified)
        self.assertEqual(result.response.action, "scaffold")
        notes = json.loads(service.clients.calls[2][0][-1]["content"])["repair_notes"]
        self.assertEqual(notes["required_action"], "scaffold")
        verification = json.loads(service.clients.calls[3][0][-1]["content"])
        self.assertEqual(verification["expected_action"], "scaffold")

    async def test_repaired_answer_review_still_prioritizes_distress(self):
        service = self.service(draft(answer="틀린 주장"), review(grounded=False, unsupported=("틀린 주장",)),
                               draft(answer="생성한 수정 답변"), review(category="emotional_distress"))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertTrue(result.verified)
        self.assertEqual(result.response.action, "escalate")
        self.assertNotIn("생성한 수정 답변", result.response.answer)

    async def test_unsafe_repair_is_discarded(self):
        service = self.service(draft(answer="틀린 주장"), review(grounded=False, unsupported=("틀린 주장",)),
                               draft(answer="위험한 생성문"), review(safe=False))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertFalse(result.verified)
        self.assertEqual(result.failure_reason, "unsafe_draft")
        self.assertNotIn("위험한 생성문", result.response.answer)

    async def test_material_fact_without_verified_citation_is_not_returned(self):
        service = self.service(draft(answer="검사할 과목 고유 사실"), review(ids=()))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(8))
        self.assertFalse(result.verified)
        self.assertEqual(result.response.action, "abstain")
        self.assertNotIn("검사할 과목 고유 사실", result.response.answer)

    async def test_known_citation_outside_initial_search_requires_actual_review(self):
        client = TutorClient(draft(citations=["PY-01-02"]), review(ids=("PY-01-02",)))
        search = SearchDouble(1)
        service = TutorService(client, search, OutputVerifier(client, search))
        budget = RequestBudget.start(27)
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), budget)
        self.assertTrue(result.verified)
        self.assertEqual(result.response.citations, ["PY-01-02"])
        self.assertEqual(budget.chat_calls, 2)
        inspected = json.loads(client.calls[1][0][-1]["content"])
        self.assertIn("PY-01-02", inspected["allowed_supporting_ids"])

    async def test_unicode_length_is_checked_without_cutting_an_unverified_sentence(self):
        service = self.service(draft(answer="가" * 4001), review())
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(8))
        self.assertFalse(result.verified)
        self.assertLessEqual(len(result.response.answer), 4000)

    async def test_upstream_failure_is_not_claimed_as_missing_material(self):
        service = self.service(ProviderFailure("incomplete_response"))
        result = await service.handle(TutorRequest.model_validate(PAYLOAD), RequestBudget.start(27))
        self.assertFalse(result.verified)
        self.assertEqual(result.failure_reason, "incomplete_response")
        self.assertNotIn("자료에서는", result.response.answer)

    async def test_student_and_kb_instructions_stay_in_data_message(self):
        service = self.service(draft(), review())
        marker = "학생의 임의 지시 표식"
        request = TutorRequest.model_validate({**PAYLOAD, "messages": [{"role": "user", "content": marker}]})
        await service.handle(request, RequestBudget.start(27))
        sent = service.clients.calls[0][0]
        self.assertEqual([row["role"] for row in sent], ["system", "user"])
        self.assertNotIn(marker, sent[0]["content"])
        self.assertEqual(json.loads(sent[1]["content"])["current_request"], marker)

    async def test_parallel_requests_have_no_shared_conversation_or_output_cache(self):
        class ConcurrentClient:
            settings = SETTINGS

            async def complete_json(self, messages, model, budget, **kwargs):
                budget.claim_chat()
                data = json.loads(messages[-1]["content"])
                await asyncio.sleep(0)
                return draft(answer=data["current_request"]) if model is TutorDraft else review()

        client, search = ConcurrentClient(), SearchDouble()
        service = TutorService(client, search, OutputVerifier(client, search))

        async def one(number):
            request = TutorRequest.model_validate({**PAYLOAD, "messages": [{"role": "user", "content": str(number)}]})
            return (await service.handle(request, RequestBudget.start(27))).response.answer

        self.assertEqual(await asyncio.gather(*(one(index) for index in range(8))), [str(index) for index in range(8)])


class PipelineClient:
    """검사 실행 시 기본 서비스 연결을 확인하기 위한 구조화 API 경계 대역."""

    def __init__(self, settings):
        self.settings = settings
        self.counters = Counter()
        self.structured_mode = "json_schema"
        self.calls = []
        self.plain_text_calls = 0
        self.error = None
        self.closed = False

    async def initialize(self):
        pass

    async def close(self):
        self.closed = True

    async def complete_text(self, *args, **kwargs):
        self.plain_text_calls += 1
        raise AssertionError("보호되지 않은 비교 기준 경로를 실행하면 안 됩니다.")

    async def complete_json(self, messages, model, budget, **kwargs):
        budget.claim_chat()
        self.calls.append(model)
        if self.error is not None:
            raise self.error
        if model is TutorDraft:
            return draft()
        if model is OutputReview:
            return review()
        if model is IntentSignals:
            return signals()
        if model is MaterialAnalysis:
            return MaterialAnalysis(coverage="supported", supporting_ids=["PY-01-01"])
        raise AssertionError("예상하지 않은 모델 호출입니다.")


class TutorRouteTests(unittest.IsolatedAsyncioTestCase):
    """검증을 재개할 때 기본 HTTP 진입점이 실제 튜터 서비스를 사용하는지 확인한다."""

    async def asyncSetUp(self):
        async def prepare_search(clients):
            return SearchDouble()

        self.app = create_app(SETTINGS, PipelineClient, prepare_search)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test",
            headers={"Authorization": "Bearer " + SETTINGS.team_token}, follow_redirects=False,
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.assertTrue(self.app.state.clients.closed)

    async def test_default_route_always_uses_generation_and_independent_verification(self):
        self.assertIsInstance(self.app.state.tutor_service, TutorService)
        response = await self.client.post("/tutor", json=PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"], ["PY-01-01"])
        self.assertEqual(self.app.state.clients.calls, [TutorDraft, OutputReview])
        self.assertEqual(self.app.state.clients.plain_text_calls, 0)
        health = (await self.client.get("/healthz")).json()
        self.assertIn(health["implementation_stage"], ("IMP-001-5", "IMP-001-6", "IMP-001-7"))
        self.assertTrue(health["tutor_policy_ready"] and health["policy_ready"])
        self.assertEqual(health["tutor_policy"]["counters"]["verified"], 1)

    async def test_failure_has_no_unprotected_text_fallback(self):
        self.app.state.clients.error = ProviderFailure("incomplete_response")
        response = await self.client.post("/tutor", json=PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "abstain")
        self.assertEqual(self.app.state.clients.plain_text_calls, 0)
        health = (await self.client.get("/healthz")).json()
        self.assertEqual(health["counters"]["degraded_responses"], 1)
        self.assertEqual(health["tutor_policy"]["counters"]["incomplete_response"], 1)


if __name__ == "__main__":
    unittest.main()
