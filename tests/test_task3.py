"""Offline tests for the /guard knowledge-base checks and English-question retrieval.

Run: python3 -m unittest discover -s tests -t . -v
"""
import asyncio
import json
import time
import unittest
from unittest import mock

from tests.test_app import MARKER, Base, chat  # sets the test environment first

from app import grounding, prompts, retrieval  # noqa: E402
from app.schemas import CATEGORIES, Message  # noqa: E402

CANDIDATE = "이 과정의 학점 예제에서는 85점 이상이면 A를 받아요. 따라서 85점은 A 조건을 만족해요."


def msgs(*turns):
    return [Message(**m) for m in chat(*turns)]


def verdict_json(category, harm_kind="none"):
    return json.dumps({"category": category, "harm_kind": harm_kind})


def ground_json(grounded, spans=()):
    return json.dumps({"grounded": grounded, "unsupported": list(spans)}, ensure_ascii=False)


class LocateSpansTests(unittest.TestCase):
    def test_exact_and_cleaned_spans(self):
        spans = ["85점 이상이면 A를 받아요", "“따라서 85점은 A 조건을 만족해요.”", "자료에 없는 말"]
        self.assertEqual(grounding.locate_spans(spans, CANDIDATE),
                         ["85점 이상이면 A를 받아요", "따라서 85점은 A 조건을 만족해요."])
        # End punctuation is trimmed only when that is what makes the span match.
        self.assertEqual(grounding.locate_spans(["A를 받아요!"], CANDIDATE), ["A를 받아요"])

    def test_nested_and_duplicate_spans(self):
        self.assertEqual(grounding.locate_spans(["85점 이상이면", "85점 이상이면 A를 받아요", "85점 이상이면"], CANDIDATE),
                         ["85점 이상이면 A를 받아요"])

    def test_cap_and_empty(self):
        self.assertEqual(grounding.locate_spans(["", "   "], CANDIDATE), [])
        many = ["이 과정의", "학점 예제에서는", "85점 이상이면", "A를 받아요", "따라서", "조건을"]
        self.assertEqual(len(grounding.locate_spans(many, CANDIDATE)), grounding.MAX_SPANS)


class GuardCandidateTests(Base):
    def guard(self, ground=None, category="normal", candidate=CANDIDATE):
        self.fake.json_script = [("text", verdict_json(category, "request"))]
        if ground is not None:
            self.fake.scripts["ground"] = ground
        return self.post("/guard", {"messages": chat(f"85점은 왜 A가 아니야 {MARKER}"),
                                    "candidate_answer": candidate}).json()

    def test_ungrounded_with_located_span(self):
        data = self.guard([("text", ground_json(False, ["85점 이상이면 A를 받아요", f"없는 말 {MARKER}"]))])
        self.assertEqual(data, {"input": {"decision": "allow", "category": "normal"},
                                "output": {"grounded": False, "unsupported": ["85점 이상이면 A를 받아요"],
                                           "safe": True, "category": "normal"}})

    def test_located_span_overrides_grounded_true(self):
        data = self.guard([("text", ground_json(True, ["85점 이상이면 A를 받아요"]))])
        self.assertEqual((data["output"]["grounded"], data["output"]["unsupported"]), (False, ["85점 이상이면 A를 받아요"]))

    def test_ungrounded_without_usable_span_marks_the_whole_answer(self):
        for spans in (["후보에 없는 구절"], []):
            with self.subTest(spans=spans):
                data = self.guard([("text", ground_json(False, spans))])
                self.assertEqual((data["output"]["grounded"], data["output"]["unsupported"]), (False, [CANDIDATE]))

    def test_grounded(self):
        data = self.guard([("text", ground_json(True))])
        self.assertEqual((data["output"]["grounded"], data["output"]["unsupported"]), (True, []))

    def test_judge_failure_keeps_the_safe_default(self):
        data = self.guard([("text", "not json"), ("raise", RuntimeError(MARKER))])
        self.assertEqual(data["output"], {"grounded": True, "unsupported": [], "safe": True, "category": "normal"})
        self.assertTrue(any("degraded=ground" in l for l in self.request_lines()))

    def test_blocked_input_skips_the_judge(self):
        data = self.guard(category="jailbreak_injection")
        self.assertEqual(data["output"], {"grounded": True, "unsupported": [], "safe": False, "category": "normal"})
        self.assertEqual(self.fake.calls_of("ground"), [])

    def test_empty_candidate_skips_the_judge(self):
        data = self.guard(candidate="")
        self.assertEqual(data["output"]["grounded"], True)
        self.assertEqual(self.fake.calls_of("ground"), [])

    def test_prompt_holds_materials_conversation_and_candidate(self):
        self.guard()
        call = self.fake.calls_of("ground")[0]
        system, user = call["messages"][0]["content"], call["messages"][1]["content"]
        self.assertNotIn(MARKER, system)
        self.assertIn("=== 섹션 ", user)
        self.assertIn(f"[후보 답변]\n{CANDIDATE}", user)
        self.assertIn(MARKER, user)
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertLessEqual(call["max_completion_tokens"], 2000)

    def test_judgment_is_never_logged(self):
        self.guard([("text", ground_json(False, ["85점 이상이면 A를 받아요"]))])
        joined = "\n".join(self.cap.lines)
        for word in ("grounded", "unsupported", "85점"):
            self.assertNotIn(word, joined)


class AgreedSpanTests(unittest.TestCase):
    def test_overlap_of_two_judges(self):
        first = ["따라서 85점은 A 조건을 만족해요."]
        second = ["85점은 A 조건을 만족해요"]
        self.assertEqual(grounding.agreed_spans(first, second, CANDIDATE), ["85점은 A 조건을 만족해요"])

    def test_each_first_span_keeps_its_best_overlap(self):
        first = ["85점 이상이면 A를 받아요", "따라서 85점은 A 조건을 만족해요."]
        second = ["이상이면 A를 받아요.", "A 조건을 만족해요"]
        self.assertEqual(grounding.agreed_spans(first, second, CANDIDATE), ["이상이면 A를 받아요", "A 조건을 만족해요"])
        for span in grounding.agreed_spans(first, second, CANDIDATE):
            self.assertIn(span, CANDIDATE)

    def test_short_or_no_overlap_falls_back_to_the_first_judge(self):
        first = ["85점 이상이면 A를 받아요"]
        self.assertEqual(grounding.agreed_spans(first, ["따라서"], CANDIDATE), first)
        self.assertEqual(grounding.agreed_spans(first, ["A를"], CANDIDATE), first)  # 2 chars < minimum

    def test_whole_answer_fallback_meets_a_real_span(self):
        self.assertEqual(grounding.agreed_spans([CANDIDATE], ["85점 이상이면"], CANDIDATE), ["85점 이상이면"])


class JurySettingTests(unittest.TestCase):
    def test_server_default_is_three_judges(self):
        # Outside Base (which pins one judge); the test environment leaves GROUND_JUDGES unset.
        self.assertEqual(grounding.config.GROUND_JUDGES, 3)


class JuryTests(Base):
    """Three judges at once; the majority decides, a tie goes to the first judge."""

    def setUp(self):
        super().setUp()
        self.jury = mock.patch.object(grounding.config, "GROUND_JUDGES", 3)
        self.jury.start()

    def tearDown(self):
        self.jury.stop()
        super().tearDown()

    def judge(self, *scripts):
        self.fake.scripts["ground"] = list(scripts)
        return asyncio.run(grounding.judge_grounding(msgs(f"85점은 왜 A가 아니야 {MARKER}"), CANDIDATE, [],
                                                     deadline=grounding.Deadline(25)))

    OK = ("text", ground_json(True))
    BAD = ("text", "not json")

    def test_all_grounded(self):
        self.assertEqual(self.judge(self.OK, self.OK, self.OK), (True, []))
        self.assertEqual(len(self.fake.calls_of("ground")), 3)

    def test_one_flag_out_of_three_stays_grounded(self):
        self.assertEqual(self.judge(self.OK, ("text", ground_json(False, ["85점 이상이면 A를 받아요"])), self.OK), (True, []))

    def test_two_flags_keep_what_they_agree_on(self):
        out = self.judge(("text", ground_json(False, ["따라서 85점은 A 조건을 만족해요."])), self.OK,
                         ("text", ground_json(False, ["85점은 A 조건을 만족해요"])))
        self.assertEqual(out, (False, ["85점은 A 조건을 만족해요"]))

    def test_three_flags_use_the_first_two(self):
        out = self.judge(("text", ground_json(False, ["85점 이상이면 A를 받아요"])),
                         ("text", ground_json(False, ["이상이면 A를 받아요."])),
                         ("text", ground_json(False, ["따라서 85점은 A 조건을 만족해요."])))
        self.assertEqual(out, (False, ["이상이면 A를 받아요"]))

    def test_tie_after_a_failed_judge_goes_to_the_first(self):
        # The fake never yields, so judge 1 (and its retry) run first, then judges 2 and 3.
        flag = ("text", ground_json(False, ["85점 이상이면 A를 받아요"]))
        self.assertEqual(self.judge(self.BAD, self.BAD, flag, self.OK), (False, ["85점 이상이면 A를 받아요"]))
        self.fake.calls.clear()
        self.assertEqual(self.judge(self.BAD, self.BAD, self.OK, flag), (True, []))

    def test_all_judges_failing_returns_none(self):
        self.assertIsNone(self.judge(*([self.BAD] * 6)))

    def test_judges_run_at_the_same_time(self):
        start = time.monotonic()
        self.judge(("sleep", 0.3), ("sleep", 0.3), ("sleep", 0.3))
        self.assertLess(time.monotonic() - start, 0.55)

    def test_two_agreeing_judges_do_not_wait_for_the_third(self):
        start = time.monotonic()
        self.assertEqual(self.judge(self.OK, self.OK, ("sleep", 5.0)), (True, []))
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertTrue(any("kind=ground status=cancelled" in line for line in self.cap.lines))

    def test_two_agreeing_flags_do_not_wait_for_the_third(self):
        start = time.monotonic()
        out = self.judge(("text", ground_json(False, ["따라서 85점은 A 조건을 만족해요."])),
                         ("text", ground_json(False, ["85점은 A 조건을 만족해요"])), ("sleep", 5.0))
        self.assertEqual(out, (False, ["85점은 A 조건을 만족해요"]))
        self.assertLess(time.monotonic() - start, 1.0)

    def test_a_split_waits_for_the_third(self):
        start = time.monotonic()
        out = self.judge(("text", ground_json(False, ["85점 이상이면 A를 받아요"])), self.OK,
                         ("delay", (0.3, ground_json(False, ["이상이면 A를 받아요."]))))
        self.assertEqual(out, (False, ["이상이면 A를 받아요"]))
        self.assertGreaterEqual(time.monotonic() - start, 0.3)

    def test_spans_pair_judges_by_launch_order(self):
        # The first judge answers last. The two flags share too little, so the first flagging
        # judge's spans are kept: that is judge 1 by launch order (judge 3 by finishing order).
        out = self.judge(("delay", (0.2, ground_json(False, ["85점 이상이면 A를 받아요"]))), self.OK,
                         ("text", ground_json(False, ["따라서 85점은 A 조건을 만족해요."])))
        self.assertEqual(out, (False, ["85점 이상이면 A를 받아요"]))

    def test_two_judge_tie_goes_to_the_first(self):
        with mock.patch.object(grounding.config, "GROUND_JUDGES", 2):
            flag = ("text", ground_json(False, ["85점 이상이면 A를 받아요"]))
            self.assertEqual(self.judge(self.OK, flag), (True, []))
            self.assertEqual(self.judge(flag, self.OK), (False, ["85점 이상이면 A를 받아요"]))

    def test_guard_route_uses_the_jury(self):
        self.fake.json_script = [("text", verdict_json("normal"))]
        self.fake.scripts["ground"] = [("text", ground_json(False, ["따라서 85점은 A 조건을 만족해요."])),
                                       ("text", ground_json(False, ["85점은 A 조건을 만족해요"])), self.OK]
        data = self.post("/guard", {"messages": chat(f"85점은 왜 A가 아니야 {MARKER}"), "candidate_answer": CANDIDATE}).json()
        self.assertEqual((data["output"]["grounded"], data["output"]["unsupported"]), (False, ["85점은 A 조건을 만족해요"]))
        self.assertEqual(len(self.fake.calls_of("ground")), 3)
        self.assertFalse(any("85점" in line for line in self.cap.lines))


class GuardMaterialTests(Base):
    def guard(self, category="normal", found=None, turns=(f"다람쥐 통장의 목표 잠금 기능은 뭐야 {MARKER}",)):
        self.fake.json_script = [("text", verdict_json(category))]
        if found is not None:
            self.fake.scripts["found"] = found
        return self.post("/guard", {"messages": chat(*turns)}).json()

    def test_not_in_material_becomes_out_of_material(self):
        data = self.guard(found=[("text", '{"found": false}')])
        self.assertEqual(data, {"input": {"decision": "allow", "category": "out_of_material"}})
        self.assertNotIn("out_of_material", "\n".join(self.cap.lines))

    def test_in_material_stays_normal(self):
        self.assertEqual(self.guard(found=[("text", '{"found": true}')])["input"]["category"], "normal")

    def test_check_failure_stays_normal(self):
        data = self.guard(found=[("text", "x"), ("text", "y")])
        self.assertEqual(data["input"], {"decision": "allow", "category": "normal"})
        self.assertTrue(any("degraded=found" in l for l in self.request_lines()))

    def test_only_normal_inputs_are_checked(self):
        for category in ("pii_disclosure", "answer_seeking", "jailbreak_injection"):
            with self.subTest(category=category):
                self.fake.calls.clear()
                self.guard(category=category)
                self.assertEqual(self.fake.calls_of("found"), [])

    def test_found_prompt_shares_the_tutor_rules(self):
        self.guard()
        call = self.fake.calls_of("found")[0]
        self.assertIn(prompts.FOUND_RULES, call["messages"][0]["content"])
        self.assertIn(prompts.FOUND_RULES, prompts.tutor_system("answer"))
        self.assertIn("[마지막 학습자 발화]", call["messages"][1]["content"])


class EnglishRetrievalTests(Base):
    QUESTION = "what is the minimum amount to open the squirrel dream deposit"

    def test_detection(self):
        self.assertTrue(retrieval.is_english(self.QUESTION))
        self.assertFalse(retrieval.is_english("야구 점수판 example의 기록 data 범위가 뭐야"))
        self.assertFalse(retrieval.is_english("print"))

    def test_translation_is_added_to_the_bm25_query(self):
        self.fake.scripts["translate"] = [("text", "다람쥐 꿈모으기 정기예금 최소 가입 금액")]
        hits = asyncio.run(retrieval.retrieve(msgs(self.QUESTION)))
        self.assertEqual(hits[0].section_id, "FIN-02-04")
        call = self.fake.calls_of("translate")[0]
        self.assertIn("다람쥐 틴즈 체크카드", call["messages"][0]["content"])  # glossary
        self.assertEqual(call["messages"][1]["content"], self.QUESTION)
        self.assertEqual(call["extra_body"], {"reasoning_effort": "minimal"})

    def test_translation_failure_keeps_bm25(self):
        self.fake.scripts["translate"] = [("raise", RuntimeError(MARKER))]
        self.assertEqual(len(asyncio.run(retrieval.retrieve(msgs(self.QUESTION)))), 5)

    def test_korean_question_is_not_translated(self):
        asyncio.run(retrieval.retrieve(msgs("다람쥐 꿈모으기 정기예금 최소 가입 금액")))
        self.assertEqual(self.fake.calls_of("translate"), [])

    def test_extra_text_steers_retrieval(self):
        hits = asyncio.run(retrieval.retrieve(msgs("이거 맞아?"), extra="다람쥐 꿈모으기 정기예금 최소 가입 금액은 10만 원"))
        self.assertEqual(hits[0].section_id, "FIN-02-04")


class GlossaryTests(unittest.TestCase):
    def test_glossary_comes_from_the_kb(self):
        lines = retrieval.GLOSSARY.split("\n")
        for name in ("- 다람쥐 틴즈 체크카드", "- 도토리 포인트", "- 꿈틀 중학교 코딩 동아리", "- 다람쥐은행"):
            self.assertIn(name, lines)
        self.assertNotIn("- 다람쥐은행은", lines)
        self.assertLess(len(retrieval.GLOSSARY), 2000)

    def test_categories_unchanged(self):
        self.assertIn("out_of_material", CATEGORIES)


if __name__ == "__main__":
    unittest.main()
