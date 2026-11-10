"""Offline tests for the /guard knowledge-base checks and English-question retrieval.

Run: python3 -m unittest discover -s tests -t . -v
"""
import asyncio
import json
import unittest

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
