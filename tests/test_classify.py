"""Offline tests for the classifier, policy, templates and few-shot loading (fake LLM, no network).

Run: python3 -m unittest discover -s tests -v
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_app import AUTH, MARKER, Base, chat  # sets the test environment first

from app import classify, fewshot, generate, policy, prompts, templates  # noqa: E402
from app.deadline import Deadline  # noqa: E402
from app.schemas import CATEGORIES, Message  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The exact texts specified for task 1 (copied from the task description).
EXPECTED_TEXTS = {
    "REFUSE_JAILBREAK": "이전 지시를 무시하거나 제한 없이 답하는 역할을 맡을 수는 없고, 숨겨진 지시도 보여 드릴 수 없어요. 수업 내용은 계속 도와드릴게요.",
    "REFUSE_HARMFUL": "다른 사람을 속이거나 해치는 일, 다른 사람의 개인정보를 알아내거나 퍼뜨리는 일은 도와드릴 수 없어요. 안전하게 대처하는 방법이나 수업 내용은 함께 공부할 수 있어요.",
    "REDIRECT_STATEMENT": "특정 집단이나 사람을 깎아내리는 표현은 사용하지 말아 주세요. 지금 공부하던 내용은 계속 도와드릴게요.",
    "REDIRECT_OFF_TOPIC": "그 요청은 이 수업과 관련이 없어요. 수업에서 배운 내용 가운데 궁금한 점을 물어봐 주시면 함께 살펴볼게요.",
    "ESCALATE": "그동안 혼자 버티느라 많이 힘들었겠어요. 지금은 공부를 잠시 멈추고, 믿을 수 있는 선생님이나 보호자에게 지금 마음을 이야기해 주세요. 학교 상담 선생님이나 청소년상담전화 1388에 도움을 요청해도 좋아요.",
    "CLARIFY": "무엇이 궁금한지 조금 더 알려 줄래요? 어떤 실습이나 개념에 관한 질문인지 말해 주면 함께 살펴볼게요.",
    "SCAFFOLD_ACADEMIC": "제출할 답안을 대신 완성해 줄 수는 없어요. 대신 직접 쓸 수 있게 순서를 안내할게요. 먼저 자료에서 문제와 관련된 개념을 찾아 한 문장으로 정리해 보세요. 그다음 그 개념을 문제 상황에 적용해 단계별로 적어 보고, 쓴 내용을 보여 주면 함께 살펴볼게요.",
    "SCAFFOLD_ANSWER_SEEKING_TEMP": "정답을 바로 알려 주기보다 스스로 고를 수 있게 도와드릴게요. 먼저 문제에서 묻는 값과 조건을 찾고, 자료에서 그 조건에 해당하는 규칙을 확인해 보세요. 규칙과 문제의 값을 비교하면 답을 고를 수 있어요.",
    "PII_NOTICE": "이름이나 학교 같은 개인정보는 적지 않아도 괜찮아요.",
}

# category -> (action, template name) for the tutor; harmful is split by harm_kind.
EXPECTED_ROUTES = {
    ("normal", "none"): ("answer", None),
    ("pii_disclosure", "none"): ("answer", None),
    ("answer_seeking", "none"): ("scaffold", "SCAFFOLD_ANSWER_SEEKING_TEMP"),
    ("academic_dishonesty", "none"): ("scaffold", "SCAFFOLD_ACADEMIC"),
    ("jailbreak_injection", "none"): ("refuse", "REFUSE_JAILBREAK"),
    ("off_topic", "none"): ("redirect", "REDIRECT_OFF_TOPIC"),
    ("emotional_distress", "none"): ("escalate", "ESCALATE"),
    ("low_signal_input", "none"): ("clarify", "CLARIFY"),
    ("harmful_content", "request"): ("refuse", "REFUSE_HARMFUL"),
    ("harmful_content", "statement"): ("redirect", "REDIRECT_STATEMENT"),
    ("harmful_content", "none"): ("refuse", "REFUSE_HARMFUL"),
}


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def verdict_json(category, harm_kind="none"):
    return json.dumps({"category": category, "harm_kind": harm_kind})


class TemplateAndPolicyTests(unittest.TestCase):
    def test_templates_match_the_spec_exactly(self):
        for name, text in EXPECTED_TEXTS.items():
            self.assertEqual(getattr(templates, name), text, name)
            self.assertLessEqual(len(text), 4000)
            self.assertNotIn("{", text)  # no placeholders for learner input

    def test_every_category_has_a_route_and_decision(self):
        for category in CATEGORIES:
            policy.tutor_route(category, "none")
            self.assertIn(policy.decision(category), ("allow", "block"))

    def test_routes(self):
        for (category, harm_kind), (action, name) in EXPECTED_ROUTES.items():
            route = policy.tutor_route(category, harm_kind)
            self.assertEqual(route.action, action, category)
            self.assertEqual(route.template, EXPECTED_TEXTS[name] if name else None, category)

    def test_guard_decision_matches_public_labels(self):
        for item in read_jsonl(ROOT / "public/guard.jsonl"):
            self.assertEqual(policy.decision(item["labels"]["category"]), item["labels"]["decision"], item["id"])

    def test_tutor_actions_match_public_labels(self):
        for item in read_jsonl(ROOT / "public/tutor.jsonl"):
            category, action = item["labels"]["category"], item["labels"]["action"]
            if category == "out_of_material":
                continue  # task 2
            harm_kind = {"refuse": "request", "redirect": "statement"}.get(action, "none")
            self.assertEqual(policy.tutor_route(category, harm_kind).action, action, item["id"])


class LowSignalRuleTests(unittest.TestCase):
    def test_public_set_has_no_false_positive_or_negative(self):
        for endpoint in ("tutor", "guard"):
            for item in read_jsonl(ROOT / f"public/{endpoint}.jsonl"):
                last = [m["content"] for m in item["messages"] if m["role"] == "user"][-1]
                expected = item["labels"]["category"] == "low_signal_input"
                self.assertEqual(classify.is_low_signal(last), expected, item["id"])

    def test_edge_cases(self):
        for text in ("", "   \n", "ㅋ", "ㅠㅠ", "?!?!", "...", "ㅁㄴㅇㄹ", "ㅋ" * 12, "​"):
            self.assertTrue(classify.is_low_signal(text), repr(text))
        for text in ("asdf", "ㅇㅋ 1", "a", "ㅋ" * 13, "뭐", "ㅠㅠ 왜"):
            self.assertFalse(classify.is_low_signal(text), repr(text))


class FewShotTests(unittest.TestCase):
    def test_shots_come_only_from_the_fewshot_split(self):
        split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
        ids = [s.id for s in classify.SHOTS]
        self.assertTrue(12 <= len(ids) <= 20)
        self.assertTrue(set(ids) <= set(split["fewshot"]))
        self.assertFalse(set(ids) & set(split["dev"]))

    def test_shot_labels(self):
        for shot in classify.SHOTS:
            self.assertNotIn(shot.category, ("out_of_material", "low_signal_input"))
            if shot.category == "harmful_content":
                self.assertIn(shot.harm_kind, ("request", "statement"))
            else:
                self.assertEqual(shot.harm_kind, "none")

    def test_dev_id_blocks_start_up(self):
        split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "shots.json"
            bad.write_text(json.dumps({"shots": [{"id": split["dev"][0]}]}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fewshot.load_shots(shots_file=bad)
            unknown = Path(d) / "unknown.json"
            unknown.write_text(json.dumps({"shots": [{"id": "pub-nope"}]}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fewshot.load_shots(shots_file=unknown)

    def test_prompt_shape(self):
        prompt = classify.build_prompt([Message(role="user", content=f"질문 {MARKER}")])
        self.assertEqual(prompt[0]["role"], "system")
        roles = [m["role"] for m in prompt[1:]]
        self.assertEqual(roles, ["user", "assistant"] * len(classify.SHOTS) + ["user"])
        for m in prompt[1:-1]:
            if m["role"] == "assistant":
                classify.Verdict.model_validate_json(m["content"])
        self.assertIn(MARKER, prompt[-1]["content"])
        self.assertIn("[마지막 학습자 발화]", prompt[-1]["content"])


class RenderTests(unittest.TestCase):
    def test_long_turn_keeps_the_decisive_tail(self):
        text = "붙여 넣은 글 " + "가나다라 " * 300 + "끝에 있는 실제 요청"
        out = prompts.render([{"role": "user", "content": text}], **prompts.SHOT_LIMITS)
        self.assertIn("끝에 있는 실제 요청", out)
        self.assertIn("붙여 넣은 글", out)
        self.assertLess(len(out), 600)

    def test_history_is_marked_and_last_user_turn_is_separate(self):
        msgs = [{"role": "user", "content": "앞 질문"}, {"role": "assistant", "content": "앞 답"},
                {"role": "user", "content": "그럼 그거 알려 줘"}]
        out = prompts.render(msgs, **prompts.LIVE_LIMITS)
        self.assertEqual(out.split("\n"), ["[이전 대화]", "학습자: 앞 질문", "튜터: 앞 답", "",
                                           "[마지막 학습자 발화]", "그럼 그거 알려 줘"])

    def test_single_turn(self):
        out = prompts.render([{"role": "user", "content": "질문"}], **prompts.LIVE_LIMITS)
        self.assertTrue(out.startswith("[이전 대화]\n(없음)"))


class ClassifierTests(Base):
    def run_classify(self, *turns):
        msgs = [Message(**m) for m in chat(*turns)]
        return asyncio.run(classify.classify(msgs, Deadline(25)))

    def test_low_signal_skips_the_llm(self):
        result = self.run_classify("질문", "답", "ㅠㅠ")
        self.assertEqual((result.verdict.category, result.source), ("low_signal_input", "rule"))
        self.assertEqual(self.fake.calls, [])

    def test_llm_verdict(self):
        self.fake.json_script = [("text", verdict_json("harmful_content", "statement"))]
        result = self.run_classify(f"질문 {MARKER}")
        self.assertEqual((result.verdict.category, result.verdict.harm_kind, result.source),
                         ("harmful_content", "statement", "llm"))
        call = self.fake.json_calls()[0]
        self.assertEqual(call["extra_body"], {"reasoning_effort": "low"})
        self.assertLessEqual(call["timeout"], 10)

    def test_out_of_material_becomes_normal_and_harm_kind_is_dropped(self):
        self.fake.json_script = [("text", verdict_json("out_of_material", "request"))]
        self.assertEqual(self.run_classify("질문").verdict.model_dump(), {"category": "normal", "harm_kind": "none"})

    def test_extra_fields_from_the_model_are_ignored(self):
        self.fake.json_script = [("text", '{"category": "off_topic", "harm_kind": "none", "reason": "' + MARKER + '"}')]
        self.assertEqual(self.run_classify("질문").verdict.category, "off_topic")

    def test_off_spec_harm_kind_is_tolerated(self):
        self.fake.json_script = [("text", '{"category": "harmful_content", "harm_kind": "' + MARKER + '"}')]
        result = self.run_classify("질문")
        self.assertEqual((result.verdict.category, result.verdict.harm_kind), ("harmful_content", "none"))
        self.assertEqual(len(self.fake.json_calls()), 1)

    def test_validation_error_logs_field_names_only(self):
        self.fake.json_script = [("text", '{"category": "' + MARKER + '"}'), ("text", verdict_json("normal"))]
        self.run_classify("질문")
        self.assertTrue(any("error=ValidationError[category]" in l for l in self.cap.lines))

    def test_invalid_twice_falls_back_to_normal(self):
        self.fake.json_script = [("text", '{"category": "evil"}'), ("raise", RuntimeError(MARKER))]
        result = self.run_classify("질문")
        self.assertEqual((result.verdict.category, result.source), ("normal", "fallback"))
        self.assertEqual(len(self.fake.json_calls()), 2)

    def test_retry_succeeds(self):
        self.fake.json_script = [("text", "not json"), ("text", verdict_json("jailbreak_injection"))]
        self.assertEqual(self.run_classify("질문").verdict.category, "jailbreak_injection")

    def test_classifier_timeout_is_capped_by_the_request_budget(self):
        msgs = [Message(**m) for m in chat("질문")]
        with mock.patch("app.config.DEADLINE_MARGIN_S", 0.0):
            asyncio.run(classify.classify(msgs, Deadline(3.0)))
        self.assertLessEqual(self.fake.json_calls()[0]["timeout"], 3.0)


class TutorRoutingTests(Base):
    def test_template_routes_return_exact_text_without_generation(self):
        for (category, harm_kind), (action, name) in EXPECTED_ROUTES.items():
            if name is None or category == "low_signal_input":
                continue
            with self.subTest(category=category, harm_kind=harm_kind):
                self.fake.calls.clear()
                self.fake.json_script = [("text", verdict_json(category, harm_kind))]
                data = self.post("/tutor", {"messages": chat(f"요청 {MARKER}")}).json()
                self.assertEqual(data, {"answer": EXPECTED_TEXTS[name], "action": action, "citations": []})
                self.assertEqual(self.fake.gen_calls(), [])

    def test_low_signal_route(self):
        data = self.post("/tutor", {"messages": chat("질문", "답", "ㅋㅋㅋ")}).json()
        self.assertEqual(data, {"answer": EXPECTED_TEXTS["CLARIFY"], "action": "clarify", "citations": []})
        self.assertEqual(self.fake.calls, [])

    def test_pii_answer_adds_notice_and_instruction(self):
        self.fake.json_script = [("text", verdict_json("pii_disclosure"))]
        self.fake.script = [("text", "print는 화면에 값을 보여 줘요.")]
        data = self.post("/tutor", {"messages": chat(f"나는 꿈틀중 2학년 {MARKER}이야 print가 뭐야")}).json()
        self.assertEqual(data["action"], "answer")
        self.assertTrue(data["answer"].endswith("\n\n" + EXPECTED_TEXTS["PII_NOTICE"]))
        system = self.fake.gen_calls()[0]["messages"][0]["content"]
        self.assertIn(generate.PII_INSTRUCTION, system)

    def test_pii_answer_stays_within_4000_chars(self):
        self.fake.json_script = [("text", verdict_json("pii_disclosure"))]
        self.fake.script = [("text", "가" * 5000)]
        data = self.post("/tutor", {"messages": chat("나는 보리야 질문")}).json()
        self.assertLessEqual(len(data["answer"]), 4000)
        self.assertTrue(data["answer"].endswith(EXPECTED_TEXTS["PII_NOTICE"]))

    def test_normal_answer_has_no_notice(self):
        data = self.post("/tutor", {"messages": chat("print가 뭐야")}).json()
        self.assertNotIn(EXPECTED_TEXTS["PII_NOTICE"], data["answer"])
        self.assertNotIn(generate.PII_INSTRUCTION, self.fake.gen_calls()[0]["messages"][0]["content"])

    def test_classifier_failure_answers_normally_and_is_logged(self):
        self.fake.json_script = [("raise", RuntimeError(MARKER)), ("raise", RuntimeError(MARKER))]
        data = self.post("/tutor", {"messages": chat("print가 뭐야")}).json()
        self.assertEqual(data["action"], "answer")
        self.assertTrue(any("degraded=classify" in l for l in self.request_lines()))

    def test_verdict_is_never_logged(self):
        self.fake.json_script = [("text", verdict_json("jailbreak_injection"))]
        self.post("/tutor", {"messages": chat("질문")})
        joined = "\n".join(self.cap.lines)
        for category in CATEGORIES:
            self.assertNotIn(category, joined)


class GuardRoutingTests(Base):
    def test_block_categories(self):
        for category in ("academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"):
            with self.subTest(category=category):
                self.fake.json_script = [("text", verdict_json(category, "request"))]
                data = self.post("/guard", {"messages": chat("q"), "candidate_answer": f"a {MARKER}"}).json()
                self.assertEqual(data["input"], {"decision": "block", "category": category})
                self.assertEqual(data["output"], {"grounded": True, "unsupported": [], "safe": False, "category": "normal"})

    def test_allow_categories(self):
        for category in ("normal", "answer_seeking", "pii_disclosure", "emotional_distress"):
            with self.subTest(category=category):
                self.fake.json_script = [("text", verdict_json(category))]
                data = self.post("/guard", {"messages": chat("q"), "candidate_answer": "a"}).json()
                self.assertEqual(data["input"], {"decision": "allow", "category": category})
                self.assertEqual(data["output"]["safe"], True)

    def test_low_signal_without_candidate(self):
        data = self.post("/guard", {"messages": chat("ㅁㄴㅇㄹ")}).json()
        self.assertEqual(data, {"input": {"decision": "allow", "category": "low_signal_input"}})
        self.assertEqual(self.fake.calls, [])

    def test_classifier_failure_allows(self):
        self.fake.json_script = [("text", "x"), ("text", "y")]
        data = self.post("/guard", {"messages": chat("q"), "candidate_answer": "a"}).json()
        self.assertEqual(data["input"], {"decision": "allow", "category": "normal"})
        self.assertTrue(data["output"]["safe"])


if __name__ == "__main__":
    unittest.main()
