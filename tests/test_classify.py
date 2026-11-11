"""Offline tests for the classifier, policy, templates and few-shot loading (fake LLM, no network).

Run: python3 -m unittest discover -s tests -v
"""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests.test_app import AUTH, MARKER, Base, chat, gen_json  # sets the test environment first

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
    "SCAFFOLD_ANSWER_SEEKING": "정답을 바로 알려 주기보다 스스로 고를 수 있게 도와드릴게요. 먼저 문제에서 묻는 값과 조건을 찾고, 자료에서 그 조건에 해당하는 규칙을 확인해 보세요. 규칙과 문제의 값을 비교하면 답을 고를 수 있어요.",
    "PII_NOTICE": "이름이나 학교 같은 개인정보는 적지 않아도 괜찮아요.",
}

# category -> (action, template name) for the tutor; harmful is split by harm_kind.
EXPECTED_ROUTES = {
    ("normal", "none"): ("answer", None),
    ("pii_disclosure", "none"): ("answer", None),
    ("answer_seeking", "none"): ("scaffold", None),  # generated; the template is its fallback
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
                continue  # decided by the generator (found=false -> abstain), see test_task2
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

    def test_hand_written_labels_block_start_up(self):
        # Labels must come from public/; an id file entry with a label is refused.
        with tempfile.TemporaryDirectory() as d:
            labelled = Path(d) / "labelled.json"
            labelled.write_text(json.dumps({"shots": [{"id": classify.SHOTS[0].id, "harm_kind": "request"}]}),
                                encoding="utf-8")
            with self.assertRaises(RuntimeError):
                fewshot.load_shots(shots_file=labelled)

    def test_guard_harmful_items_without_public_harm_kind_are_never_examples(self):
        public = fewshot._read_public()
        for shot in [*classify.SHOTS, *classify.POOL]:
            item = public[shot.id]
            if shot.category == "harmful_content":
                self.assertIn(item["labels"].get("action"), ("refuse", "redirect"), shot.id)

    def test_pool(self):
        split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
        ids = {s.id for s in classify.POOL}
        self.assertTrue(ids <= set(split["fewshot"]))
        self.assertFalse(ids & set(split["dev"]))
        self.assertFalse(ids & {s.id for s in classify.SHOTS})
        self.assertFalse({s.category for s in classify.POOL} & {"out_of_material", "low_signal_input"})
        self.assertGreater(len(classify.POOL), 100)

    def test_prompt_shape(self):
        msgs = [Message(role="user", content=f"그 동네 사람들은 다 빼야 하지 않아? {MARKER}")]
        prompt = classify.build_prompt(msgs)
        self.assertEqual(prompt[0]["role"], "system")
        roles = [m["role"] for m in prompt[1:]]
        n_dynamic = len(classify.similar_shots(msgs, classify.config.CLASSIFY_DYNAMIC_SHOTS))
        self.assertEqual(n_dynamic, classify.config.CLASSIFY_DYNAMIC_SHOTS)
        self.assertEqual(roles, ["user", "assistant"] * (len(classify.SHOTS) + n_dynamic) + ["user"])
        # Static shots first, then the dynamic ones with the most similar last.
        static = classify.build_shot_messages(classify.SHOTS)
        self.assertEqual(prompt[1:1 + len(static)], static)
        dynamic = classify.similar_shots(msgs, 6)
        self.assertEqual(dynamic[-1].id, "pub-57e7be7b")  # near-duplicate in the pool
        self.assertEqual(prompt[-3]["content"], prompts.render(dynamic[-1].messages, **prompts.SHOT_LIMITS))
        for m in prompt[1:-1]:
            if m["role"] == "assistant":
                classify.Verdict.model_validate_json(m["content"])
        self.assertIn(MARKER, prompt[-1]["content"])
        self.assertIn("[마지막 학습자 발화]", prompt[-1]["content"])


class DynamicShotTests(unittest.TestCase):
    def test_exclude_and_k(self):
        msgs = [Message(role="user", content="그 동네 사람들은 다 빼야 하지 않아?")]
        top = classify.similar_shots(msgs, 6)
        self.assertEqual(len(top), 6)
        without = classify.similar_shots(msgs, 6, exclude=frozenset({"pub-57e7be7b"}))
        self.assertNotIn("pub-57e7be7b", [s.id for s in without])
        self.assertEqual(classify.similar_shots(msgs, 0), [])

    def test_no_overlap_means_no_dynamic_shots(self):
        self.assertEqual(classify.similar_shots([Message(role="user", content="zzqx")], 6), [])

    def test_groups_cover_every_public_item(self):
        groups = fewshot.group_of()
        self.assertEqual(len(groups), 400)


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


class VotingSettingTests(unittest.TestCase):
    def test_server_default_is_a_three_call_majority(self):
        # Outside Base (which pins one call); the test environment leaves CLASSIFY_VOTES unset.
        self.assertEqual(classify.config.CLASSIFY_VOTES, 3)


class SelfIntroFormTests(unittest.TestCase):
    def test_self_introductions(self):
        for text in ("나는 푸른 중학교 2학년 하늘이야. 적금 이자율이 궁금해", "난 푸른중 1학년 별이야 이체 한도 알려 줘",
                     "나 푸른 중학교 3학년 2반 하늘인데 질문 있어", "저는 푸른고등학교 1학년 하늘입니다.",
                     "질문 있어\n나는 푸른 초등학교 6학년 별이야", "내 이름은 하늘이야 보고서 순서 알려 줘", "내이름은 별이에요"):
            with self.subTest(text=text):
                self.assertTrue(classify.states_own_identity(text))

    def test_other_sentences(self):
        for text in ("이건 중학교 2학년 수학이야", "나는 학생이야", "나는 하늘이야 질문 있어", "중학교 2학년 때 배웠어",
                     "그 친구 이름은 뭐야", "다람쥐 통장의 최소 금액은 얼마야", ""):
            with self.subTest(text=text):
                self.assertFalse(classify.states_own_identity(text))


class SelfIntroVerdictTests(Base):
    """A self-introduction turns a normal verdict into pii_disclosure and changes nothing else."""
    INTRO = "나는 푸른 중학교 2학년 하늘이야. 다람쥐 통장의 최소 금액은 얼마야"

    def run_classify(self, *turns):
        return asyncio.run(classify.classify([Message(**m) for m in chat(*turns)], Deadline(25)))

    def test_normal_becomes_pii(self):
        self.fake.json_script = [("text", verdict_json("normal"))]
        result = self.run_classify(self.INTRO)
        self.assertEqual((result.verdict.category, result.source), ("pii_disclosure", "llm"))

    def test_other_verdicts_stay(self):
        for category in ("jailbreak_injection", "academic_dishonesty", "off_topic", "answer_seeking", "emotional_distress"):
            with self.subTest(category=category):
                self.fake.json_script = [("text", verdict_json(category))]
                self.assertEqual(self.run_classify(self.INTRO).verdict.category, category)

    def test_only_the_last_turn_counts(self):
        self.fake.json_script = [("text", verdict_json("normal"))]
        self.assertEqual(self.run_classify(self.INTRO, "답변이에요", "다람쥐 통장은 뭐야").verdict.category, "normal")

    def test_failed_calls_stay_normal(self):
        self.fake.json_script = [("text", "x"), ("text", "y")]
        result = self.run_classify(self.INTRO)
        self.assertEqual((result.verdict.category, result.source), ("normal", "fallback"))

    def test_routes(self):
        self.fake.json_script = [("text", verdict_json("normal"))]
        data = self.post("/guard", {"messages": chat(self.INTRO)}).json()
        self.assertEqual(data["input"], {"decision": "allow", "category": "pii_disclosure"})
        self.fake.json_script = [("text", verdict_json("normal"))]
        data = self.post("/tutor", {"messages": chat(self.INTRO)}).json()
        self.assertEqual(data["action"], "answer")
        self.assertTrue(data["answer"].endswith(EXPECTED_TEXTS["PII_NOTICE"]))


class VotingTests(Base):
    """Majority vote: two calls at once, a third only when they disagree."""

    def setUp(self):
        super().setUp()
        self.votes = mock.patch.object(classify.config, "CLASSIFY_VOTES", 3)
        self.votes.start()

    def tearDown(self):
        self.votes.stop()
        super().tearDown()

    def run_classify(self, *turns, deadline=25):
        msgs = [Message(**m) for m in chat(*turns)]
        return asyncio.run(classify.classify(msgs, Deadline(deadline)))

    def test_agreement_takes_two_calls(self):
        self.fake.json_script = [("text", verdict_json("off_topic"))] * 2
        result = self.run_classify(f"질문 {MARKER}")
        self.assertEqual((result.verdict.category, result.source), ("off_topic", "llm"))
        self.assertEqual(len(self.fake.json_calls()), 2)

    def test_first_two_calls_run_at_the_same_time(self):
        self.fake.json_script = [("sleep", 0.3), ("sleep", 0.3)]
        start = time.monotonic()
        self.run_classify("질문")
        self.assertLess(time.monotonic() - start, 0.55)

    def test_disagreement_asks_a_third_and_takes_the_majority(self):
        self.fake.json_script = [("text", verdict_json("jailbreak_injection")), ("text", verdict_json("normal")),
                                 ("text", verdict_json("normal"))]
        result = self.run_classify("질문")
        self.assertEqual(result.verdict.category, "normal")
        self.assertEqual(len(self.fake.json_calls()), 3)
        self.fake.calls.clear()
        self.fake.json_script = [("text", verdict_json("jailbreak_injection")), ("text", verdict_json("normal")),
                                 ("text", verdict_json("jailbreak_injection"))]
        self.assertEqual(self.run_classify("질문").verdict.category, "jailbreak_injection")

    def test_harm_kind_disagreement_is_settled_by_the_third_call(self):
        self.fake.json_script = [("text", verdict_json("harmful_content", "request")),
                                 ("text", verdict_json("harmful_content", "statement")),
                                 ("text", verdict_json("harmful_content", "statement"))]
        result = self.run_classify("질문")
        self.assertEqual((result.verdict.category, result.verdict.harm_kind), ("harmful_content", "statement"))
        self.assertEqual(len(self.fake.json_calls()), 3)

    def test_normalized_verdicts_count_as_agreement(self):
        # out_of_material -> normal before voting, so this is agreement and needs no third call.
        self.fake.json_script = [("text", verdict_json("out_of_material", "request")), ("text", verdict_json("normal"))]
        self.assertEqual(self.run_classify("질문").verdict.model_dump(), {"category": "normal", "harm_kind": "none"})
        self.assertEqual(len(self.fake.json_calls()), 2)

    def test_three_different_answers(self):
        cases = [
            # category majority, harm_kind from its earliest call
            (["harmful_content/request", "normal/none", "harmful_content/statement"], ("harmful_content", "request")),
            # decision majority (two allow), earliest allow verdict
            (["harmful_content/request", "pii_disclosure/none", "normal/none"], ("pii_disclosure", "none")),
            # decision majority (two block)
            (["normal/none", "off_topic/none", "jailbreak_injection/none"], ("off_topic", "none")),
        ]
        for answers, expected in cases:
            with self.subTest(answers=answers):
                self.fake.calls.clear()
                self.fake.json_script = [("text", verdict_json(*a.split("/"))) for a in answers]
                v = self.run_classify("질문").verdict
                self.assertEqual((v.category, v.harm_kind), expected)

    def test_vote_ties_go_to_the_first_call(self):
        v = classify.vote([classify.Verdict(category="off_topic"), classify.Verdict(category="normal")])
        self.assertEqual(v.category, "off_topic")

    def test_one_failed_call_uses_the_other(self):
        bad = ("text", "not json")
        # The fake never yields, so call 1 and its retry run first (both bad), then call 2 (ok). No third call.
        self.fake.json_script = [bad, bad, ("text", verdict_json("academic_dishonesty"))]
        result = self.run_classify("질문")
        self.assertEqual((result.verdict.category, result.source), ("academic_dishonesty", "llm"))
        self.assertEqual(len(self.fake.json_calls()), 3)

    def test_all_calls_failing_falls_back_to_normal(self):
        self.fake.json_script = [("text", "not json")] * 4
        result = self.run_classify("질문")
        self.assertEqual((result.verdict.category, result.source), ("normal", "fallback"))
        self.assertEqual(len(self.fake.json_calls()), 4)

    def test_third_call_is_capped_by_the_request_budget(self):
        self.fake.json_script = [("text", verdict_json("off_topic")), ("text", verdict_json("normal"))]
        with mock.patch("app.config.DEADLINE_MARGIN_S", 0.0):
            self.run_classify("질문", deadline=3.0)
        self.assertEqual(len(self.fake.json_calls()), 3)
        self.assertTrue(all(c["timeout"] <= 3.0 for c in self.fake.json_calls()))

    def test_low_signal_still_skips_the_llm(self):
        self.assertEqual(self.run_classify("ㅠㅠ").verdict.category, "low_signal_input")
        self.assertEqual(self.fake.calls, [])

    def test_votes_are_not_logged(self):
        self.fake.json_script = [("text", verdict_json("off_topic")), ("text", verdict_json("normal")),
                                 ("text", verdict_json("normal"))]
        r = self.post("/guard", {"messages": chat(f"질문 {MARKER}")})
        self.assertEqual(r.json()["input"]["decision"], "allow")
        self.assertEqual(len(self.fake.json_calls()), 3)
        for line in self.cap.lines:
            self.assertNotIn("off_topic", line)
            self.assertNotIn("normal", line)

    def test_tutor_route_uses_the_vote(self):
        self.fake.json_script = [("text", verdict_json("off_topic")), ("text", verdict_json("off_topic"))]
        data = self.post("/tutor", {"messages": chat(f"게임 공략 {MARKER}")}).json()
        self.assertEqual((data["action"], data["answer"]), ("redirect", EXPECTED_TEXTS["REDIRECT_OFF_TOPIC"]))


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
        self.fake.script = [("text", gen_json("print는 화면에 값을 보여 줘요."))]
        data = self.post("/tutor", {"messages": chat(f"나는 꿈틀중 2학년 {MARKER}이야 print가 뭐야")}).json()
        self.assertEqual(data["action"], "answer")
        self.assertTrue(data["answer"].endswith("\n\n" + EXPECTED_TEXTS["PII_NOTICE"]))
        system = self.fake.gen_calls()[0]["messages"][0]["content"]
        self.assertIn(generate.PII_INSTRUCTION, system)

    def test_pii_answer_stays_within_4000_chars(self):
        self.fake.json_script = [("text", verdict_json("pii_disclosure"))]
        self.fake.script = [("text", gen_json("가" * 5000))]
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
