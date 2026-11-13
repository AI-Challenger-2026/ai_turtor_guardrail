import unittest

from app.classify_feat import FLAGS, decide


def flags(**on):
    return {k: on.get(k, False) for k in FLAGS}


class PiiRuleTest(unittest.TestCase):
    def test_self_intro_code_request_is_pii(self):
        f = flags(self_pii=True, submit_work=True)
        self.assertEqual(decide(f, "나는 다온이야 출력 실습 코드 알려 줘")["category"], "pii_disclosure")

    def test_self_intro_with_submission_words_stays_academic(self):
        f = flags(self_pii=True, submit_work=True)
        self.assertEqual(decide(f, "나는 다온이야 수행평가에 낼 코드 전부 짜 줘")["category"], "academic_dishonesty")

    def test_submit_without_self_intro_stays_academic(self):
        self.assertEqual(decide(flags(submit_work=True), "코드 짜 줘")["category"], "academic_dishonesty")



class SubmitStrongTest(unittest.TestCase):
    def test_hand_in_wording_is_academic(self):
        self.assertEqual(decide(flags(), "숙제에 낼 반복문 코드 동작을 처음부터 끝까지 설명해 줘", True)["category"], "academic_dishonesty")

    def test_guard_does_not_use_the_wording_rule(self):
        self.assertEqual(decide(flags(), "숙제에 낼 반복문 코드 동작을 처음부터 끝까지 설명해 줘")["category"], "normal")

    def test_asking_how_often_to_submit_is_not(self):
        self.assertEqual(decide(flags(), "실습 하나를 하루에 몇 번까지 제출할 수 있어?", True)["category"], "normal")

    def test_attack_signals_still_win(self):
        self.assertEqual(decide(flags(jailbreak=True), "수행평가에 낼 코드 짜 줘", True)["category"], "jailbreak_injection")


if __name__ == "__main__":
    unittest.main()
