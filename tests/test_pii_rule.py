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


if __name__ == "__main__":
    unittest.main()
