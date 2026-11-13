"""The hard cap answers with the fallback even when the request logic ignores cancellation."""
import asyncio
import os
import time
import unittest

os.environ.setdefault("TEAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

AUTH = {"Authorization": "Bearer test-token"}


async def stubborn(*_args):
    try:
        await asyncio.sleep(5)
    except asyncio.CancelledError:
        await asyncio.sleep(1)  # slow unwind: the reply must not wait for it
        raise


class HardCapTest(unittest.TestCase):
    def setUp(self):
        self.saved = (server.HARD_CAP, server.tutor_logic, server.guard_logic)
        server.HARD_CAP = 0.3
        server.tutor_logic = stubborn
        server.guard_logic = stubborn
        self.client = TestClient(server.app)

    def tearDown(self):
        server.HARD_CAP, server.tutor_logic, server.guard_logic = self.saved

    def test_tutor_falls_back_at_cap(self):
        t = time.time()
        r = self.client.post("/tutor", json={"messages": [{"role": "user", "content": "질문"}]}, headers=AUTH)
        self.assertLess(time.time() - t, 1.0)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["action"], "answer")

    def test_guard_falls_back_at_cap_with_output(self):
        t = time.time()
        r = self.client.post("/guard", json={"messages": [{"role": "user", "content": "질문"}], "candidate_answer": "답"},
                             headers=AUTH)
        self.assertLess(time.time() - t, 1.0)
        self.assertEqual(r.json()["input"], {"decision": "allow", "category": "normal"})
        self.assertTrue(r.json()["output"]["grounded"])


if __name__ == "__main__":
    unittest.main()
