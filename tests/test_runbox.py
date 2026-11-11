"""런박스 인계·지문·생략·실패 경계를 대역으로 검사할 코드를 준비한다."""

import json
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import httpx

from tests.test_foundation import SETTINGS
from tutor_core.identity import IDENTITY_VERSION, fingerprint_paths
from tools.runbox import check_live_server, inspect_git_delivery, main, restart_result


def live_result(instance="a", *, cached=True, embeddings=0, passed=True):
    health = {"instance_id": instance * 24, "retrieval": {"cache_state": "hit" if cached else "built"},
              "provider_counters": {"embedding_attempts": embeddings}}
    return {"passed": passed, "startup_health": health, "health": health,
            "live_posts_executed": True, "upstream_calls_observed": True}


def environment_result():
    return {"python310_syntax": True, "pip_check_passed": True, "required_packages_present": True,
            "missing_delivery_files": [], "kb_inventory_matches": True, "packages": [{"matches": True}],
            "runbox_runtime_matches": True, "runtime_sha256": "a" * 64, "delivery_sha256": "b" * 64,
            "git_delivery_complete": True, "git": {"commit": "c" * 40}}


class IdentityTests(unittest.TestCase):
    def test_windows_and_linux_line_endings_have_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_bytes("값 = 1\r\n# 한글 주석\r\n".encode("utf-8"))
            windows = fingerprint_paths([source], root)
            source.write_bytes("값 = 1\n# 한글 주석\n".encode("utf-8"))
            self.assertEqual(windows, fingerprint_paths([source], root))
            source.write_bytes("값 = 2\n# 한글 주석\n".encode("utf-8"))
            self.assertNotEqual(windows, fingerprint_paths([source], root))

    def test_external_path_is_rejected_before_content_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root.parent / "not-a-project-source.py"
            with patch.object(Path, "read_text", side_effect=AssertionError("내용 읽기 금지")) as read:
                with self.assertRaises(ValueError):
                    fingerprint_paths([external], root)
                read.assert_not_called()


class DeliveryTests(unittest.TestCase):
    def test_staged_but_uncommitted_source_is_not_cloneable_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "server.py", root / "tools" / "runbox.py"]

            def git(arguments, **kwargs):
                command = arguments[1:]
                if command[0] == "rev-parse":
                    output = ("a" * 40 + "\n").encode()
                elif command[0] == "ls-tree":
                    output = b"server.py\0"
                elif command[0] == "status":
                    output = b"A  tools/runbox.py\0"
                elif ".env" in command:
                    output = b""
                else:
                    output = b"server.py\0tools/runbox.py\0"
                return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr=b"")

            with patch("tools.runbox.delivery_paths", return_value=paths), patch("tools.runbox.subprocess.run", side_effect=git):
                result = inspect_git_delivery(root)
        self.assertEqual(result["untracked_delivery_files"], [])
        self.assertEqual(result["files_absent_from_commit"], ["tools/runbox.py"])
        self.assertFalse(result["working_tree_matches_commit"])
        self.assertFalse(result["git_delivery_complete"])

    def test_restart_requires_distinct_instances_and_cache_without_kb_rebuild(self):
        self.assertTrue(restart_result([live_result("a"), live_result("b")])["passed"])
        self.assertFalse(restart_result([live_result("a"), live_result("a")])["passed"])
        self.assertFalse(restart_result([live_result("a"), live_result("b", embeddings=1)])["passed"])
        self.assertFalse(restart_result([live_result("a")])["passed"])

    def test_mismatched_server_is_not_given_tutor_or_guard_requests(self):
        requests = []
        health = {"ready": True, "spec_version": "1.0", "implementation_stage": "IMP-001-7",
                  "structured_output": "json_schema", "runtime_sha256": "b" * 64,
                  "runtime_identity_version": IDENTITY_VERSION, "instance_id": "c" * 24,
                  "policy_ready": True, "input_policy_ready": True, "output_policy_ready": True,
                  "tutor_policy_ready": True, "retrieval_ready": True,
                  "provider_counters": {"chat_successes": 1}, "counters": {}}

        def transport(request):
            requests.append(request.method)
            return httpx.Response(200, json=health)

        with httpx.Client(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
            result = check_live_server(client, "a" * 64)
        self.assertFalse(result["passed"])
        self.assertFalse(result["live_posts_executed"])
        self.assertEqual(requests, ["GET"])


class RunboxCommandTests(unittest.TestCase):
    def test_local_environment_check_never_loads_credentials_or_verifies_deployment(self):
        saved = []
        with patch("tools.runbox.inspect_environment", return_value=environment_result()), \
             patch("tools.runbox.sources_unchanged", return_value=True), \
             patch("tools.runbox.inspect_git_delivery", return_value={"commit": "c" * 40, "git_delivery_complete": True}), \
             patch("tools.runbox.Settings.from_environment", side_effect=AssertionError("자격 증명 로딩 금지")) as settings, \
             patch("tools.runbox.write_report", side_effect=lambda report, path: saved.append(json.loads(json.dumps(report)))):
            code = main(["--local-only"])
        self.assertEqual(code, 0)
        settings.assert_not_called()
        self.assertFalse(saved[-1]["deployment_verified"])
        self.assertFalse(saved[-1]["api_attempted"])

    def test_restart_only_measurement_does_not_claim_tunnel_verified(self):
        saved = []
        with patch("tools.runbox.inspect_environment", return_value=environment_result()), \
             patch("tools.runbox.sources_unchanged", return_value=True), \
             patch("tools.runbox.inspect_git_delivery", return_value={"commit": "c" * 40, "git_delivery_complete": True}), \
             patch("tools.runbox.Settings.from_environment", return_value=SETTINGS), \
             patch("tools.runbox.httpx.Client"), \
             patch("tools.runbox.managed_server", side_effect=lambda *args: nullcontext()), \
             patch("tools.runbox.check_live_server", side_effect=[live_result("a"), live_result("b")]), \
             patch("tools.runbox.write_report", side_effect=lambda report, path: saved.append(json.loads(json.dumps(report)))):
            code = main(["--start-server"])
        self.assertEqual(code, 1)
        self.assertTrue(saved[-1]["restart"]["passed"])
        self.assertTrue(saved[-1]["requested_checks_passed"])
        self.assertFalse(saved[-1]["deployment_verified"])
        self.assertIn("tunnel", saved[-1]["missing_deployment_checks"])

    def test_failed_first_start_prevents_second_start_and_tunnel(self):
        saved = []
        with patch("tools.runbox.inspect_environment", return_value=environment_result()), \
             patch("tools.runbox.sources_unchanged", return_value=True), \
             patch("tools.runbox.inspect_git_delivery", return_value={"commit": "c" * 40, "git_delivery_complete": True}), \
             patch("tools.runbox.Settings.from_environment", return_value=SETTINGS), \
             patch("tools.runbox.httpx.Client"), \
             patch("tools.runbox.managed_server", side_effect=lambda *args: nullcontext()), \
             patch("tools.runbox.check_live_server", return_value=live_result(passed=False)) as probe, \
             patch("tools.runbox.write_report", side_effect=lambda report, path: saved.append(json.loads(json.dumps(report)))):
            code = main(["--start-server", "--tunnel-url", "https://example.tunnel.elice.io/proxy/8000"])
        self.assertEqual(code, 1)
        self.assertEqual(probe.call_count, 1)
        self.assertNotIn("tunnel", saved[-1])
        self.assertFalse(saved[-1]["deployment_verified"])

    def test_full_deployment_requires_final_sources_to_match(self):
        for stable in (True, False):
            with self.subTest(stable=stable):
                saved = []
                with patch("tools.runbox.inspect_environment", return_value=environment_result()), \
                     patch("tools.runbox.sources_unchanged", side_effect=[True, True, True, stable]), \
                     patch("tools.runbox.inspect_git_delivery", return_value={"commit": "c" * 40, "git_delivery_complete": True}), \
                     patch("tools.runbox.Settings.from_environment", return_value=SETTINGS), \
                     patch("tools.runbox.httpx.Client"), \
                     patch("tools.runbox.managed_server", side_effect=lambda *args: nullcontext()), \
                     patch("tools.runbox.check_live_server", side_effect=[live_result("a"), live_result("b"), live_result("c")]), \
                     patch("tools.runbox.write_report", side_effect=lambda report, path: saved.append(json.loads(json.dumps(report)))):
                    code = main(["--start-server", "--tunnel-url", "https://example.tunnel.elice.io/proxy/8000"])
                self.assertEqual(code, 0 if stable else 1)
                self.assertEqual(saved[-1]["deployment_verified"], stable)
                self.assertEqual(saved[-1]["source_unchanged"], stable)
                self.assertNotIn("example.tunnel.elice.io", json.dumps(saved[-1]))


if __name__ == "__main__":
    unittest.main()
