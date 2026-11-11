"""런박스 패키지·재시작·터널을 검사하며 환경변수와 메타데이터만 사용한다."""

import argparse
import ast
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from tools.evaluate import EvaluationFailure, managed_server, read_health, timestamp, write_report
from tools.release_checks import call_endpoint, counter_delta
from tutor_core.config import ConfigurationError, Settings
from tutor_core.identity import IDENTITY_VERSION, ROOT, delivery_fingerprint, delivery_paths, runtime_fingerprint, runtime_paths
from tutor_core.knowledge import load_knowledge
from tutor_core.llm import silence_transport_logs


def validate_tunnel_url(value: str) -> str:
    try:
        url = urlsplit(value)
        match = re.fullmatch(r"/proxy/([0-9]+)/?", url.path)
        valid = (url.scheme == "https" and url.hostname is not None and url.hostname.endswith(".tunnel.elice.io")
                 and url.port in (None, 443) and url.username is None and url.password is None
                 and not url.query and not url.fragment and match is not None
                 and 1024 <= int(match.group(1)) <= 65535 and int(match.group(1)) != 8080)
    except ValueError:
        valid = False
    if not valid:
        raise EvaluationFailure("invalid_runbox_tunnel_url")
    return value.rstrip("/") + "/"


def inspect_git_delivery(root: Path = ROOT) -> dict:
    names = [path.relative_to(root).as_posix() for path in delivery_paths(root)]

    def git(*arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False, timeout=15)
            return result.returncode, result.stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.TimeoutExpired):
            return -1, ""

    head_status, head = git("rev-parse", "--verify", "HEAD")
    head = head.strip()
    tracked_status, tracked_text = git("ls-files", "-z", "--", *names)
    committed_status, committed_text = git("ls-tree", "-r", "--full-tree", "--name-only", "-z", "HEAD", "--", *names)
    dirty_status, dirty_text = git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *names)
    credential_status, credential_text = git(
        "ls-files", "-z", "--", ".env", ".env.*", "submission.json",
        ":(glob)**/.env", ":(glob)**/.env.*", ":(glob)**/submission.json",
        "*.pem", "*.p12", "*.pfx", "id_rsa", "id_ed25519",
    )
    # 비밀값 파일은 이름과 추적 여부만 확인하며 내용을 열지 않는다.
    forbidden = sorted(set(filter(None, credential_text.split("\0"))))
    untracked = sorted(set(names) - set(tracked_text.split("\0")))
    uncommitted = sorted(set(names) - set(committed_text.split("\0")))
    available = all(code == 0 for code in (head_status, tracked_status, committed_status, dirty_status, credential_status))
    commit = head if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) else None
    return {"metadata_available": available, "commit": commit,
            "untracked_delivery_files": untracked, "files_absent_from_commit": uncommitted,
            "working_tree_matches_commit": available and not dirty_text,
            "known_secret_paths_tracked": forbidden,
            "git_delivery_complete": available and commit is not None and not untracked and not uncommitted
            and not dirty_text and not forbidden}


def inspect_environment(root: Path = ROOT) -> dict:
    paths = delivery_paths(root)
    if any(path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) for path in paths):
        raise EvaluationFailure("unsafe_delivery_path")
    package_rows = []
    for line in (root / "requirements-runbox.txt").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, expected = line.strip().split("==", 1)
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        package_rows.append({"package": name, "expected": expected, "installed": installed, "matches": installed == expected})
    try:
        pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, check=False, timeout=60)
        pip_passed = pip_check.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        pip_passed = False
    missing = [path.relative_to(root).as_posix() for path in paths if not path.is_file()]
    syntax_failures = []
    for path in paths:
        if path.suffix == ".py" and path.is_file():
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), feature_version=(3, 10))
            except SyntaxError:
                syntax_failures.append(path.relative_to(root).as_posix())
    knowledge = load_knowledge(root / "kb")
    git = inspect_git_delivery(root)
    required_packages = {"fastapi", "uvicorn", "openai", "pydantic", "httpx", "starlette"}
    kb_matches = len(knowledge.sections) == 48 and knowledge.table_count == 23 and knowledge.duplicate_html_sections == 4
    return {"python": platform.python_version(), "system": platform.system(), "architecture": platform.machine(),
            "reference_python": "3.10.14", "python_matches_reference": platform.python_version() == "3.10.14",
            "runbox_runtime_matches": sys.version_info[:2] == (3, 10) and platform.system() == "Linux"
            and platform.machine().lower() in ("x86_64", "amd64"),
            "python310_syntax": not syntax_failures, "syntax_failures": syntax_failures, "packages": package_rows,
            "required_packages_present": required_packages.issubset({row["package"] for row in package_rows}),
            "pip_check_passed": pip_passed, "missing_delivery_files": missing,
            "runtime_sha256": runtime_fingerprint(root), "runtime_identity_version": IDENTITY_VERSION,
            "delivery_sha256": delivery_fingerprint(root) if not missing else None,
            "kb_sections": len(knowledge.sections), "kb_tables": knowledge.table_count,
            "kb_duplicate_html_sections": knowledge.duplicate_html_sections, "kb_inventory_matches": kb_matches,
            "git": git, "git_delivery_complete": git["git_delivery_complete"]}


def check_live_server(client, expected_hash: str) -> dict:
    health = read_health(client)
    checks = {
        "matching_runtime": health.get("runtime_sha256") == expected_hash,
        "identity_format": health.get("runtime_identity_version") == IDENTITY_VERSION,
        "instance_identified": health.get("instance_id") is not None,
        "components_ready": all(health.get(name) for name in (
            "policy_ready", "retrieval_ready", "input_policy_ready", "output_policy_ready", "tutor_policy_ready")),
        "structured_probe_observed": health.get("provider_counters", {}).get("chat_successes", 0) > 0,
    }
    result = {"checks": checks, "passed": False, "startup_health": health, "health": health,
              "live_posts_executed": False, "upstream_calls_observed": False}
    if not all(checks.values()):
        result["stopped_reason"] = "server_identity_or_readiness_mismatch"
        return result
    missing = client.get("/healthz", headers={"Authorization": ""})
    wrong = client.get("/healthz", headers={"Authorization": "Basic invalid-runbox-check"})
    messages = [{"role": "user", "content": "파이썬 반복문의 역할을 짧게 설명해 주세요."}]
    tutor, tutor_elapsed, _ = call_endpoint(client, "tutor", messages)
    guard, guard_elapsed, _ = call_endpoint(client, "guard", messages, candidate="", has_candidate=True)
    payload = {"spec_version": "1.0", "session_id": "runbox-check", "messages": messages}
    trailing = [client.post(path, json=payload) for path in ("/tutor/", "/guard/")]
    after = read_health(client)
    try:
        degraded = counter_delta(health, after)
        stable = True
    except EvaluationFailure:
        degraded, stable = None, False
    observed = stable and after.get("provider_counters", {}).get("chat_attempts", 0) > health.get("provider_counters", {}).get("chat_attempts", 0)
    checks.update({
        "missing_auth_rejected": missing.status_code == 401,
        "wrong_auth_rejected": wrong.status_code == 401,
        "no_trailing_slash_redirect": all(response.status_code == 404 and "location" not in response.headers for response in trailing),
        "tutor_json": tutor is not None and 0 < len(tutor.answer) <= 4000,
        "tutor_action": tutor is not None and tutor.action == "answer",
        "guard_input": guard is not None and guard.input.category == "normal" and guard.input.decision == "allow",
        "guard_empty_output": guard is not None and guard.output is not None
        and guard.output.grounded and guard.output.safe and not guard.output.unsupported,
        "same_instance": stable, "matching_runtime_after": after.get("runtime_sha256") == expected_hash
        and after.get("runtime_identity_version") == IDENTITY_VERSION,
        "no_degraded_responses": degraded == 0, "upstream_calls_observed": observed,
        "under_30_seconds": tutor_elapsed < 30 and guard_elapsed < 30,
    })
    result.update(passed=all(checks.values()), live_posts_executed=True, upstream_calls_observed=observed,
                  tutor_elapsed_seconds=round(tutor_elapsed, 3), guard_elapsed_seconds=round(guard_elapsed, 3), health=after)
    return result


def sources_unchanged(environment: dict, root: Path = ROOT) -> bool:
    return (runtime_fingerprint(root) == environment["runtime_sha256"]
            and environment.get("delivery_sha256") is not None
            and delivery_fingerprint(root) == environment["delivery_sha256"])


def restart_result(runs: list[dict]) -> dict:
    complete = len(runs) == 2
    different = complete and all(run["startup_health"].get("instance_id") for run in runs)
    different = bool(different and runs[0]["startup_health"]["instance_id"] != runs[1]["startup_health"]["instance_id"])
    cached = complete and runs[1]["startup_health"].get("retrieval", {}).get("cache_state") == "hit"
    no_kb_rebuild = complete and runs[1]["startup_health"].get("provider_counters", {}).get("embedding_attempts", 0) == 0
    return {"runs": list(runs), "different_instances": different, "cache_reused": cached,
            "no_kb_embedding_rebuild": no_kb_rebuild,
            "passed": complete and different and cached and no_kb_rebuild and all(run["passed"] for run in runs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="비밀값 파일을 읽지 않는 런박스 재현 확인")
    parser.add_argument("--local-only", action="store_true", help="API 없이 현재 패키지·자료·Git 반입 상태만 확인")
    parser.add_argument("--start-server", action="store_true", help="전용 포트에서 두 번 기동해 재시작과 캐시를 확인")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tunnel-url", help="사용자 런박스의 https 터널 기본 주소, /proxy/8000까지 지정")
    parser.add_argument("--expected-commit", help="대조할 전체 Git 커밋 해시, 키나 토큰을 지정하지 않음")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or args.port == 8080:
        parser.error("사용 가능한 1024~65535 포트를 지정하고 8080은 제외하세요.")
    if args.local_only and (args.start_server or args.tunnel_url):
        parser.error("--local-only와 실제 서버 검사는 함께 사용할 수 없습니다.")
    if not args.local_only and not (args.start_server or args.tunnel_url):
        parser.error("--local-only, --start-server, --tunnel-url 중 검사 범위를 지정하세요.")
    if args.expected_commit is not None and not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", args.expected_commit):
        parser.error("--expected-commit에는 전체 Git 커밋 해시를 지정하세요.")
    silence_transport_logs()
    report = {"stage": "IMP-001-7", "started_at": timestamp(), "passed": False, "api_attempted": False,
              "api_mode": "offline" if args.local_only else "real", "scope": "local_environment_only" if args.local_only else "deployment_checks",
              "private_evaluation_used": False, "request_response_bodies_saved": False, "deployment_verified": False,
              "requested_checks_passed": False, "excluded_checks": ["public_quality", "concurrency64", "submission"],
              "requested_checks": {"local_only": args.local_only, "restart": args.start_server, "tunnel": bool(args.tunnel_url)}}
    environment = None

    def checkpoint(name):
        report["active_check"] = name
        report["updated_at"] = timestamp()
        write_report(report, ROOT / "output" / "tests" / "runbox-latest.json")

    try:
        target = validate_tunnel_url(args.tunnel_url) if args.tunnel_url else None
        checkpoint("environment")
        environment = inspect_environment()
        report["environment"] = environment
        preliminary = (environment["python310_syntax"] and environment["pip_check_passed"]
                       and environment["required_packages_present"] and not environment["missing_delivery_files"]
                       and environment["kb_inventory_matches"] and all(row["matches"] for row in environment["packages"]))
        report["environment_checks_passed"] = preliminary
        report["expected_commit_matches"] = args.expected_commit is None or environment["git"]["commit"] == args.expected_commit.lower()
        checkpoint("environment_complete")
        if not report["expected_commit_matches"]:
            raise EvaluationFailure("git_commit_mismatch")
        if args.local_only:
            report["requested_checks_passed"] = preliminary
        else:
            if not environment["runbox_runtime_matches"] or not preliminary:
                raise EvaluationFailure("runbox_environment_mismatch")
            settings = Settings.from_environment()
            expected = environment["runtime_sha256"]
            if args.start_server:
                runs = []
                report["restart"] = restart_result(runs)
                for index in range(2):
                    if not sources_unchanged(environment):
                        raise EvaluationFailure("deployment_source_changed")
                    checkpoint("restart_" + str(index + 1))
                    report["api_attempted"] = True
                    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=30, follow_redirects=False, trust_env=False,
                                      headers={"Authorization": "Bearer " + settings.team_token}) as client:
                        with managed_server(True, args.port, client):
                            result = check_live_server(client, expected)
                            runs.append(result)
                            report["restart"] = restart_result(runs)
                            checkpoint("restart_" + str(index + 1) + "_complete")
                    if not result["passed"]:
                        raise EvaluationFailure("restart_probe_failed")
                if not report["restart"]["passed"]:
                    raise EvaluationFailure("restart_or_cache_not_verified")
            if target is not None:
                if not sources_unchanged(environment):
                    raise EvaluationFailure("deployment_source_changed")
                checkpoint("tunnel")
                with httpx.Client(base_url=target, timeout=30, follow_redirects=False, trust_env=False,
                                  headers={"Authorization": "Bearer " + settings.team_token}) as client:
                    report["tunnel"] = check_live_server(client, expected)
                report["api_attempted"] = report["api_attempted"] or report["tunnel"]["live_posts_executed"]
                checkpoint("tunnel_complete")
            report["requested_checks_passed"] = ((not args.start_server or report.get("restart", {}).get("passed", False))
                                                  and (target is None or report.get("tunnel", {}).get("passed", False)))
        report["active_check"] = None
    except ConfigurationError as exc:
        report["error"] = {"type": "ConfigurationError", "detail": str(exc)}
    except EvaluationFailure as exc:
        report["error"] = {"type": "EvaluationFailure", "reason": str(exc)}
    except KeyboardInterrupt:
        report["error"] = {"type": "Interrupted"}
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__}
    if environment is not None:
        try:
            report["source_unchanged"] = sources_unchanged(environment)
            final_git = inspect_git_delivery()
            report["final_git"] = final_git
            report["same_commit"] = final_git["commit"] == environment["git"]["commit"]
            report["requested_checks_passed"] = (report["requested_checks_passed"] and report["source_unchanged"]
                                                  and report["same_commit"] and "error" not in report)
            report["deployment_verified"] = (not args.local_only and report["requested_checks_passed"]
                                             and environment["git_delivery_complete"] and final_git["git_delivery_complete"]
                                             and report.get("restart", {}).get("passed", False)
                                             and report.get("tunnel", {}).get("passed", False))
            report["passed"] = report["requested_checks_passed"] if args.local_only else report["deployment_verified"]
        except Exception as exc:
            report["passed"] = report["deployment_verified"] = False
            report["requested_checks_passed"] = False
            report["finalization_error"] = {"type": type(exc).__name__}
    report["missing_deployment_checks"] = [name for name in ("restart", "tunnel") if not report.get(name, {}).get("passed", False)]
    report["api_usage_confirmed"] = (report.get("tunnel", {}).get("upstream_calls_observed", False)
                                     or any(run.get("upstream_calls_observed", False) for run in report.get("restart", {}).get("runs", [])))
    report["finished_at"] = timestamp()
    write_report(report, ROOT / "output" / "tests" / "runbox-latest.json")
    print("요청한 범위 검사: " + ("통과" if report["requested_checks_passed"] else "확인 필요"), flush=True)
    print("실제 런박스 배포 검증: " + ("통과" if report["deployment_verified"] else "미완료"), flush=True)
    print("보고서: output/tests/runbox-latest.json", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
