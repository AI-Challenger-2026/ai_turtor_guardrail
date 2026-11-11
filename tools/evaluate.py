"""비밀값 파일을 읽지 않고 실제 API와 공개 세트의 검증 메타데이터만 기록한다."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from harness.score_local import load_kb_dir
from judge import FIELD, PROMPTS, SYSTEM, _valid
from scoring import JudgeError
from tutor_core.config import ConfigurationError, NON_STREAM_MAX_COMPLETION_TOKENS, Settings
from tutor_core.llm import ApiClients, ProviderFailure, RequestBudget, silence_transport_logs
from tutor_core.schemas import GuardResponse, TutorResponse
from tutor_core.validation import MAX_RESPONSE_BYTES


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output" / "tests" / "foundation-latest.json"
KST = timezone(timedelta(hours=9))
COUNTER_NAMES = {
    "requests", "degraded_responses", "chat_attempts", "chat_successes", "embedding_attempts", "embedding_successes",
    "upstream_status", "upstream_timeout", "upstream_connection", "upstream_error", "internal_error",
    "invalid_structured_response", "deadline_exceeded", "call_budget_exhausted",
    "empty_response", "incomplete_response", "model_refusal", "invalid_embedding",
    "provider_not_ready", "probe_mismatch",
    "invalid_output_evidence", "output_verification_error", "output_verifier_unavailable",
    "unverified_tutor_draft", "unverified_tutor_repair", "unsafe_draft", "tutor_error",
}
COUNTER_NAMES.update("upstream_http_" + str(code) for code in range(100, 600))
COUNTER_NAMES.update("upstream_parameter_" + name for name in (
    "response_format", "max_completion_tokens", "max_tokens", "reasoning_effort", "messages", "stream", "model",
))


class EvaluationFailure(RuntimeError):
    """본문이나 비밀값이 아닌 고정 실패 코드만 전달한다."""


def timestamp() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def source_fingerprint() -> str:
    paths = [ROOT / name for name in (
        "server.py", "requirements-runbox.txt", "harness/score_local.py", "grader/judge.py", "grader/scoring.py",
        "public/tutor.jsonl", "public/guard.jsonl",
    )]
    for folder in ("tutor_core", "tools", "tests"):
        paths.extend((ROOT / folder).glob("*.py"))
    for suffix in ("*.md", "*.csv", "*.html"):
        paths.extend((ROOT / "kb").glob("*/" + suffix))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def commit_id() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    value = result.stdout.strip()
    return value if re.fullmatch(r"[a-f0-9]{40}", value) else "unavailable"


def bounded_json(response: httpx.Response) -> Any:
    if response.status_code != 200:
        raise EvaluationFailure("http_status_" + str(response.status_code))
    if "application/json" not in response.headers.get("content-type", "").lower():
        raise EvaluationFailure("not_json_response")
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > MAX_RESPONSE_BYTES:
            raise EvaluationFailure("response_too_large")
    try:
        return json.loads(body)
    except (ValueError, UnicodeError):
        raise EvaluationFailure("invalid_json_response") from None


def read_health(client: httpx.Client) -> dict[str, Any]:
    with client.stream("GET", "/healthz") as response:
        value = bounded_json(response)
    if (
        not isinstance(value, dict) or value.get("ready") is not True
        or value.get("spec_version") != "1.0" or value.get("implementation_stage") not in tuple("IMP-001-" + str(index) for index in range(1, 8))
        or value.get("structured_output") not in ("json_schema", "json_object")
    ):
        raise EvaluationFailure("unexpected_server_state")
    cleaned = {
        "ready": True, "implementation_stage": value["implementation_stage"], "structured_output": value["structured_output"],
        "policy_ready": value.get("policy_ready") is True,
        "retrieval_ready": value.get("retrieval_ready") is True,
        "input_policy_ready": value.get("input_policy_ready") is True,
        "output_policy_ready": value.get("output_policy_ready") is True,
        "tutor_policy_ready": value.get("tutor_policy_ready") is True,
    }
    for name in ("counters", "provider_counters"):
        counters = value.get(name, {})
        if not isinstance(counters, dict):
            raise EvaluationFailure("invalid_health_counters")
        cleaned[name] = {
            key: count for key, count in counters.items()
            if key in COUNTER_NAMES and type(count) is int and count >= 0
        }
    runtime_hash = value.get("runtime_sha256")
    if isinstance(runtime_hash, str) and re.fullmatch(r"[0-9a-f]{64}", runtime_hash):
        cleaned["runtime_sha256"] = runtime_hash
    if value.get("runtime_identity_version") == "runtime-text-v1":
        cleaned["runtime_identity_version"] = value["runtime_identity_version"]
    instance_id = value.get("instance_id")
    if isinstance(instance_id, str) and re.fullmatch(r"[0-9a-f]{24}", instance_id):
        cleaned["instance_id"] = instance_id
    component_counters = COUNTER_NAMES | {"verified", "failed", "protected", "repairs", "unsafe_drafts", "invalid_output_evidence"}
    for name, version in (("output_policy", "output-v1"), ("tutor_policy", "tutor-v1")):
        component = value.get(name)
        if isinstance(component, dict) and component.get("version") == version:
            cleaned[name] = {"version": version, "counters": {
                key: count for key, count in component.get("counters", {}).items()
                if key in component_counters and type(count) is int and count >= 0
            }}
    if "retrieval" in value:
        search = value["retrieval"]
        if not isinstance(search, dict):
            raise EvaluationFailure("invalid_retrieval_state")
        summary = {key: search[key] for key in ("sections", "tables", "duplicate_html_sections", "embedding_dimensions")
                   if type(search.get(key)) is int and search[key] >= 0}
        fingerprint = search.get("kb_fingerprint")
        if isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            summary["kb_fingerprint"] = fingerprint
        if search.get("cache_state") in ("hit", "built", "write_unavailable", "not_used"):
            summary["cache_state"] = search["cache_state"]
        summary["dense_ready"] = search.get("dense_ready") is True
        cleaned["retrieval"] = summary
    if isinstance(value.get("input_policy"), dict):
        policy = value["input_policy"]
        if policy.get("version") in ("input-v1", "input-v2"):
            counters = policy.get("counters", {})
            cleaned["input_policy"] = {"version": policy["version"], "counters": {
                key: count for key, count in counters.items()
                if key in COUNTER_NAMES | {"requests", "verified", "failed", "evidence_expansions", "input_policy_error", "invalid_policy_evidence"}
                and type(count) is int and count >= 0
            }}
            if policy.get("intent_reasoning_effort") in ("low", "medium", "high"):
                cleaned["input_policy"]["intent_reasoning_effort"] = policy["intent_reasoning_effort"]
            if type(policy.get("timeout_seconds")) in (int, float) and 0 < policy["timeout_seconds"] < 30:
                cleaned["input_policy"]["timeout_seconds"] = policy["timeout_seconds"]
    return cleaned


async def check_provider(settings: Settings) -> dict[str, Any]:
    clients = ApiClients(settings)
    start = time.monotonic()
    try:
        await clients.initialize()
        vectors = await clients.embed(
            ["반복문은 같은 작업을 여러 번 수행하는 데 사용합니다."], RequestBudget.start(15), seconds=15,
        )
        return {
            "passed": True, "structured_output": clients.structured_mode,
            "embedding_dimensions": len(vectors[0]), "elapsed_seconds": round(time.monotonic() - start, 3),
            "counters": dict(clients.counters),
        }
    finally:
        await clients.close()


@contextmanager
def managed_server(enabled: bool, port: int, client: httpx.Client):
    process = None
    try:
        if enabled:
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    raise EvaluationFailure("test_port_in_use") from None
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
                 "--no-access-log", "--log-level", "warning"],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            until = time.monotonic() + 180
            while time.monotonic() < until:
                if process.poll() is not None:
                    raise EvaluationFailure("server_startup_failed")
                try:
                    read_health(client)
                    break
                except (httpx.HTTPError, EvaluationFailure):
                    time.sleep(0.25)
            else:
                raise EvaluationFailure("server_startup_timeout")
        else:
            read_health(client)
        yield
    finally:
        # 이 명령이 만든 검증 서버만 종료하고 기존 개발 서버는 건드리지 않는다.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def check_contract(client: httpx.Client) -> dict[str, Any]:
    checks = []
    payload = {"spec_version": "1.0", "session_id": "foundation-contract", "messages": [{"role": "user", "content": "학습을 시작하려고 해요. 짧게 격려해 주세요."}]}

    def record(name, condition):
        checks.append({"check": name, "passed": bool(condition)})

    before = read_health(client)
    for path in ("/tutor", "/guard", "/healthz"):
        for token in ("", "Bearer invalid-foundation-token"):
            response = client.request("GET" if path == "/healthz" else "POST", path, json=payload, headers={"Authorization": token})
            record(path[1:] + ("_missing_auth" if not token else "_wrong_auth"), response.status_code == 401)
    for path in ("/tutor/", "/guard/"):
        response = client.post(path, json=payload)
        record(path[1:-1] + "_no_redirect", response.status_code == 404 and "location" not in response.headers)
    for name, candidate in (("missing", None), ("empty", ""), ("present", "학습을 응원합니다.")):
        request = dict(payload)
        if candidate is not None:
            request["candidate_answer"] = candidate
        with client.stream("POST", "/guard", json=request) as response:
            body = bounded_json(response)
        parsed = GuardResponse.model_validate(body)
        record("guard_candidate_" + name, (parsed.output is not None) == (candidate is not None))
        if name == "empty":
            record("guard_empty_output", parsed.output.grounded and parsed.output.safe and not parsed.output.unsupported)
    with client.stream("POST", "/tutor", json=payload) as response:
        body = bounded_json(response)
    parsed = TutorResponse.model_validate(body)
    record("tutor_json_contract", bool(parsed.answer) and len(parsed.answer) <= 4000)
    for name, invalid in (
        ("invalid_role", {**payload, "messages": [{"role": "system", "content": "request-marker-never-echo"}]}),
        ("null_candidate", {**payload, "candidate_answer": None}),
    ):
        response = client.post("/guard", json=invalid)
        record(name, response.status_code == 422 and "request-marker-never-echo" not in response.text)
    after = read_health(client)
    degraded = after["counters"].get("degraded_responses", 0) - before["counters"].get("degraded_responses", 0)
    record("no_degraded_contract_response", degraded == 0)
    return {
        "passed": all(row["passed"] for row in checks), "checks": checks,
        "degraded_responses": degraded, "health_after": after,
    }


def evaluate_public(client: httpx.Client, settings: Settings, workers: int) -> dict[str, Any]:
    from tools.release_checks import counter_delta, score_public_cases

    public = []
    for name in ("tutor", "guard"):
        public.extend(json.loads(line) for line in (ROOT / "public" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip())
    dataset_counts = Counter(row["endpoint"] for row in public)
    if dataset_counts != {"tutor": 200, "guard": 200}:
        raise EvaluationFailure("unexpected_public_dataset_size")
    judge_counts = Counter()
    lock = threading.Lock()
    started = time.monotonic()
    before = read_health(client)

    with httpx.Client(timeout=30, follow_redirects=False, trust_env=False) as judge_client:
        def judge(kind, payload):
            # 주최측 판정 지시문·필드 검증을 그대로 재사용하고 전송 경계만 제한한다.
            body = {
                "model": settings.chat_model, "stream": False, "max_completion_tokens": NON_STREAM_MAX_COMPLETION_TOKENS,
                "reasoning_effort": "low", "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM + "\n" + PROMPTS[kind]},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            }
            with lock:
                judge_counts["calls"] += 1
            try:
                with judge_client.stream(
                    "POST", settings.chat_base_url + "/chat/completions", json=body,
                    headers={"Authorization": "Bearer " + settings.chat_api_key},
                ) as response:
                    completion = bounded_json(response)
                choice = completion["choices"][0]
                if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
                    raise EvaluationFailure("incomplete_judge_response")
                value = json.loads(choice["message"]["content"])[FIELD[kind]]
                if not _valid(kind, value):
                    raise EvaluationFailure("invalid_judge_field")
                if kind == "complied" and value:
                    with lock:
                        judge_counts["attack_compliance"] += 1
                return {FIELD[kind]: value}
            except Exception:
                with lock:
                    judge_counts["errors"] += 1
                raise JudgeError("공개 판정 실패") from None

        # 실제 판정 모델을 호출할 수 없으면 부분 점수를 전체 점수로 보고하지 않는다.
        judge("faithful", {"answer": "질문을 한 가지씩 정리해 볼까요?", "evidence": []})
        measured = score_public_cases(public, client, judge, load_kb_dir(ROOT / "kb"), workers)
    after = read_health(client)
    degraded = counter_delta(before, after)
    score = measured["score"]
    return {
        **measured, "dataset": "organizer_public", "items_count": len(public), "workers": workers,
        "judge": {"enabled": True, "calls": judge_counts["calls"], "errors": judge_counts["errors"],
                  "attack_compliance": judge_counts["attack_compliance"],
                  "prompts": "organizer", "reasoning_effort": "low", "max_completion_tokens": NON_STREAM_MAX_COMPLETION_TOKENS,
                  "timeout_seconds": 30, "retries": 0},
        "degraded_responses": degraded,
        "elapsed_with_judge_seconds": round(time.monotonic() - started, 3),
        "operational_pass": not measured["endpoint_errors"] and degraded == 0 and measured["deadline_violations"] == 0
        and score["judge_errors"] == 0 and score["invalid"] == 0,
    }


def write_report(report: dict[str, Any], destination: Path = REPORT) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.stem + "-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_retrieval(args, report, settings):
    from tools.retrieval_checks import evaluate_retrieval

    result = asyncio.run(evaluate_retrieval(settings, ROOT / "public" / "tutor.jsonl", args.workers, args.lexical_only))
    report["retrieval"] = result
    if args.start_server:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{args.port}", timeout=30, follow_redirects=False, trust_env=False,
            headers={"Authorization": "Bearer " + settings.team_token},
        ) as client:
            with managed_server(True, args.port, client):
                report["startup_server"] = read_health(client)
                actual = report["startup_server"].get("retrieval", {})
                expected = result["knowledge"]
                verified = (
                    report["startup_server"]["implementation_stage"] in tuple("IMP-001-" + str(index) for index in range(2, 8))
                    and report["startup_server"]["retrieval_ready"] and actual.get("dense_ready")
                    and actual.get("kb_fingerprint") == expected["kb_fingerprint"]
                    and actual.get("sections") == expected["sections"]
                    and actual.get("tables") == expected["tables"]
                    and actual.get("cache_state") == "hit"
                    and report["startup_server"]["provider_counters"].get("embedding_attempts", 0) == 0
                )
                report["server_startup_cache_verified"] = bool(verified)
                if not verified:
                    raise EvaluationFailure("retrieval_startup_not_verified")
    report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
    report["passed"] = (
        report["source_unchanged"] and result["operational_pass"]
        and (args.lexical_only or result["evaluation"]["quality_target_met"])
    )


def run_foundation(args, report, settings):
    print("실제 채팅 구조화 응답·임베딩 API를 확인합니다.", flush=True)
    report["api_attempted"] = True
    report["provider"] = asyncio.run(check_provider(settings))
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", timeout=30, follow_redirects=False, trust_env=False,
        headers={"Authorization": "Bearer " + settings.team_token},
    ) as client:
        print("서버 준비·인증·JSON 계약을 확인합니다.", flush=True)
        with managed_server(args.start_server, args.port, client):
            report["server"] = read_health(client)
            report["contract"] = check_contract(client)
            report["final_server"] = report["contract"]["health_after"]
            if not report["contract"]["passed"]:
                raise EvaluationFailure("contract_checks_failed")
            print("공개 세트 400개와 실제 판정 모델로 기반 점수를 측정합니다.", flush=True)
            report["baseline"] = evaluate_public(client, settings, args.workers)
            report["final_server"] = read_health(client)
    report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
    report["passed"] = report["source_unchanged"] and report["baseline"]["operational_pass"]


def run_policy(args, report, settings):
    from tools.policy_checks import evaluate_policy

    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", timeout=30, follow_redirects=False, trust_env=False,
        headers={"Authorization": "Bearer " + settings.team_token},
    ) as client:
        with managed_server(args.start_server, args.port, client):
            report["policy"] = evaluate_policy(client, ROOT / "public" / "guard.jsonl", args.workers)
    report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
    report["passed"] = report["source_unchanged"] and report["policy"]["operational_pass"] and report["policy"]["quality_target_met"]


def ensure_source(report):
    report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
    if not report["source_unchanged"]:
        raise EvaluationFailure("evaluation_source_changed")


def repeated_public(client, settings, args, report, checkpoint):
    from tools.release_checks import release_targets

    runs = []
    repeat = {"runs": runs, "requested_runs": args.repeat, "two_runs_verified": False,
              "measurement_complete": False, "stopped_reason": None}
    report["release"] = repeat
    initial_instance = read_health(client).get("instance_id")
    for index in range(args.repeat):
        ensure_source(report)
        print(f"공개 전체 평가 {index + 1}/{args.repeat}를 시작합니다.", flush=True)
        result = evaluate_public(client, settings, args.workers)
        result["source_unchanged"] = source_fingerprint() == report["source_sha256"]
        result["same_server_instance"] = initial_instance is not None and read_health(client).get("instance_id") == initial_instance
        result["targets"] = {**release_targets(result), "source_unchanged": result["source_unchanged"],
                             "same_server_instance": result["same_server_instance"]}
        result["quality_target_met"] = all(result["targets"].values())
        runs.append(result)
        checkpoint()
        if not result["operational_pass"] or not result["source_unchanged"] or not result["same_server_instance"]:
            repeat["stopped_reason"] = "operational_or_identity_failure"
            break
    repeat["measurement_complete"] = len(runs) == args.repeat
    repeat["two_runs_verified"] = len(runs) == 2 and all(run["quality_target_met"] for run in runs)
    if len(runs) == 1 and args.repeat == 1:
        repeat["stopped_reason"] = "second_run_not_requested"
    checkpoint()
    return repeat["two_runs_verified"]


def run_extended(args, report, settings):
    from tools.fault_checks import evaluate_faults
    from tools.policy_checks import evaluate_policy
    from tools.release_checks import counter_delta, evaluate_load, evaluate_output, evaluate_smoke

    def checkpoint():
        report["updated_at"] = timestamp()
        write_report(report, ROOT / "output" / "tests" / (args.stage + "-latest.json"))

    def begin(name):
        ensure_source(report)
        report["passed"] = False
        report["active_check"] = name
        checkpoint()

    if args.stage == "acceptance":
        begin("faults")
        report["faults"] = evaluate_faults()
        checkpoint()
        if not report["faults"]["passed"]:
            raise EvaluationFailure("offline_fault_checks_failed")
        begin("provider")
        report["api_attempted"] = True
        report["provider"] = asyncio.run(check_provider(settings))
        checkpoint()
    report["api_attempted"] = True
    report["code_alignment_verified"] = args.start_server
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", timeout=30, follow_redirects=False, trust_env=False,
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=128),
        headers={"Authorization": "Bearer " + settings.team_token},
    ) as client:
        with managed_server(args.start_server, args.port, client):
            health = read_health(client)
            required = ("input_policy_ready", "output_policy_ready")
            if args.stage != "output":
                required += ("tutor_policy_ready",)
            if not all(health.get(name) for name in required):
                raise EvaluationFailure("full_pipeline_not_ready")
            if args.stage in ("release", "load", "acceptance") and not health.get("instance_id"):
                raise EvaluationFailure("stage6_server_required")

            if args.stage in ("release", "acceptance"):
                begin("contract")
                report["contract"] = check_contract(client)
                checkpoint()
                if not report["contract"]["passed"]:
                    raise EvaluationFailure("contract_checks_failed")
            if args.stage in ("smoke", "acceptance"):
                begin("smoke")
                report["smoke"] = evaluate_smoke(client, ROOT / "kb", args.workers)
                checkpoint()
                if not report["smoke"]["operational_pass"]:
                    raise EvaluationFailure("smoke_operational_failure")
                report["passed"] = report["smoke"]["behavior_pass"]
            if args.stage == "acceptance":
                begin("policy")
                report["policy"] = evaluate_policy(client, ROOT / "public" / "guard.jsonl", args.workers)
                checkpoint()
                if not report["policy"]["operational_pass"]:
                    raise EvaluationFailure("policy_operational_failure")
            if args.stage in ("output", "acceptance"):
                begin("output")
                report["output"] = evaluate_output(client, ROOT / "public" / "guard.jsonl", args.workers)
                checkpoint()
                if not report["output"]["operational_pass"]:
                    raise EvaluationFailure("output_operational_failure")
                report["passed"] = report["output"]["quality_target_met"]
            if args.stage in ("release", "acceptance"):
                begin("release")
                report["passed"] = repeated_public(client, settings, args, report, checkpoint)
                if not report["release"]["measurement_complete"] or report["release"]["stopped_reason"] == "operational_or_identity_failure":
                    raise EvaluationFailure("release_measurement_incomplete")
            if args.stage in ("load", "acceptance"):
                begin("load")
                def load_checkpoint(result):
                    report["load"] = result
                    checkpoint()
                    ensure_source(report)

                report["load"] = evaluate_load(client, ROOT / "public", full=args.full_load, seed=args.seed,
                                                checkpoint=load_checkpoint)
                report["passed"] = report["load"]["quality_target_met"]
            final_health = read_health(client)
            counter_delta(health, final_health)
            report["final_server"] = final_health
            if args.stage == "acceptance":
                report["acceptance_targets"] = {
                    "faults": report["faults"]["passed"], "provider": report["provider"]["passed"],
                    "contract": report["contract"]["passed"], "smoke": report["smoke"]["behavior_pass"],
                    "policy": report["policy"]["quality_target_met"], "output": report["output"]["quality_target_met"],
                    "repeat": report["release"]["two_runs_verified"], "load": report["load"]["quality_target_met"],
                }
                report["passed"] = all(report["acceptance_targets"].values())
    ensure_source(report)
    if args.stage in ("release", "load", "acceptance"):
        report["passed"] = report["passed"] and report["code_alignment_verified"]
    report["active_check"] = None
    checkpoint()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="프로세스 환경변수만 사용하는 실제 API·공개 세트 검증")
    parser.add_argument("stage", choices=("foundation", "retrieval", "policy", "smoke", "output", "release", "load", "faults", "acceptance"))
    parser.add_argument("--start-server", action="store_true", help="전용 검증 서버를 시작하고 종료 시 정리")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--lexical-only", action="store_true", help="API 없이 KB 무결성과 어휘 검색 기준만 측정")
    parser.add_argument("--diagnostics", action="store_true", help="공개 입력 오류 사례의 의도 속성만 비교")
    parser.add_argument("--repeat", type=int, choices=(1, 2), default=2, help="release 전체 평가 반복 횟수, 기본 두 번")
    parser.add_argument("--full-load", action="store_true", help="load·acceptance의 단계적 검사 통과 후 혼합 600건 실행")
    parser.add_argument("--seed", type=int, default=0, help="유한 부하의 공개 문항 순서를 재현하는 난수 시드")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("포트는 1024~65535 범위여야 합니다.")
    if args.lexical_only and (args.stage != "retrieval" or args.start_server):
        parser.error("--lexical-only는 retrieval 단독 검사에서만 사용할 수 있습니다.")
    if args.diagnostics and (args.stage != "policy" or args.start_server or args.lexical_only):
        parser.error("--diagnostics는 policy 단독 검사에서만 사용할 수 있습니다.")
    if args.full_load and args.stage not in ("load", "acceptance"):
        parser.error("--full-load는 load·acceptance에서만 사용할 수 있습니다.")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("시드는 0~4294967295 범위여야 합니다.")
    if args.stage == "faults" and args.start_server:
        parser.error("faults는 실제 서버를 시작하지 않습니다.")
    silence_transport_logs()
    report: dict[str, Any] = {
        "stage": {"foundation": "IMP-001-1", "retrieval": "IMP-001-2", "policy": "IMP-001-3", "output": "IMP-001-4",
                  "smoke": "IMP-001-5", "release": "IMP-001-6", "load": "IMP-001-6", "faults": "IMP-001-6",
                  "acceptance": "IMP-001-6"}[args.stage], "started_at": timestamp(), "passed": False,
        "command": args.stage, "api_mode": "offline" if args.lexical_only or args.stage == "faults" else "real",
        "python_version": sys.version.split()[0], "platform": sys.platform,
        "api_attempted": False, "private_evaluation_used": False,
        "request_response_bodies_saved": False,
    }
    if args.stage == "acceptance":
        report["scope"] = "local_acceptance"
        report["excluded_checks"] = ["runbox_runtime", "tunnel", "submission"]
    try:
        settings = None if args.lexical_only or args.stage == "faults" else Settings.from_environment()
        report["source_sha256"] = source_fingerprint()
        report["base_commit"] = commit_id()
        if args.stage == "faults":
            from tools.fault_checks import evaluate_faults
            report["faults"] = evaluate_faults()
            ensure_source(report)
            report["passed"] = report["faults"]["passed"]
        elif args.stage == "retrieval":
            print("KB 준비·캐시·공개 근거 검색을 확인합니다.", flush=True)
            report["api_attempted"] = not args.lexical_only
            run_retrieval(args, report, settings)
        elif args.stage == "policy" and args.diagnostics:
            from tools.policy_diagnostics import diagnose_intent
            print("선택된 공개 오류의 의도 속성을 비교합니다.", flush=True)
            report["api_attempted"] = True
            report["diagnostics"] = asyncio.run(diagnose_intent(settings, ROOT, args.workers))
            report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
            report["passed"] = report["source_unchanged"] and all("error" not in row for row in report["diagnostics"]["items"])
        elif args.stage == "policy":
            print("실제 입력 분류와 공개 입력 정책을 확인합니다.", flush=True)
            report["api_attempted"] = True
            run_policy(args, report, settings)
        elif args.stage in ("smoke", "output", "release", "load", "acceptance"):
            run_extended(args, report, settings)
        else:
            run_foundation(args, report, settings)
    except ConfigurationError as exc:
        report["passed"] = False
        report["error"] = {"type": "ConfigurationError", "detail": str(exc)}
        print(str(exc), flush=True)
    except (ProviderFailure, EvaluationFailure) as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "reason": str(exc)}
        if isinstance(exc, ProviderFailure) and exc.status_code is not None:
            report["error"]["status_code"] = exc.status_code
        print("검증 중단: " + str(exc), flush=True)
    except KeyboardInterrupt:
        report["passed"] = False
        report["error"] = {"type": "Interrupted"}
        print("검증을 중단했습니다.", flush=True)
    except Exception as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__}
        print("검증 중단: " + type(exc).__name__, flush=True)
    finally:
        if "source_sha256" in report:
            try:
                report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
            except Exception:
                report["source_unchanged"] = False
            report["passed"] = report["passed"] and report["source_unchanged"]
            if not report["source_unchanged"] and "release" in report:
                report["release"]["two_runs_verified"] = False
                report["release"]["stopped_reason"] = "final_source_changed"
            if "acceptance_targets" in report:
                report["acceptance_targets"]["source_unchanged"] = report["source_unchanged"]
                report["acceptance_targets"]["new_server"] = report["code_alignment_verified"]
                report["acceptance_targets"]["repeat"] = report["release"]["two_runs_verified"]
                report["passed"] = report["passed"] and all(report["acceptance_targets"].values())
        report["finished_at"] = timestamp()
        destination = args.stage + ("-diagnostics" if args.diagnostics else "")
        write_report(report, ROOT / "output" / "tests" / (destination + "-latest.json"))
    return finish_report(report, args.stage + ("-diagnostics" if args.diagnostics else ""))


def finish_report(report, stage):
    if "baseline" in report:
        score = report["baseline"]["score"]
        print(f"기반 점수: 총점 {score['total']}, 튜터 {score['tutor']}, 가드 {score['guard']}", flush=True)
    label = "진단 실행 결과: " if "diagnostics" in report else "검증 결과: "
    print(label + ("통과" if report["passed"] else "확인 필요"), flush=True)
    if "retrieval" in report:
        print("검색 Recall@k: " + json.dumps(report["retrieval"]["evaluation"]["recall_at_k"], ensure_ascii=False), flush=True)
    if "policy" in report:
        print("입력 block F1: " + str(report["policy"]["block_f1"]) + ", 비정상 범주 정확도: " + str(report["policy"]["non_normal_category_accuracy"]), flush=True)
    if "release" in report:
        for index, run in enumerate(report["release"]["runs"], start=1):
            score = run["score"]
            print(f"공개 평가 {index}: 총점 {score['total']}, 튜터 {score['tutor']}, 가드 {score['guard']}", flush=True)
        if report["release"].get("stopped_reason") == "second_run_not_requested":
            print("단일 측정만 완료했으며 최종 두 번 평가 조건은 미충족입니다.", flush=True)
    if "faults" in report:
        print("통제된 장애 검사 수: " + str(report["faults"]["tests_run"]), flush=True)
    print("보고서: output/tests/" + stage + "-latest.json", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    # 실행 모듈을 보조 검사에서 가져올 때 동일한 예외·함수 정의를 사용한다.
    sys.modules.setdefault("tools.evaluate", sys.modules[__name__])
    raise SystemExit(main())
