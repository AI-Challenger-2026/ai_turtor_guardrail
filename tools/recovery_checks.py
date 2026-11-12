"""공개 복구 실패와 보호 대조군을 제한적으로 재검사하고 본문 없는 결과를 남긴다."""

import argparse
import json
from collections import Counter
from datetime import datetime

import httpx

from tools.evaluate import (
    KST, ROOT, EvaluationFailure, commit_id, interactive_environment, load_kb_dir, make_judge,
    managed_server, read_health, source_fingerprint, timestamp, write_report,
)
from tools.release_checks import latency_summary, score_public_cases
from tutor_core.config import ConfigurationError, Settings
from tutor_core.identity import runtime_fingerprint
from tutor_core.llm import silence_transport_logs


REFERENCE_FILES = ("diagnose-20261111-214443.json", "diagnose-20261112-102606.json")


def select_cases(public, baseline, previous):
    """이전 공개 실패 유형과 범주별 대조군을 고르며 정답을 서버에 전달하지 않는다."""
    by_id = {item["id"]: item for item in public}
    selected = {row["id"] for row in previous["diagnostics"]["items"] if row["endpoint"] == "tutor"
                and ("stage_failure" in row["flags"] or "attack_compliance" in row["flags"])}
    selected.update(row["id"] for row in baseline["diagnostics"]["items"] if "attack_compliance" in row["flags"])
    if not selected or not selected.issubset(by_id):
        raise EvaluationFailure("focus_public_id_mismatch")
    controls = set()
    for item in public:
        if item["endpoint"] == "tutor":
            key = ("tutor", item["labels"]["category"])
        else:
            labels = item["labels"]
            key = ("guard", "candidate", labels.get("grounded"), labels.get("safe")) if "candidate_answer" in item else ("guard", labels["decision"])
        if key not in controls:
            controls.add(key)
            selected.add(item["id"])
    if len(selected) > 80:
        raise EvaluationFailure("focus_scope_too_large")
    return [item for item in public if item["id"] in selected]


def paired_summary(measured, ids):
    rows = [row for row in measured["diagnostics"]["items"] if row["id"] in ids and row["endpoint"] == "tutor"]
    return {
        "tutor_count": len(rows),
        "tutor_failures": sum(any(event["stage"] == "tutor" and event.get("verified") is False for event in row["events"]) for row in rows),
        "attack_compliance": sum("attack_compliance" in row["flags"] for row in rows),
        "repair_attempts": sum(any(event["stage"] == "tutor_repair" for event in row["events"]) for row in rows),
        "successful_repair_generations": sum(any(event["stage"] == "tutor_repair" and event["status"] == "ok" for event in row["events"]) for row in rows),
        "latency_seconds": latency_summary([row["server_elapsed_seconds"] for row in rows]),
    }


def references():
    reports = []
    for name in REFERENCE_FILES:
        path = ROOT / "output" / "tests" / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise EvaluationFailure("unsafe_reference_path")
        if not path.is_file():
            raise EvaluationFailure("public_reference_report_missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not value.get("measurement_complete") or value.get("private_evaluation_used") is not False:
            raise EvaluationFailure("public_reference_report_invalid")
        reports.append(value["baseline"])
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description="2단계의 공개 복구 실패·보호 대조군만 실제 API로 재검사")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)
    silence_transport_logs()
    report = {"stage": "IMP-002-2", "scope": "selected_public_recovery_cases", "started_at": timestamp(),
              "passed": False, "measurement_complete": False, "api_mode": "real", "api_attempted": False,
              "private_evaluation_used": False, "request_response_bodies_saved": False,
              "selection_bias": "previous_failures_and_controls_not_full_public_score"}
    destination = ROOT / "output" / "tests" / "recovery-latest.json"
    try:
        if args.interactive:
            interactive_environment()
        settings = Settings.from_environment()
        baseline, previous = references()
        public = [json.loads(line) for name in ("tutor", "guard")
                  for line in (ROOT / "public" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
        if Counter(item["endpoint"] for item in public) != {"tutor": 200, "guard": 200}:
            raise EvaluationFailure("unexpected_public_dataset_size")
        items = select_cases(public, baseline, previous)
        ids = {item["id"] for item in items}
        report.update(source_sha256=source_fingerprint(), base_commit=commit_id(), references=list(REFERENCE_FILES),
                      items_count=len(items), workers=4, active_check="server_startup")
        write_report(report, destination)
        report["judge"] = {"source": "organizer_make_judge", "stream": True, "max_completion_tokens": 4000,
                           "timeout_seconds": 60, "retries": 2, "reasoning_effort_override": None}
        print(f"공개 복구·보호 사례 {len(items)}건을 동시 작업 4개로 확인합니다.", flush=True)
        judge = make_judge(settings.chat_base_url, settings.chat_model, settings.chat_api_key, timeout=60, retries=2)
        with httpx.Client(base_url="http://127.0.0.1:8765", timeout=30, trust_env=False, follow_redirects=False,
                          headers={"Authorization": "Bearer " + settings.team_token}) as client:
            report["api_attempted"] = True
            with managed_server(True, 8765, client, application="tools.diagnostic_server:app"):
                initial = read_health(client)
                if initial.get("runtime_sha256") != runtime_fingerprint():
                    raise EvaluationFailure("diagnostic_runtime_mismatch")
                report.update(initial_server=initial, active_check="selected_public_scoring")
                write_report(report, destination)

                def checkpoint(progress):
                    report["progress"] = progress
                    write_report(report, destination)

                measured = score_public_cases(items, client, judge, load_kb_dir(ROOT / "kb"), 4,
                                              diagnostics=True, checkpoint=checkpoint)
                final = read_health(client)
                report.update(measured=measured, final_server=final)
        report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
        report["measurement_complete"] = (report["source_unchanged"] and bool(initial.get("instance_id")) and initial.get("instance_id") == final.get("instance_id")
                                          and measured["metrics"]["unique_observed_count"] == len(items)
                                          and measured["diagnostics"]["trace_complete_count"] == len(items)
                                          and measured["score"]["judge_errors"] == 0)
        pairs = {"baseline": paired_summary(baseline, ids), "previous": paired_summary(previous, ids), "current": paired_summary(measured, ids)}
        report["paired"] = pairs
        report["endpoint_contract_pass"] = (not measured["endpoint_errors"] and measured["score"]["invalid"] == 0
                                             and measured["deadline_violations"] == 0)
        report["eligible_for_full_evaluation"] = (report["measurement_complete"]
            and report["endpoint_contract_pass"]
            and pairs["current"]["tutor_failures"] < pairs["previous"]["tutor_failures"]
            and pairs["current"]["attack_compliance"] <= pairs["baseline"]["attack_compliance"])
        report.update(passed=report["measurement_complete"], passed_scope="selected_measurement_only", active_check=None)
    except ConfigurationError as exc:
        report["error"] = {"type": "ConfigurationError", "detail": str(exc)}
    except EvaluationFailure as exc:
        report["error"] = {"type": "EvaluationFailure", "reason": str(exc)}
    except KeyboardInterrupt:
        report["error"] = {"type": "Interrupted"}
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__}
    finally:
        if "source_sha256" in report:
            report["source_unchanged"] = source_fingerprint() == report["source_sha256"]
            if not report["source_unchanged"]:
                report.update(passed=False, measurement_complete=False, eligible_for_full_evaluation=False)
        report["finished_at"] = timestamp()
        report["passed"] = report["passed"] and "error" not in report
        report["archive"] = "recovery-" + datetime.now(KST).strftime("%Y%m%d-%H%M%S") + ".json"
        write_report(report, ROOT / "output" / "tests" / report["archive"])
        write_report(report, destination)
        print("보고서: output/tests/recovery-latest.json", flush=True)
        print("측정 완료 여부: " + str(report["passed"]), flush=True)
        print("전체 재평가 진행 근거: " + str(report.get("eligible_for_full_evaluation", False)), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
