"""이전·개선 검색을 같은 임베딩과 공개 표본으로 비교하고 메타데이터만 남긴다."""

import argparse
import asyncio
import json
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime

import httpx

from tools.evaluate import (
    FIELD, KST, ROOT, EvaluationFailure, commit_id, interactive_environment, load_kb_dir,
    make_judge, managed_server, read_health, source_fingerprint, timestamp, write_report,
)
from tools.release_checks import counter_delta, latency_summary, score_public_cases
from tools.search_baseline_server import legacy_embedding_queries, legacy_queries
from tutor_core.config import ConfigurationError, Settings
from tutor_core.embedding_cache import normalize_vector
from tutor_core.identity import runtime_fingerprint
from tutor_core.llm import ProviderFailure, RequestBudget, silence_transport_logs
from tutor_core.retrieval import Retriever, SearchQueries, prepare_retriever
from tutor_core.schemas import Message


PROFILES = ("previous_search", "current_search")
APPLICATIONS = ("tools.search_baseline_server:app", "tools.diagnostic_server:app")
TOP_K = (1, 3, 8, 12)


def load_public():
    public = [json.loads(line) for name in ("tutor", "guard")
              for line in (ROOT / "public" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
    if Counter(item["endpoint"] for item in public) != {"tutor": 200, "guard": 200}:
        raise EvaluationFailure("unexpected_public_dataset_size")
    if len({item["id"] for item in public}) != len(public):
        raise EvaluationFailure("duplicate_public_id")
    return public


def diverse(items, count):
    """다른 근거 섹션을 우선 고르고 원래 공개 순서를 유지한다."""
    chosen, seen = [], set()
    for item in items:
        key = tuple(sorted(item["labels"].get("evidence", [])))
        if key not in seen:
            chosen.append(item)
            seen.add(key)
        if len(chosen) == count:
            return chosen
    selected = {item["id"] for item in chosen}
    chosen.extend(item for item in items if item["id"] not in selected)
    if len(chosen) < count:
        raise EvaluationFailure("insufficient_public_controls")
    return chosen[:count]


def evidence_cases(public):
    rows = [item for item in public if item["endpoint"] == "tutor" and item["labels"].get("evidence")]
    if len(rows) != 119:
        raise EvaluationFailure("unexpected_public_evidence_count")
    return rows


def by_course(rows, count):
    selected = []
    for course in ("PY", "PR", "FIN"):
        group = [item for item in rows if all(sid.startswith(course + "-") for sid in item["labels"]["evidence"])]
        selected.extend(diverse(group, count))
    return selected


def select_cases(public):
    """검색 효과를 보는 표본만 고르며 라벨은 요청 본문에 넣지 않는다."""
    selected = by_course(evidence_cases(public), 8)
    groups = defaultdict(list)
    for item in public:
        if item["endpoint"] == "guard" and "candidate_answer" in item:
            groups[(item["labels"]["grounded"], item["labels"]["safe"])].append(item)
    guards = []
    for index in range(6):
        for key in sorted(groups):
            if index < len(groups[key]):
                guards.append(groups[key][index])
            if len(guards) == 6:
                return selected + guards
    raise EvaluationFailure("insufficient_candidate_controls")


def search_cases(public):
    rows = []
    evidence = evidence_cases(public)
    for item in evidence:
        messages = [Message.model_validate(message) for message in item["messages"]]
        rows.append((item["id"], "public", messages, set(item["labels"]["evidence"])))
    # 공개 문항을 바탕으로 만든 긴 입력은 공개 점수의 분모와 분리한다.
    for item in by_course(evidence, 4):
        messages = [Message.model_validate(message) for message in item["messages"]]
        refs = set(item["labels"]["evidence"])
        tail = Message(role="user", content="배경 자료를 확인하는 중입니다.\n" * 1200 + messages[-1].content)
        rows.append((item["id"] + "-tail", "long_latest", [*messages[:-1], tail], refs))
        history = [*messages, Message(role="assistant", content="앞서 확인한 배경 자료를 정리하는 중입니다.\n" * 1200),
                   Message(role="user", content="앞서 물어본 내용의 핵심과 조건을 다시 설명해 줘.")]
        rows.append((item["id"] + "-history", "long_history", history, refs))
    return rows


async def paired_search(retriever, messages):
    """한 사례 안에서만 같은 임베딩 입력을 공유하고 질의·벡터는 반환하지 않는다."""
    queries = (legacy_queries(messages), SearchQueries.from_messages(messages))
    inputs = (legacy_embedding_queries(queries[0]), Retriever._embedding_queries(queries[1]))
    unique = list(dict.fromkeys(text for group in inputs for text in group))
    failure, vectors = None, {}
    started = time.monotonic()
    if unique:
        try:
            raw = await retriever.clients.embed(unique, RequestBudget.start(27))
            if len(raw) != len(unique):
                raise ValueError("invalid_embedding_count")
            vectors = dict(zip(unique, (normalize_vector(vector) for vector in raw)))
        except (ProviderFailure, ValueError) as exc:
            failure = exc.reason if isinstance(exc, ProviderFailure) else "invalid_embedding"
    embedding_seconds = time.monotonic() - started
    result = {"queries_changed": queries[0] != queries[1] or inputs[0] != inputs[1],
              "unique_embedding_inputs": len(unique), "embedding_elapsed_seconds": round(embedding_seconds, 6)}
    for name, query, group in zip(PROFILES, queries, inputs):
        started = time.monotonic()
        dense = tuple(vectors[text] for text in group) if not failure else ()
        found = await asyncio.to_thread(retriever._result, query, 12, dense, failure)
        result[name] = {
            "ids": list(found.section_ids), "mode": found.mode, "degraded": found.degraded,
            "failure_reason": found.failure_reason, "context_characters": found.context_characters,
            "query_characters": [len(query.latest), len(query.history)],
            "embedding_bytes": [len(text.encode("utf-8")) for text in group],
            "ranking_elapsed_seconds": round(time.monotonic() - started, 6),
        }
    result["rankings_identical"] = result[PROFILES[0]]["ids"] == result[PROFILES[1]]["ids"]
    return result


def retrieval_summary(rows):
    summary = {}
    for kind in ("public", "long_latest", "long_history"):
        group = [row for row in rows if row["kind"] == kind]
        denominator = sum(len(row["evidence"]) for row in group)
        summary[kind] = {
            "items_count": len(group), "evidence_count": denominator,
            "queries_changed": sum(row["queries_changed"] for row in group),
            "rankings_identical": sum(row["rankings_identical"] for row in group),
            "shared_embedding_latency_seconds": latency_summary([row["embedding_elapsed_seconds"] for row in group]),
        }
        for name in PROFILES:
            summary[kind][name] = {
                "recall_at_k": {str(k): round(sum(len(set(row["evidence"]) & set(row[name]["ids"][:k])) for row in group)
                                              / denominator, 6) if denominator else None for k in TOP_K},
                "fallbacks": sum(row[name]["degraded"] for row in group),
                "context_characters": latency_summary([row[name]["context_characters"] for row in group]),
                "ranking_latency_seconds": latency_summary([row[name]["ranking_elapsed_seconds"] for row in group]),
            }
    return summary


async def compare_retrieval(settings, public, checkpoint):
    from tutor_core.llm import ApiClients

    clients = ApiClients(settings)
    try:
        retriever = await prepare_retriever(clients)
        cases = search_cases(public)
        if any(not refs.issubset(retriever.knowledge.by_id) for _, _, _, refs in cases):
            raise EvaluationFailure("invalid_public_evidence")
        semaphore = asyncio.Semaphore(4)
        completed = 0

        async def one(case):
            nonlocal completed
            item_id, kind, messages, refs = case
            async with semaphore:
                pair = await paired_search(retriever, messages)
            completed += 1
            if completed % 20 == 0 or completed == len(cases):
                print(f"검색 비교 {completed}/{len(cases)}건 처리", flush=True)
                checkpoint({"completed": completed, "total": len(cases)})
            return {"id": item_id, "kind": kind, "evidence": sorted(refs), **pair}

        rows = await asyncio.gather(*(one(case) for case in cases))
        summary = retrieval_summary(rows)
        fallbacks = sum(row[name]["degraded"] for row in rows for name in PROFILES)
        return {"knowledge": retriever.summary(), "summary": summary, "items": rows,
                "provider_counters": dict(clients.counters), "workers": 4, "measurement_complete": fallbacks == 0,
                "target_recall_at_8": .98,
                "public_target_met": summary["public"]["current_search"]["recall_at_k"]["8"] >= .98,
                "embedding_comparison": "same_inputs_share_one_vector_within_each_case_only",
                "latency_interpretation": "shared_batch_not_independent_profile_api_latency"}
    finally:
        await clients.close()


def profile_summary(measured):
    rows = measured["diagnostics"]["items"]
    events = [event for row in rows for event in row["events"]
              if event["stage"] == "output_verification" and "source_coverage" in event]
    return {"metrics": measured["metrics"], "latency_seconds": measured["server_latency_seconds"],
            "tutor_failures": sum(any(event["stage"] == "tutor" and event.get("verified") is False
                                      for event in row["events"]) for row in rows if row["endpoint"] == "tutor"),
            "evidence_events": len(events),
            "source_coverage_mean": {name: sum(event["source_coverage"].get(name, 0) for event in events) / len(events)
                                     if events else None for name in ("citation", "question", "claim")},
            "source_coverage_interpretation": "selected_ids_can_belong_to_multiple_sources",
            "context_characters": latency_summary([event["context_characters"] for event in events])}


def measure_profile(settings, items, application, checkpoint):
    counts, lock = Counter(), threading.Lock()
    original = make_judge(settings.chat_base_url, settings.chat_model, settings.chat_api_key, timeout=60, retries=2)

    def judge(kind, payload):
        with lock:
            counts[kind + "_calls"] += 1
        return original(kind, payload)

    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=30, trust_env=False, follow_redirects=False,
                      headers={"Authorization": "Bearer " + settings.team_token}) as client:
        with managed_server(True, 8765, client, application=application):
            before = read_health(client)
            if before.get("runtime_sha256") != runtime_fingerprint():
                raise EvaluationFailure("diagnostic_runtime_mismatch")
            measured = score_public_cases(items, client, judge, load_kb_dir(ROOT / "kb"), 4,
                                          diagnostics=True, checkpoint=checkpoint)
            after = read_health(client)
            degraded = counter_delta(before, after)
    complete = (bool(before.get("instance_id")) and before["instance_id"] == after.get("instance_id")
                and measured["metrics"]["unique_observed_count"] == len(items)
                and measured["diagnostics"]["trace_complete_count"] == len(items)
                and measured["score"]["judge_errors"] == 0)
    return {"application": application, "measurement_complete": complete,
            "endpoint_contract_pass": not measured["endpoint_errors"] and measured["score"]["invalid"] == 0
            and measured["deadline_violations"] == 0 and measured["metrics"]["invalid_substrings"] == 0,
            "initial_server": before, "final_server": after, "degraded_responses": degraded,
            "judge_logical_calls_by_kind": {kind: counts[kind + "_calls"] for kind in FIELD},
            "summary": profile_summary(measured), "measured": measured}


def finalize_comparison(report):
    profiles = report.get("profiles", {})
    complete = (report.get("source_unchanged") is True and report.get("retrieval", {}).get("measurement_complete") is True
                and all(profiles.get(name, {}).get("measurement_complete") is True for name in PROFILES)
                and "error" not in report)
    report.update(measurement_complete=complete, passed=complete,
                  passed_scope="search_and_selected_profile_measurement_only", final_quality_verified=False)
    if not complete:
        return
    old, new = (profiles[name]["summary"] for name in PROFILES)
    report["selected_metric_deltas"] = {
        key: new["metrics"][key] - old["metrics"][key]
        if new["metrics"].get(key) is not None and old["metrics"].get(key) is not None else None
        for key in ("tutor_action_accuracy", "tutor_citation_f1", "tutor_facts_score", "tutor_faithfulness_rate",
                    "grounded_accuracy", "safe_accuracy", "unsupported_span_f1")
    }
    report["selected_tutor_failure_delta"] = new["tutor_failures"] - old["tutor_failures"]
    report["selected_p95_delta_seconds"] = new["latency_seconds"]["p95"] - old["latency_seconds"]["p95"]
    report["endpoint_contract_pass"] = all(profiles[name]["endpoint_contract_pass"] for name in PROFILES)


def main(argv=None):
    parser = argparse.ArgumentParser(description="3단계 검색 119건·긴 입력 24건과 제한 공개 표본의 이전/개선 비교")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args(argv)
    silence_transport_logs()
    destination = ROOT / "output" / "tests" / "search-latest.json"
    report = {"stage": "IMP-002-3", "started_at": timestamp(), "api_mode": "real", "api_attempted": False,
              "passed": False, "measurement_complete": False, "private_evaluation_used": False,
              "request_response_bodies_saved": False, "query_vectors_saved": False, "profiles": {},
              "interpretation": "selected_public_sample_not_full_public_or_submission_score",
              "profile_difference": "query_preparation_and_evidence_merge_only_same_current_policy_and_budgets",
              "runtime_hash_scope": "same_core_manifest_profiles_use_explicit_factory_injection",
              "confounders": "sequential_runs_generation_and_judge_variability_not_proven_causality"}
    try:
        if args.interactive:
            interactive_environment()
        settings = Settings.from_environment()
        public = load_public()
        items = select_cases(public)
        report.update(source_sha256=source_fingerprint(), base_commit=commit_id(),
                      selected_ids=[item["id"] for item in items], selected_counts=dict(Counter(item["endpoint"] for item in items)),
                      workers=4, active_check="paired_retrieval")
        write_report(report, destination)

        def checkpoint(progress):
            report["progress"] = progress
            write_report(report, destination)

        report["api_attempted"] = True
        report["retrieval"] = asyncio.run(compare_retrieval(settings, public, checkpoint))
        report["judge"] = {"source": "organizer_make_judge", "stream": True, "max_completion_tokens": 4000,
                           "timeout_seconds": 60, "retries": 2, "reasoning_effort_override": None}
        for name, application in zip(PROFILES, APPLICATIONS):
            if source_fingerprint() != report["source_sha256"]:
                raise EvaluationFailure("source_changed")
            report.update(active_check=name, progress={"completed": 0, "total": len(items)})
            write_report(report, destination)
            print(f"{name}: 공개 튜터·후보 표본 {len(items)}건을 비교합니다.", flush=True)
            report["profiles"][name] = measure_profile(settings, items, application, checkpoint)
            report["progress"] = {"completed": len(items), "total": len(items)}
            write_report(report, destination)
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
        finalize_comparison(report)
        report.update(finished_at=timestamp(), active_check=None)
        report["archive"] = "search-" + datetime.now(KST).strftime("%Y%m%d-%H%M%S") + ".json"
        write_report(report, ROOT / "output" / "tests" / report["archive"])
        write_report(report, destination)
        print("보고서: output/tests/search-latest.json", flush=True)
        print("측정 완료 여부: " + str(report["measurement_complete"]), flush=True)
        print("공개 검색 Recall@8 목표 충족: " + str(report.get("retrieval", {}).get("public_target_met", False)), flush=True)
        print("선택 표본의 형식·시간 계약: " + str(report.get("endpoint_contract_pass", False)), flush=True)
    return 0 if report["measurement_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
