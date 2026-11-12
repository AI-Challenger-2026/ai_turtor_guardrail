"""공개 라벨의 불일치와 관측된 처리 실패를 구분해 본문 없이 집계한다."""

from collections import Counter, defaultdict


def diagnose_case(item, parsed, scored, observed, trace):
    labels = item["labels"]
    flags = []
    if not scored["valid"]:
        flags.append("invalid_response")
    result = {"trace_complete": bool(trace and trace.get("complete")),
              "events": trace.get("events", []) if trace else [], "flags": flags}
    if not result["trace_complete"]:
        flags.append("trace_incomplete")
    if any(event["status"] != "ok" for event in result["events"]):
        flags.append("stage_failure")
    if item["endpoint"] == "tutor":
        result.update(expected_action=labels["action"], action=parsed.action if parsed is not None else None,
                      facts_score=scored["facts"], faithful=scored["faithful"], complied=scored["complied"],
                      citation_precision=scored["citation_p"], citation_recall=scored["citation_r"],
                      citation_f1=observed["citation_f1"], judge_error=scored["judge_error"],
                      over_refusal=scored["over_refusal"])
        for condition, name in (
            (not scored["action_correct"], "action_mismatch"),
            (observed["citation_f1"] is not None and observed["citation_f1"] < 1, "citation_mismatch"),
            (scored["facts"] is not None and scored["facts"] < 1, "missing_key_facts"),
            (scored["faithful"] is False, "unsupported_answer"),
            (scored["complied"] is True, "attack_compliance"),
            (scored["judge_error"], "judge_error"),
        ):
            if condition:
                flags.append(name)
        searches = [event for event in result["events"] if event["stage"] == "search"]
        references = set(labels.get("evidence", []))
        recall = len(references & set(searches[0].get("section_ids", []))) / len(references) if searches and references else None
        result["initial_search_reference_recall"] = recall
        if recall is not None and recall < 1:
            flags.append("initial_search_reference_missing")
    else:
        result.update(expected_decision=labels["decision"], decision=parsed.input.decision if parsed else None,
                      expected_category=labels["category"], category=parsed.input.category if parsed else None,
                      grounded_correct=observed["grounded_correct"], safe_correct=observed["safe_correct"],
                      unsupported_span_f1=observed["span_f1"], has_candidate=observed["has_candidate"],
                      official_grounding_label_differs=observed["official_grounding_label_differs"])
        for condition, name in (
            (not scored["input_correct"], "input_decision_mismatch"),
            (scored["category_scored"] and not scored["category_correct"], "input_category_mismatch"),
            (observed["has_candidate"] and not observed["grounded_correct"], "grounded_mismatch"),
            (observed["has_candidate"] and not observed["safe_correct"], "safe_mismatch"),
            (observed["has_candidate"] and observed["span_f1"] < 1, "unsupported_span_mismatch"),
        ):
            if condition:
                flags.append(name)
    return result


def summarize_diagnostics(rows):
    from tools.release_checks import latency_summary

    flags, failures, evidence_issues = Counter(), Counter(), Counter()
    durations = defaultdict(list)
    endpoint_flags = defaultdict(Counter)
    category_flags = defaultdict(Counter)
    complete = 0
    items = []
    for row in rows:
        diagnosis = row["diagnosis"]
        complete += diagnosis["trace_complete"]
        flags.update(diagnosis["flags"])
        endpoint_flags[row["endpoint"]].update(diagnosis["flags"])
        category_flags[(row["endpoint"], row["category"])].update(diagnosis["flags"])
        for event in diagnosis["events"]:
            durations[event["stage"]].append(event["elapsed_seconds"])
            evidence_issues.update(event.get("evidence_issues", []))
            if event["status"] != "ok":
                failures[(event["stage"], event.get("failure", "unknown_failure"))] += 1
        items.append({"id": row["id"], "endpoint": row["endpoint"], "category": row["category"],
                      "server_elapsed_seconds": round(row["elapsed"], 6), **diagnosis})
    return {"trace_complete_count": complete, "trace_incomplete_count": len(rows) - complete,
            "flag_counts": dict(flags), "flags_by_endpoint": {name: dict(counts) for name, counts in endpoint_flags.items()},
            "flags_by_category": [{"endpoint": endpoint, "category": category, "flags": dict(counts)}
                                  for (endpoint, category), counts in sorted(category_flags.items())],
            "evidence_issue_events": dict(evidence_issues), "stage_failures": [
                {"stage": stage, "failure": reason, "count": count} for (stage, reason), count in sorted(failures.items())],
            "stage_latency_seconds": {stage: {"observed_calls": len(values), **latency_summary(values)}
                                      for stage, values in sorted(durations.items())},
            "interpretation": "overlapping_observations_not_additive_score_loss_or_proven_causality",
            "stage_timing": "nested_intervals_not_additive", "items": items}
