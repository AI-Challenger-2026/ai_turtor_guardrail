"""Summarize a saved server log after a run (no LLM calls, no contact with the server).

    python3 eval/log_summary.py logs/8000.log --last-run
    python3 eval/log_summary.py logs/8000.log --since "2026-11-11 05:00:00" --until "2026-11-11 05:10:00"
    python3 eval/log_summary.py logs/8000.log --last-run --row      # one row for docs/SUBMISSIONS.md

The report holds counts and latencies only and never prints a line of the log. A line that is not
in the allowlisted metadata form is counted, so anything unexpected shows up as a non-zero audit.
Read a submission's log only after its grading has finished.
"""
import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ALLOWED = {"event", "endpoint", "status", "latency_ms", "body_bytes", "input_chars", "n_msgs",
           "outcome", "degraded", "error", "kind", "attempt", "hedge", "kb_sections", "kb_chunks", "embeddings"}
EVENT_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} (event=.*)$")
UVICORN_LINE = re.compile(r"^(?:INFO|WARNING):\s+(?:Started server process|Waiting for application|"
                          r"Application (?:startup|shutdown)|Uvicorn running on|Shutting down|"
                          r"Finished server process)")
# Brackets appear in error=ValidationError[field], which names failing fields and never values.
FIELD = re.compile(r"^([a-z_]+)=([\w./,\[\]-]*)$")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
JUDGE_STAGES = {"classify", "ground", "found"}
GENERATE_TIMEOUT = "generate_timeout"
# Tutor answers given without a generated answer: plain fallback text, extract from the top
# section, or the fixed academic-dishonesty text.
GENERATE_OUTCOMES = {"fallback_generate", "fallback_extractive", "fallback_academic"}


def parse(lines):
    """Return (events, audit). An event is its fields plus a `t` datetime."""
    events = []
    audit = Counter(event_lines=0, uvicorn_lines=0, other_lines=0, bad_fields=0)
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if UVICORN_LINE.match(line):
            audit["uvicorn_lines"] += 1
            continue
        match = EVENT_LINE.match(line)
        if match is None:
            audit["other_lines"] += 1
            continue
        fields = [FIELD.match(token) for token in match.group(2).split(" ")]
        if not all(f is not None and f.group(1) in ALLOWED for f in fields):
            audit["bad_fields"] += 1
            continue
        audit["event_lines"] += 1
        events.append({**{f.group(1): f.group(2) for f in fields},
                       "t": datetime.strptime(match.group(1), TIME_FORMAT)})
    return events, audit


def last_run_start(events, gap_s):
    """Time of the first request after the last silence of at least `gap_s` seconds."""
    times = sorted(e["t"] for e in events if e.get("event") == "request")
    if not times:
        return None
    start = times[0]
    for before, after in zip(times, times[1:]):
        if after - before >= timedelta(seconds=gap_s):
            start = after
    return start


def percentile(values, q):
    """Value at index int(q * n), the definition behind the latencies already in PROGRESS.md."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(events):
    requests = [e for e in events if e.get("event") == "request"]
    calls = [e for e in events if e.get("event") == "llm"]
    latency = {}
    for endpoint in sorted({e.get("endpoint", "other") for e in requests}):
        ms = [int(e["latency_ms"]) for e in requests
              if e.get("endpoint") == endpoint and e.get("status") == "200"]
        if ms:
            latency[endpoint] = {"n": len(ms), "p50": round(percentile(ms, 0.5) / 1000, 1),
                                 "p95": round(percentile(ms, 0.95) / 1000, 1), "max": round(max(ms) / 1000, 1)}
    fallback = Counter(deadline=0, generate=0, judge=0, other=0)
    degraded = Counter()
    for e in requests:
        stages = set(filter(None, e.get("degraded", "").split(",")))
        degraded.update(stages)
        outcome = e.get("outcome", "")
        # A failed speculative generation is also marked degraded when a fixed text was sent
        # anyway, so only requests whose outcome is a fallback count here.
        if outcome.startswith("fallback_"):
            if outcome == "fallback_deadline" or GENERATE_TIMEOUT in stages:
                fallback["deadline"] += 1
            elif outcome in GENERATE_OUTCOMES:
                fallback["generate"] += 1
            else:
                fallback["other"] += 1
        if stages & JUDGE_STAGES:
            fallback["judge"] += 1
    by_kind = {}
    for e in calls:
        by_kind.setdefault(e.get("kind", "other"), Counter())[e.get("status", "other")] += 1
    times = [e["t"] for e in requests]
    return {
        "from": min(times).strftime(TIME_FORMAT) if times else None,
        "to": max(times).strftime(TIME_FORMAT) if times else None,
        "requests": dict(Counter(e.get("endpoint", "other") for e in requests)),
        "status": dict(Counter(e.get("status", "other") for e in requests)),
        "latency_s": latency,
        "fallback": dict(fallback),
        "outcomes": dict(Counter(e.get("outcome", "none") for e in requests)),
        "degraded": dict(degraded),
        "llm": {"calls": len(calls),
                "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
                "retries": sum(int(e.get("attempt", "1")) > 1 for e in calls),
                "hedges": sum(e.get("hedge") == "1" for e in calls),
                "cancelled": sum(e.get("status") == "cancelled" for e in calls),
                "api_errors": dict(Counter(e.get("error", "other") for e in calls if e.get("status") == "error")),
                "invalid": sum(e.get("status") == "invalid" for e in calls)},
    }


def _latency_text(latency, endpoint):
    x = latency.get(endpoint)
    return f"{x['p50']}/{x['p95']}/{x['max']}" if x else "-"


def line(report, audit):
    f, llm = report["fallback"], report["llm"]
    kinds = ", ".join(f"{k} {sum(v.values())}" for k, v in llm["by_kind"].items())
    return (f"{report['from']} ~ {report['to']} | 요청 {report['requests']} 상태 {report['status']} | "
            f"대체 응답 마감 {f['deadline']}, 생성 실패 {f['generate']}, 판정 실패 {f['judge']}, 그 외 {f['other']} | "
            f"API 오류 {sum(llm['api_errors'].values())} {llm['api_errors']}, 형식 실패 {llm['invalid']} | "
            f"호출 {llm['calls']} ({kinds}), 재시도 {llm['retries']}, 헤지 {llm['hedges']}, 취소 {llm['cancelled']} | "
            f"튜터 p50/p95/최대 {_latency_text(report['latency_s'], '/tutor')}초 | "
            f"가드 p50/p95/최대 {_latency_text(report['latency_s'], '/guard')}초 | "
            f"감사 허용 밖 줄 {audit['other_lines']}, 허용 밖 필드 {audit['bad_fields']}")


def row(report):
    """A docs/SUBMISSIONS.md row; the fields only the result message knows stay as `?`."""
    f, llm = report["fallback"], report["llm"]
    cells = ["?", report["from"], "?", "?", "?", "?", "?",
             f"{f['deadline']}/{f['generate']}/{f['judge']}/{f['other']}", str(sum(llm["api_errors"].values())),
             _latency_text(report["latency_s"], "/tutor"), _latency_text(report["latency_s"], "/guard"),
             str(llm["calls"]), ""]
    return "| " + " | ".join(cells) + " |"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize a saved server log; counts and latencies only")
    ap.add_argument("log")
    ap.add_argument("--since", help="keep events from this time, YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--until", help="keep events up to this time, YYYY-MM-DD HH:MM:SS")
    ap.add_argument("--last-run", action="store_true", help="keep only the last burst of requests")
    ap.add_argument("--gap", type=float, default=120, help="seconds of silence that separate runs (default 120)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--row", action="store_true", help="print a docs/SUBMISSIONS.md table row")
    a = ap.parse_args(argv)
    with Path(a.log).open(encoding="utf-8", errors="replace") as log:
        events, audit = parse(log)
    since = datetime.strptime(a.since, TIME_FORMAT) if a.since else None
    until = datetime.strptime(a.until, TIME_FORMAT) if a.until else None
    if since or until:
        events = [e for e in events if (not since or e["t"] >= since) and (not until or e["t"] <= until)]
    if a.last_run:
        start = last_run_start(events, a.gap)
        events = [e for e in events if start is not None and e["t"] >= start]
    report = summarize(events)
    if a.json:
        print(json.dumps({**report, "audit": dict(audit)}, ensure_ascii=False, indent=1))
    elif a.row:
        print(row(report))
    else:
        print(line(report, audit))


if __name__ == "__main__":
    main()
