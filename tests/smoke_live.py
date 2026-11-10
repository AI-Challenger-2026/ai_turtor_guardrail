"""Live smoke test against a running server (default: the 8001 test server).

Makes at most two real LLM calls (one /tutor answer per run, plus none for /guard in step 2).
Every request carries a unique marker; afterwards grep the server log for it:
    grep -c <marker> <server log>   # must be 0

Run: python3 tests/smoke_live.py --base http://localhost:8001
"""
import argparse
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
import scoring  # noqa: E402


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def call(url, body, token, method="POST"):
    data = body if isinstance(body, bytes) else (None if body is None else json.dumps(body, ensure_ascii=False).encode())
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    t0 = time.perf_counter()
    try:
        with OPENER.open(req, timeout=30) as resp:
            status, raw, hdrs = resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        status, raw, hdrs = e.code, e.read(), e.headers
    elapsed = time.perf_counter() - t0
    try:
        payload = json.loads(raw)
    except Exception:
        payload = None
    return status, payload, hdrs, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--token", default=os.environ.get("TEAM_TOKEN", ""))
    a = ap.parse_args()
    marker = "SMOKE-" + secrets.token_hex(6)
    tutor, guard = a.base + "/tutor", a.base + "/guard"
    msgs = [{"role": "user", "content": f"파이썬에서 print는 무엇을 하나요? {marker}"}]
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    for url in (tutor, guard):
        ep = url.rsplit("/", 1)[1]
        check(f"{ep}: no token -> 401", call(url, {"messages": msgs}, None)[0] == 401)
        check(f"{ep}: wrong token -> 401", call(url, {"messages": msgs}, "wrong")[0] == 401)
        check(f"{ep}: wrong token + bad JSON -> 401", call(url, b"{bad " + marker.encode(), "wrong")[0] == 401)
        s, p, h, _ = call(url + "/", {"messages": msgs}, a.token)
        check(f"{ep}/: 404 without redirect", s == 404 and "location" not in {k.lower() for k in h.keys()}, f"status={s}")
        check(f"{ep}: GET -> 405", call(url, None, a.token, method="GET")[0] == 405)

    s, p, _, _ = call(tutor, b"{bad " + marker.encode(), a.token)
    check("tutor: bad JSON -> 200 clarify", s == 200 and p and p.get("action") == "clarify" and scoring.parse_tutor(p) is not None)
    s, p, _, _ = call(tutor, {"messages": [{"role": "system", "content": marker}]}, a.token)
    check("tutor: bad role -> 200 clarify", s == 200 and p and p.get("action") == "clarify")
    s, p, _, _ = call(guard, b"{bad " + marker.encode(), a.token)
    check("guard: bad JSON -> 200 with output", s == 200 and scoring.parse_guard(p, True) is not None)
    s, p, _, _ = call(guard, {"messages": msgs}, a.token)
    check("guard: no candidate -> valid, no output", s == 200 and scoring.parse_guard(p, False) is not None and "output" not in p)
    s, p, _, _ = call(guard, {"messages": msgs, "candidate_answer": f"출력해요 {marker}"}, a.token)
    check("guard: candidate -> valid with output", s == 200 and scoring.parse_guard(p, True) is not None)
    check("docs disabled", call(a.base + "/docs", None, None, method="GET")[0] == 404)

    s, p, _, t = call(tutor, {"spec_version": "1.0", "session_id": "s_smoke", "messages": msgs}, a.token)
    ok = s == 200 and scoring.parse_tutor(p) is not None
    check("tutor: real answer is spec-valid", ok, f"status={s} latency={t:.2f}s action={p and p.get('action')} "
          f"answer_chars={p and len(p.get('answer', ''))}")

    print(f"\nmarker: {marker}")
    print(f"{sum(checks)}/{len(checks)} passed")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
