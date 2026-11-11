"""실전 조건 부하 테스트: 공개 세트로 약 600개 요청을 동시 N개로 보내 시간·형식 오류를 본다(판정 모델 없음).

  python3 scripts/load_test.py --base https://<런박스>.tunnel.elice.io/proxy/8000 --n 600 --concurrency 64
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
import scoring  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=30)
    a = ap.parse_args()
    tok = os.environ["TEAM_TOKEN"]
    pub = {e: [json.loads(l) for l in (ROOT / f"public/{e}.jsonl").read_text().splitlines() if l.strip()] for e in ("tutor", "guard")}
    jobs = []
    i = 0
    while len(jobs) < a.n:  # 튜터·가드를 번갈아 섞어 실제처럼 동시에 들어오게
        e = "tutor" if i % 2 == 0 else "guard"
        p = pub[e][(i // 2) % len(pub[e])]
        jobs.append((e, p))
        i += 1

    def one(job):
        e, p = job
        body = {"spec_version": "1.0", "session_id": "load", "messages": p["messages"]}
        if p.get("candidate_answer"):
            body["candidate_answer"] = p["candidate_answer"]
        req = urllib.request.Request(f"{a.base}/{e}", data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                code, raw = r.status, r.read(256 * 1024 + 1)
            obj = json.loads(raw)
            ok = (scoring.parse_tutor(obj) if e == "tutor" else scoring.parse_guard(obj, bool(p.get("candidate_answer")))) is not None
            res = "ok" if ok else "format_error"
        except urllib.error.HTTPError as ex:
            code, res = ex.code, f"http_{ex.code}"
        except Exception as ex:  # noqa: BLE001
            code, res = 0, "timeout" if "timed out" in str(ex) else type(ex).__name__
        return e, res, time.time() - t

    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        out = list(ex.map(one, jobs))
    wall = time.time() - t0
    print(f"요청 {len(out)}개, 동시 {a.concurrency}, 전체 {wall:.1f}초, 처리량 {len(out) / wall:.1f} req/s")
    print("결과:", dict(Counter(r for _, r, _ in out)))
    for e in ("tutor", "guard"):
        lat = sorted(d for ee, _, d in out if ee == e)
        print(f"  {e}: p50 {lat[len(lat) // 2]:.1f}s  p95 {lat[int(len(lat) * .95)]:.1f}s  max {lat[-1]:.1f}s  (>25s: {sum(d > 25 for d in lat)})")


if __name__ == "__main__":
    main()
