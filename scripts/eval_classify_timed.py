"""분류기 평가 + 문항별 지연. 설정은 환경변수(CLASSIFY_EFFORT, CLASSIFY_VOTES)로 바꾼다.

  CLASSIFY_EFFORT=medium CLASSIFY_VOTES=1 python3 scripts/eval_classify_timed.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.classify import classify  # noqa: E402
from app.policy import guard_decision, tutor_action  # noqa: E402


async def main():
    rows = [json.loads(l) for e in ("tutor", "guard") for l in (ROOT / f"public/{e}.jsonl").read_text().splitlines() if l.strip()]
    sem = asyncio.Semaphore(32)

    async def one(r):
        async with sem:
            t = time.time()
            p = await classify(r["messages"])
            return p, time.time() - t

    t0 = time.time()
    res = await asyncio.gather(*[one(r) for r in rows])
    wall = time.time() - t0
    preds = [p for p, _ in res]
    lat = sorted(d for _, d in res)
    gold = [r["labels"]["category"] for r in rows]
    pc = [p["category"] for p in preds]
    nn = [(g, p) for g, p in zip(gold, pc) if g != "normal"]
    g_rows = [(r, p) for r, p in zip(rows, preds) if r["endpoint"] == "guard"]
    tp = sum(r["labels"]["decision"] == "block" and guard_decision(p["category"]) == "block" for r, p in g_rows)
    fp = sum(r["labels"]["decision"] == "allow" and guard_decision(p["category"]) == "block" for r, p in g_rows)
    fn = sum(r["labels"]["decision"] == "block" and guard_decision(p["category"]) == "allow" for r, p in g_rows)
    gnn = [(r, p) for r, p in g_rows if r["labels"]["category"] != "normal"]
    t_rows = [(r, p) for r, p in zip(rows, preds) if r["endpoint"] == "tutor"]
    act = sum(r["labels"]["action"] == tutor_action(p["category"], p["harmful_type"]) for r, p in t_rows) / len(t_rows)
    cfg = f"effort={os.environ.get('CLASSIFY_EFFORT', 'low')} votes={os.environ.get('CLASSIFY_VOTES', '3')}"
    print(f"[{cfg}] block F1 {2 * tp / (2 * tp + fp + fn):.3f} (FP {fp} FN {fn}) | 범주(normal 제외) {sum(g == p for g, p in nn) / len(nn):.3f} "
          f"| 가드 범주 {sum(r['labels']['category'] == p['category'] for r, p in gnn) / len(gnn):.3f} | 튜터 action {act:.3f} "
          f"| 지연 p50 {lat[len(lat) // 2]:.1f}s p95 {lat[int(len(lat) * .95)]:.1f}s max {lat[-1]:.1f}s | 전체 {wall:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
