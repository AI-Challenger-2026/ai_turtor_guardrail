"""가드 block 판정의 표 분포 분석: 막는 신호(jailbreak, harm, hate, submit_work, off_topic)에 몇 표가 모였나.
정답이 block인 문항과 allow인 문항에서 표 분포(0~VOTES표)를 비교한다.

  python3 scripts/vote_analysis.py --votes 3 --runs 2
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import classify_feat  # noqa: E402
from app.classify import render, EFFORT, MAX_TOKENS, TIMEOUT  # noqa: E402
from app.llm import chat_json  # noqa: E402

BLOCK_FLAGS = ("jailbreak", "harm", "hate", "submit_work", "off_topic")
PRE = ("distress", "low_signal")  # decide()에서 block 신호보다 먼저 보는 신호


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "public/guard.jsonl").read_text().splitlines() if l.strip()]
    sem = asyncio.Semaphore(32)

    async def one(r):
        async with sem:
            outs = await asyncio.gather(*[chat_json(classify_feat.SYSTEM, render(r["messages"]), effort=EFFORT,
                                                    max_tokens=MAX_TOKENS, timeout=TIMEOUT) for _ in range(a.votes)],
                                        return_exceptions=True)
        outs = [o for o in outs if isinstance(o, dict)]
        # 문항이 block으로 판정되려면: 앞선 신호(distress, low_signal)가 과반이 아니고 block 신호 중 하나가 과반
        blk = max(sum(o.get(f) is True for o in outs) for f in BLOCK_FLAGS) if outs else 0
        return blk, len(outs)

    dist = Counter()
    for _ in range(a.runs):
        res = await asyncio.gather(*[one(r) for r in rows])
        for r, (blk, n) in zip(rows, res):
            dist[(r["labels"]["decision"], blk)] += 1
    print(f"표 수 {a.votes}, {a.runs}회 × 가드 {len(rows)}문항")
    for gold in ("block", "allow"):
        tot = sum(v for (g, _), v in dist.items() if g == gold)
        print(f"  정답 {gold:5s}: " + " | ".join(f"{k}표 {dist[(gold, k)]:3d}" for k in range(a.votes + 1)) + f"  (총 {tot})")


if __name__ == "__main__":
    asyncio.run(main())
