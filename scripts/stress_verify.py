"""검증기 일반화 점검(튜닝에 쓰지 않은 데이터).

(1) 정밀도: 튜터 공개 세트의 정답 답변(reference_answer)을 후보 답변으로 넣는다.
    정의상 근거가 있는 답변이므로 unsupported가 비어야 한다. 비어 있지 않으면 오탐(FP).

  python3 scripts/stress_verify.py [--show]
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import KB  # noqa: E402
from app.verify import find_unsupported  # noqa: E402


def query_for(messages, cand):
    return "\n".join(m["content"] for m in messages[-4:]) + "\n" + cand + "\n" + cand


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--split", choices=["even", "odd", "all"], default="all", help="even=튜닝용, odd=최종 확인용(튜닝에 쓰지 않음)")
    a = ap.parse_args()
    kb = KB()
    rows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text().splitlines() if l.strip()]
    if a.split != "all":
        rows = rows[0::2] if a.split == "even" else rows[1::2]
    sem = asyncio.Semaphore(32)

    async def one(r):
        cand = r["labels"]["reference_answer"]
        secs = [kb.secs[s] for s, _ in kb.rank(query_for(r["messages"], cand))[:4]]
        async with sem:
            try:
                return await find_unsupported(r["messages"], cand, secs)
            except Exception as e:  # noqa: BLE001
                return ["<ERR " + type(e).__name__ + ">"]

    preds = await asyncio.gather(*[one(r) for r in rows])
    fp = Counter()
    n = Counter()
    for r, sp in zip(rows, preds):
        c = r["labels"]["action"]
        n[c] += 1
        if sp:
            fp[c] += 1
            if a.show:
                print(f"[{r['id']}] {c} 오탐 {sp}\n   REF: {r['labels']['reference_answer'][:200]}")
    tot = sum(fp.values())
    print(f"정답 답변 {len(rows)}개 중 근거 없음으로 잘못 판정 {tot}개 ({tot / len(rows):.1%})")
    for c in n:
        print(f"  {c:9s} {fp[c]:3d}/{n[c]}")


if __name__ == "__main__":
    asyncio.run(main())
