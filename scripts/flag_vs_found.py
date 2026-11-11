"""분류 결과(막는 범주)와 답변 생성기의 found 판단을 교차해, 'found=true이면 정상으로 되돌리기' 규칙의 손익을 본다.
공개 400문항(튜터+가드), 분류와 생성을 서버와 같은 설정으로 실행.

  python3 scripts/flag_vs_found.py --runs 2
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.answer import grounded_answer  # noqa: E402
from app.classify import classify  # noqa: E402
from app.kb import KB, build_query  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    kb = KB(use_embeddings=False)
    rows = [json.loads(l) for e in ("tutor", "guard") for l in (ROOT / f"public/{e}.jsonl").read_text().splitlines() if l.strip()]
    sem = asyncio.Semaphore(32)

    async def one(r):
        async def gen():
            try:
                g = await grounded_answer(r["messages"], [kb.secs[kb.rank(build_query(r["messages"], "all"))[0][0]]], "answer")
                return g["found"]
            except Exception:  # noqa: BLE001
                return None
        async with sem:
            c, f = await asyncio.gather(classify(r["messages"]), gen())
        return c["category"], f

    tab = Counter()
    for _ in range(a.runs):
        res = await asyncio.gather(*[one(r) for r in rows])
        for r, (cat, found) in zip(rows, res):
            if cat in ("off_topic", "jailbreak_injection", "harmful_content", "academic_dishonesty", "answer_seeking"):
                gold = r["labels"]["category"]
                tab[(cat, found, "정답도 같은 범주" if gold == cat else f"정답 {gold}")] += 1
    print(f"{a.runs}회 합계. (분류 결과, 생성기 found, 정답) 별 건수")
    for k in sorted(tab, key=lambda x: (x[0], str(x[1]), x[2])):
        print(f"  {k[0]:20s} found={str(k[1]):5s} {k[2]:22s} {tab[k]}")


if __name__ == "__main__":
    asyncio.run(main())
