"""'자료에 있나'(found) 판단만 따로 평가(인용 검증 전용 확인 실험은 DEVLOG 8장, 커밋 afb4d93): 튜터 normal·pii(있어야 함) vs out_of_material(없어야 함).
서버 없이 검색 + 답변 생성 함수를 직접 부른다.

  python3 scripts/eval_found.py [--k 1] [--show]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.answer import grounded_answer  # noqa: E402
from app.kb import KB, build_query  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--split", choices=["even", "odd", "all"], default="all")
    a = ap.parse_args()
    kb = KB()
    rows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["labels"]["category"] in ("normal", "pii_disclosure", "out_of_material")]
    if a.split != "all":
        rows = rows[0::2] if a.split == "even" else rows[1::2]
    sem = asyncio.Semaphore(32)

    async def one(r):
        secs = [kb.secs[s] for s, _ in kb.rank(build_query(r["messages"], "all"))[:a.k]]
        mode = "answer"
        async with sem:
            try:
                return await grounded_answer(r["messages"], secs, mode)
            except Exception as e:  # noqa: BLE001
                return {"found": None, "answer": f"<ERR {type(e).__name__}>"}

    outs = await asyncio.gather(*[one(r) for r in rows])
    pos = [(r, o) for r, o in zip(rows, outs) if r["labels"]["category"] != "out_of_material"]
    neg = [(r, o) for r, o in zip(rows, outs) if r["labels"]["category"] == "out_of_material"]
    tp = sum(o["found"] is True for _, o in pos)
    tn = sum(o["found"] is False for _, o in neg)
    print(f"있어야 함(normal·pii) {tp}/{len(pos)} 있음 판정 | 없어야 함(out_of_material) {tn}/{len(neg)} 없음 판정 | 실패 {sum(o['found'] is None for o in outs)}")
    if a.show:
        for r, o in pos:
            if o["found"] is not True:
                print(f"  [놓침 {r['id']}] {r['messages'][-1]['content'][:100]!r} → {o['answer'][:100]!r}")
        for r, o in neg:
            if o["found"] is not False:
                print(f"  [지어냄 {r['id']}] {r['messages'][-1]['content'][:100]!r} → {o['answer'][:120]!r}")


if __name__ == "__main__":
    asyncio.run(main())
