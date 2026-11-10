"""검색만 따로 평가: 정답 근거가 있는 튜터 문항에서 정답 섹션의 recall@k, MRR.

  python3 scripts/eval_retrieval.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import KB, build_query, embed_async  # noqa: E402


def load():
    rows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["labels"].get("evidence")]


async def main():
    kb = KB()
    rows = load()
    print(f"섹션 {len(kb.secs)}개, 청크 {len(kb.chunks)}개, 평가 문항 {len(rows)}개")
    for qmode in ["last", "last+users", "all"]:
        qs = [build_query(r["messages"], qmode) for r in rows]
        sem = asyncio.Semaphore(16)

        async def emb(q):
            async with sem:
                return await embed_async(q)
        vecs = await asyncio.gather(*[emb(q) for q in qs])
        for name, kw in [("bm25", dict(qvec=None)), ("emb", dict(w_bm25=0.0)), ("hybrid", {})]:
            hits = {1: 0, 3: 0, 5: 0, 8: 0}
            mrr = 0.0
            for r, q, v in zip(rows, qs, vecs):
                gold = r["labels"]["evidence"][0]
                qv = None if name == "bm25" else v
                ranked = [s for s, _ in kb.rank(q, qvec=qv, **{k: x for k, x in kw.items() if k != "qvec"})]
                pos = ranked.index(gold) + 1 if gold in ranked else 999
                mrr += 1 / pos
                for k in hits:
                    hits[k] += pos <= k
            n = len(rows)
            print(f"  query={qmode:10s} {name:6s} " + " ".join(f"R@{k}={hits[k] / n:.3f}" for k in hits) + f" MRR={mrr / n:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
