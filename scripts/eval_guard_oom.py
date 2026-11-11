"""실험 C: 분류기가 normal로 본 입력에 '자료에 답이 있나'(답변 생성기의 found)를 적용해 out_of_material을 붙이면
범주 정확도(normal 제외 문항만 채점)가 얼마나 오르는지. 판정(allow/block)은 바뀌지 않는다.
"""
import asyncio, json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.classify import classify  # noqa: E402
from app.answer import grounded_answer  # noqa: E402
from app.kb import KB, build_query  # noqa: E402


async def main():
    kb = KB(use_embeddings=False) if "use_embeddings" in KB.__init__.__code__.co_varnames else KB()
    rows = [json.loads(l) for e in ("tutor", "guard") for l in (ROOT / f"public/{e}.jsonl").read_text().splitlines() if l.strip()]
    sem = asyncio.Semaphore(32)

    async def one(r):
        async def gen():
            try:
                return await grounded_answer(r["messages"], [kb.secs[kb.rank(build_query(r["messages"], "all"))[0][0]]], "answer")
            except Exception:  # noqa: BLE001
                return None  # 실패하면 범주를 바꾸지 않음(서버와 같은 처리)
        async with sem:
            c, g = await asyncio.gather(classify(r["messages"]), gen())
        new = "out_of_material" if c["category"] == "normal" and g is not None and not g["found"] else c["category"]
        return c["category"], new

    res = await asyncio.gather(*[one(r) for r in rows])
    for name, sel in [("가드 세트", lambda r: r["endpoint"] == "guard"), ("전체 400", lambda r: True)]:
        idx = [i for i, r in enumerate(rows) if sel(r) and r["labels"]["category"] != "normal"]
        b = sum(rows[i]["labels"]["category"] == res[i][0] for i in idx) / len(idx)
        a = sum(rows[i]["labels"]["category"] == res[i][1] for i in idx) / len(idx)
        print(f"{name}: 범주 정확도(normal 제외, n={len(idx)}) {b:.3f} → {a:.3f}")
    ch = Counter((rows[i]["labels"]["category"], res[i][0] == rows[i]["labels"]["category"], res[i][1] == rows[i]["labels"]["category"])
                 for i in range(len(rows)) if res[i][0] != res[i][1])
    print("바뀐 문항(정답 범주, 전 맞음, 후 맞음):", dict(ch))

asyncio.run(main())
