"""가드 out_of_material 범주 판정 오프라인 평가(DEVLOG 20장). 가드 공개 200문항.
분류는 18장 캐시(5표 과반)를 쓰고, found는 실제 생성 호출로 계산한다(약 200회).

  python3 scripts/eval_oom_guard.py CACHE.json
"""
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sweep_p4 as P  # noqa: E402
from app.answer import grounded_answer  # noqa: E402
from app.kb import KB, build_query  # noqa: E402
from app.policy import guard_decision  # noqa: E402


async def main(cache):
    d = json.loads(Path(cache).read_text(encoding="utf-8"))
    src = {r["id"]: r for r in P.load("guard")}
    rows = [(src[c["id"]], c["votes"]) for c in d["votes"] if c["ep"] == "guard"]
    kb = KB(); sem = asyncio.Semaphore(32)

    async def found(r):
        secs = [kb.secs[s] for s, _ in kb.rank(build_query(r["messages"], "all"))[:1]]
        async with sem:
            try:
                return (await grounded_answer(r["messages"], secs, "answer"))["found"]
            except Exception:  # noqa: BLE001
                return None
    fs = await asyncio.gather(*[found(r) for r, _ in rows])
    before = after = n = 0; flips = Counter(); bad = Counter(); per = Counter(); tot = Counter()
    for (r, v), f in zip(rows, fs):
        g = r["labels"]["category"]; base = P.decide_from(v, 3, 3)["category"]
        new = "out_of_material" if (base == "normal" and f is False) else base
        if g != "normal":
            n += 1; before += base == g; after += new == g; tot[g] += 1; per[g] += new == g
            if base == g and new != g:
                bad[g] += 1
        elif new != base:
            flips["정상→OOM(점수 영향 없음)"] += 1
    print(f"범주 정확도(정답 비정상 {n}개): 현재 {before / n:.3f} → {after / n:.3f}")
    print("범주별(적용 후): " + " ".join(f"{k[:8]}{per[k]}/{tot[k]}" for k in tot))
    print("기존 정답이 깨진 문항(손해):", dict(bad) or 0, "| 정상 입력 flip:", dict(flips), "| found 실패(None):", sum(f is None for f in fs))
    blk = sum(guard_decision(P.decide_from(v, 3, 3)["category"]) != guard_decision("out_of_material" if (P.decide_from(v, 3, 3)["category"] == "normal" and f is False) else P.decide_from(v, 3, 3)["category"]) for (r, v), f in zip(rows, fs))
    print("판정(block/allow)이 바뀐 문항(0이어야 함):", blk)


asyncio.run(main(sys.argv[1]))
