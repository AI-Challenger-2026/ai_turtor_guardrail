"""답변 생성기의 found(자료에 답이 있나) 판단이 문항마다 얼마나 흔들리는지 기록하고, 규칙별 결과를 계산한다.
대상: 정답 범주가 normal / pii_disclosure / out_of_material인 공개 문항(튜터+가드).

  python3 scripts/found_samples.py --collect OUT.json --k 3
  python3 scripts/found_samples.py --eval OUT.json
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CATS = ("normal", "pii_disclosure", "out_of_material")


def rows():
    out = []
    for e in ("tutor", "guard"):
        for l in (ROOT / f"public/{e}.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["labels"]["category"] in CATS:
                out.append(r)
    return out


async def collect(path, k):
    from app.answer import grounded_answer
    from app.kb import KB, build_query
    kb = KB(use_embeddings=False)
    rs = rows()
    sem = asyncio.Semaphore(32)

    async def one(r):
        sec = [kb.secs[kb.rank(build_query(r["messages"], "all"))[0][0]]]

        async def g():
            async with sem:
                try:
                    return (await grounded_answer(r["messages"], sec, "answer"))["found"]
                except Exception:  # noqa: BLE001
                    return None
        return await asyncio.gather(*[g() for _ in range(k)])
    res = await asyncio.gather(*[one(r) for r in rs])
    Path(path).write_text(json.dumps([{"id": r["id"], "endpoint": r["endpoint"], "cat": r["labels"]["category"], "found": f}
                                      for r, f in zip(rs, res)], ensure_ascii=False))
    print("저장", path, len(res))


def evaluate(path):
    data = json.loads(Path(path).read_text())
    # 흔들림 분포
    stab = Counter()
    for d in data:
        f = [x for x in d["found"] if x is not None]
        stab[(d["cat"] == "out_of_material", sum(f), len(f))] += 1
    print("found=True 횟수 분포 (자료 없음 여부, True 횟수, 유효 표본):")
    for k in sorted(stab):
        print(f"   {'자료없음' if k[0] else '정상/개인정보'}  True {k[1]}/{k[2]}: {stab[k]}")

    def rule_found(f, rule):
        f = [x for x in f if x is not None]
        if not f:
            return True
        if rule == "first":
            return f[0]
        if rule == "and2":
            return all(f[:2])
        if rule == "or2":
            return any(f[:2])
        if rule == "maj3":
            return sum(f[:3]) * 2 > len(f[:3])
        if rule == "all3":
            return all(f[:3])
    print("\n[튜터] 규칙별 action 결과 (정상·개인정보는 found=True가 정답, 자료 없음은 False가 정답)")
    for rule in ("first", "and2", "or2", "maj3", "all3"):
        t = [d for d in data if d["endpoint"] == "tutor"]
        pos_ok = sum(rule_found(d["found"], rule) for d in t if d["cat"] != "out_of_material")
        neg_ok = sum(not rule_found(d["found"], rule) for d in t if d["cat"] == "out_of_material")
        npos = sum(d["cat"] != "out_of_material" for d in t)
        nneg = sum(d["cat"] == "out_of_material" for d in t)
        # 문항당 대략 손익: 정상 맞으면 +0.75(action·사실·인용), 자료없음 맞으면 +0.5(action·근거 충실)
        print(f"   {rule:6s} 정상 {pos_ok}/{npos} | 자료없음 {neg_ok}/{nneg} | 가중 점수 {0.75 * pos_ok + 0.5 * neg_ok:.2f}")
    print("\n[가드 범주] 자료 없음 문항 탐지(정상 문항은 범주 미채점이라 손해 없음): 'k번 중 한 번이라도 False면 자료 없음'")
    g = [d for d in data if d["endpoint"] == "guard" and d["cat"] == "out_of_material"]
    for k in (1, 2, 3):
        ok = sum(any(x is False for x in d["found"][:k]) for d in g)
        print(f"   k={k}: {ok}/{len(g)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--eval")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.k))
    if a.eval:
        evaluate(a.eval)
