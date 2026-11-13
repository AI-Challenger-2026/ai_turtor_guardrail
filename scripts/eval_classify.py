"""분류기만 따로 공개 세트(tutor+guard 400문항)에 돌려 범주 정확도와 혼동 행렬을 본다.
서버·판정 모델 없이 빠르게 반복하기 위한 개발용 스크립트.

  python3 scripts/eval_classify.py [--only guard|tutor] [--show-errors]
"""
import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.classify import classify  # noqa: E402
from app.policy import guard_decision, tutor_action  # noqa: E402


def load(which):
    rows = []
    for e in which:
        rows += [json.loads(l) for l in (ROOT / "public" / f"{e}.jsonl").read_text().splitlines() if l.strip()]
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["guard", "tutor"])
    ap.add_argument("--show-errors", action="store_true")
    a = ap.parse_args()
    rows = load([a.only] if a.only else ["tutor", "guard"])
    sem = asyncio.Semaphore(32)

    async def one(r):
        async with sem:
            return await classify(r["messages"])

    preds = await asyncio.gather(*[one(r) for r in rows])
    gold = [r["labels"]["category"] for r in rows]
    pc = [p["category"] for p in preds]

    acc = sum(g == p for g, p in zip(gold, pc)) / len(rows)
    nn = [(g, p) for g, p in zip(gold, pc) if g != "normal"]
    print(f"범주 정확도 {acc:.3f} (n={len(rows)}), normal 제외 {sum(g == p for g, p in nn) / len(nn):.3f} (n={len(nn)})")

    # 가드 block F1 (가드 세트만)
    g_rows = [(r, p) for r, p in zip(rows, preds) if r["endpoint"] == "guard"]
    if g_rows:
        tp = sum(r["labels"]["decision"] == "block" and guard_decision(p["category"]) == "block" for r, p in g_rows)
        fp = sum(r["labels"]["decision"] == "allow" and guard_decision(p["category"]) == "block" for r, p in g_rows)
        fn = sum(r["labels"]["decision"] == "block" and guard_decision(p["category"]) == "allow" for r, p in g_rows)
        print(f"가드 block F1 {2 * tp / (2 * tp + fp + fn):.3f} (TP {tp}, FP {fp}, FN {fn})")
    # 튜터 action 정확도 (튜터 세트만)
    t_rows = [(r, p) for r, p in zip(rows, preds) if r["endpoint"] == "tutor"]
    if t_rows:
        ok = sum(r["labels"]["action"] == tutor_action(p["category"], p["harmful_type"]) for r, p in t_rows)
        print(f"튜터 action 정확도(분류 기준) {ok / len(t_rows):.3f} (n={len(t_rows)})")

    print("\n범주별 재현율 (정답 범주 기준)")
    for c, n in Counter(gold).most_common():
        hit = sum(g == c and p == c for g, p in zip(gold, pc))
        wrong = Counter(p for g, p in zip(gold, pc) if g == c and p != c)
        print(f"  {c:22s} {hit:3d}/{n:<3d} 오분류 {dict(wrong)}")
    if a.show_errors:
        print("\n오분류 사례")
        for r, g, p in zip(rows, gold, preds):
            if g != p["category"]:
                print(f"  [{r['id']}] {g} → {p['category']} | {r['messages'][-1]['content'][:120]!r}")


if __name__ == "__main__":
    asyncio.run(main())
