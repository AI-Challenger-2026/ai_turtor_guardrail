"""가드 검증기에 주는 문맥(섹션 선택) 비교 실험. DEVLOG 13장.

변형마다 두 가지를 잰다.
  (a) 공개 가드 후보 답변 중 막지 않는 입력의 것: 근거 없음 탐지·구간 F1·출력 판정
  (b) 튜터 정답 답변 200개(정의상 근거 있음): 오탐률. even=튜닝용, odd=확인용

  python3 scripts/eval_verify_context.py --variants new4,old4,union,new6,new8 --reps 2
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
import scoring  # noqa: E402
from app.kb import KB  # noqa: E402
from app.policy import BLOCK_CATEGORIES  # noqa: E402
from app.verify import find_unsupported_vote  # noqa: E402


def query_for(messages, cand):
    return "\n".join(m["content"] for m in messages[-4:]) + "\n" + cand + "\n" + cand


def pick(kb, q, variant):
    def top(qtf, k):
        return [s for s, _ in kb.rank(q, qtf=qtf)[:k]]
    if variant == "new4": ids = top(True, 4)
    elif variant == "old4": ids = top(False, 4)
    elif variant == "new6": ids = top(True, 6)
    elif variant == "new8": ids = top(True, 8)
    elif variant == "union":
        ids = []
        for s in top(True, 4) + top(False, 4):
            if s not in ids:
                ids.append(s)
    else:
        raise SystemExit("unknown variant " + variant)
    return [kb.secs[s] for s in ids]


async def run(kb, variant, guard_rows, tutor_rows, sem, effort=None, votes=1):
    lat = []

    async def call(msgs, cand):
        secs = pick(kb, query_for(msgs, cand), variant)
        async with sem:
            t0 = time.time()
            try:
                return await find_unsupported_vote(msgs, cand, secs, effort=effort, votes=votes)
            except Exception:  # noqa: BLE001
                return None
            finally:
                lat.append(time.time() - t0)
    g = await asyncio.gather(*[call(r["messages"], r["candidate_answer"]) for r in guard_rows])
    t = await asyncio.gather(*[call(r["messages"], r["labels"]["reference_answer"]) for r in tutor_rows])
    tp = fp = fn = tn = 0
    tot, f1s = [], []
    for r, sp in zip(guard_rows, g):
        sp = sp or []
        L = r["labels"]
        gold, pred = not L["grounded"], bool(sp)
        f1 = scoring.span_f1(sp, L["unsupported"])
        tot.append((float((not pred) == L["grounded"]) + f1 + 1.0) / 3)
        tp += gold and pred; fp += (not gold) and pred; fn += gold and not pred; tn += (not gold) and not pred
        if gold:
            f1s.append(f1)
    fpe = sum(1 for i, sp in enumerate(t) if sp and i % 2 == 0)
    fpo = sum(1 for i, sp in enumerate(t) if sp and i % 2 == 1)
    err = sum(x is None for x in g + t)
    return dict(variant=variant, out=sum(tot) / len(tot), tp=tp, fn=fn, fp=fp, tn=tn, f1=sum(f1s) / len(f1s),
                ref_fp_even=fpe, ref_fp_odd=fpo, n_even=(len(t) + 1) // 2, n_odd=len(t) // 2, err=err,
                p50=sorted(lat)[len(lat) // 2], p95=sorted(lat)[int(len(lat) * .95)], mx=max(lat))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="new4,old4,union,new6,new8")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--votes", type=int, default=1, help="검증 다수결 표 수")
    ap.add_argument("--effort", default=None, help="검증기 추론 수준(기본: VERIFY_EFFORT)")
    a = ap.parse_args()
    kb = KB()
    grows = [json.loads(l) for l in (ROOT / "public/guard.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    grows = [r for r in grows if "candidate_answer" in r and r["labels"]["category"] not in BLOCK_CATEGORIES]
    trows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sem = asyncio.Semaphore(32)
    print(f"가드 후보 답변(막지 않는 입력) {len(grows)}개, 튜터 정답 답변 {len(trows)}개")
    for rep in range(a.reps):
        res = await asyncio.gather(*[run(kb, v, grows, trows, sem, a.effort, a.votes) for v in a.variants.split(",")])
        for r in res:
            print(f"[votes{a.votes}] rep{rep} {r['variant']:6s} 출력판정 {r['out']:.3f} | 탐지 TP{r['tp']} FN{r['fn']} FP{r['fp']} TN{r['tn']} | 구간F1 {r['f1']:.3f} "
                  f"| 정답답변 오탐 짝 {r['ref_fp_even']}/{r['n_even']} 홀 {r['ref_fp_odd']}/{r['n_odd']} | err {r['err']} | 지연 p50 {r['p50']:.1f} p95 {r['p95']:.1f} max {r['mx']:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
