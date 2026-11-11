"""가드 후보 답변 판정(근거 여부·근거 없는 구간)만 따로 평가. 공개 가드 세트의 후보 답변 86개.
입력 판정과 분리해서 보려고 safe 항목은 정답 라벨 기준(=입력 block 여부)으로 두고, 근거 판정 품질만 잰다.

  python3 scripts/eval_guard_output.py [--k 4] [--show]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
import scoring  # noqa: E402
from app.kb import KB  # noqa: E402
from app.verify import find_unsupported  # noqa: E402
from app.policy import BLOCK_CATEGORIES  # noqa: E402


def query_for(messages, cand):
    return "\n".join(m["content"] for m in messages[-4:]) + "\n" + cand + "\n" + cand


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    kb = KB()
    rows = [json.loads(l) for l in (ROOT / "public/guard.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "candidate_answer" in r]
    sem = asyncio.Semaphore(32)

    async def one(r):
        L = r["labels"]
        if L["category"] in BLOCK_CATEGORIES:  # 막을 입력의 후보 답변: 근거 판정 생략(라벨상 grounded=True)
            return []
        secs = [kb.secs[s] for s, _ in kb.rank(query_for(r["messages"], r["candidate_answer"]))[:a.k]]
        async with sem:
            try:
                return await find_unsupported(r["messages"], r["candidate_answer"], secs)
            except Exception as e:  # noqa: BLE001
                print("err", type(e).__name__)
                return []

    preds = await asyncio.gather(*[one(r) for r in rows])
    tot, tp, fp, fn, tn = [], 0, 0, 0, 0
    span_f1_ung = []
    for r, sp in zip(rows, preds):
        L = r["labels"]
        gold_ung = not L["grounded"]
        pred_ung = bool(sp)
        g_ok = (not pred_ung) == L["grounded"]
        f1 = scoring.span_f1(sp, L["unsupported"])
        tot.append((float(g_ok) + f1 + 1.0) / 3)  # safe는 정답으로 가정
        tp += gold_ung and pred_ung; fp += (not gold_ung) and pred_ung; fn += gold_ung and not pred_ung; tn += (not gold_ung) and not pred_ung
        if gold_ung:
            span_f1_ung.append(f1)
        if a.show and (g_ok is False or f1 < 0.5):
            print(f"[{r['id']}] gold {'UNG' if gold_ung else 'ok '} pred {sp} | gold spans {L['unsupported']} | f1 {f1:.2f}")
            print("   CAND:", r["candidate_answer"][:200].replace("\n", " "))
    n = len(rows)
    print(f"후보 답변 {n}개: 출력 판정 평균 {sum(tot) / n:.3f} (safe 정답 가정)")
    print(f"  근거 없음 탐지: TP {tp} FN {fn} | 정상을 근거없음으로 FP {fp} TN {tn}")
    print(f"  근거 없는 문항의 구간 F1 평균 {sum(span_f1_ung) / len(span_f1_ung):.3f} (n={len(span_f1_ung)})")
    base = sum(((1.0 if r['labels']['grounded'] else 0.0) + (1.0 if r['labels']['grounded'] else 0.0) + 1) / 3 for r in rows) / n
    print(f"  (비교: 항상 grounded=True·[]로 답할 때 {base:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
