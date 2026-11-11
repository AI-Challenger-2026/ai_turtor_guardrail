"""근거 없는 구간 경계 분석. 검증기 출력을 저장해 두고(--collect), 후처리 방식별 구간 F1을 결정적으로 비교한다(--eval).
공개 가드 세트의 후보 답변 중 막을 입력이 아닌 것(67개: 근거 있음 32, 근거 없음 35)만 대상.

  python3 scripts/span_analysis.py --collect OUT.json
  python3 scripts/span_analysis.py --eval OUT.json [OUT2.json ...]
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
import scoring  # noqa: E402


def rows():
    out = []
    for i, l in enumerate((ROOT / "public/guard.jsonl").read_text().splitlines()):
        r = json.loads(l)
        if "candidate_answer" in r and r["labels"]["decision"] == "allow":
            out.append(r)
    return out


async def collect(path):
    from app.kb import KB
    from app.verify import find_unsupported
    kb = KB(use_embeddings=False) if "use_embeddings" in KB.__init__.__code__.co_varnames else KB()
    rs = rows()
    sem = asyncio.Semaphore(32)

    async def one(r):
        q = "\n".join(m["content"] for m in r["messages"][-4:]) + "\n" + r["candidate_answer"] * 2
        secs = [kb.secs[s] for s, _ in kb.rank(q)[:4]]
        async with sem:
            try:
                return await find_unsupported(r["messages"], r["candidate_answer"], secs)
            except Exception:  # noqa: BLE001
                return []
    preds = await asyncio.gather(*[one(r) for r in rs])
    Path(path).write_text(json.dumps([{"id": r["id"], "pred": p} for r, p in zip(rs, preds)], ensure_ascii=False))
    print("저장:", path, len(preds))


# ---------- 후처리 방식 ----------
SENT_END = re.compile(r"[.!?。]|(?<=요)(?=\s)|(?<=다)(?=\s)")


def sentences(text):
    parts, start = [], 0
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        parts.append((m.start(), m.end()))
    return parts


def expand(cand, span, how):
    i = cand.find(span)
    if i < 0 or how == "raw":
        return span
    j = i + len(span)
    if how == "to_clause_end":  # 구간 끝을 다음 어절 경계까지(서술어 포함) 늘림: 문장 끝 또는 쉼표까지
        m = re.search(r"[.,!?\n]", cand[j:])
        end = j + (m.start() if m else len(cand) - j)
        return cand[i:end].strip()
    if how == "to_sentence":
        for s, e in sentences(cand):
            if s <= i < e:
                return cand[s:e].strip()
        return span
    if how == "plus_word":  # 뒤 한 어절만 더
        m = re.match(r"\S*\s*\S+", cand[j:])
        return (cand[i:j + (m.end() if m else 0)]).strip()
    return span


def evaluate(paths):
    rs = {r["id"]: r for r in rows()}
    runs = [json.loads(Path(p).read_text()) for p in paths]
    for how in ["raw", "plus_word", "to_clause_end", "to_sentence"]:
        res = {"even": [], "odd": []}
        lens = []
        for run in runs:
            for k, x in enumerate(run):
                r = rs[x["id"]]
                cand = r["candidate_answer"]
                pred = [expand(cand, s, how) for s in x["pred"]]
                gold = r["labels"]["unsupported"]
                f1 = scoring.span_f1(pred, gold)
                grounded_ok = (not pred) == r["labels"]["grounded"]
                res["even" if k % 2 == 0 else "odd"].append((float(grounded_ok) + f1 + 1.0) / 3)
                if gold and pred:
                    lens.append(len(scoring.normalize("".join(pred))) / max(1, len(scoring.normalize("".join(gold)))))
        e, o = res["even"], res["odd"]
        lens.sort()
        print(f"{how:14s} 출력 판정 평균 짝 {sum(e) / len(e):.4f} / 홀 {sum(o) / len(o):.4f} | 예측/정답 구간 길이비 중앙값 {lens[len(lens) // 2] if lens else 0:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--eval", nargs="+")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect))
    if a.eval:
        evaluate(a.eval)
