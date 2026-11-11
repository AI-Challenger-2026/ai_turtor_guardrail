"""제목·소제목 필드 가중 스윕(LLM 호출 없음). DEVLOG 23장.
섹션 점수 = 본문 점수(청크 BM25 최댓값, 현재 방식) + lam × 제목·소제목 BM25. 짝/홀 문항으로 나눠 본다.

  python3 scripts/sweep_title.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import BM25, KB, build_query  # noqa: E402


def main():
    kb = KB()
    sids = list(kb.secs)
    heads = [" ".join(re.findall(r"^#+ .*", kb.secs[s]["text"], re.M)) + " " + kb.secs[s]["title"] for s in sids]
    hb = BM25(heads, k1=1.2, b=0.5)
    rows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["labels"].get("evidence")]
    gold = [r["labels"]["evidence"][0] if isinstance(r["labels"]["evidence"], list) else r["labels"]["evidence"] for r in rows]
    body = []
    for r in rows:
        d = dict(kb.rank(build_query(r["messages"], "all")))
        body.append([d.get(s, 0.0) for s in sids])
    hq = {}
    for mode in ("all", "last"):
        hq[mode] = [hb.scores(build_query(r["messages"], mode if mode == "all" else "last")) for r in rows]

    def evaluate(lam, mode, norm, idx):
        hit = mrr = 0
        for i in idx:
            b, h = body[i], hq[mode][i]
            if norm:
                mb, mh = max(b) or 1, max(h) or 1
                sc = [x / mb + lam * y / mh for x, y in zip(b, h)]
            else:
                sc = [x + lam * y for x, y in zip(b, h)]
            order = sorted(range(len(sids)), key=lambda k: -sc[k])
            pos = [sids[k] for k in order].index(gold[i]) + 1
            hit += pos == 1; mrr += 1 / pos
        return hit, mrr / len(idx)

    allx, ev, od = list(range(len(rows))), list(range(0, len(rows), 2)), list(range(1, len(rows), 2))
    print(f"근거 있는 튜터 문항 {len(rows)}개 (짝 {len(ev)}, 홀 {len(od)}), 섹션 {len(sids)}개")
    print("질의 정규화 λ  | R@1 전체 | R@1 짝 | R@1 홀 | MRR 전체")
    for mode in ("all", "last"):
        for norm in (False, True):
            for lam in (0, 0.1, 0.25, 0.5, 1, 2):
                a, e, o = evaluate(lam, mode, norm, allx), evaluate(lam, mode, norm, ev), evaluate(lam, mode, norm, od)
                print(f"{mode:4s} {'norm' if norm else 'raw ':4s} {lam:<4} | {a[0]:3d}/{len(allx)} {a[0]/len(allx):.3f} | {e[0]:2d}/{len(ev)} | {o[0]:2d}/{len(od)} | {a[1]:.3f}")


main()
