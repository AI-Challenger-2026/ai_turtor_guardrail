"""검색 설정 탐색: 질의 구성·토큰화·BM25 파라미터·섹션 점수 방식 조합의 R@1을 짝수(튜닝)/홀수(확인) 문항으로 나눠 본다.
판정 모델·서버 없이 돈다(임베딩 옵션만 API 사용).

  python3 scripts/sweep_retrieval.py
"""
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import load_sections, make_chunks  # noqa: E402

_TOKEN_RE = re.compile(r"[가-힣]+|[a-zA-Z_][a-zA-Z_0-9]*|\d+(?:\.\d+)?")


def tokenize(text, grams=(2,)):
    toks = []
    for w in _TOKEN_RE.findall(text.lower()):
        if "가" <= w[0] <= "힣":
            if len(w) == 1:
                toks.append(w)
            for n in grams:
                toks.extend(w[i:i + n] for i in range(len(w) - n + 1))
        else:
            toks.append(w)
    return toks


class BM25:
    def __init__(self, docs, grams, k1, b):
        self.grams, self.k1, self.b = grams, k1, b
        self.tfs = [Counter(tokenize(d, grams)) for d in docs]
        self.lens = [sum(t.values()) for t in self.tfs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, qtf):
        out = []
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            out.append(sum(self.idf[t] * f * (self.k1 + 1) / (f + norm) * qc for t, qc in qtf.items() if (f := tf.get(t))))
        return out


def query_tf(messages, grams, last_w, hist_n, with_asst):
    users = [m["content"] for m in messages if m["role"] == "user"]
    last = users[-1]
    hist = [m["content"] for m in messages[:-1] if with_asst or m["role"] == "user"][-hist_n:] if hist_n else []
    q = Counter()
    for t in tokenize(last, grams):
        q[t] += last_w
    for h in hist:
        for t in tokenize(h, grams):
            q[t] += 1
    return q


def main():
    secs = load_sections()
    chunks = make_chunks(secs)
    rows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["labels"].get("evidence")]
    halves = {"even": rows[0::2], "odd": rows[1::2]}
    results = []
    grid = dict(grams=[(2,), (1, 2), (2, 3)], k1=[1.2, 1.6], b=[0.5, 0.75], unit=["chunk_max", "section"],
                last_w=[1, 2, 3], hist_n=[0, 3, 6, 10], with_asst=[True, False])
    cache = {}
    for g, k1, b, unit in itertools.product(grid["grams"], grid["k1"], grid["b"], grid["unit"]):
        docs = [c["text"] for c in chunks] if unit == "chunk_max" else [s["text"] for s in secs.values()]
        dsid = [c["sid"] for c in chunks] if unit == "chunk_max" else list(secs)
        bm = BM25(docs, g, k1, b)
        for lw, hn, wa in itertools.product(grid["last_w"], grid["hist_n"], grid["with_asst"]):
            if hn == 0 and wa is False:
                continue
            acc = {}
            for hname, hrows in halves.items():
                hit = 0
                for r in hrows:
                    sc = bm.scores(query_tf(r["messages"], g, lw, hn, wa))
                    best = defaultdict(lambda: -1.0)
                    for sid, v in zip(dsid, sc):
                        best[sid] = max(best[sid], v)
                    hit += max(best, key=best.get) == r["labels"]["evidence"][0]
                acc[hname] = hit / len(hrows)
            results.append((acc["even"], acc["odd"], dict(grams=g, k1=k1, b=b, unit=unit, last_w=lw, hist_n=hn, with_asst=wa)))
    results.sort(key=lambda x: -x[0])
    cur = [x for x in results if x[2] == dict(grams=(2,), k1=1.2, b=0.75, unit="chunk_max", last_w=1, hist_n=6, with_asst=True)]
    print(f"설정 {len(results)}개 탐색. 현재 설정 근사(대화 전체 6턴, 마지막 1배): " + (f"짝 {cur[0][0]:.3f} / 홀 {cur[0][1]:.3f}" if cur else "-"))
    print("짝수(튜닝) R@1 상위 15:")
    for e, o, cfg in results[:15]:
        print(f"  짝 {e:.3f}  홀 {o:.3f}  {cfg}")


if __name__ == "__main__":
    main()
