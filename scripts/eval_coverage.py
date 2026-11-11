"""P2 오프라인 분석(LLM 호출 없음): '질문이 자료에 있나'를 코드 신호로 가를 수 있나. DEVLOG 15장.

신호(모두 KB와 마지막 학습자 발화에서만 계산, 클수록 '자료에 없을 가능성'):
  novel_frac  마지막 발화의 고유 토큰 중 KB 어디에도 없는 비율
  novel_n     KB 어디에도 없는 고유 토큰 수
  miss_top    1 - (BM25 1위 청크 점수 / 질의 토큰이 모두 맞을 때의 이론 점수)
  miss_sec    1 - (한 섹션이 덮는 질의 토큰 질량 최댓값 / 전체 질량), KB에 없는 토큰은 최대 idf로 계산

양성 = out_of_material, 음성 = 자료로 답하는 질문(normal, pii_disclosure, answer_seeking).
AUC와, 튜터 세트에서 고른 임계값을 가드 세트에 그대로 적용한 결과(세트 간 일반화)를 본다.

  python3 scripts/eval_coverage.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import KB, tokenize  # noqa: E402

NEG = {"normal", "pii_disclosure", "answer_seeking"}


def load(name):
    return [json.loads(l) for l in (ROOT / "public" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def auc(pos, neg):
    """양성 점수가 음성보다 클 확률(동점 0.5)."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_f1(pos, neg):
    best = (0.0, None)
    for th in sorted(set(pos + neg)):
        tp = sum(p >= th for p in pos); fp = sum(n >= th for n in neg); fn = len(pos) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, th)
    return best


def at(th, pos, neg):
    tp = sum(p >= th for p in pos); fp = sum(n >= th for n in neg)
    return tp, len(pos), fp, len(neg)


def main():
    kb = KB()
    bm = kb.bm25
    vocab = set(bm.idf)
    max_idf = max(bm.idf.values())
    sec_tokens = {sid: set(tokenize(s["text"])) for sid, s in kb.secs.items()}

    def feats(messages):
        users = [m["content"] for m in messages if m["role"] == "user"]
        last = users[-1] if users else ""
        toks = set(tokenize(last))
        if not toks:
            return dict(novel_frac=0.0, novel_n=0, miss_top=0.0, miss_sec=0.0)
        absent = [t for t in toks if t not in vocab]
        w = {t: bm.idf.get(t, max_idf) for t in toks}
        total = sum(w.values())
        theory = (bm.k1 + 1) * total
        top = max(bm.scores(last, qtf=False))
        cover = max(sum(w[t] for t in toks if t in st) / total for st in sec_tokens.values())
        return dict(novel_frac=len(absent) / len(toks), novel_n=len(absent),
                    miss_top=1 - top / theory, miss_sec=1 - cover)

    data = {}
    for name in ("tutor", "guard"):
        rows = [r for r in load(name) if r["labels"]["category"] == "out_of_material" or r["labels"]["category"] in NEG]
        pos, neg = [], []
        for r in rows:
            f = feats(r["messages"])
            (pos if r["labels"]["category"] == "out_of_material" else neg).append(f)
        data[name] = (pos, neg)
        print(f"{name}: 없음(양성) {len(pos)}개, 자료로 답하는 질문(음성) {len(neg)}개")

    print("\nAUC (양성=자료에 없는 질문)")
    for k in ("novel_frac", "novel_n", "miss_top", "miss_sec"):
        a = [auc([f[k] for f in data[n][0]], [f[k] for f in data[n][1]]) for n in ("tutor", "guard")]
        print(f"  {k:10s} 튜터 {a[0]:.3f}  가드 {a[1]:.3f}")

    print("\n세트 간 일반화: 튜터 세트에서 F1 최대 임계값 → 가드 세트에 적용 (TP/양성, FP/음성)")
    for k in ("novel_frac", "novel_n", "miss_top", "miss_sec"):
        tp_, tn_ = data["tutor"]
        f1, th = best_f1([f[k] for f in tp_], [f[k] for f in tn_])
        t = at(th, [f[k] for f in tp_], [f[k] for f in tn_])
        g = at(th, [f[k] for f in data["guard"][0]], [f[k] for f in data["guard"][1]])
        print(f"  {k:10s} th={th:.3f} 튜터 F1 {f1:.2f} (TP {t[0]}/{t[1]}, FP {t[2]}/{t[3]}) → 가드 TP {g[0]}/{g[1]}, FP {g[2]}/{g[3]}")

    print("\n오탐을 거의 허용하지 않는 꼬리: 튜터 음성에서 FP가 k개 이하가 되는 임계값 → 가드 세트에 적용")
    for k in ("novel_frac", "miss_top", "miss_sec"):
        tp_, tn_ = data["tutor"]
        negs = sorted((f[k] for f in tn_), reverse=True)
        for maxfp in (0, 1, 2, 3):
            th = negs[maxfp] + 1e-9
            a = at(th, [f[k] for f in tp_], [f[k] for f in tn_])
            g = at(th, [f[k] for f in data["guard"][0]], [f[k] for f in data["guard"][1]])
            print(f"  {k:10s} FP≤{maxfp}: th={th:.3f} 튜터 TP {a[0]}/{a[1]} | 가드 TP {g[0]}/{g[1]}, FP {g[2]}/{g[3]}")


if __name__ == "__main__":
    main()
