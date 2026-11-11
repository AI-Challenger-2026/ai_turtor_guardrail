"""검색 점수 분포 모양으로 '자료에 없는 질문'을 가를 수 있나(LLM 호출 없음). DEVLOG 21장.
섹션별 BM25 점수 벡터의 모양(1위 격차, z점수, 엔트로피, 지니 등)이 정상 질문과 자료 없음 질문에서 다른지 AUC로 본다.
양성 = out_of_material, 음성 = normal·pii_disclosure·answer_seeking. 튜터 세트에서 고른 임계값을 가드 세트에 적용해 일반화도 본다.

  python3 scripts/eval_shape.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import KB, build_query  # noqa: E402

NEG = {"normal", "pii_disclosure", "answer_seeking"}


def load(name):
    return [json.loads(l) for l in (ROOT / "public" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def shape(scores):
    v = sorted(scores, reverse=True)
    t1, t2 = v[0], v[1]
    mean, sd = st.mean(v), st.pstdev(v) or 1e-9
    s = sum(v) or 1e-9
    p = [x / s for x in v if x > 0]
    ent = -sum(x * math.log(x) for x in p) / math.log(len(v))
    gini = sum(abs(a - b) for a in v for b in v) / (2 * len(v) * s)
    top5 = sum(v[:5]) / s
    return {
        "1위/2위(작을수록 평평)": t2 / t1 if t1 else 1.0,
        "z(1위)(작을수록 평평)": -(t1 - mean) / sd,
        "엔트로피(클수록 평평)": ent,
        "지니(작을수록 평평)": -gini,
        "상위5 점유율(작을수록 평평)": -top5,
        "1위 점수(작을수록 약함)": -t1,
    }


def best_th(pos, neg):
    best = (0.0, None)
    for th in sorted(set(pos + neg)):
        tp = sum(p >= th for p in pos); fp = sum(n >= th for n in neg); fn = len(pos) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, th)
    return best


def main():
    kb = KB(use_embeddings=False)
    data = {}
    for name in ("tutor", "guard"):
        rows = [r for r in load(name) if r["labels"]["category"] in NEG or r["labels"]["category"] == "out_of_material"]
        for qmode in ("all", "last"):
            pos, neg = [], []
            for r in rows:
                q = build_query(r["messages"], qmode)
                f = shape(list(kb._section_scores(kb.bm25.scores(q)).values()))
                (pos if r["labels"]["category"] == "out_of_material" else neg).append(f)
            data[(name, qmode)] = (pos, neg)
        print(f"{name}: 자료 없음 {len(data[(name, 'all')][0])}, 자료로 답함 {len(data[(name, 'all')][1])}")
    print("\n특징 | AUC 튜터(전체대화/마지막발화) | AUC 가드(전체대화/마지막발화) | 튜터 F1최대→가드 적용(TP,FP)")
    for k in shape([1.0, 0.5, 0.2, 0.1]):
        a = [auc([f[k] for f in data[(n, m)][0]], [f[k] for f in data[(n, m)][1]]) for n in ("tutor", "guard") for m in ("all", "last")]
        tp_, tn_ = data[("tutor", "all")]
        f1, th = best_th([f[k] for f in tp_], [f[k] for f in tn_])
        gp, gn = data[("guard", "all")]
        g = (sum(f[k] >= th for f in gp), len(gp), sum(f[k] >= th for f in gn), len(gn))
        print(f"{k:28s} | {a[0]:.3f}/{a[1]:.3f} | {a[2]:.3f}/{a[3]:.3f} | F1 {f1:.2f} → TP {g[0]}/{g[1]} FP {g[2]}/{g[3]}")


main()
