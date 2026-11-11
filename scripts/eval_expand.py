"""문서 확장 효과 오프라인 평가(LLM 호출 없음): 정답 섹션 R@1/MRR, 짝/홀, 영어 질문, 가드 검증기 문맥 변화. DEVLOG 22장.
  python3 scripts/eval_expand.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import KB, build_query  # noqa: E402

VARIANTS = {"기준(확장 없음)": "", "한국어 질문": "ko_questions", "영어 용어": "en_terms", "영어 용어+질문": "en_terms,en_questions",
            "한국어 질문+영어 용어": "ko_questions,en_terms", "전부": "ko_questions,en_terms,en_questions"}


def load(n):
    return [json.loads(l) for l in (ROOT / "public" / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def is_en(row):
    last = [m["content"] for m in row["messages"] if m["role"] == "user"][-1]
    return len(re.findall(r"[A-Za-z]", last)) > 2 * len(re.findall(r"[가-힣]", last))


def main():
    rows = [r for r in load("tutor") if r["labels"].get("evidence")]
    gold = [r["labels"]["evidence"][0] if isinstance(r["labels"]["evidence"], list) else r["labels"]["evidence"] for r in rows]
    en = [i for i, r in enumerate(rows) if is_en(r)]
    cands = [r for r in load("guard") if "candidate_answer" in r]
    print(f"근거 있는 튜터 문항 {len(rows)}개 (영어 질문 {len(en)}개), 가드 후보 답변 {len(cands)}개")
    print("변형 | R@1 전체 | 짝 | 홀 | MRR | 영어 R@1 | 가드 검증기 문맥(상위4) 달라진 후보 수")
    base_ctx = None
    for name, kinds in VARIANTS.items():
        kb = KB(use_embeddings=False, expand=kinds)
        pos = []
        for r, g in zip(rows, gold):
            ids = [s for s, _ in kb.rank(build_query(r["messages"], "all"))]
            pos.append(ids.index(g) + 1)
        ctx = []
        for r in cands:
            q = "\n".join(m["content"] for m in r["messages"][-4:]) + "\n" + r["candidate_answer"] + "\n" + r["candidate_answer"]
            ctx.append([s for s, _ in kb.rank(q)[:4]])
        if base_ctx is None:
            base_ctx = ctx
        ch = sum(set(a) != set(b) for a, b in zip(ctx, base_ctx))
        h = lambda idx: sum(pos[i] == 1 for i in idx)
        al, ev, od = range(len(rows)), range(0, len(rows), 2), range(1, len(rows), 2)
        print(f"{name:18s} | {h(al)}/{len(rows)} | {h(ev)}/{len(ev)} | {h(od)}/{len(od)} | {sum(1 / p for p in pos) / len(pos):.3f} | {h(en)}/{len(en)} | {ch}/{len(cands)}")


main()
