"""검색만 따로 평가: 정답 근거가 있는 튜터 문항에서 정답 섹션의 recall@k, MRR(BM25, 질의 구성별).
LLM·API 호출 없음(결정적). 임베딩·하이브리드 비교는 DEVLOG 2장(커밋 bb4c9cb 버전 스크립트)에 기록.

  python3 scripts/eval_retrieval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.kb import KB, build_query  # noqa: E402


def main():
    kb = KB()
    rows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["labels"].get("evidence")]
    print(f"섹션 {len(kb.secs)}개, 청크 {len(kb.chunks)}개, 평가 문항 {len(rows)}개")
    for qmode in ["last", "last+users", "all"]:
        hits = {1: 0, 3: 0, 5: 0, 8: 0}
        mrr = 0.0
        for r in rows:
            gold = r["labels"]["evidence"][0]
            ranked = [s for s, _ in kb.rank(build_query(r["messages"], qmode))]
            pos = ranked.index(gold) + 1 if gold in ranked else 999
            mrr += 1 / pos
            for k in hits:
                hits[k] += pos <= k
        n = len(rows)
        print(f"  query={qmode:10s} " + " ".join(f"R@{k}={hits[k] / n:.3f}" for k in hits) + f" MRR={mrr / n:.3f}")


if __name__ == "__main__":
    main()
