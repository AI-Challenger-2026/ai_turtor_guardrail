"""가드 문항별 진단: block 판정(TP/FP/FN), 범주, 후보 답변 판정(근거 여부·구간 F1·안전)을 쪼개서 본다.
로컬 채점기와 같은 채점 함수를 쓰고, 실행 중인 서버를 호출한다.

  python3 scripts/diagnose_guard.py --url http://localhost:8001/guard [--out OUT.jsonl] [--show]
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
import scoring  # noqa: E402
from score_local import as_scoring_input  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001/guard")
    ap.add_argument("--out")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "public/guard.jsonl").read_text().splitlines() if l.strip()]
    tok = os.environ["TEAM_TOKEN"]

    def one(p):
        item, v = as_scoring_input(p)
        body = {"spec_version": "1.0", "session_id": "diag", "messages": v["messages"]}
        if v["candidate_answer"]:
            body["candidate_answer"] = v["candidate_answer"]
        req = urllib.request.Request(a.url, data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
        except Exception:  # noqa: BLE001
            resp = None
        return p, resp, scoring.score_guard(item, v, resp)

    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(one, rows))
    gs = [r for _, _, r in res]
    tp = sum(r["gold_block"] and r["input_correct"] for r in gs)
    fp = sum((not r["gold_block"]) and not r["input_correct"] for r in gs)
    fn = sum(r["gold_block"] and not r["input_correct"] for r in gs)
    cat = [r["category_correct"] for r in gs if r["category_scored"]]
    outs = [r for r in gs if r["has_output"]]
    out_avg = sum((float(r["output_correct"]) + r["span_f1"] + float(r["safe_correct"])) / 3 for r in outs) / len(outs)
    print(f"가드 점수 {scoring.guard_score(gs):.2f} | block F1 {2 * tp / (2 * tp + fp + fn):.3f} (TP {tp} FP {fp} FN {fn}) "
          f"| 범주 {sum(cat) / len(cat):.3f} (n={len(cat)}) | 출력 판정 {out_avg:.3f} (n={len(outs)})")
    # 출력 판정 손실 분해
    c = Counter()
    for p, resp, r in res:
        if not r["has_output"]:
            continue
        L = p["labels"]
        if not r["safe_correct"]:
            c["안전 판정 틀림(=block 판정 오류 연동)"] += 1
        if not r["output_correct"]:
            c["근거 없음 놓침" if not L["grounded"] else "정상 답변 오탐"] += 1
        elif not L["grounded"] and r["span_f1"] < 0.99:
            c["탐지했지만 구간 불완전"] += 1
    print("출력 판정 손실 유형:", dict(c))
    print("범주 오답(정답→예측):", dict(Counter((p["labels"]["category"], (resp or {}).get("input", {}).get("category"))
                                                for p, resp, r in res if r["category_scored"] and not r["category_correct"])))
    if a.show:
        for p, resp, r in res:
            if not r["input_correct"]:
                print(f"  [{'FP' if not r['gold_block'] else 'FN'} {p['id']}] {p['labels']['category']} → {(resp or {}).get('input')} | {p['messages'][-1]['content'][:110]!r}")
    if a.out:
        with open(a.out, "w") as f:
            for p, resp, r in res:
                f.write(json.dumps({"id": p["id"], "resp": resp, "r": r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
