"""튜터 문항별 채점 항목을 쪼개서 본다(action, 인용 P/R, 정답 사실, 근거 충실, 응함 여부).
로컬 채점기와 같은 채점 함수(grader/scoring.py)와 판정 모델을 쓴다. 실행 중인 서버를 호출한다.

  python3 scripts/diagnose_tutor.py [--category normal] [--limit 0] [--show all|bad]
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
import scoring  # noqa: E402
from judge import make_judge  # noqa: E402
from score_local import as_scoring_input, load_kb_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/tutor")
    ap.add_argument("--category")
    ap.add_argument("--show", default="bad", choices=["all", "bad", "none"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", help="문항별 결과를 jsonl로 저장(개발용, 공개 세트만)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "public/tutor.jsonl").read_text().splitlines() if l.strip()]
    if a.category:
        rows = [r for r in rows if r["labels"]["category"] == a.category]
    kb = load_kb_dir(ROOT / "kb")
    judge = make_judge(os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_MODEL"], os.environ["OPENAI_API_KEY"])
    tok = os.environ["TEAM_TOKEN"]

    def one(p):
        item, v = as_scoring_input(p)
        body = json.dumps({"spec_version": "1.0", "session_id": "diag", "messages": v["messages"]}, ensure_ascii=False).encode()
        req = urllib.request.Request(a.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            resp = {"error": type(e).__name__}
        r = scoring.score_tutor(item, v, resp, judge, kb)
        return p, resp, r, scoring.tutor_points(r)

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, rows))

    agg = defaultdict(list)
    for p, resp, r, pts in res:
        c = p["labels"]["category"]
        agg[c].append((pts, r))
    print(f"{'범주':22s} {'n':>3s} {'점수':>6s} {'action':>7s} {'citF1':>6s} {'facts':>6s} {'faith':>6s} {'complied':>8s}")
    for c, lst in sorted(agg.items(), key=lambda x: -len(x[1])):
        def m(key, fn):
            vals = [fn(r) for _, r in lst if r.get(key) is not None]
            return f"{sum(vals) / len(vals):.2f}" if vals else "  -"
        cf = lambda r: (2 * r["citation_p"] * r["citation_r"] / (r["citation_p"] + r["citation_r"])) if (r["citation_p"] or 0) + (r["citation_r"] or 0) else 0.0
        print(f"{c:22s} {len(lst):3d} {100 * sum(p for p, _ in lst) / len(lst):6.1f} "
              f"{m('action_correct', lambda r: float(r['action_correct'])):>7s} {m('citation_p', cf):>6s} "
              f"{m('facts', lambda r: r['facts']):>6s} {m('faithful', lambda r: float(r['faithful'])):>6s} "
              f"{m('complied', lambda r: float(r['complied'])):>8s}")
    allp = [pts for *_, pts in res]
    print(f"튜터 전체 {100 * sum(allp) / len(allp):.2f} (n={len(allp)}), 판정 오류 {sum(r['judge_error'] for _, _, r, _ in res)}")
    print("action 혼동(정답→예측):", dict(Counter((p['labels']['action'], resp.get('action')) for p, resp, r, _ in res if not r['action_correct'])))

    if a.show != "none":
        for p, resp, r, pts in res:
            if a.show == "bad" and pts >= 0.99:
                continue
            L = p["labels"]
            print(f"\n[{p['id']}] {L['category']} 점수 {pts:.2f} | action {L['action']}→{resp.get('action')} "
                  f"| 근거 {L.get('evidence')}→{resp.get('citations')} | facts {r['facts']} faith {r['faithful']} complied {r['complied']}")
            print("  Q:", p["messages"][-1]["content"][:160].replace("\n", " / "))
            print("  A:", str(resp.get("answer", ""))[:260].replace("\n", " / "))
            if L.get("gold_facts"):
                print("  GOLD:", L["gold_facts"])
    if a.out:
        with open(a.out, "w") as f:
            for p, resp, r, pts in res:
                f.write(json.dumps({"id": p["id"], "category": p["labels"]["category"], "points": pts, "resp": resp, "r": r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
