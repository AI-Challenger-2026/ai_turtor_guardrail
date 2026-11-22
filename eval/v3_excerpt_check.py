"""Answers with and without the appended material sentences, scored by the local judge.

    python3 eval/v3_excerpt_check.py R1.json R2.json --out OUT.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

from app.excerpt import with_excerpt
from app.kb import load_sections

PII_NOTE = "이름이나 학교 같은 개인정보는 채팅에 적지 않아도 괜찮아요."


def grams(text, n=3):
    s = "".join(text.split())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def fact_overlap(facts, answer):
    a = grams(answer)
    shares = [len(grams(f) & a) / max(1, len(grams(f))) for f in facts]
    return sum(shares) / len(shares) if shares else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from judge import make_judge
    from score_local import load_kb_dir
    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    judge = make_judge(**env)
    kb_grader = load_kb_dir(ROOT / "kb")
    secs = load_sections()
    items = {}
    for line in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        items[r["id"]] = r
    jobs = []
    for run, path in enumerate(a.dumps):
        for d in json.load(open(path, encoding="utf-8")):
            if d["pred_action"] != "answer" or not d["cited"] or d["gold_action"] != "answer":
                continue
            r = items[d["id"]]
            q = [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
            ans = d["answer"]
            note = ans.endswith(PII_NOTE)
            base = ans[: -len(PII_NOTE)].rstrip() if note else ans
            new = with_excerpt(base, secs[d["cited"][0]]["text"], q)
            if note:
                new = new.rstrip() + " " + PII_NOTE
            jobs.append({"run": run, "id": d["id"], "orig": ans, "new": new, "changed": new != ans,
                         "facts_gold": r["labels"].get("gold_facts") or [], "cited": d["cited"]})

    def judge_one(job):
        ev = [kb_grader[s] for s in job["cited"] if s in kb_grader]
        out = {}
        for ver in ("orig", "new"):
            try:
                if job["facts_gold"]:
                    out[f"facts_{ver}"] = float(judge("facts", {"gold_facts": job["facts_gold"], "answer": job[ver], "evidence": ev})["score"])
                out[f"faithful_{ver}"] = bool(judge("faithful", {"answer": job[ver], "evidence": ev})["ok"])
            except Exception as e:
                out[f"error_{ver}"] = type(e).__name__
        return out

    changed = [j for j in jobs if j["changed"]]
    with ThreadPoolExecutor(a.workers) as ex:
        for j, res in zip(changed, ex.map(judge_one, changed)):
            j.update(res)
    Path(a.out).write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    calls = sum((2 if j["facts_gold"] else 1) * 2 for j in changed)
    print(f"answered items {len(jobs)}, with excerpt {len(changed)}, judge calls about {calls}")
    for run in range(len(a.dumps)):
        sel = [j for j in changed if j["run"] == run and not any(k.startswith("error") for k in j)]
        f = [j for j in sel if "facts_orig" in j]
        print(f"run {run + 1}: judged {len(sel)}  facts {sum(j['facts_orig'] for j in f):.1f} -> {sum(j['facts_new'] for j in f):.1f} /{len(f)}"
              f"  faithful {sum(j['faithful_orig'] for j in sel)} -> {sum(j['faithful_new'] for j in sel)} /{len(sel)}"
              f"  gold 3-gram share {sum(fact_overlap(j['facts_gold'], j['orig']) for j in f) / max(1, len(f)):.3f}"
              f" -> {sum(fact_overlap(j['facts_gold'], j['new']) for j in f) / max(1, len(f)):.3f}")
        d = sum((j.get("facts_new", 0) - j.get("facts_orig", 0)) + (j["faithful_new"] - j["faithful_orig"]) for j in sel) / 4
        print(f"   item-points change {d:+.2f} over {len(jobs) // len(a.dumps)} answered items")


if __name__ == "__main__":
    main()
