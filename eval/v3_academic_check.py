"""Academic-dishonesty replies: generated hint vs the fixed reply.

    python3 eval/v3_academic_check.py --collect OUT.json --runs 2
    python3 eval/v3_academic_check.py --judge OUT.json
    python3 eval/v3_academic_check.py --eval OUT.json
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

CATS = ("academic_dishonesty", "answer_seeking")
FIXED_JUDGES = 2


def items():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for ep in ("tutor", "guard"):
        for line in (ROOT / f"public/{ep}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["labels"]["category"] in CATS:
                out.append({"id": r["id"], "endpoint": ep, "split": where[r["id"]], "cat": r["labels"]["category"],
                            "messages": r["messages"], "evidence": r["labels"].get("evidence") or [],
                            "gold_facts": r["labels"].get("gold_facts") or []})
    return out


async def collect(path, runs, concurrency, kb_en):
    import server
    from app.answer import grounded_answer
    from app.classify import classify
    if kb_en:
        server.KB_INDEX.set_english(json.loads(Path(kb_en).read_text(encoding="utf-8")))
    rows = items()
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        secs = server.retrieve(r["messages"])
        async with sem:
            out = {"top1": secs[0]["id"]}
            try:
                c = await classify(r["messages"])
                out["pred"] = c["category"] if "flags" in c else None
            except Exception as e:
                out["pred"], out["cls_error"] = None, type(e).__name__
            try:
                g = await grounded_answer(r["messages"], secs, "academic")
                out.update(found=g["found"], answer=g["answer"], cited=[secs[0]["id"]] if g["found"] else [])
            except Exception as e:
                out.update(found=None, gen_error=type(e).__name__)
            return out

    jobs = [(r, k) for r in rows for k in range(runs)]
    res = await asyncio.gather(*[one(r) for r, _ in jobs])
    by = {r["id"]: {**r, "runs": []} for r in rows}
    for (r, _), x in zip(jobs, res):
        by[r["id"]]["runs"].append(x)
    Path(path).write_text(json.dumps(list(by.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", path, len(rows), "items x", runs, "runs; generation failures", sum(x["found"] is None for x in res),
          "classify failures", sum(x["pred"] is None for x in res))


def judge_all(path, workers):
    from judge import make_judge
    from score_local import load_kb_dir
    from server import ACADEMIC_FALLBACK
    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    j = make_judge(**env)
    kb = load_kb_dir(ROOT / "kb")
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def score(r, answer, cited):
        ev = [kb[s] for s in cited if s in kb]
        out = {}
        try:
            if r["cat"] == "academic_dishonesty":
                out["complied"] = bool(j("complied", {"messages": r["messages"], "answer": answer})["complied"])
            out["faithful"] = bool(j("faithful", {"answer": answer, "evidence": ev})["ok"])
            if r["gold_facts"]:
                out["facts"] = float(j("facts", {"gold_facts": r["gold_facts"], "answer": answer, "evidence": ev})["score"])
        except Exception as e:
            out["error"] = type(e).__name__
        return out

    jobs = []
    for r in data:
        for x in r["runs"]:
            if x.get("answer"):
                jobs.append((r, x, "gen", x["answer"], x["cited"]))
            for _ in range(FIXED_JUDGES):
                jobs.append((r, x, "fixed", ACADEMIC_FALLBACK, []))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: score(t[0], t[3], t[4]), jobs))
    for r in data:
        for x in r["runs"]:
            x["judged"] = {"gen": [], "fixed": []}
    for (r, x, which, _, _), s in zip(jobs, res):
        x["judged"][which].append(s)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("judged", len(jobs), "replies; errors", sum("error" in s for s in res))


def points(r, cited, judged):
    from scoring import tutor_points
    pts = []
    for s in judged:
        if "error" in s:
            continue
        ev = set(r["evidence"])
        rr = {"valid": True, "action_correct": True, "complied": s.get("complied"), "facts": s.get("facts"),
              "faithful": s.get("faithful"), "citation_p": None, "citation_r": None}
        if ev:
            hit = len(ev & set(cited))
            rr.update(citation_p=hit / len(cited) if cited else 0.0, citation_r=hit / len(ev))
        pts.append(tutor_points(rr))
    return sum(pts) / len(pts) if pts else None


def evaluate(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for split in ("fewshot", "dev", "all"):
        sel = [r for r in data if split == "all" or r["split"] == split]
        print(f"== {split}: {len(sel)} items")
        for cat in CATS:
            c = Counter(x["pred"] for r in sel if r["cat"] == cat for x in r["runs"])
            print(f"  gold {cat}: predicted {dict(c)}")
        n = Counter()
        gen = Counter()
        fix = Counter()
        comp = Counter()
        for r in sel:
            for x in r["runs"]:
                jd = x.get("judged") or {}
                g, f = points(r, x.get("cited", []), jd.get("gen", [])), points(r, [], jd.get("fixed", []))
                for s in jd.get("gen", []):
                    if "complied" in s:
                        comp["gen_n"] += 1
                        comp["gen_c"] += s["complied"]
                for s in jd.get("fixed", []):
                    if "complied" in s:
                        comp["fix_n"] += 1
                        comp["fix_c"] += s["complied"]
                if g is None or f is None:
                    continue
                key = r["cat"]
                n[key] += 1
                gen[key] += g
                fix[key] += f
                if x["pred"] == "academic_dishonesty":
                    n["pred_acad"] += 1
                    gen["pred_acad"] += g
                    fix["pred_acad"] += f
        print(f"  complied: generated {comp['gen_c']}/{comp['gen_n']}  fixed {comp['fix_c']}/{comp['fix_n']}")
        for key in (*CATS, "pred_acad"):
            if n[key]:
                print(f"  {key:22s} n={n[key]:3d}  generated {gen[key] / n[key]:.3f}  fixed {fix[key] / n[key]:.3f}"
                      f"  change if fixed {fix[key] - gen[key]:+.2f} item-points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--judge")
    ap.add_argument("--eval")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--kb-en")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.runs, a.concurrency, a.kb_en))
    if a.judge:
        judge_all(a.judge, a.workers)
    if a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
