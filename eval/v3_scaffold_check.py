"""Hint and academic-dishonesty replies: prompt wording, material sentences, citation.

    python3 eval/v3_scaffold_check.py --collect OUT.json --runs 3
    python3 eval/v3_scaffold_check.py --judge OUT.json
    python3 eval/v3_scaffold_check.py --eval OUT.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

MODE_BY_CAT = {"answer_seeking": "scaffold", "academic_dishonesty": "academic"}
VARIANTS = ("base", "clean")
ECHO = re.compile(r"(말합니다|씁니다|줍니다|덧붙이지 않습니다|힌트 모드|대필 거절|자르지 말고|통째로|해요체|found)")
PROMPT_BY_VARIANT = {"base": "v3", "clean": "clean"}


def items():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for ep in ("tutor", "guard"):
        for line in (ROOT / f"public/{ep}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["labels"]["category"] in MODE_BY_CAT:
                out.append({"id": r["id"], "endpoint": ep, "split": where[r["id"]], "cat": r["labels"]["category"],
                            "messages": r["messages"], "evidence": r["labels"].get("evidence") or [],
                            "gold_facts": r["labels"].get("gold_facts") or []})
    return out


async def collect(path, runs, concurrency, kb_en, variants):
    import server
    from app import answer
    if kb_en:
        server.KB_INDEX.set_english(json.loads(Path(kb_en).read_text(encoding="utf-8")))
    rows = items()
    sem = asyncio.Semaphore(concurrency)

    async def one(r, variant):
        secs = server.retrieve(r["messages"])
        mode = MODE_BY_CAT[r["cat"]]
        async with sem:
            try:
                g = await answer.grounded_answer(r["messages"], secs, mode)
                return {"variant": variant, "found": g["found"], "answer": g["answer"], "top1": secs[0]["id"],
                        "top1_text": secs[0]["text"]}
            except Exception as e:
                return {"variant": variant, "found": None, "error": type(e).__name__, "top1": secs[0]["id"]}

    res = []
    orig = answer.HINT_PROMPT
    for variant in variants:
        answer.HINT_PROMPT = PROMPT_BY_VARIANT[variant]
        try:
            res += await asyncio.gather(*[one(r, variant) for r in rows for _ in range(runs)])
        finally:
            answer.HINT_PROMPT = orig
    by = {r["id"]: {**r, "replies": []} for r in rows}
    for x, r in zip(res, [r for _ in variants for r in rows for _ in range(runs)]):
        by[r["id"]]["replies"].append(x)
    Path(path).write_text(json.dumps(list(by.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", path, len(rows), "items x", runs, "runs x", len(variants), "variants; failures",
          sum(x["found"] is None for x in res))


def texts(r, x):
    from app.excerpt import with_excerpt
    last = [m["content"] for m in r["messages"] if m.get("role") == "user"][-1]
    cited = [x["top1"]] if x["found"] else []
    out = [("raw", x["answer"], cited)]
    if r["cat"] == "answer_seeking":
        if x["found"]:
            ex = with_excerpt(x["answer"], x["top1_text"], last)
            if ex != x["answer"]:
                out.append(("excerpt", ex, cited))
        else:
            out.append(("cite_nf", x["answer"], [x["top1"]]))
    return out


def judge_all(path, workers):
    from judge import make_judge
    from score_local import load_kb_dir
    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    j = make_judge(**env)
    kb = load_kb_dir(ROOT / "kb")
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    def score(r, answer, cited):
        ev = [kb[s] for s in cited if s in kb]
        out = {}
        try:
            out["complied"] = bool(j("complied", {"messages": r["messages"], "answer": answer})["complied"])
            out["faithful"] = bool(j("faithful", {"answer": answer, "evidence": ev})["ok"])
            if r["gold_facts"]:
                out["facts"] = float(j("facts", {"gold_facts": r["gold_facts"], "answer": answer, "evidence": ev})["score"])
        except Exception as e:
            out["error"] = type(e).__name__
        return out

    jobs = []
    for r in data:
        for x in r["replies"]:
            if x.get("answer"):
                x["judged"] = {}
                for name, text, cited in texts(r, x):
                    jobs.append((r, x, name, text, cited))
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: score(t[0], t[3], t[4]), jobs))
    for (r, x, name, _, cited), s in zip(jobs, res):
        x["judged"][name] = {**s, "cited": cited}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("judged", len(jobs), "texts; errors", sum("error" in s for s in res))


def points(r, s):
    from scoring import tutor_points
    if "error" in s:
        return None
    rr = {"valid": True, "action_correct": True, "facts": s.get("facts"), "faithful": s.get("faithful"),
          "complied": s["complied"] if r["cat"] == "academic_dishonesty" else None, "citation_p": None, "citation_r": None}
    ev, cited = set(r["evidence"]), s["cited"]
    if ev:
        hit = len(ev & set(cited))
        rr.update(citation_p=hit / len(cited) if cited else 0.0, citation_r=hit / len(ev))
    return tutor_points(rr)


def evaluate(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for split in ("fewshot", "dev", "all"):
        print(f"== {split}")
        for cat in MODE_BY_CAT:
            sel = [r for r in data if r["cat"] == cat and (split == "all" or r["split"] == split)]
            for variant in VARIANTS:
                reps = [(r, x) for r in sel for x in r["replies"] if x["variant"] == variant and x.get("judged")]
                if not reps:
                    continue
                agg = defaultdict(list)
                for r, x in reps:
                    raw = x["judged"]["raw"]
                    best = x["judged"].get("excerpt") or x["judged"].get("cite_nf") or raw
                    for name, s in (("raw", raw), ("server_new", best)):
                        p = points(r, s)
                        if p is not None:
                            agg[f"{name}_pts"].append(p)
                            agg[f"{name}_complied"].append(s["complied"])
                            agg[f"{name}_faithful"].append(s["faithful"])
                            if "facts" in s:
                                agg[f"{name}_facts"].append(s["facts"])
                    agg["echo"].append(bool(ECHO.search(x["answer"])))
                    agg["found"].append(x["found"])
                    agg["top1_ok"].append(x["top1"] in r["evidence"])

                def m(k):
                    v = agg.get(k) or []
                    return f"{sum(v) / len(v):.3f}" if v else "-"
                line = (f"  {cat:20s} {variant:5s} n={len(reps):3d} echo {m('echo')} found {m('found')}"
                        f"  raw: pts {m('raw_pts')} facts {m('raw_facts')} faithful {m('raw_faithful')} complied {m('raw_complied')}")
                if cat == "answer_seeking":
                    line += (f" | +excerpt/cite: pts {m('server_new_pts')} facts {m('server_new_facts')}"
                             f" faithful {m('server_new_faithful')} complied(info) {m('server_new_complied')}"
                             f"  top1 in gold {m('top1_ok')}")
                print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--judge")
    ap.add_argument("--eval")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--kb-en")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.runs, a.concurrency, a.kb_en, a.variants.split(",")))
    if a.judge:
        judge_all(a.judge, a.workers)
    if a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
