"""Where do the guard's on-and-off errors come from? Eval-only (2026-11-20).

    python3 eval/v4_guard_flaky.py --collect TRACE.json --runs R1.json R2.json ... [--k 5]
    python3 eval/v4_guard_flaky.py --eval TRACE.json

Items: public guard items whose decision or category was wrong in at least one of the given
eval/v4_low_rounds.py result files. Each is sent K times through the server's guard decision path,
step by step, with the server's own functions and settings: the three classifier votes
(app.classify_feat.SYSTEM, same call settings, app.normalize input rules), the per-flag majority and
app.classify_feat.decide, the material check (server._material_found: retrieval + generator found),
the off_topic recheck, the decision table and the out_of_material switch. The verifier (spans) is
left out: it does not change decision or category. Each step's result is kept, so a wrong final
answer can be traced to the step that varied.

Result (2026-11-20, the 20 guard items wrong at least once in four whole-public runs, 5 traces each,
about 400 calls): of 50 wrong final answers, the classifier majority varied in 19, was steady but
wrong in 17, the material check (found) varied in 7 (category only), the off_topic recheck flipped 2.
These items sit near 50/50 per vote, on both sides of it. Resampling each item's 15 votes: right
decisions 12.05 (1 vote), 12.07 (3), 12.05 (5), 11.88 (15), 11.71 (31) of 20. Making the answer steady
fixes it to the majority side, which is wrong about as often as right, so more votes do not help.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TEAM_TOKEN", "eval-only")


def flaky_ids(paths):
    ids = set()
    for p in paths:
        for d in json.loads(Path(p).read_text(encoding="utf-8")):
            r = d["r"]
            if d["endpoint"] == "guard" and (not r["input_correct"] or (r["category_scored"] and not r["category_correct"])):
                ids.add(d["id"])
    return ids


async def trace_once(messages, sem):
    import server
    from app import classify, classify_feat, normalize
    from app.llm import chat_json
    from app.policy import guard_decision
    async with sem:
        deadline = time.time() + server.REQUEST_BUDGET
        found_task = asyncio.create_task(server._material_found(messages, deadline))
        votes = None
        if normalize.is_low_signal(messages):
            flags = {k: k == "low_signal" for k in classify_feat.FLAGS}
        else:
            kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT,
                      deadline=deadline, hedge_after=7.0, kind="classify")
            text = classify.render(normalize.for_classifier(messages))
            outs = await asyncio.gather(*[chat_json(classify_feat.SYSTEM, text, **kw) for _ in range(classify.VOTES)],
                                        return_exceptions=True)
            votes = [{f: o.get(f) is True for f in classify_feat.FLAGS} if isinstance(o, dict) else None for o in outs]
            ok = [v for v in votes if v is not None]
            flags = {f: sum(v[f] for v in ok) * 2 > len(ok) for f in classify_feat.FLAGS} if ok else None
        pre = classify_feat.decide(flags)["category"] if flags else "normal"
        found = await found_task
        category = pre
        if server.OFFTOPIC_RECHECK and category == "off_topic" and found is True:
            category = "normal"
        decision = guard_decision(category)
        if category == "normal" and found is False:
            category = "out_of_material"
        return {"votes": votes, "pre": pre, "found": found, "category": category, "decision": decision}


async def collect(out_path, run_paths, k, concurrency):
    ids = flaky_ids(run_paths)
    rows = [json.loads(l) for l in (ROOT / "public/guard.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["id"] in ids]
    sem = asyncio.Semaphore(concurrency)
    t = time.monotonic()
    res = await asyncio.gather(*[asyncio.gather(*[trace_once(r["messages"], sem) for _ in range(k)]) for r in rows])
    out = [{"id": r["id"], "gold_decision": r["labels"]["decision"], "gold_category": r["labels"]["category"], "traces": tr}
           for r, tr in zip(rows, res)]
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"saved {out_path}: {len(rows)} items x {k}, {time.monotonic() - t:.0f} s")


def evaluate(path):
    from app.classify_feat import FLAGS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    causes = Counter()
    print("per item: gold | final decision right k/K | category right k/K | what varied")
    for d in data:
        tr = d["traces"]
        k = len(tr)
        dec_ok = sum(t["decision"] == d["gold_decision"] for t in tr)
        cat_ok = sum(t["category"] == d["gold_category"] for t in tr)
        pres = Counter(t["pre"] for t in tr)
        founds = Counter(str(t["found"]) for t in tr)
        flips = sum(t["pre"] == "off_topic" and t["category"] != "off_topic" for t in tr)
        # flags whose vote count differs across traces (the classifier's own wobble)
        wobble = []
        for f in FLAGS:
            counts = Counter(sum(v[f] for v in t["votes"] if v) for t in tr if t["votes"])
            if len(counts) > 1:
                wobble.append(f"{f}{dict(sorted(counts.items()))}")
        print(f"  {d['id']} gold {d['gold_decision']:5s}/{d['gold_category']:20s} decision {dec_ok}/{k}  category {cat_ok}/{k}  "
              f"pre {dict(pres)}  found {dict(founds)}  recheck flips {flips}  votes " + (", ".join(wobble) or "steady"))
        for t in tr:
            if t["decision"] != d["gold_decision"]:
                if t["pre"] == "off_topic" and t["category"] == "normal":
                    causes["recheck turned off_topic into normal"] += 1
                elif len(pres) > 1:
                    causes["classifier majority varied"] += 1
                else:
                    causes["classifier steady but wrong"] += 1
            elif t["category"] != d["gold_category"] and d["gold_category"] != "normal":
                if d["gold_category"] == "out_of_material" or t["category"] == "out_of_material":
                    causes["category: material check (found) varied"] += 1
                elif len(pres) > 1:
                    causes["category: classifier majority varied"] += 1
                else:
                    causes["category: classifier steady but wrong"] += 1
    print("\nwrong final answers by cause:", dict(causes.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--eval")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.runs, a.k, a.concurrency))
    elif a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
