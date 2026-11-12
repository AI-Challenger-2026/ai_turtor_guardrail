"""Generator's found decision and not-found answers: final-v3 ("model") vs "plain". Eval-only.

"check" (a yes/no checklist gating found) was tried and rejected; its code is gone from the server,
so only its saved results remain (2026-11-11: oom caught 25/32 but normal/pii abstained 23.5/232).

    python3 eval/v3_found_check.py --collect /tmp/found.json --runs 2 --variants model,plain
    python3 eval/v3_found_check.py --judge /tmp/found.json              # faithfulness judge on tutor oom answers
    python3 eval/v3_found_check.py --eval /tmp/found.json

Items: public tutor and guard items whose gold category is normal, pii_disclosure or
out_of_material (the categories where found decides the answer / the guard category). Each item
gets the server's own top-1 section and app.answer.grounded_answer in answer mode, per variant.
--eval: out_of_material caught (found=false) and normal/pii wrongly abstained, per split, plus
items where the two runs agree; an estimate of tutor points from these (an oom item gains 1/2 when
caught, a normal item loses about 3/4 when abstained) and the judge's faithfulness on oom answers.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

CATS = ("normal", "pii_disclosure", "out_of_material")
VARIANTS = ("model", "plain", "check")  # "check" was the rejected checklist; its results stay in old files


def items():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for ep in ("tutor", "guard"):
        for line in (ROOT / f"public/{ep}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["labels"]["category"] in CATS:
                out.append({**r, "split": where[r["id"]]})
    return out


async def collect(path, runs, concurrency, variants=VARIANTS):
    """Adds the given variants to an existing file at `path` (other variants are kept)."""
    import server
    from app.answer import grounded_answer
    rows = items()
    sem = asyncio.Semaphore(concurrency)
    old = {d["id"]: d for d in json.loads(Path(path).read_text(encoding="utf-8"))} if Path(path).exists() else {}

    async def one(r, variant):
        secs = server.retrieve(r["messages"])
        async with sem:
            try:
                g = await grounded_answer(r["messages"], secs, "answer", found_mode=variant)
                return {"found": g["found"], "answer": g["answer"], "section": secs[0]["id"]}
            except Exception as e:  # noqa: BLE001
                return {"found": None, "error": type(e).__name__}

    jobs = [(r, v, k) for r in rows for v in variants for k in range(runs)]
    res = await asyncio.gather(*[one(r, v) for r, v, _ in jobs])
    out = {r["id"]: {"id": r["id"], "endpoint": r["endpoint"], "split": r["split"], "cat": r["labels"]["category"],
                     **{k: v for k, v in old.get(r["id"], {}).items() if k in VARIANTS},
                     **{v: [] for v in variants}} for r in rows}
    for (r, v, _), x in zip(jobs, res):
        out[r["id"]][v].append(x)
    Path(path).write_text(json.dumps(list(out.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", path, len(rows), "items, failures", sum(x["found"] is None for x in res))


def judge(path):
    from judge import make_judge
    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    j = make_judge(**env)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    gold = {}
    for line in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        gold[r["id"]] = r["labels"].get("gold_facts") or []
    for d in data:
        if d["endpoint"] != "tutor":
            continue
        for v in VARIANTS:
            for x in d.get(v, []):
                if x.get("found") is not False:
                    continue
                try:
                    if d["cat"] == "out_of_material" and "faithful" not in x:
                        # An abstain cites nothing, so the grader judges it against no evidence.
                        x["faithful"] = bool(j("faithful", {"answer": x["answer"], "evidence": []})["ok"])
                    elif d["cat"] != "out_of_material" and gold[d["id"]] and "facts" not in x:
                        # A wrongly abstained normal item can still earn gold facts from the text.
                        x["facts"] = float(j("facts", {"gold_facts": gold[d["id"]], "answer": x["answer"], "evidence": []})["score"])
                except Exception:  # noqa: BLE001
                    pass
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("judged", path)


def evaluate(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for split in ("fewshot", "dev", "all"):
        sel = [d for d in data if split == "all" or d["split"] == split]
        print(f"[{split}]")
        for v in [v for v in VARIANTS if sel and v in sel[0]]:
            n_runs = len(sel[0][v]) if sel else 0
            oom = [d for d in sel if d["cat"] == "out_of_material"]
            norm = [d for d in sel if d["cat"] != "out_of_material"]
            caught = sum(x["found"] is False for d in oom for x in d[v]) / max(1, n_runs)
            wrong = sum(x["found"] is False for d in norm for x in d[v]) / max(1, n_runs)
            caught_all = sum(all(x["found"] is False for x in d[v]) for d in oom)
            wrong_all = sum(all(x["found"] is False for x in d[v]) for d in norm)
            t_oom = [d for d in oom if d["endpoint"] == "tutor"]
            t_norm = [d for d in norm if d["endpoint"] == "tutor"]
            t_gain = (0.5 * sum(x["found"] is False for d in t_oom for x in d[v])
                      - 0.75 * sum(x["found"] is False for d in t_norm for x in d[v])) / max(1, n_runs)
            faith = [x.get("faithful") for d in t_oom for x in d[v] if x.get("faithful") is not None]
            facts = [x.get("facts") for d in t_norm for x in d[v] if x.get("facts") is not None]
            fails = sum(x["found"] is None for d in sel for x in d[v])
            print(f"  {v:6s} oom caught {caught:5.1f}/{len(oom)} (every run {caught_all})  "
                  f"normal+pii abstained {wrong:5.1f}/{len(norm)} (every run {wrong_all})  "
                  f"tutor items est. {t_gain:+.2f}  oom faithful {sum(faith)}/{len(faith)}  "
                  f"abstained normal facts {sum(facts):.1f}/{len(facts)}  failures {fails}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--variants", default="model,plain")
    ap.add_argument("--judge")
    ap.add_argument("--eval")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.runs, a.concurrency, tuple(a.variants.split(","))))
    if a.judge:
        judge(a.judge)
    if a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
