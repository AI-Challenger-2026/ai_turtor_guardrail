"""Guard candidate-answer judge, run in-process on public guard items (no HTTP server).

Only items whose gold decision is allow and that carry a candidate answer: the judge's part of
the guard output score, assuming the input decision is right (so `safe` is right). The judge
uses no public items as examples, so every fewshot-split item is fair to tune on; the dev
split stays for end-of-step checks.

Per run: grounded accuracy (confusion), span F1 on gold-ungrounded items, and the item
output score (grounded correct + span F1 + safe correct) / 3. Prints ids and numbers only.

  python3 eval/ground_report.py --split fewshot --effort low --runs 2
  python3 eval/ground_report.py --split fewshot --effort low --votes 2   # flag if any call flags
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grader"))

import scoring  # noqa: E402

from app import config, grounding, retrieval  # noqa: E402
from app.deadline import Deadline  # noqa: E402
from app.schemas import Message  # noqa: E402


def load_items(split: str):
    ids = set(json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))[split])
    items = [json.loads(x) for x in (ROOT / "public/guard.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return [it for it in items if it["id"] in ids and it.get("candidate_answer") and it["labels"]["decision"] == "allow"]


async def judge(msgs, candidate, hits, votes):
    outs = await asyncio.gather(*(grounding.judge_grounding(msgs, candidate, hits, deadline=Deadline(25))
                                  for _ in range(votes)))
    outs = [o for o in outs if o is not None]
    if not outs:
        return None
    flagged = [o for o in outs if not o[0]]
    if not flagged:
        return True, []
    spans = []
    for _, s in flagged:
        spans += s
    return False, grounding.locate_spans(spans, candidate)


async def one_run(items, hits_by_id, votes, sem):
    async def one(it):
        msgs = [Message(**m) for m in it["messages"]]
        async with sem:
            t0 = time.monotonic()
            out = await judge(msgs, it["candidate_answer"], hits_by_id[it["id"]], votes)
            took = time.monotonic() - t0
        gold_grounded = it["labels"]["grounded"] is not False
        gold_spans = it["labels"].get("unsupported") or []
        grounded, spans = out if out is not None else (True, [])
        correct = grounded == gold_grounded
        f1 = scoring.span_f1(spans, gold_spans)
        return {"id": it["id"], "gold": gold_grounded, "pred": grounded, "failed": out is None,
                "spans": spans, "gold_spans": gold_spans, "candidate_len": len(it["candidate_answer"]),
                "span_f1": f1, "score": (float(correct) + f1 + 1.0) / 3, "latency": took}
    return await asyncio.gather(*(one(it) for it in items))


def report(rows, label):
    n = len(rows)
    ung = [r for r in rows if not r["gold"]]
    conf = {(g, p): sum(r["gold"] == g and r["pred"] == p for r in rows) for g in (True, False) for p in (True, False)}
    lat = sorted(r["latency"] for r in rows)
    print(f"{label}: {n} items, failed {sum(r['failed'] for r in rows)}; "
          f"gold grounded -> pred grounded {conf[(True, True)]} / ungrounded {conf[(True, False)]}; "
          f"gold ungrounded -> pred grounded {conf[(False, True)]} / ungrounded {conf[(False, False)]}; "
          f"span F1 on gold ungrounded {statistics.mean(r['span_f1'] for r in ung):.3f}; "
          f"output score {statistics.mean(r['score'] for r in rows):.4f}; "
          f"latency p50 {statistics.median(lat):.1f}s max {lat[-1]:.1f}s")
    misses = [r["id"] for r in rows if r["gold"] != r["pred"]]
    print("   wrong:", " ".join(misses) if misses else "-")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="fewshot", choices=("fewshot", "dev"))
    ap.add_argument("--effort", default=config.GROUND_EFFORT, choices=("minimal", "low", "medium"))
    ap.add_argument("--votes", type=int, default=1, help="judge calls per item; ungrounded if any call says so")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dump", default="", help="eval-only JSON of per-item spans (public items)")
    a = ap.parse_args()
    config.GROUND_EFFORT = a.effort
    await retrieval.build_embeddings()
    items = load_items(a.split)
    sem = asyncio.Semaphore(a.workers)
    hits = await asyncio.gather(*(retrieval.retrieve([Message(**m) for m in it["messages"]], extra=it["candidate_answer"])
                                  for it in items))
    hits_by_id = {it["id"]: h for it, h in zip(items, hits)}
    for run in range(1, a.runs + 1):
        rows = await one_run(items, hits_by_id, a.votes, sem)
        report(rows, f"[{a.split} effort={a.effort} votes={a.votes} run {run}]")
        if a.dump:
            Path(a.dump.replace(".json", f"_run{run}.json")).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
