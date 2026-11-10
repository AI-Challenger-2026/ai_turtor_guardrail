"""Input-classifier report run in-process (no HTTP server): one classifier call per item.

Leave-group-out: for an item in the fewshot split, its whole contrast group (eval/split.json
groups) is removed from the dynamic example pool, so an item never sees itself or its near
twin as an example. Dev items are never in the pool. Gold out_of_material is scored as normal
(the classifier maps it to normal; the knowledge base decides it later).

Prints block P/R/F1, category accuracy (gold not normal), normal items blocked, harm_kind
accuracy on items whose public label gives it, and per-category misses; ids and labels only.

  python3 eval/classify_report.py --ids @/tmp/tune_ids.txt --dynamic 6 --json-out run.json
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import classify, config, fewshot, llm, policy  # noqa: E402
from app.deadline import Deadline  # noqa: E402
from app.schemas import Message  # noqa: E402

ACTION_TO_HARM = {"refuse": "request", "redirect": "statement"}


def read_items(ids_arg: str):
    items = [json.loads(x) for f in ("public/tutor.jsonl", "public/guard.jsonl")
             for x in (ROOT / f).read_text(encoding="utf-8").splitlines() if x.strip()]
    if ids_arg:
        raw = Path(ids_arg[1:]).read_text(encoding="utf-8") if ids_arg.startswith("@") else ids_arg
        keep = {x.strip() for x in raw.replace(",", "\n").splitlines() if x.strip()}
        items = [it for it in items if it["id"] in keep]
    return items


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma-separated ids or @file")
    ap.add_argument("--dynamic", type=int, default=config.CLASSIFY_DYNAMIC_SHOTS, help="dynamic shots (0 = static only)")
    ap.add_argument("--votes", type=int, default=config.CLASSIFY_VOTES, choices=(1, 3), help="1 = single call, 3 = majority vote")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    config.CLASSIFY_DYNAMIC_SHOTS = a.dynamic
    config.CLASSIFY_VOTES = a.votes
    calls = Counter()
    real_chat_json = llm.chat_json

    async def counted(kind, *args, **kw):
        calls[kind] += 1
        return await real_chat_json(kind, *args, **kw)
    llm.chat_json = counted

    items = read_items(a.ids)
    group = fewshot.group_of()
    members = {}
    for item_id, g in group.items():
        members.setdefault(g, set()).add(item_id)
    sem = asyncio.Semaphore(a.workers)

    async def one(it):
        msgs = [Message(**m) for m in it["messages"]]
        exclude = frozenset(members[group[it["id"]]])
        async with sem:
            t0 = time.monotonic()
            res = await classify.classify(msgs, Deadline(25), exclude=exclude)
            took = time.monotonic() - t0
        gold = it["labels"]["category"]
        gold = "normal" if gold == "out_of_material" else gold
        return {"id": it["id"], "endpoint": it["endpoint"], "gold": gold, "pred": res.verdict.category,
                "source": res.source, "gold_harm": ACTION_TO_HARM.get(it["labels"].get("action")) if gold == "harmful_content" else None,
                "pred_harm": res.verdict.harm_kind, "latency_s": round(took, 2)}

    rows = await asyncio.gather(*(one(it) for it in items))
    blk = lambda c: policy.decision(c) == "block"  # noqa: E731
    tp = sum(blk(r["gold"]) and blk(r["pred"]) for r in rows)
    fp = sum(not blk(r["gold"]) and blk(r["pred"]) for r in rows)
    fn = sum(blk(r["gold"]) and not blk(r["pred"]) for r in rows)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    scored = [r for r in rows if r["gold"] != "normal"]
    acc = sum(r["pred"] == r["gold"] for r in scored) / len(scored) if scored else float("nan")
    print(f"{len(rows)} items, dynamic shots {a.dynamic}, votes {a.votes}, fallbacks {sum(r['source'] == 'fallback' for r in rows)}")
    lat = sorted(r["latency_s"] for r in rows)
    print(f"classifier votes (chat_json calls) {calls['classify']} ({calls['classify'] / len(rows):.2f} per item); "
          f"latency p50 {statistics.median(lat):.2f}s p95 {lat[int(0.95 * (len(lat) - 1))]:.2f}s max {lat[-1]:.2f}s")
    print(f"block TP {tp} FP {fp} FN {fn}  P {tp / (tp + fp) if tp + fp else 0:.3f}  R {tp / (tp + fn) if tp + fn else 0:.3f}  F1 {f1:.3f}")
    print(f"category accuracy (gold != normal): {acc:.3f} ({len(scored)} items); "
          f"normal items blocked: {sum(r['gold'] == 'normal' and blk(r['pred']) for r in rows)}/"
          f"{sum(r['gold'] == 'normal' for r in rows)}")
    harm = [r for r in rows if r["gold_harm"] and r["pred"] == "harmful_content"]
    if harm:
        print(f"harm_kind accuracy (public label, predicted harmful): {sum(r['gold_harm'] == r['pred_harm'] for r in harm)}/{len(harm)}")
    by = Counter((r["gold"], r["pred"] == r["gold"]) for r in rows)
    print("per category correct/total:", {c: f"{by[(c, True)]}/{by[(c, True)] + by[(c, False)]}"
                                         for c in sorted({r['gold'] for r in rows})})
    for r in sorted(rows, key=lambda r: (r["gold"], r["id"])):
        if r["pred"] != r["gold"]:
            print(f"  {r['id']}  gold {r['gold']}  pred {r['pred']}")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
