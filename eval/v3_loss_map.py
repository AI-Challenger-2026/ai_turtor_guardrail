"""Where tutor points are lost, from eval/tutor_breakdown.py dumps (no API calls).

    python3 eval/v3_loss_map.py R1.json R2.json
"""
import json
import sys
from collections import Counter, defaultdict

PARTS = ("action", "citation", "facts", "faithful", "complied")


def item_losses(r):
    if r["complied"]:
        return {"complied": 1.0}
    k = sum(v is not None for v in r["parts"].values())
    return {p: (1 - v) / k for p, v in r["parts"].items() if v is not None and v < 1}


def main(paths):
    runs = [{r["id"]: r for r in json.load(open(p, encoding="utf-8"))} for p in paths]
    ids = sorted(set.intersection(*[set(run) for run in runs]))
    n = len(ids)
    points = [100 * sum(run[i]["points"] for i in ids) / n for run in runs]
    print(f"items {n}, runs {len(runs)}, tutor points per run {[round(p, 2) for p in points]}, "
          f"judge errors {[sum(run[i]['judge_error'] for i in ids) for run in runs]}")

    mean_loss = defaultdict(float)
    stable_loss = defaultdict(float)
    cat_n = Counter(runs[0][i]["cat"] for i in ids)
    for i in ids:
        cat = runs[0][i]["cat"]
        per_run = [item_losses(run[i]) for run in runs]
        for part in PARTS:
            vals = [lo.get(part, 0.0) for lo in per_run]
            mean_loss[(cat, part)] += 100 * sum(vals) / len(vals) / n
            stable_loss[(cat, part)] += 100 * min(vals) / n

    cats = sorted(cat_n, key=lambda c: -sum(mean_loss[(c, p)] for p in PARTS))
    print(f"\nloss in tutor points (mean of runs; stable part in brackets)")
    print(f"{'category':20s} {'n':>3s} {'total':>12s}  " + "  ".join(f"{p:>13s}" for p in PARTS))
    for c in cats + ["ALL"]:
        sel = cats if c == "ALL" else [c]
        tot = sum(mean_loss[(x, p)] for x in sel for p in PARTS)
        st = sum(stable_loss[(x, p)] for x in sel for p in PARTS)
        cells = [f"{sum(mean_loss[(x, p)] for x in sel):5.2f} [{sum(stable_loss[(x, p)] for x in sel):5.2f}]" for p in PARTS]
        nn = sum(cat_n[x] for x in sel)
        print(f"{c:20s} {nn:3d} {tot:5.2f} [{st:5.2f}]  " + "  ".join(cells))

    print("\naction errors (gold -> predicted: items, item-runs wrong)")
    pairs = defaultdict(lambda: [set(), 0])
    for i in ids:
        for run in runs:
            r = run[i]
            if r["pred_action"] != r["gold_action"]:
                key = (r["cat"], r["gold_action"], r["pred_action"])
                pairs[key][0].add(i)
                pairs[key][1] += 1
    for (cat, g, p), (items, k) in sorted(pairs.items(), key=lambda x: -x[1][1]):
        print(f"  {cat:20s} {g:>8s} -> {str(p):8s} items {len(items)}, wrong in {k} item-runs")

    print("\ncitation misses on answered items (cited vs evidence), item-runs:",
          sum(1 for i in ids for run in runs
              if run[i]["parts"]["citation"] is not None and run[i]["parts"]["citation"] < 1
              and run[i]["pred_action"] == run[i]["gold_action"]))


if __name__ == "__main__":
    main(sys.argv[1:])
