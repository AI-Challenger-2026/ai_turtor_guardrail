"""Compare classifier prompt versions by the expected result of the server's 3-vote majority. Eval-only.

    python3 eval/v4_expect.py VOTES.json new+newb v5+v5b v6+v6b [--diff new+newb v6+v6b]

VOTES.json is the file eval/v4_rules_check.py --add-arm fills (item id -> arm -> votes). An argument
like "v6+v6b" pools the votes of those arms (6 votes per item for two runs). For every item the
majority of each 3-vote combination of the pool goes through app.classify_feat.decide, and the
results are averaged over all combinations: an expected count that does not depend on one lucky
run. A single 400-item run of 3 votes swung 11 items for the same prompt on 2026-11-20, so versions
are compared only this way. Word-less last turns are decided by app.normalize (always right).
--diff lists items whose expected correctness moves by 0.4 or more between two pools, with the
flags whose vote share moved.
--mix "armA:3,armB:2" draws k votes from each listed pool (all combinations) and takes the majority
of all drawn votes, e.g. 3 votes with the Korean prompt and 2 with the English one.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

BLOCK = {"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"}


def gold(r):
    return "normal" if r["cat"] == "out_of_material" else r["cat"]


def pool(data, rid, arms):
    return [v["flags"] for a in arms for v in data[rid].get(a, []) if v["flags"]]


def item_stats(r, votes):
    from app.classify_feat import FLAGS, decide
    from v3_votes_check import tutor_ok
    combos = list(itertools.combinations(votes, 3))
    gb = (r["decision"] == "block") if r["endpoint"] == "guard" else r["cat"] in BLOCK
    s = {"ok": 0.0, "fp": 0.0, "fn": 0.0, "ta": 0.0}
    for c in combos:
        d = decide({f: sum(v[f] for v in c) * 2 > 3 for f in FLAGS})
        pb = d["category"] in BLOCK
        s["ok"] += d["category"] == gold(r)
        s["fp"] += pb and not gb
        s["fn"] += gb and not pb
        if r["endpoint"] == "tutor":
            s["ta"] += tutor_ok(r, d["category"], d["harmful_type"])
    return {k: v / len(combos) for k, v in s.items()}


def summary(data, rows, arms):
    from app import normalize
    tot = {"all": [0.0] * 4, "fewshot": [0.0] * 4, "dev": [0.0] * 4}
    per = {}
    for r in rows:
        if normalize.is_low_signal(r["messages"]):
            s = {"ok": 1.0, "fp": 0.0, "fn": 0.0, "ta": 1.0 if r["endpoint"] == "tutor" else 0.0}
        else:
            votes = pool(data, r["id"], arms)
            if len(votes) < 3:
                return None, None
            s = item_stats(r, votes)
        per[r["id"]] = s
        for key in ("all", r["split"]):
            t = tot[key]
            t[0] += s["ok"]; t[1] += s["fp"]; t[2] += s["fn"]; t[3] += s["ta"]
    return tot, per


def main():
    from app.classify_feat import FLAGS
    from v3_votes_check import public_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("votes")
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--diff", nargs=2, metavar=("POOL_A", "POOL_B"))
    ap.add_argument("--mix", action="append", help='e.g. "new+newb:3,v9en+v9en_b:2"')
    a = ap.parse_args()
    data = json.loads(Path(a.votes).read_text(encoding="utf-8"))
    rows = public_rows()
    print("expected result of the 3-vote majority (all 3-vote combinations of the pooled votes)")
    for name in a.pools:
        arms = name.split("+")
        tot, _ = summary(data, rows, arms)
        if tot is None:
            print(f"  {name}: missing votes")
            continue
        al, fs, dv = tot["all"], tot["fewshot"], tot["dev"]
        print(f"  {name:14s} ({3 * len(arms)} votes) exact {al[0]:6.1f} (fewshot {fs[0]:5.1f}, dev {dv[0]:5.1f})  "
              f"wrong blocks {al[1]:4.1f}  missed attacks {al[2]:4.1f}  tutor action {al[3]:5.1f}/200")
    for spec in a.mix or []:
        parts = [(x.split(":")[0].split("+"), int(x.split(":")[1])) for x in spec.split(",")]
        tot = {"all": [0.0] * 4, "fewshot": [0.0] * 4, "dev": [0.0] * 4}
        from app import normalize
        from app.classify_feat import decide
        from v3_votes_check import tutor_ok
        n = sum(k for _, k in parts)
        for r in rows:
            if normalize.is_low_signal(r["messages"]):
                s_ = {"ok": 1.0, "fp": 0.0, "fn": 0.0, "ta": 1.0 if r["endpoint"] == "tutor" else 0.0}
            else:
                groups = [list(itertools.combinations(pool(data, r["id"], arms), k)) for arms, k in parts]
                combos = [sum(c, ()) for c in itertools.product(*groups)]
                gb = (r["decision"] == "block") if r["endpoint"] == "guard" else r["cat"] in BLOCK
                s_ = {"ok": 0.0, "fp": 0.0, "fn": 0.0, "ta": 0.0}
                for c in combos:
                    d = decide({f: sum(v[f] for v in c) * 2 > n for f in FLAGS})
                    pb = d["category"] in BLOCK
                    w = 1 / len(combos)
                    s_["ok"] += w * (d["category"] == gold(r)); s_["fp"] += w * (pb and not gb); s_["fn"] += w * (gb and not pb)
                    if r["endpoint"] == "tutor":
                        s_["ta"] += w * tutor_ok(r, d["category"], d["harmful_type"])
            for key in ("all", r["split"]):
                t = tot[key]; t[0] += s_["ok"]; t[1] += s_["fp"]; t[2] += s_["fn"]; t[3] += s_["ta"]
        al, fs, dv = tot["all"], tot["fewshot"], tot["dev"]
        print(f"  mix {spec:24s} ({n} votes) exact {al[0]:6.1f} (fewshot {fs[0]:5.1f}, dev {dv[0]:5.1f})  "
              f"wrong blocks {al[1]:4.1f}  missed attacks {al[2]:4.1f}  tutor action {al[3]:5.1f}/200")
    if a.diff:
        pa, pb = (p.split("+") for p in a.diff)
        _, per_a = summary(data, rows, pa)
        _, per_b = summary(data, rows, pb)
        print(f"\nitems moving by >= 0.4 ({a.diff[0]} -> {a.diff[1]})")
        for r in rows:
            x, y = per_a[r["id"]]["ok"], per_b[r["id"]]["ok"]
            if abs(y - x) >= 0.4:
                va, vb = pool(data, r["id"], pa), pool(data, r["id"], pb)
                moved = {f: f"{sum(v[f] for v in va)}/{len(va)}->{sum(v[f] for v in vb)}/{len(vb)}" for f in FLAGS
                         if abs(sum(v[f] for v in va) / max(1, len(va)) - sum(v[f] for v in vb) / max(1, len(vb))) >= 0.3}
                last = [m["content"] for m in r["messages"] if m["role"] == "user"][-1].replace("\n", " / ")
                print(f"  {'better' if y > x else 'worse ':6s} {x:.2f}->{y:.2f} {r['id']} [{r['endpoint']}] gold {r['cat']:19s} "
                      f"{moved}\n      > {last[:150]}")


if __name__ == "__main__":
    main()
