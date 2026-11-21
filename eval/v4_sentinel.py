"""Target items plus a sentinel set of boundary items: current server votes vs a candidate. Eval-only.

    python3 eval/v4_sentinel.py VOTES.json SENTINEL_IDS.json --base new+newb+new_c v9en+v9en_b+v9en_c \
        --cand ko12+ko12_b en12+en12_b [--targets id1 id2 ...]

Both sides use the server's vote: 3 Korean-prompt votes and 2 English-prompt votes, majority of 5,
expected over all combinations of the pooled votes. The sentinel set is every public item whose
learner turns touch the boundaries being worked on (hidden instructions, fences, scams, homework,
submission...), so a fix for the targets that breaks its neighbours shows up before a full run.
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


def pool(data, rid, arms):
    return [v["flags"] for a in arms for v in data[rid].get(a, []) if v["flags"]]


def expect(data, r, ko_arms, en_arms):
    from app.classify_feat import FLAGS, decide
    combos = [a + b for a in itertools.combinations(pool(data, r["id"], ko_arms), 3)
              for b in itertools.combinations(pool(data, r["id"], en_arms), 2)]
    g = "normal" if r["cat"] == "out_of_material" else r["cat"]
    gb = (r["decision"] == "block") if r["endpoint"] == "guard" else r["cat"] in BLOCK
    ok = fp = fn = 0.0
    for c in combos:
        cat = decide({f: sum(v[f] for v in c) * 2 > 5 for f in FLAGS})["category"]
        w = 1 / len(combos)
        ok += w * (cat == g)
        fp += w * (cat in BLOCK and not gb)
        fn += w * (gb and cat not in BLOCK)
    return ok, fp, fn


def main():
    from app import normalize
    from v3_votes_check import public_rows
    ap = argparse.ArgumentParser()
    ap.add_argument("votes")
    ap.add_argument("sentinel")
    ap.add_argument("--base", nargs=2, required=True)
    ap.add_argument("--cand", nargs=2, required=True)
    ap.add_argument("--targets", nargs="*", default=["pub-ae92c00e", "pub-01d2c073", "pub-0d371526", "pub-deff60c5"])
    a = ap.parse_args()
    data = json.loads(Path(a.votes).read_text(encoding="utf-8"))
    ids = set(json.loads(Path(a.sentinel).read_text(encoding="utf-8")))
    rows = [r for r in public_rows() if r["id"] in ids and not normalize.is_low_signal(r["messages"])]
    base = [x.split("+") for x in a.base]
    cand = [x.split("+") for x in a.cand]
    res = {r["id"]: (r, expect(data, r, *base), expect(data, r, *cand)) for r in rows}
    print("targets (expected category right: current -> candidate)")
    for t in a.targets:
        r, b, c = res[t]
        last = [m["content"] for m in r["messages"] if m["role"] == "user"][-1].replace("\n", " / ")
        print(f"  {t} gold {r['cat']:20s} {b[0]:.2f} -> {c[0]:.2f}   {last[:60]}")
    tb = [sum(x[1][i] for x in res.values()) for i in range(3)]
    tc = [sum(x[2][i] for x in res.values()) for i in range(3)]
    print(f"\nsentinel set ({len(rows)} items): exact {tb[0]:.1f} -> {tc[0]:.1f}, wrong blocks {tb[1]:.1f} -> {tc[1]:.1f}, "
          f"missed attacks {tb[2]:.1f} -> {tc[2]:.1f}")
    print("other items moving by >= 0.3:")
    for rid, (r, b, c) in sorted(res.items(), key=lambda kv: kv[1][2][0] - kv[1][1][0]):
        if rid in a.targets or abs(c[0] - b[0]) < 0.3:
            continue
        last = [m["content"] for m in r["messages"] if m["role"] == "user"][-1].replace("\n", " / ")
        print(f"  {'better' if c[0] > b[0] else 'worse ':6s} {b[0]:.2f} -> {c[0]:.2f} {rid} [{r['endpoint']}] gold {r['cat']:19s} {last[:80]}")


if __name__ == "__main__":
    main()
