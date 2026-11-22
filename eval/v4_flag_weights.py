"""How Korean and English classifier votes combine, per signal (no API calls).

    python3 eval/v4_flag_weights.py VOTES.json
"""
import itertools
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

BLOCK = {"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"}
OPTIONS = ("mix", "ko", "en_both", "en_any", "w_en2", "w_ko2")


def flag_value(opt, ko, en):
    if opt == "mix":
        return ko + en >= 3
    if opt == "ko":
        return ko >= 2
    if opt == "en_both":
        return en == 2
    if opt == "en_any":
        return en >= 1
    if opt == "w_en2":
        return ko + 2 * en >= 4
    if opt == "w_ko2":
        return 2 * ko + en >= 5
    raise ValueError(opt)


def draws_for(data, rid, ko_arms, en_arms, n, rng):
    ko = [v["flags"] for a in ko_arms for v in data[rid].get(a, []) if v["flags"]]
    en = [v["flags"] for a in en_arms for v in data[rid].get(a, []) if v["flags"]]
    ko_c = list(itertools.combinations(ko, 3))
    en_c = list(itertools.combinations(en, 2))
    pairs = [(a, b) for a in ko_c for b in en_c]
    if len(pairs) > n:
        pairs = rng.sample(pairs, n)
    from app.classify_feat import FLAGS
    return [({f: sum(v[f] for v in a) for f in FLAGS}, {f: sum(v[f] for v in b) for f in FLAGS}) for a, b in pairs]


def score(items, rule):
    from app.classify_feat import FLAGS, decide
    from v3_votes_check import tutor_ok
    ex = fp = fn = ta = 0.0
    for r, draws in items:
        g = "normal" if r["cat"] == "out_of_material" else r["cat"]
        gb = (r["decision"] == "block") if r["endpoint"] == "guard" else r["cat"] in BLOCK
        w = 1 / len(draws)
        for ko, en in draws:
            d = decide({f: flag_value(rule[f], ko[f], en[f]) for f in FLAGS})
            pb = d["category"] in BLOCK
            ex += w * (d["category"] == g)
            fp += w * (pb and not gb)
            fn += w * (gb and not pb)
            if r["endpoint"] == "tutor":
                ta += w * tutor_ok(r, d["category"], d["harmful_type"])
    return ex, fp, fn, ta


def choose(items):
    from app.classify_feat import FLAGS
    rule = {f: "mix" for f in FLAGS}
    best = score(items, rule)[0]
    for _ in range(3):
        changed = False
        for f in FLAGS:
            for opt in OPTIONS:
                cand = {**rule, f: opt}
                s = score(items, cand)[0]
                if s > best + 1e-9:
                    best, rule, changed = s, cand, True
        if not changed:
            break
    return rule


def show(name, items, rule):
    ex, fp, fn, ta = score(items, rule)
    print(f"  {name:44s} exact {ex:6.1f}/{len(items)}  wrong blocks {fp:4.1f}  missed attacks {fn:4.1f}  tutor action {ta:5.1f}")


def main():
    from app import normalize
    from app.classify_feat import FLAGS
    from v3_votes_check import public_rows
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    rows = [r for r in public_rows() if not normalize.is_low_signal(r["messages"])]
    rng = random.Random(20261120)
    KO, EN = ["new", "newb", "new_c"], ["v9en", "v9en_b", "v9en_c"]
    allv = [(r, draws_for(data, r["id"], KO, EN, 300, rng)) for r in rows]
    mix = {f: "mix" for f in FLAGS}
    print("all votes (9 Korean, 9 English per item); word-less items left out (always right)")
    show("mix for every signal (server)", allv, mix)
    rule_all = choose(allv)
    show("best per signal, chosen on all 400 (optimistic)", allv, rule_all)
    print("    chosen:", {f: o for f, o in rule_all.items() if o != "mix"})

    print("\n1) choose on one half, score the other")
    for a, b in (("fewshot", "dev"), ("dev", "fewshot")):
        tr = [x for x in allv if x[0]["split"] == a]
        te = [x for x in allv if x[0]["split"] == b]
        rule = choose(tr)
        print(f"   chosen on {a}: {{{', '.join(f'{f}: {o}' for f, o in rule.items() if o != 'mix')}}}")
        show(f"   {b}: mix", te, mix)
        show(f"   {b}: chosen", te, rule)

    print("\n2) choose with runs 1-2, score with run 3 only")
    first = [(r, draws_for(data, r["id"], KO[:2], EN[:2], 300, rng)) for r, _ in allv]
    third = [(r, draws_for(data, r["id"], KO[2:], EN[2:], 300, rng)) for r, _ in allv]
    rule = choose(first)
    print("   chosen:", {f: o for f, o in rule.items() if o != "mix"})
    show("   run 3: mix", third, mix)
    show("   run 3: chosen", third, rule)


if __name__ == "__main__":
    main()
