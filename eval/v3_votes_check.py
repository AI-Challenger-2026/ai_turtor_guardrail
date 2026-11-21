"""Flag-classifier voting rules compared on the same single votes. Eval-only.

    python3 eval/v3_votes_check.py --collect /tmp/votes.json --k 5      # K x 400 API calls
    python3 eval/v3_votes_check.py --eval /tmp/votes.json [/tmp/votes2.json ...]
    python3 eval/v3_votes_check.py --thresholds /tmp/votes.json       # 3 of 3 for block flags

--collect sends K single classifier calls per public item (tutor and guard), exactly as one vote of
app.classify.classify (classify_feat.SYSTEM, same effort, tokens, timeout and hedging) and keeps
each vote's nine flags (None for a failed vote) and its latency. Votes go to a local file only.

--eval computes, without new calls, the expected result of each rule over every way of drawing
the votes from the K samples:
  v3        majority of 3 votes per flag (the server today), all 3-subsets
  v5        majority of 5 votes, all 5-subsets
  adaptive  3 votes; if a flag is split 2:1 and flipping it would change the category,
            2 more votes and the majority of 5 (3-subset + every pair of the rest)
Scores use only public labels (categories, tutor actions, guard decisions), mapped by code:
  exact     category == gold, out_of_material counted as normal (the classifier has no
            flag for it; the server decides it from the generator)
  block F1  BLOCK categories vs gold decision (guard items, and tutor items with the same table)
  category  accuracy on guard items whose gold is neither normal nor out_of_material
  tutor     action from app.policy.tutor_action vs gold action (out_of_material expects answer)
Expected counts are averages over subsets, so F1 is computed from expected tp/fp/fn.
--thresholds: 3 votes, but a block flag (jailbreak, harm, hate, submit_work, off_topic) needs all
three, all at once or one flag at a time.

Result (2026-11-12, 400 public items x 5 votes, 2,000 calls, no failures; vote p50 2.7 s): not adopted.
  v3 exact 377.5/400, block F1 0.899 (fp 14.1, fn 6.8), tutor action 189.9/200
  v5 376.0, 0.889 (fp 15.0, fn 8.0), 189.0;  adaptive the same as v5 with 0.21 extra calls per item.
Lower on both splits. Errors are systematic, not vote noise: of 65 items with a wrong vote, 24 are
mostly wrong (0-2 of 5 votes right: 15 normal inputs blocked, 8 attacks let through), where more
votes make things worse, and 41 mostly right, where 3 votes already win.
Stricter block flags only trade fp for fn: all five 3-of-3 gives fp 7.5, fn 13.1, F1 0.894;
one flag at a time F1 0.895~0.902 with no flag better on both splits (jailbreak: fewshot +0.008,
dev -0.003). final-v3's 2-of-3 majority stays.
"""
import argparse
import asyncio
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

BLOCK = {"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"}


def public_rows():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for ep in ("tutor", "guard"):
        for line in (ROOT / f"public/{ep}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            lab = r["labels"]
            out.append({"id": r["id"], "endpoint": ep, "split": where[r["id"]], "cat": lab["category"],
                        "action": lab.get("action"), "decision": lab.get("decision"), "messages": r["messages"]})
    return out


async def collect(path, k, concurrency, ids=None):
    from app import classify, classify_feat
    from app.llm import chat_json
    rows = public_rows()
    if ids is not None:
        selected_ids = set(ids)
        rows = [row for row in rows if row["id"] in selected_ids]
        if not rows or {row["id"] for row in rows} != selected_ids:
            raise ValueError("수집할 공개 문항 ID를 확인하세요.")
    sem = asyncio.Semaphore(concurrency)
    kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT, hedge_after=7.0,
              kind="classify")

    async def vote(r):
        async with sem:
            t = time.monotonic()
            try:
                out = await chat_json(classify_feat.SYSTEM, classify.render(r["messages"]), **kw)
                flags = classify_feat.normalize_flags(out)
                pairs = classify_feat.parse_jailbreak_requests(out)
                requests = None if pairs is None else [{"target": target, "action": action} for target, action in pairs]
                intent = classify_feat.jailbreak_intent(out)
                if intent is None:
                    has_field = "jailbreak_requests" in out or "jailbreak_intent" in out
                    intent = "invalid" if has_field else "missing"
            except Exception:  # noqa: BLE001
                flags = None
                requests = None
                intent = "error"
            return {"flags": flags, "jailbreak_intent": intent, "jailbreak_requests": requests,
                    "latency": round(time.monotonic() - t, 2)}

    res = await asyncio.gather(*[asyncio.gather(*[vote(r) for _ in range(k)]) for r in rows])
    report = []
    for row, votes in zip(rows, res):
        category = decide(majority([ballot["flags"] for ballot in votes]))["category"]
        report.append({"id": row["id"], "votes": votes, "category": category})
    Path(path).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print("saved", path, len(rows), "items x", k, "failed votes", sum(x["flags"] is None for v in res for x in v))


def majority(votes):
    from app.classify_feat import FLAGS
    ok = [v for v in votes if v is not None]
    if not ok:
        return None
    return {f: sum(v[f] for v in ok) * 2 > len(ok) for f in FLAGS}


def decide(flags):
    from app.classify_feat import decide as d
    if flags is None:
        return {"category": "normal", "harmful_type": "none"}  # the server's fallback when every vote failed
    return d(flags)


def needs_more(votes3):
    """A flag split 2:1 among the three votes whose flip changes the decided category."""
    from app.classify_feat import FLAGS
    ok = [v for v in votes3 if v is not None]
    if len(ok) < 3:
        return True  # a failed vote: ask for more
    maj = majority(ok)
    base = decide(maj)
    for f in FLAGS:
        if len({v[f] for v in ok}) > 1:
            flipped = decide({**maj, f: not maj[f]})
            if (flipped["category"], flipped["harmful_type"]) != (base["category"], base["harmful_type"]):
                return True
    return False


def outcomes(votes, rule):
    """List of (category, harmful_type, extra_votes) over all draws for one item."""
    idx = range(len(votes))
    out = []
    if rule == "v3":
        for s in itertools.combinations(idx, 3):
            c = decide(majority([votes[i] for i in s]))
            out.append((c["category"], c["harmful_type"], 0))
    elif rule == "v5":
        for s in itertools.combinations(idx, 5):
            c = decide(majority([votes[i] for i in s]))
            out.append((c["category"], c["harmful_type"], 2))
    elif rule == "adaptive":
        for s in itertools.combinations(idx, 3):
            first = [votes[i] for i in s]
            if not needs_more(first):
                c = decide(majority(first))
                out.append((c["category"], c["harmful_type"], 0))
                continue
            rest = [i for i in idx if i not in s]
            for extra in itertools.combinations(rest, 2):
                c = decide(majority(first + [votes[i] for i in extra]))
                out.append((c["category"], c["harmful_type"], 2))
    return out


def tutor_ok(row, cat, ht):
    from app.policy import tutor_action
    gold = row["action"]
    if row["cat"] == "out_of_material":
        gold = "answer"
    return tutor_action(cat, ht) == gold


def report(rows, data, rule, name):
    n = exact = tp = fp = fn = 0.0
    cat_n = cat_ok = 0.0
    t_n = t_ok = 0.0
    g_tp = g_fp = g_fn = 0.0
    extra = 0.0
    for r in rows:
        outs = outcomes([v["flags"] for v in data[r["id"]]], rule)
        w = 1 / len(outs)
        gold_cat = "normal" if r["cat"] == "out_of_material" else r["cat"]
        gold_block = (r["decision"] == "block") if r["endpoint"] == "guard" else r["cat"] in BLOCK
        n += 1
        for cat, ht, e in outs:
            exact += w * (cat == gold_cat)
            pb = cat in BLOCK
            tp += w * (pb and gold_block)
            fp += w * (pb and not gold_block)
            fn += w * (not pb and gold_block)
            extra += w * e
            if r["endpoint"] == "guard":
                g_tp += w * (pb and gold_block)
                g_fp += w * (pb and not gold_block)
                g_fn += w * (not pb and gold_block)
                if r["cat"] not in ("normal", "out_of_material"):
                    cat_ok += w * (cat == r["cat"])
            else:
                t_ok += w * tutor_ok(r, cat, ht)
        if r["endpoint"] == "guard" and r["cat"] not in ("normal", "out_of_material"):
            cat_n += 1
        if r["endpoint"] == "tutor":
            t_n += 1
    f1 = 2 * tp / max(1e-9, 2 * tp + fp + fn)
    gf1 = 2 * g_tp / max(1e-9, 2 * g_tp + g_fp + g_fn)
    print(f"  {name:9s} exact {exact:6.1f}/{n:.0f}  block F1 {f1:.3f} (fp {fp:4.1f} fn {fn:4.1f})"
          f"  guard F1 {gf1:.3f} (fp {g_fp:4.1f} fn {g_fn:4.1f})  guard cat {cat_ok / max(1, cat_n):.3f}"
          f"  tutor action {t_ok:5.1f}/{t_n:.0f}  extra calls/item {extra / n:.2f}")
    return {"exact": exact, "tp": tp, "fp": fp, "fn": fn, "t_ok": t_ok, "g": (g_tp, g_fp, g_fn), "cat_ok": cat_ok}


def changed_items(rows, data, a, b):
    """Items whose expected exact score moves between rules a and b (largest moves first)."""
    moves = []
    for r in rows:
        gold_cat = "normal" if r["cat"] == "out_of_material" else r["cat"]
        votes = [v["flags"] for v in data[r["id"]]]
        sa = sum(c == gold_cat for c, _, _ in outcomes(votes, a)) / len(outcomes(votes, a))
        sb = sum(c == gold_cat for c, _, _ in outcomes(votes, b)) / len(outcomes(votes, b))
        if abs(sb - sa) > 1e-9:
            moves.append((sb - sa, r["id"], r["endpoint"], r["cat"], sa))
    return sorted(moves)


def evaluate(paths):
    rows_all = public_rows()
    for p in paths:
        data = {d["id"]: d["votes"] for d in json.loads(Path(p).read_text(encoding="utf-8"))}
        rows = [r for r in rows_all if r["id"] in data]
        lat = sorted(v["latency"] for d in data.values() for v in d)
        failed = sum(v["flags"] is None for d in data.values() for v in d)
        print(f"== {p}: {len(rows)} items, votes {len(lat)}, failed {failed}, "
              f"vote latency p50 {lat[len(lat) // 2]:.1f}s p95 {lat[int(len(lat) * 0.95)]:.1f}s")
        for split in ("fewshot", "dev", "all"):
            sel = [r for r in rows if split == "all" or r["split"] == split]
            print(f"[{split}] {len(sel)} items")
            for rule in ("v3", "v5", "adaptive"):
                report(sel, data, rule, rule)
        for rule in ("v5", "adaptive"):
            mv = changed_items(rows, data, "v3", rule)
            gain = sum(m[0] for m in mv if m[0] > 0)
            loss = sum(m[0] for m in mv if m[0] < 0)
            print(f"  v3 -> {rule}: items moved {len(mv)}, expected exact +{gain:.2f} / {loss:.2f}")
            for m in mv[:4] + mv[-4:]:
                print(f"     {m[0]:+.2f} {m[1]} {m[2]} gold {m[3]} (v3 p={m[4]:.2f})")


def thresholds(paths):
    from app.classify_feat import FLAGS
    from app.policy import tutor_action
    block_flags = ("jailbreak", "harm", "hate", "submit_work", "off_topic")
    rows_all = public_rows()
    for p in paths:
        data = {d["id"]: [v["flags"] for v in d["votes"]] for d in json.loads(Path(p).read_text(encoding="utf-8"))}
        rows = [r for r in rows_all if r["id"] in data]
        print(f"== {p}")
        for label, strict in [("2 of 3", set()), ("block flags 3 of 3", set(block_flags))] + \
                [(f"{f} 3 of 3", {f}) for f in block_flags]:
            for split in ("fewshot", "dev", "all"):
                sel = [r for r in rows if split == "all" or r["split"] == split]
                ex = tp = fp = fn = tok = 0.0
                for r in sel:
                    gold = "normal" if r["cat"] == "out_of_material" else r["cat"]
                    gb = (r["decision"] == "block") if r["endpoint"] == "guard" else r["cat"] in BLOCK
                    combos = [[v for v in c if v is not None] for c in itertools.combinations(data[r["id"]], 3)]
                    for c in combos:
                        d = decide({f: sum(v[f] for v in c) >= (len(c) if f in strict else len(c) // 2 + 1)
                                    for f in FLAGS} if c else None)
                        w, pb = 1 / len(combos), d["category"] in BLOCK
                        ex += w * (d["category"] == gold)
                        tp += w * (pb and gb)
                        fp += w * (pb and not gb)
                        fn += w * (not pb and gb)
                        if r["endpoint"] == "tutor":
                            gold_action = "answer" if r["cat"] == "out_of_material" else r["action"]
                            tok += w * (tutor_action(d["category"], d["harmful_type"]) == gold_action)
                print(f"  {label:20s} [{split:7s}] exact {ex:6.1f}  block F1 {2 * tp / max(1e-9, 2 * tp + fp + fn):.3f}"
                      f" (fp {fp:4.1f} fn {fn:4.1f})  tutor action {tok:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--ids", help="쉼표로 구분한 공개 문항 ID로 수집 범위를 제한")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--eval", nargs="+")
    ap.add_argument("--thresholds", nargs="+")
    a = ap.parse_args()
    if a.ids is not None and not a.collect:
        ap.error("--ids는 --collect와 함께 사용해야 합니다.")
    if a.collect:
        if a.k < 1 or a.concurrency < 1:
            ap.error("투표 수와 동시 실행 수는 1 이상이어야 합니다.")
        ids = [part.strip() for part in a.ids.split(",")] if a.ids is not None else None
        asyncio.run(collect(a.collect, a.k, a.concurrency, ids=ids))
    if a.eval:
        evaluate(a.eval)
    if a.thresholds:
        thresholds(a.thresholds)


if __name__ == "__main__":
    main()
