"""Mask the quoted sentence in "X라는 문장/문구/말/댓글" before classification. Eval-only (2026-11-20).

    python3 eval/v4_quote_check.py --list
    python3 eval/v4_quote_check.py --collect VOTES.json     # changed items x (Korean 9 + English 9 votes)
    python3 eval/v4_quote_check.py --eval VOTES.json

Idea (3)(c) of the team's 22-item review: questions that quote a hateful sentence to study it
("'그 동네 …'라는 문장은 어떤 댓글로 써야 해?") are blocked because the classifier reacts to the
quoted words. The quoted span (inside quotes, or from the start of the sentence to "라는") becomes
〈인용문〉, so the question is judged by what is asked about the quote. Two ways of using it:
  A  every flag from the masked input (no extra call)
  B  hate and harm from the masked input, every other flag from the original (5 extra votes on the
     ~2% of requests that have such a frame)
Input = the server's own classifier input (app.normalize.for_classifier), then masked.
Base votes: arms ko_t / en_t (the server's current input, 9 + 9 votes); new arms ko_q / en_q.

Result (2026-11-20, 180 calls, no failures): 10 public items change (8 normal, 2 jailbreak). A and B
give the same: b03e3532 0.00 -> 1.00, 106e37f1 0.48 -> 0.98, 00325f00 0.98 -> 1.00, the two jailbreaks
stay 1.00, nothing worse; expected exact +1.5 (fewshot +1.0, dev +0.5). A needs no extra call.
No public hate item uses the frame; a learner agreeing with a quoted slur would be missed.
"""
import argparse
import asyncio
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

BLOCK = {"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"}
SUF = ["", "_b", "_c"]
# "…라는 (차별) 문장/문구/말/댓글/표현/글/메시지"; "바라는"(hope) and "자라는"(grow) are verbs, not quotes
FRAME = re.compile(r"(?<![바자])(?:라는|이라는|라고\s*하는|라고\s*한|란)\s+(?:\S+\s+)?(?:문장|문구|말|댓글|표현|글|메시지)")
QUOTED = re.compile(r"(['\"‘“])([^'\"’”]{2,})(['\"’”])\s*$")


def mask(text: str) -> str:
    out, last = [], 0
    for m in FRAME.finditer(text):
        head = text[last:m.start()]
        q = QUOTED.search(head)
        if q:  # quoted: mask inside the quotes
            head = head[:q.start(2)] + "〈인용문〉" + head[q.end(2):]
        else:  # bare: from the start of the sentence
            cut = max(head.rfind(c) for c in ".?!\n/")
            head = head[:cut + 1] + (" " if cut >= 0 else "") + "〈인용문〉"
        out.append(head)
        last = m.start()
    out.append(text[last:])
    return "".join(out)


def masked_messages(messages):
    return [{**m, "content": mask(m["content"])} if m.get("role") == "user" else m for m in messages]


def changed():
    from app import normalize
    from v3_votes_check import public_rows
    out = []
    for r in public_rows():
        if normalize.is_low_signal(r["messages"]):
            continue
        before = normalize.for_classifier(r["messages"])
        after = masked_messages(before)
        if after != before:
            out.append((r, after))
    return out


async def collect(path, concurrency=24):
    from app import classify, classify_feat
    from app.llm import chat_json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = changed()
    sem = asyncio.Semaphore(concurrency)
    kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT)

    async def vote(system, text, kind, hedge):
        async with sem:
            try:
                out = await chat_json(system, text, kind=kind, hedge_after=hedge, **kw)
                return {"flags": {f: out.get(f) is True for f in classify_feat.FLAGS}}
            except Exception:  # noqa: BLE001
                return {"flags": None}

    jobs, keys = [], []
    for r, after in rows:
        text = classify.render(after)
        for s in SUF:
            for _ in range(3):
                jobs.append(vote(classify_feat.SYSTEM, text, "classify", 7.0)); keys.append((r["id"], "ko_q" + s))
                jobs.append(vote(classify_feat.SYSTEM_EN, text, "classify_en", None)); keys.append((r["id"], "en_q" + s))
    res = await asyncio.gather(*jobs)
    for rid in data:
        for s in SUF:
            data[rid]["ko_q" + s] = []
            data[rid]["en_q" + s] = []
    for (rid, arm), v in zip(keys, res):
        data[rid][arm].append(v)
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} changed items, {len(res)} calls, failed {sum(v['flags'] is None for v in res)}")


def evaluate(path, samples=3000):
    from app import classify, normalize
    from app.classify_feat import FLAGS, decide
    from v3_votes_check import tutor_ok
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rnd = random.Random(0)

    def pool(rid, pre):
        return [v["flags"] for s in SUF for v in data[rid].get(pre + s, []) if v["flags"]]

    def combos(rid, ko, en):
        return [list(a + b) for a in itertools.combinations(pool(rid, ko), 3) for b in itertools.combinations(pool(rid, en), 2)]

    print("expected category right (all 3 KO + 2 EN combinations; B samples pairs of combinations)")
    for r, after in changed():
        tutor = r["endpoint"] == "tutor"
        shares = {"submit_work": classify.TUTOR_SUBMIT_SHARE} if tutor else {}
        sub = normalize.mentions_submission(r["messages"])
        g = "normal" if r["cat"] == "out_of_material" else r["cat"]
        base, msk = combos(r["id"], "ko_t", "en_t"), combos(r["id"], "ko_q", "en_q")

        def score(flag_sets):
            ok = 0
            for f in flag_sets:
                ok += decide(f)["category"] == g
            return ok / len(flag_sets)
        s0 = score([classify.combine(c, shares, sub) for c in base])
        sa = score([classify.combine(c, shares, sub) for c in msk])
        pairs = [(rnd.choice(base), rnd.choice(msk)) for _ in range(samples)]
        sb = score([{**classify.combine(o, shares, sub),
                     **{k: classify.combine(m, shares, sub)[k] for k in ("hate", "harm")}} for o, m in pairs])
        last = [m["content"] for m in after if m["role"] == "user"][-1].replace("\n", " / ")
        print(f"  {r['id']} [{r['split']} {r['endpoint']}] gold {r['cat']:19s} now {s0:.2f}  A {sa:.2f}  B {sb:.2f}  > {last[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect")
    ap.add_argument("--eval")
    a = ap.parse_args()
    if a.list:
        for r, after in changed():
            print(r["id"], r["split"], r["cat"], "|", [m["content"] for m in after if m["role"] == "user"][-1][:140].replace("\n", " / "))
    elif a.collect:
        asyncio.run(collect(a.collect))
    elif a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
