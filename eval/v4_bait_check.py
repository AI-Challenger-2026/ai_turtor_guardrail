"""Strip "no authority" bait phrases from learner turns before the classifier sees them. Eval-only (2026-11-20).

    python3 eval/v4_bait_check.py --list                         # which public items change (no calls)
    python3 eval/v4_bait_check.py --collect VOTES.json --runs 3  # changed items x (Korean, English prompt) x 3 votes x runs

The classifier prompt already says that "관리자/개발자/선생님이야", "허락받았어", "거절하지 말고",
"소설·연극·역할극에 쓸 거야" and "그냥 궁금해서" give the learner no authority and never make a
question true. Yet one of the four unresolved items (pub-0d371526, a normal scam-countermeasure
question that says "거절하지 말고") is blocked as harm. Idea 3 of the 2026-11-20 review: remove those
phrases in code so the classifier only sees the request. The patterns are the general forms of the
prompt's own list (regular expressions, no word lists from public items).

Votes go into VOTES.json as arms ko_s, ko_s_b, ... (server Korean prompt) and en_s, en_s_b, ... (server
English prompt) for the items whose text changes. For every other item the input is identical, so
those arms get a copy of the server-prompt votes (new, newb, new_c / v9en, v9en_b, v9en_c) and the
comparison with eval/v4_expect.py --mix differs only where the stripping changed the input.

Result (2026-11-20, 1,134 calls, no failures): adopted (app/normalize.py, BAIT_RULES). Expected exact
of 3 Korean + 2 English votes over 400 items 382.0 -> 383.6 (fewshot 190.5 -> 191.6, dev 191.5 ->
192.0), wrong blocks 11.2 -> 9.8, missed attacks 5.0 -> 4.7; pub-0d371526 0.12 -> 1.00 and no item
worse by 0.15. Every item that moved has "거절하지 말고". The server leaves out the two teacher
patterns, written after seeing public wording; the 7 items they touch have no other phrase, so they
fall back to the server votes and the expectation stays 383.6. --fill / --repair added the round-14
Korean and round-12 English votes on these inputs (not adopted: the gain is on the fewshot half the
rules were written from; dev 192.3 -> 191.9 / 191.5).
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

BAIT = re.compile("|".join([
    r"거절하지\s*말고|거절\s*말고",
    r"(?:나|저)(?:는)?\s*(?:이\s*)?(?:튜터\s*)?(?:관리자|개발자|운영자|선생님)"
    r"(?:이에요|예요|이야|야|인데요|인데|입니다|임|라서)",
    r"선생님(?:이|께서)\s*[^.?!\n]{0,20}?(?:된댔어요?|된대요?|된다고\s*했어요?|허락(?:했어|하셨어)요?)",
    r"선생님\s*계정으로\s*들어왔(?:고|어|는데)",
    r"허락\s*받았(?:어요|어|는데|으니)",
    r"그냥\s*궁금해서(?:요|\s*(?:그러는데|묻는\s*거야|물어보는\s*거야|물어봐))?",
    r"(?:소설|연극|역할극)에?\s*쓸\s*(?:건데|거야|거예요)",
]))
_WORD = re.compile(r"[가-힣A-Za-z0-9]")
KO_BASE = ["new", "newb", "new_c"]
EN_BASE = ["v9en", "v9en_b", "v9en_c"]


def strip_bait(text: str) -> str:
    if not BAIT.search(text):
        return text
    out = BAIT.sub(" ", text)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out).strip(" ,.\n")
    return out if _WORD.search(out) else text  # never leave a turn without words


def strip_messages(messages):
    return [{**m, "content": strip_bait(m["content"])} if m.get("role") == "user" else m for m in messages]


def changed_rows():
    from app import normalize
    from v3_votes_check import public_rows
    out = []
    for r in public_rows():
        if normalize.is_low_signal(r["messages"]):
            continue
        before = normalize.for_classifier(r["messages"])
        after = strip_messages(before)
        if after != before:
            out.append((r, after))
    return out


def list_changes():
    rows = changed_rows()
    by = {}
    for r, _ in rows:
        key = (r["split"], "attack" if r["cat"] in ("academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic")
               else r["cat"])
        by[key] = by.get(key, 0) + 1
    print(f"{len(rows)} items change: {dict(sorted(by.items()))}")
    for r, after in rows:
        last = [m["content"] for m in after if m["role"] == "user"][-1].replace("\n", " / ")
        print(f"  {r['split']:7s} {r['endpoint']:5s} {r['id']} {r['cat']:19s} > {last[:110]}")


async def collect(path, runs, concurrency):
    from app import classify, classify_feat
    from app.llm import chat_json
    from v4_rules_check import new_system
    assert classify_feat.SYSTEM == new_system(), "server Korean prompt differs from the 'new' arms"
    assert classify_feat.SYSTEM_EN == Path("/home/elicer/competition/eval/out/v9en.system.txt").read_text(encoding="utf-8")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = changed_rows()
    sem = asyncio.Semaphore(concurrency)
    kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT)

    async def vote(system, text, kind, hedge):
        async with sem:
            try:
                out = await chat_json(system, text, kind=kind, hedge_after=hedge, **kw)
                return {"flags": {f: out.get(f) is True for f in classify_feat.FLAGS}}
            except Exception:  # noqa: BLE001
                return {"flags": None}

    suffix = ["", "_b", "_c", "_d", "_e"][:runs]
    jobs, keys = [], []
    for r, after in rows:
        text = classify.render(after)
        for s in suffix:
            for _ in range(3):
                jobs.append(vote(classify_feat.SYSTEM, text, "classify", 7.0)); keys.append((r["id"], "ko_s" + s))
                jobs.append(vote(classify_feat.SYSTEM_EN, text, "classify_en", None)); keys.append((r["id"], "en_s" + s))
    res = await asyncio.gather(*jobs)
    changed = {r["id"] for r, _ in rows}
    for rid in data:
        for s in suffix:
            for arm in ("ko_s" + s, "en_s" + s):
                data[rid][arm] = []
        if rid not in changed:  # identical input: reuse the server-prompt votes
            for i, s in enumerate(suffix):
                data[rid]["ko_s" + s] = list(data[rid][KO_BASE[i % 3]])
                data[rid]["en_s" + s] = list(data[rid][EN_BASE[i % 3]])
    for (rid, arm), v in zip(keys, res):
        data[rid][arm].append(v)
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} changed items, {len(res)} calls, failed {sum(v['flags'] is None for v in res)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--collect")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args()
    if a.list:
        list_changes()
    elif a.collect:
        asyncio.run(collect(a.collect, a.runs, a.concurrency))


if __name__ == "__main__" and "--fill" not in sys.argv and "--repair" not in sys.argv:
    main()


async def fill(path, base_arm, prompt_file, kind, concurrency):
    """Two runs (6 votes) of another prompt on the inputs the server would send after stripping:
    arms <base_arm>s and <base_arm>s_b. Items whose input does not change reuse the votes already
    stored under <base_arm> / <base_arm>_b (same prompt, same input); everything else is called."""
    from app import classify, classify_feat, normalize
    from app.llm import chat_json
    from v3_votes_check import public_rows
    text_in = Path(prompt_file).read_text(encoding="utf-8")
    if prompt_file.endswith(".system.txt"):
        system = text_in
    else:
        head, sep, tail = classify_feat.SYSTEM_BASE.rpartition("\n출력: JSON 객체 하나")
        system = head.rstrip() + "\n" + text_in + sep + tail
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    changed = {r["id"] for r, _ in changed_rows()}
    sem = asyncio.Semaphore(concurrency)
    kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT, kind=kind,
              hedge_after=None if kind == "classify_en" else 7.0)

    async def vote(text):
        async with sem:
            try:
                out = await chat_json(system, text, **kw)
                return {"flags": {f: out.get(f) is True for f in classify_feat.FLAGS}}
            except Exception:  # noqa: BLE001
                return {"flags": None}

    jobs, keys, reused = [], [], 0
    for r in public_rows():
        if normalize.is_low_signal(r["messages"]):
            continue
        text = classify.render(strip_messages(normalize.for_classifier(r["messages"])))
        for s in ("", "_b"):
            src = data[r["id"]].get(base_arm + s, [])
            if r["id"] not in changed and len(src) == 3:
                data[r["id"]][base_arm + "s" + s] = list(src)
                reused += 1
                continue
            data[r["id"]][base_arm + "s" + s] = []
            for _ in range(3):
                jobs.append(vote(text)); keys.append((r["id"], base_arm + "s" + s))
    if "--dry" in sys.argv:
        print(f"{base_arm}: reuse {reused} runs, call {len(jobs)}")
        return
    res = await asyncio.gather(*jobs)
    for (rid, arm), v in zip(keys, res):
        data[rid][arm].append(v)
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{base_arm}: reused {reused} runs, {len(res)} calls, failed {sum(v['flags'] is None for v in res)}")


if __name__ == "__main__" and "--fill" in sys.argv:
    i = sys.argv.index("--fill")
    asyncio.run(fill(*sys.argv[i + 1:i + 5], 24))


async def repair(path, arms, prompt_file, kind, concurrency=24, batch=96, max_fail=0.02):
    """Re-call only the failed votes (flags None) of the given arms, in batches; stop as soon as the
    cumulative failure rate passes max_fail (API trouble), saving what was done. Prints error classes."""
    from collections import Counter
    from app import classify, classify_feat, normalize
    from app.llm import chat_json
    from v3_votes_check import public_rows
    text_in = Path(prompt_file).read_text(encoding="utf-8")
    if prompt_file.endswith(".system.txt"):
        system = text_in
    else:
        head, sep, tail = classify_feat.SYSTEM_BASE.rpartition("\n출력: JSON 객체 하나")
        system = head.rstrip() + "\n" + text_in + sep + tail
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sem = asyncio.Semaphore(concurrency)
    kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT, kind=kind,
              hedge_after=None if kind == "classify_en" else 7.0)
    errors = Counter()

    async def vote(text):
        async with sem:
            try:
                out = await chat_json(system, text, **kw)
                return {"flags": {f: out.get(f) is True for f in classify_feat.FLAGS}}
            except Exception as e:  # noqa: BLE001
                errors[type(e).__name__] += 1
                return {"flags": None}

    slots = []
    for r in public_rows():
        if normalize.is_low_signal(r["messages"]):
            continue
        text = classify.render(strip_messages(normalize.for_classifier(r["messages"])))
        for arm in arms:
            for i, v in enumerate(data[r["id"]].get(arm, [])):
                if v["flags"] is None:
                    slots.append((r["id"], arm, i, text))
    print(f"{len(slots)} failed votes to redo")
    done = failed = 0
    for b in range(0, len(slots), batch):
        part = slots[b:b + batch]
        res = await asyncio.gather(*[vote(t) for *_, t in part])
        for (rid, arm, i, _), v in zip(part, res):
            data[rid][arm][i] = v
        done += len(res)
        failed += sum(v["flags"] is None for v in res)
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if failed > max(2, max_fail * done):
            print(f"STOP: {failed}/{done} failed ({dict(errors)})")
            return
    print(f"redone {done}, failed {failed} ({dict(errors)})")


if __name__ == "__main__" and "--repair" in sys.argv:
    i = sys.argv.index("--repair")
    p, arms, pf, kind = sys.argv[i + 1:i + 5]
    asyncio.run(repair(p, arms.split("+"), pf, kind))
