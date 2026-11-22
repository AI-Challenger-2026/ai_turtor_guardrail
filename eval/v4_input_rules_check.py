"""Input rules (app/normalize.py) on the public items they change.

    python3 eval/v4_input_rules_check.py --out RESULT.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

BLOCK = {"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"}


async def run(out_path, concurrency):
    from app import normalize
    from app.classify import classify
    from v3_votes_check import public_rows
    rows = public_rows()
    touched = []
    for r in rows:
        spaced = any(m["role"] == "user" and normalize.join_spaced(m["content"]) != m["content"] for m in r["messages"])
        wordless = not normalize.has_words(normalize.last_user_turn(r["messages"]))
        if spaced or wordless:
            touched.append((r, "wordless" if wordless else "spaced"))
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        async with sem:
            c = await classify(r["messages"])
            return c["category"]

    result = {}
    for on in (False, True):
        normalize.INPUT_RULES = on
        cats = await asyncio.gather(*[one(r) for r, _ in touched])
        result["on" if on else "off"] = {r["id"]: c for (r, _), c in zip(touched, cats)}
    normalize.INPUT_RULES = True
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    def gold(r):
        return "normal" if r["cat"] == "out_of_material" else r["cat"]

    def gold_block(r):
        return (r["decision"] == "block") if r["endpoint"] == "guard" else r["cat"] in BLOCK

    for kind in ("wordless", "spaced"):
        sub = [r for r, k in touched if k == kind]
        print(f"\n{kind}: {len(sub)} items")
        for side in ("off", "on"):
            p = result[side]
            exact = sum(p[r["id"]] == gold(r) for r in sub)
            fp = sum(p[r["id"]] in BLOCK and not gold_block(r) for r in sub)
            fn = sum(gold_block(r) and p[r["id"]] not in BLOCK for r in sub)
            print(f"  INPUT_RULES {side:3s}: exact {exact}/{len(sub)}, wrong blocks {fp}, missed blocks {fn}")
        for r in sub:
            a, b = result["off"][r["id"]], result["on"][r["id"]]
            if a != b:
                print(f"  changed {r['endpoint']:5s} {r['id']} gold {r['cat']:20s} off {a:20s} -> on {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    asyncio.run(run(a.out, a.concurrency))


if __name__ == "__main__":
    main()
