"""Classifier outputs on public items.

    python3 eval/v3_classify_samples.py --collect OUT.json
    python3 eval/v3_classify_samples.py --eval OUT.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLOCK = {"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"}

_NAME_END = r"(?:이야|야|이에요|예요|입니다|이고|인데)"
_SCHOOL_GRADE_NAME = r"(?:초등학교|중학교|고등학교|[가-힣]{1,8}(?:초|중|고))\s*\d\s*학년(?:\s*\d+\s*반)?\s*[가-힣]{1,5}" + _NAME_END
_SELF_INTRO = re.compile(r"(?:^|\s)(?:나는|난|나|저는|전|저)\s+[^\n]{0,20}?" + _SCHOOL_GRADE_NAME
                         + r"|내\s*이름은\s*[가-힣]{1,5}" + _NAME_END)


def last_user(messages):
    users = [m["content"] for m in messages if m.get("role") == "user"]
    return users[-1] if users else ""


def rows():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for ep in ("tutor", "guard"):
        for line in (ROOT / f"public/{ep}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out.append({**r, "split": where[r["id"]]})
    return out


async def collect(path, concurrency, only_pattern=False):
    os.environ.setdefault("TEAM_TOKEN", "eval-only")
    from app.classify import classify
    rs = rows()
    if only_pattern:
        rs = [r for r in rs if _SELF_INTRO.search(last_user(r["messages"]))]
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        async with sem:
            c = await classify(r["messages"])
        return {"category": c["category"], "harmful_type": c["harmful_type"], "flags": c.get("flags")}

    res = await asyncio.gather(*[one(r) for r in rs])
    Path(path).write_text(json.dumps([{"id": r["id"], **x} for r, x in zip(rs, res)], ensure_ascii=False),
                          encoding="utf-8")
    print("saved", path, len(rs))


def with_rule(r, pred):
    if pred["flags"] is not None and pred["category"] == "normal" and _SELF_INTRO.search(last_user(r["messages"])):
        return "pii_disclosure"
    return pred["category"]


def report(sel, preds, name):
    scored = [r for r in sel if r["labels"]["category"] != "normal"]
    cat_acc = sum(preds[r["id"]] == r["labels"]["category"] for r in scored) / max(1, len(scored))
    tp = sum(r["labels"]["category"] in BLOCK and preds[r["id"]] in BLOCK for r in sel)
    fp = sum(r["labels"]["category"] not in BLOCK and preds[r["id"]] in BLOCK for r in sel)
    fn = sum(r["labels"]["category"] in BLOCK and preds[r["id"]] not in BLOCK for r in sel)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    pii = [r for r in sel if r["labels"]["category"] == "pii_disclosure"]
    pii_hit = sum(preds[r["id"]] == "pii_disclosure" for r in pii)
    print(f"  {name:6s} category acc {cat_acc:.3f} ({len(scored)} scored)  block F1 {f1:.3f} (fp {fp}, fn {fn})  "
          f"pii {pii_hit}/{len(pii)}")


def evaluate(path):
    data = {d["id"]: d for d in json.loads(Path(path).read_text(encoding="utf-8"))}
    rs = [r for r in rows() if r["id"] in data]
    print("classifier failed (no flags):", sum(data[r["id"]]["flags"] is None for r in rs))
    for split in ("fewshot", "dev", "all"):
        sel = [r for r in rs if split == "all" or r["split"] == split]
        base = {r["id"]: data[r["id"]]["category"] for r in sel}
        rule = {r["id"]: with_rule(r, data[r["id"]]) for r in sel}
        changed = [r for r in sel if base[r["id"]] != rule[r["id"]]]
        fixed = sum(r["labels"]["category"] == "pii_disclosure" for r in changed)
        print(f"[{split}] items {len(sel)}  rule changes {len(changed)}: fixed {fixed}, wrong {len(changed) - fixed} "
              f"(wrong by gold: {dict(Counter(r['labels']['category'] for r in changed if r['labels']['category'] != 'pii_disclosure'))})")
        report(sel, base, "v3")
        report(sel, rule, "+rule")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--only-pattern", action="store_true", help="classify only items the rule could change")
    ap.add_argument("--eval")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.concurrency, a.only_pattern))
    if a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
