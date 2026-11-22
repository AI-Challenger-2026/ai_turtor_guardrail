"""Classifier with similar examples added to the prompt (not used by the server).

    python3 eval/v3_dynshot_check.py --collect OUT.json --variants base,shots --split fewshot
    python3 eval/v3_dynshot_check.py --eval OUT.json
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

BLOCK = {"academic_dishonesty", "harmful_content", "jailbreak_injection", "off_topic"}
CAT_FLAG = {"emotional_distress": "distress", "low_signal_input": "low_signal", "jailbreak_injection": "jailbreak",
            "academic_dishonesty": "submit_work", "off_topic": "off_topic", "answer_seeking": "answer_only",
            "pii_disclosure": "self_pii", "normal": None}
K = 6
HEAD = ("[참고: 표현이 비슷한 공개 예시의 마지막 학습자 발화와 정답 신호. 표현이 비슷해도 실제로 시키는 일이 다르면 "
        "다르게 판단한다]")


def user_turns(messages):
    return [m.get("content", "") for m in messages if m.get("role") == "user"]


def similarity_text(messages):
    turns = user_turns(messages)
    return "\n".join(turns + turns[-1:])


def short(text, head=140, tail=100):
    t = " ".join(text.split())
    return t if len(t) <= head + tail + 5 else f"{t[:head]} … {t[-tail:]}"


def public_rows():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    group = {i: g["group"] for g in split["groups"] for i in g["ids"]}
    out = []
    for ep in ("tutor", "guard"):
        for line in (ROOT / f"public/{ep}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out.append({"id": r["id"], "endpoint": ep, "split": where[r["id"]], "group": group[r["id"]],
                        "cat": r["labels"]["category"], "action": r["labels"].get("action"), "messages": r["messages"]})
    return out


def label(r):
    if r["cat"] == "harmful_content":
        return {"refuse": "harm", "redirect": "hate"}.get(r["action"], "harm 또는 hate")
    flag = CAT_FLAG[r["cat"]]
    return flag or "없음"


class Pool:
    def __init__(self, rows):
        from app.kb import BM25
        self.items = [r for r in rows if r["split"] == "fewshot" and r["cat"] != "out_of_material"]
        self.bm25 = BM25([similarity_text(r["messages"]) for r in self.items])

    def similar(self, messages, exclude_group=None, k=K):
        scores = self.bm25.scores(similarity_text(messages))
        ranked = sorted((i for i, s in enumerate(scores) if s > 0 and self.items[i]["group"] != exclude_group),
                        key=lambda i: (-scores[i], self.items[i]["id"]))
        return [self.items[i] for i in reversed(ranked[:k])]

    def block(self, messages, exclude_group=None):
        shots = self.similar(messages, exclude_group)
        if not shots:
            return ""
        lines = [HEAD] + [f'- "{short(user_turns(s["messages"])[-1])}" → true인 신호: {label(s)}' for s in shots]
        return "\n".join(lines) + "\n\n[분류할 대화]\n"


async def classify_with(messages, prefix):
    from app import classify, classify_feat
    from app.llm import chat_json
    kw = dict(effort=classify.EFFORT, max_tokens=classify.MAX_TOKENS, timeout=classify.TIMEOUT, hedge_after=7.0,
              kind="classify")
    user = prefix + classify.render(messages)
    outs = await asyncio.gather(*[chat_json(classify_feat.SYSTEM, user, **kw) for _ in range(classify.VOTES)],
                                return_exceptions=True)
    outs = [o for o in outs if isinstance(o, dict)]
    if not outs:
        return {"category": None, "flags": None}
    flags = {k: sum(o.get(k) is True for o in outs) * 2 > len(outs) for k in classify_feat.FLAGS}
    return {**classify_feat.decide(flags), "flags": flags, "votes": len(outs)}


async def collect(path, variants, split, concurrency):
    import time
    rows = public_rows()
    pool = Pool(rows)
    sel = [r for r in rows if split == "all" or r["split"] == split]
    sem = asyncio.Semaphore(concurrency)

    async def one(r, variant):
        prefix = pool.block(r["messages"], r["group"]) if variant == "shots" else ""
        async with sem:
            t = time.monotonic()
            c = await classify_with(r["messages"], prefix)
            return {"id": r["id"], "variant": variant, **c, "latency": round(time.monotonic() - t, 2),
                    "shots": [s["id"] for s in pool.similar(r["messages"], r["group"])] if variant == "shots" else []}

    res = await asyncio.gather(*[one(r, v) for v in variants for r in sel])
    Path(path).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("saved", path, len(res), "classifications; failed", sum(x["category"] is None for x in res))


def expected(cat):
    return "normal" if cat == "out_of_material" else cat


def report(rows, preds, name):
    sel = [r for r in rows if r["id"] in preds]
    exact = sum(preds[r["id"]] == expected(r["cat"]) for r in sel)
    scored = [r for r in sel if r["cat"] not in ("normal", "out_of_material")]
    cat_acc = sum(preds[r["id"]] == r["cat"] for r in scored) / max(1, len(scored))
    tp = sum(r["cat"] in BLOCK and preds[r["id"]] in BLOCK for r in sel)
    fp = sum(r["cat"] not in BLOCK and preds[r["id"]] in BLOCK for r in sel)
    fn = sum(r["cat"] in BLOCK and preds[r["id"]] not in BLOCK for r in sel)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    pii = [r for r in sel if r["cat"] == "pii_disclosure"]
    print(f"  {name:22s} n={len(sel)} exact {exact}  category acc {cat_acc:.3f}  block F1 {f1:.3f} (fp {fp}, fn {fn})"
          f"  pii {sum(preds[r['id']] == 'pii_disclosure' for r in pii)}/{len(pii)}")


def evaluate(paths):
    rows = public_rows()
    by_id = {r["id"]: r for r in rows}
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    for split in ("fewshot", "dev"):
        print(f"== {split}")
        sp = [r for r in rows if r["split"] == split]
        for k, res in enumerate(runs):
            for variant in ("base", "shots"):
                preds = {x["id"]: x["category"] for x in res if x["variant"] == variant and x["category"]
                         and by_id[x["id"]]["split"] == split}
                if preds:
                    lat = sorted(x["latency"] for x in res if x["variant"] == variant and by_id[x["id"]]["split"] == split)
                    report(sp, preds, f"run{k + 1} {variant} p50 {lat[len(lat) // 2]:.1f}s")
        for k, res in enumerate(runs):
            b = {x["id"]: x["category"] for x in res if x["variant"] == "base"}
            s = {x["id"]: x["category"] for x in res if x["variant"] == "shots"}
            ch = [i for i in b if i in s and by_id[i]["split"] == split and b[i] != s[i]]
            fixed = sum(s[i] == expected(by_id[i]["cat"]) for i in ch)
            broke = sum(b[i] == expected(by_id[i]["cat"]) for i in ch)
            if ch:
                print(f"  run{k + 1} changed {len(ch)}: fixed {fixed}, broke {broke}, "
                      f"by gold {dict(Counter(by_id[i]['cat'] for i in ch))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--variants", default="base,shots")
    ap.add_argument("--split", default="fewshot", choices=("fewshot", "dev", "all"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--eval", nargs="+")
    a = ap.parse_args()
    if a.collect:
        asyncio.run(collect(a.collect, a.variants.split(","), a.split, a.concurrency))
    if a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
