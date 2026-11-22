"""Retrieval for English questions over a translated knowledge base.

    python3 eval/v3_bilingual.py --translate-kb KB_EN.json
    python3 eval/v3_bilingual.py --eval KB_EN.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

from app.kb import BM25, KB, build_query

TRANSLATE_SYS = """Translate the Korean course material below into natural English for a search index.
Keep code, numbers, units, section IDs and English words exactly as they are.
For Korean names of banks, products, people, schools, places, tools and exercises, translate the name by
its meaning and add the romanized Korean in parentheses, e.g. a name meaning "blue bird bank" becomes
"Blue Bird Bank (Parangsae Eunhaeng)". Translate table cells too. Output only the translation."""

QUERY_SYS = """Translate the student's Korean message into the casual English a middle-school student would type.
Keep code, numbers and English words as they are. Output only the translation."""


async def _translate(texts, system, concurrency=16):
    from app.llm import chat
    sem = asyncio.Semaphore(concurrency)

    async def one(t):
        async with sem:
            try:
                return await chat(system, t, effort="minimal", max_tokens=2000, timeout=60, kind="translate")
            except Exception as e:
                return None

    return await asyncio.gather(*[one(t) for t in texts])


def translate_kb(path):
    kb = KB()
    out = asyncio.run(_translate([c["text"] for c in kb.chunks], TRANSLATE_SYS))
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print("saved", path, "chunks", len(out), "calls", len(out), "failures", sum(x is None for x in out))


def items():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for line in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["labels"].get("evidence"):
            last = [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
            out.append({"id": r["id"], "split": where[r["id"]], "messages": r["messages"], "gold": r["labels"]["evidence"],
                        "english": not re.search(r"[가-힣]", last)})
    return out


def translate_queries(path):
    rows = [r for r in items() if not r["english"]]
    lasts = [[m["content"] for m in r["messages"] if m["role"] == "user"][-1] for r in rows]
    out = asyncio.run(_translate(lasts, QUERY_SYS))
    Path(path).write_text(json.dumps({r["id"]: t for r, t in zip(rows, out)}, ensure_ascii=False, indent=0), encoding="utf-8")
    print("saved", path, "queries", len(out), "calls", len(out), "failures", sum(x is None for x in out))


def with_last(messages, last):
    idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
    return [*messages[:idx], {"role": "user", "content": last}, *messages[idx + 1:]]


def rank(bm, sids, q):
    best = defaultdict(lambda: -1e9)
    for sid, s in zip(sids, bm.scores(q)):
        best[sid] = max(best[sid], s)
    return [sid for sid, _ in sorted(best.items(), key=lambda x: -x[1])]


def rrf(*lists, k=60):
    f = defaultdict(float)
    for lst in lists:
        for r, sid in enumerate(lst):
            f[sid] += 1 / (k + r + 1)
    return [sid for sid, _ in sorted(f.items(), key=lambda x: -x[1])]


def evaluate(kb_path, q_path=None):
    kb = KB()
    en = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    en = [t or "" for t in en]
    sids = kb.chunk_sids
    bm_en = BM25(en)
    bm_bi = BM25([c["text"] + "\n" + t for c, t in zip(kb.chunks, en)])
    rows = items()
    sets = {"real": [(r, r["messages"]) for r in rows if r["english"]]}
    if q_path:
        q = json.loads(Path(q_path).read_text(encoding="utf-8"))
        sets["translated"] = [(r, with_last(r["messages"], q[r["id"]])) for r in rows if q.get(r["id"])]
        sets["korean (bi on Korean)"] = [(r, r["messages"]) for r in rows if not r["english"]]
    for name, sel in sets.items():
        hits = defaultdict(lambda: defaultdict(int))
        n = defaultdict(int)
        for r, msgs in sel:
            query = build_query(msgs, "all")
            ko = [sid for sid, _ in kb.rank(query)]
            e = rank(bm_en, sids, query)
            ranks = {"ko": ko, "en": e, "rrf": rrf(ko, e), "bi": rank(bm_bi, sids, query)}
            n[r["split"]] += 1
            for k, lst in ranks.items():
                hits[k][r["split"]] += lst[0] in r["gold"]
                hits[k]["top3"] += bool(set(lst[:3]) & set(r["gold"]))
            if name == "real":
                print("  ", r["id"], r["gold"], {k: v[0] for k, v in ranks.items()})
        tot = n["fewshot"] + n["dev"]
        print(f"[{name}] n fewshot {n['fewshot']} dev {n['dev']}")
        for k, h in hits.items():
            print(f"   {k:4s} top-1 fewshot {h['fewshot']:3d}  dev {h['dev']:3d}  all {h['fewshot'] + h['dev']:3d}/{tot}  top-3 {h['top3']}/{tot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translate-kb")
    ap.add_argument("--translate-queries")
    ap.add_argument("--eval")
    ap.add_argument("--queries")
    a = ap.parse_args()
    if a.translate_kb:
        translate_kb(a.translate_kb)
    if a.translate_queries:
        translate_queries(a.translate_queries)
    if a.eval:
        evaluate(a.eval, a.queries)


if __name__ == "__main__":
    main()
