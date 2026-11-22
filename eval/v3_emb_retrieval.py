"""Retrieval: BM25 vs embeddings (not used by the server).

    python3 eval/v3_emb_retrieval.py --collect OUT.json
    python3 eval/v3_emb_retrieval.py --eval OUT.json
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TEAM_TOKEN", "eval-only")

from app.kb import KB, build_query

EMB_MODEL = os.environ.get("EMBEDDING_MODEL", "openai/text-embedding-3-small")


def items():
    split = json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))
    where = {i: "fewshot" for i in split["fewshot"]} | {i: "dev" for i in split["dev"]}
    out = []
    for line in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["labels"].get("evidence"):
            last = [m["content"] for m in r["messages"] if m["role"] == "user"][-1]
            out.append({"id": r["id"], "split": where[r["id"]], "messages": r["messages"], "gold": r["labels"]["evidence"],
                        "english": not re.search(r"[가-힣]", last), "last": last})
    return out


def embed(texts):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["EMBEDDING_API_KEY"], base_url=os.environ["EMBEDDING_BASE_URL"], timeout=60, max_retries=2)
    out, calls = [], 0
    for i in range(0, len(texts), 64):
        r = c.embeddings.create(model=EMB_MODEL, input=texts[i:i + 64])
        calls += 1
        out.extend(d.embedding for d in r.data)
    return [_norm(v) for v in out], calls


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def collect(path):
    kb = KB()
    rows = items()
    chunk_vecs, c1 = embed([c["text"] for c in kb.chunks])
    q_all, c2 = embed([build_query(r["messages"], "all") for r in rows])
    q_last, c3 = embed([r["last"] for r in rows])
    data = {"chunks": chunk_vecs, "items": [{**r, "q_all": a, "q_last": b} for r, a, b in zip(rows, q_all, q_last)]}
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    print("saved", path, "items", len(rows), "embedding calls", c1 + c2 + c3)


def emb_rank(kb, chunk_vecs, q):
    best = defaultdict(lambda: -1e9)
    for sid, v in zip(kb.chunk_sids, chunk_vecs):
        s = sum(a * b for a, b in zip(v, q))
        if s > best[sid]:
            best[sid] = s
    return [sid for sid, _ in sorted(best.items(), key=lambda x: -x[1])]


def rrf(*lists, k=60):
    f = defaultdict(float)
    for lst in lists:
        for r, sid in enumerate(lst):
            f[sid] += 1 / (k + r + 1)
    return [sid for sid, _ in sorted(f.items(), key=lambda x: -x[1])]


def evaluate(path):
    kb = KB()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cv = data["chunks"]
    res = defaultdict(lambda: defaultdict(int))
    misses = []
    for r in data["items"]:
        bm = [sid for sid, _ in kb.rank(build_query(r["messages"], "all"))]
        ea = emb_rank(kb, cv, r["q_all"])
        el = emb_rank(kb, cv, r["q_last"])
        ranks = {"bm25": bm, "emb_all": ea, "emb_last": el, "rrf_all": rrf(bm, ea), "rrf_last": rrf(bm, el)}
        ranks["route_all"] = ea if r["english"] else bm
        ranks["route_last"] = el if r["english"] else bm
        ranks["route_rrf"] = rrf(bm, ea) if r["english"] else bm
        lang = "en" if r["english"] else "ko"
        for name, lst in ranks.items():
            hit = lst[0] in r["gold"]
            for key in (f"{lang}/{r['split']}", f"{lang}/all", "all/all"):
                res[name][key] += hit
        for key in (f"{lang}/{r['split']}", f"{lang}/all", "all/all"):
            res["_n"][key] += 1
        if r["english"]:
            misses.append((r["id"], r["split"], r["gold"], bm[0], ea[0], el[0]))
    keys = ["ko/fewshot", "ko/dev", "ko/all", "en/fewshot", "en/dev", "en/all", "all/all"]
    print("top-1 correct".ljust(12) + "".join(k.rjust(12) for k in keys))
    print("n".ljust(12) + "".join(str(res["_n"][k]).rjust(12) for k in keys))
    for name in [n for n in res if n != "_n"]:
        print(name.ljust(12) + "".join(str(res[name][k]).rjust(12) for k in keys))
    print("English items: id split gold | bm25 emb_all emb_last")
    for m in misses:
        print("  ", *m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect")
    ap.add_argument("--eval")
    a = ap.parse_args()
    if a.collect:
        collect(a.collect)
    if a.eval:
        evaluate(a.eval)


if __name__ == "__main__":
    main()
