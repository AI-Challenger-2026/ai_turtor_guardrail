"""Retrieval recall on labeled tutor items (items whose label has an evidence section).

Reports BM25 alone, embeddings alone and the server's candidate set (`retrieval.combine`):
recall@1/3/5 of the gold section and MRR, then the items whose gold section is not in the
server's top-k. Embedding calls: the KB chunks once and one query per item, batched; no chat
calls. `--vec-cache` keeps vectors in a local eval-only file so reruns make no calls.
Prints ids only, never text.

Run from the repo root with the .env loaded:
  python3 eval/retrieval_report.py --split fewshot
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config, retrieval  # noqa: E402
from app.schemas import Message  # noqa: E402


def load_items(split: str):
    items = [json.loads(x) for x in (ROOT / "public/tutor.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = set(json.loads((ROOT / "eval/split.json").read_text(encoding="utf-8"))[split])
    return [it for it in items if it["id"] in ids and it["labels"].get("evidence")]


def metrics(ranked, gold):
    pos = [r.index(g) if g in r else 99 for r, g in zip(ranked, gold)]
    n = len(pos)
    out = {f"R@{k}": sum(p < k for p in pos) / n for k in (1, 3, 5)}
    out["MRR"] = sum(1 / (p + 1) for p in pos) / n
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="fewshot", choices=["fewshot", "dev"])
    ap.add_argument("--vec-cache", default="", help="eval-only JSON file to reuse embeddings across runs")
    a = ap.parse_args()
    if not config.EMBED_ENABLED:
        sys.exit("EMBEDDING_* settings are missing (load .env first)")

    items = load_items(a.split)
    gold = [it["labels"]["evidence"][0] for it in items]
    convs = [[Message(**m) for m in it["messages"]] for it in items]
    print(f"split {a.split}: {len(items)} items with an evidence section")

    cache_path = Path(a.vec_cache) if a.vec_cache else None
    cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}

    async def emb(texts, kind):
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            vecs = await retrieval.embed_texts(missing, kind=kind, timeout=60, retries=2)
            if vecs is None:
                sys.exit("embedding failed")
            cache.update(zip(missing, vecs))
        return [cache[t] for t in texts]

    chunk_vecs = await emb([c for _, c in retrieval.CHUNKS], "embed_kb")
    qvecs = await emb([retrieval.embed_query_text(m) for m in convs], "embed_query")
    if cache_path:
        cache_path.write_text(json.dumps(cache))

    k = config.RETRIEVE_K
    bm = [retrieval.rank_bm25(retrieval.query_text(m)) for m in convs]
    vec = [retrieval.rank_vector(q, chunk_vecs) for q in qvecs]
    server = [retrieval.combine(b, v, k) for b, v in zip(bm, vec)]
    variants = {
        "bm25 only": [[s for s, _ in r] for r in bm],
        "embedding only": [[s for s, _ in r] for r in vec],
        f"server (bm25 top{retrieval.BM25_TAKE} + vector {retrieval.VECTOR_TAKE}, k={k})": [[s for s, _, _ in r] for r in server],
    }
    print(f"{'variant':44s}  R@1    R@3    R@5    MRR")
    for name, ranked in variants.items():
        m = metrics(ranked, gold)
        print(f"{name:44s}  {m['R@1']:.3f}  {m['R@3']:.3f}  {m['R@5']:.3f}  {m['MRR']:.3f}")
    print(f"gold section not in the server's top-{k}:")
    for it, r, g in zip(items, server, gold):
        if g not in [s for s, _, _ in r]:
            print(f"  {it['id']}  {it['labels']['category']:15s} gold {g}  bm25 rank "
                  f"{[s for s, _ in bm[items.index(it)]].index(g) + 1}  top {[s for s, _, _ in r][:3]}")


if __name__ == "__main__":
    asyncio.run(main())
