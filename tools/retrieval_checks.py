"""공개 근거 라벨로 검색을 평가하고 본문 없는 진단 결과를 반환한다."""

import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

from tutor_core.embedding_cache import EmbeddingCache
from tutor_core.knowledge import load_knowledge
from tutor_core.llm import ApiClients, RequestBudget
from tutor_core.retrieval import DEFAULT_CACHE_PATH, DEFAULT_KB_ROOT, Retriever, prepare_retriever
from tutor_core.schemas import Message


TOP_K = (1, 3, 4, 6, 8, 12)


def _read_public(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        evidence = set(item["labels"].get("evidence") or [])
        if evidence:
            rows.append((item["id"], [Message.model_validate(message) for message in item["messages"]], evidence))
    return rows


async def evaluate_retrieval(settings, public_path: Path, workers: int, lexical_only: bool = False) -> dict:
    clients = None
    started = time.monotonic()
    try:
        if lexical_only:
            kb = await asyncio.to_thread(load_knowledge, DEFAULT_KB_ROOT)
            retriever = await asyncio.to_thread(Retriever, kb)
        else:
            clients = ApiClients(settings)
            # 검색 검사는 임베딩만 사용한다. 채팅 API 확인은 선택한 서버 기동에서 수행한다.
            retriever = await prepare_retriever(clients)
        prepared_seconds = time.monotonic() - started
        rows = _read_public(public_path)
        if not rows or any(not evidence.issubset(retriever.knowledge.by_id) for _, _, evidence in rows):
            raise ValueError("invalid_public_evidence")
        cache_verified = False
        if clients is not None:
            loaded = await asyncio.to_thread(EmbeddingCache(DEFAULT_CACHE_PATH, settings.embedding_model).load, retriever.knowledge)
            cache_verified = loaded is not None
        semaphore = asyncio.Semaphore(workers)
        completed = 0
        results = []

        async def one(row):
            nonlocal completed
            item_id, messages, evidence = row
            lexical = await asyncio.to_thread(retriever.lexical, messages, top_k=12)
            result = lexical
            elapsed = 0.0
            if not lexical_only:
                async with semaphore:
                    start = time.monotonic()
                    result = await retriever.search(messages, RequestBudget.start(27), top_k=12)
                    elapsed = time.monotonic() - start
            completed += 1
            if completed % 20 == 0 or completed == len(rows):
                print(f"공개 근거 검색 {completed}/{len(rows)}개 확인", flush=True)
            # 공개 ID와 KB ID만 남기고 대화와 질의 벡터는 보관하지 않는다.
            return {
                "id": item_id, "evidence": sorted(evidence), "lexical_ids": list(lexical.section_ids),
                "hybrid_ids": list(result.section_ids) if not lexical_only else [],
                "semantic_ids": list(result.semantic_section_ids),
                "mode": result.mode, "degraded": result.degraded, "failure_reason": result.failure_reason,
                "elapsed_seconds": round(elapsed, 4),
                "latest_non_hangul": re.search(r"[\uac00-\ud7a3]", messages[-1].content) is None,
            }

        results = await asyncio.gather(*(one(row) for row in rows))
        total = sum(len(row["evidence"]) for row in results)

        def metrics(key, subset):
            denominator = sum(len(row["evidence"]) for row in subset)
            return {str(k): round(sum(len(set(row["evidence"]) & set(row[key][:k])) for row in subset) / denominator, 6)
                    if denominator else None for k in TOP_K}

        variants = ("lexical_ids",) if lexical_only else ("lexical_ids", "semantic_ids", "hybrid_ids")
        recalls = {name.removesuffix("_ids"): metrics(name, results) for name in variants}
        non_hangul = [row for row in results if row["latest_non_hangul"]]
        fallbacks = Counter(row["failure_reason"] for row in results if row["degraded"])
        key = "lexical_ids" if lexical_only else "hybrid_ids"
        misses = [{"id": row["id"], "evidence": row["evidence"], "retrieved": row[key][:8],
                   "expanded": row[key], "latest_non_hangul": row["latest_non_hangul"]}
                  for row in results if not set(row["evidence"]).issubset(row[key][:8])]
        quality = recalls["lexical" if lexical_only else "hybrid"]["8"] >= 0.98
        return {
            "knowledge": retriever.summary(), "preparation_seconds": round(prepared_seconds, 3),
            "cache_roundtrip_verified": cache_verified,
            "evaluation": {
                "dataset": "organizer_public_tutor_evidence", "items_count": len(rows), "evidence_count": total,
                "workers": workers, "recall_at_k": recalls,
                "latest_non_hangul_count": len(non_hangul),
                "latest_non_hangul_recall_at_k": {name.removesuffix("_ids"): metrics(name, non_hangul) for name in variants},
                "query_fallbacks": dict(fallbacks), "quality_target_recall_at_8": 0.98,
                "quality_target_met": quality, "misses_at_8": misses, "items": results,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            "provider_counters": dict(clients.counters) if clients is not None else {},
            "operational_pass": lexical_only or (cache_verified and not fallbacks),
        }
    finally:
        if clients is not None:
            await clients.close()
