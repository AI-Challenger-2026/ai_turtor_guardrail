"""공개 오류 사례의 의도 속성만 비교하며 모델 요약·본문은 저장하지 않는다."""

import asyncio
import json
import time
from pathlib import Path

from tutor_core.llm import ApiClients, ProviderFailure, RequestBudget
from tutor_core.policy import InputPolicy, IntentSignals, analyze_intent
from tutor_core.schemas import Message


async def diagnose_intent(settings, root: Path, workers: int) -> dict:
    selected_ids = set()
    for name in ("policy-run4.json", "policy-run5.json"):
        report = json.loads((root / "output" / "tests" / name).read_text(encoding="utf-8"))
        selected_ids.update(row["id"] for row in report.get("policy", {}).get("public_errors", []))
    public = [json.loads(line) for line in (root / "public" / "guard.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in public if row["id"] in selected_ids]
    if not selected:
        raise ValueError("no_public_errors_to_diagnose")
    clients = ApiClients(settings)
    semaphore = asyncio.Semaphore(min(workers, 4))
    try:
        await clients.initialize()

        async def one(row, effort):
            messages = [Message.model_validate(message) for message in row["messages"]]
            async with semaphore:
                start = time.monotonic()
                try:
                    signals = await clients.complete_json(
                        InputPolicy.intent_messages(messages), IntentSignals, RequestBudget.start(24),
                        seconds=24, max_tokens=2000, reasoning_effort=effort,
                    )
                    analysis = analyze_intent(signals)
                    return {"id": row["id"], "gold_category": row["labels"]["category"],
                            "expected_intent": "normal" if row["labels"]["category"] == "out_of_material" else row["labels"]["category"],
                            "intent_category": analysis.category, "reasoning_effort": effort,
                            "signals": signals.model_dump(exclude={"task_summary"}),
                            "elapsed_seconds": round(time.monotonic() - start, 3)}
                except ProviderFailure as exc:
                    return {"id": row["id"], "gold_category": row["labels"]["category"],
                            "reasoning_effort": effort, "error": exc.reason, "status_code": exc.status_code}

        rows = await asyncio.gather(*(one(row, effort) for row in selected for effort in ("low", "medium")))
        return {"scope": "selected_public_errors_intent_only", "full_public_quality_measured": False,
                "selected_public_count": len(selected), "comparison_count": len(rows),
                "task_summary_saved": False, "items": rows, "provider_counters": dict(clients.counters)}
    finally:
        await clients.close()
