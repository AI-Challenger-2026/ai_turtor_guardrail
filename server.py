"""Tutor and guard endpoints.

Every authorized request gets HTTP 200 with a schema-valid JSON body; errors and timeouts fall
back to a safe default. Logs carry metadata only (see app.logs).
"""
import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import logs
from app.answer import grounded_answer
from app.excerpt import EXCERPT_HEAD, with_excerpt
from app.extract import extractive_answer
from app.classify import classify
from app.kb import KB, build_query
from app.kb_en import english_turn, translate_chunks
from app.llm import DeadlineExceeded
from app.policy import guard_decision, template_for, tutor_action
from app.schemas import GuardResponse, TutorResponse
from app.verify import find_unsupported, find_unsupported_jury

logs.setup()
_t_start = time.perf_counter()
KB_INDEX = KB(use_embeddings=False)
logs.startup_line(kb_sections=len(KB_INDEX.secs), kb_chunks=len(KB_INDEX.chunks),
                  latency_ms=round((time.perf_counter() - _t_start) * 1000))

TOKEN = os.environ["TEAM_TOKEN"]
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."
MAX_ANSWER = 4000
TOP_K = int(os.environ.get("TOP_K", "1"))
VERIFY_TOP_K = int(os.environ.get("VERIFY_TOP_K", "4"))
CITE_MODE = os.environ.get("CITE_MODE", "bm25")
OFFTOPIC_RECHECK = os.environ.get("OFFTOPIC_RECHECK", "1") == "1"
EXTRACTIVE_FALLBACK = os.environ.get("EXTRACTIVE_FALLBACK", "1") == "1"
SPECULATIVE = os.environ.get("SPECULATIVE", "1") == "1"
GUARD_OOM = os.environ.get("GUARD_OOM", "1") == "1"
GUARD_OOM_MODE = os.environ.get("GUARD_OOM_MODE", "parallel")
# Seconds per request, below the 30 s limit.
REQUEST_BUDGET = float(os.environ.get("REQUEST_BUDGET", "23"))
CLASSIFY_BUDGET = float(os.environ.get("CLASSIFY_BUDGET", "12"))
GROUND_JUDGES = int(os.environ.get("GROUND_JUDGES", "3"))
KB_ENGLISH = os.environ.get("KB_ENGLISH", "1") == "1"
EXCERPT = os.environ.get("EXCERPT", "1") == "1"
HINT_MATERIAL = os.environ.get("HINT_MATERIAL", "1") == "1"


def last_user_turn(messages) -> str:
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    return users[-1] if users else ""


# English questions are searched over each chunk plus its English translation,
# made with the same model at start-up and kept in memory.
async def build_english_index() -> None:
    t0 = time.perf_counter()
    english = await translate_chunks([c["text"] for c in KB_INDEX.chunks])
    n = KB_INDEX.set_english(english)
    logs.index_line(kb_en_chunks=n, kb_chunks=len(KB_INDEX.chunks), latency_ms=round((time.perf_counter() - t0) * 1000))


@asynccontextmanager
async def lifespan(_app):
    asyncio.get_running_loop().set_exception_handler(logs.loop_exception)
    if KB_ENGLISH:
        try:
            await build_english_index()
        except Exception as e:
            logs.index_line(kb_en_chunks=0, kb_chunks=len(KB_INDEX.chunks), latency_ms=0, error=type(e).__name__)
    yield


app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def authorized(request: Request) -> bool:
    got = request.headers.get("authorization", "")
    return secrets.compare_digest(got.encode(), f"Bearer {TOKEN}".encode())


# A malformed body is read as empty values so the request still gets an answer.
async def read_payload(request: Request):
    try:
        data = json.loads(await request.body() or b"{}")
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    msgs = []
    for m in data.get("messages") or []:
        if isinstance(m, dict):
            role = "assistant" if m.get("role") == "assistant" else "user"
            content = m.get("content")
            msgs.append({"role": role, "content": content if isinstance(content, str) else ("" if content is None else str(content))})
    cand = data.get("candidate_answer")
    if cand is not None and not isinstance(cand, str):
        cand = str(cand)
    return msgs, cand


def retrieve(messages) -> List[Dict[str, str]]:
    ranked = KB_INDEX.rank(build_query(messages, "all"), bilingual=english_turn(messages))
    return [KB_INDEX.secs[sid] for sid, _ in ranked[:TOP_K]]


PII_NOTE = "이름이나 학교 같은 개인정보는 채팅에 적지 않아도 괜찮아요."
ACADEMIC_FALLBACK = ("평가나 숙제에 제출할 답안은 대신 써 줄 수 없어요. 문제에서 무엇을 요구하는지 한 문장으로 적고, "
                     "관련 실습 코드를 다시 실행해 보며 각 줄이 무엇을 하는지 스스로 정리해 보세요. 직접 쓴 답을 보여 주면 힌트를 줄게요.")


async def _generate(messages, secs, mode, deadline):
    try:
        return await grounded_answer(messages, secs, mode, deadline=deadline)
    except Exception as e:
        timed_out = isinstance(e, (DeadlineExceeded, asyncio.TimeoutError))
        logs.degrade("generate_timeout" if timed_out else "generate", e)
        return None


async def tutor_logic(messages, t0) -> Dict:
    deadline = t0 + REQUEST_BUDGET
    secs = retrieve(messages)
    # Generate while classifying; dropped if the category gets a fixed reply.
    gen_task = asyncio.create_task(_generate(messages, secs, "answer", deadline)) if SPECULATIVE else None
    c = await classify(messages, deadline=min(deadline, t0 + CLASSIFY_BUDGET))
    # Not off-topic if the material answers the question.
    if OFFTOPIC_RECHECK and c["category"] == "off_topic" and gen_task is not None:
        g0 = await gen_task
        if g0 is not None and g0["found"]:
            c = {**c, "category": "normal"}
    action = tutor_action(c["category"], c["harmful_type"])
    citations: List[str] = []
    fixed = template_for(c["category"], c["harmful_type"])
    if fixed is not None:
        if gen_task:
            gen_task.cancel()
        return {"answer": fixed[:MAX_ANSWER], "action": action, "citations": citations}
    mode = "answer"
    if action == "scaffold":
        mode = "academic" if c["category"] == "academic_dishonesty" else "scaffold"
    if action == "scaffold" or gen_task is None:
        if gen_task:
            gen_task.cancel()
        g = await _generate(messages, secs, mode, deadline)
    else:
        g = await gen_task
    # Generation failed: answer with the top section's own sentences.
    if g is None:
        if c["category"] == "academic_dishonesty" and EXTRACTIVE_FALLBACK:
            logs.set_outcome("fallback_academic")
            return {"answer": ACADEMIC_FALLBACK, "action": action, "citations": citations}
        ext = extractive_answer(messages, secs[0]) if EXTRACTIVE_FALLBACK and "flags" in c else ""
        if ext:
            logs.set_outcome("fallback_extractive")
            return {"answer": ext[:MAX_ANSWER], "action": action, "citations": [secs[0]["id"]]}
        logs.set_outcome("fallback_generate")
        return {"answer": FALLBACK, "action": action, "citations": citations}
    answer = g["answer"]
    if g["found"]:
        citations = [secs[0]["id"]] if CITE_MODE == "bm25" else [g["evidence"]]
        if EXCERPT and (mode == "answer" or (mode == "scaffold" and HINT_MATERIAL)):
            answer = with_excerpt(answer, secs[0]["text"], last_user_turn(messages))
    elif action == "answer":
        action = "abstain"
    elif mode == "scaffold" and HINT_MATERIAL:
        citations = [secs[0]["id"]]
    if c["category"] == "pii_disclosure" and "개인정보" not in answer:
        answer = answer.rstrip() + " " + PII_NOTE
    return {"answer": answer[:MAX_ANSWER], "action": action, "citations": citations}


def retrieve_for_candidate(messages, cand: str) -> List[Dict[str, str]]:
    q = "\n".join(m["content"] for m in messages[-4:]) + "\n" + cand + "\n" + cand
    return [KB_INDEX.secs[sid] for sid, _ in KB_INDEX.rank(q)[:VERIFY_TOP_K]]


async def _verify(messages, cand: str, deadline) -> Optional[List[str]]:
    try:
        sections = retrieve_for_candidate(messages, cand)
        if GROUND_JUDGES > 1:
            return await find_unsupported_jury(messages, cand, sections, deadline=deadline, judges=GROUND_JUDGES)
        return await find_unsupported(messages, cand, sections, deadline=deadline)
    except Exception as e:
        logs.degrade("ground", e)
        return None


async def _material_found(messages, deadline) -> Optional[bool]:
    try:
        g = await grounded_answer(messages, retrieve(messages), "answer", deadline=deadline, kind="found")
        return bool(g["found"])
    except Exception as e:
        logs.degrade("found", e)
        return None


async def guard_logic(messages, cand: Optional[str], t0) -> Dict:
    deadline = t0 + REQUEST_BUDGET
    # Classification, candidate check and material check run concurrently.
    found_task = asyncio.create_task(_material_found(messages, deadline)) if GUARD_OOM and GUARD_OOM_MODE == "parallel" else None
    if cand is not None and cand.strip():
        c, spans = await asyncio.gather(classify(messages, deadline=deadline), _verify(messages, cand, deadline))
    else:
        c, spans = await classify(messages, deadline=deadline), []
    category = c["category"]
    # Not off-topic if the material answers the question.
    if OFFTOPIC_RECHECK and category == "off_topic" and found_task is not None:
        if await found_task is True:
            category = "normal"
    decision = guard_decision(category)
    if GUARD_OOM and GUARD_OOM_MODE != "parallel" and category == "normal":
        if await _material_found(messages, deadline) is False:
            category = "out_of_material"
    # A normal question the material does not cover is out_of_material.
    if found_task is not None:
        if category == "normal" and not found_task.cancelled():
            if await found_task is False:
                category = "out_of_material"
        else:
            found_task.cancel()
    result = {"input": {"decision": decision, "category": category}}
    if cand is not None:
        if decision == "block":
            # An answer to a request that must be blocked is unsafe.
            out = {"grounded": True, "unsupported": [], "safe": False}
        else:
            # A failed check leaves the answer grounded.
            spans = spans or []
            out = {"grounded": not spans, "unsupported": spans, "safe": True}
        result["output"] = {**out, "category": category}
    return result


# Metadata only: sizes, counts, status and stage names.
def _log_request(endpoint: str, status: int, t0: float, body_bytes: int, messages, cand, info: dict) -> None:
    try:
        logs.request_line(endpoint=endpoint, status=status, latency_ms=round((time.time() - t0) * 1000),
                          body_bytes=body_bytes,
                          input_chars=sum(len(m["content"]) for m in messages) + len(cand or ""),
                          n_msgs=len(messages), outcome=info.get("outcome", "ok"),
                          degraded=info.get("degraded"), error=info.get("error"))
    except Exception:
        pass


@app.post("/tutor")
async def tutor(request: Request):
    t0 = time.time()
    info = logs.start_request()
    if not authorized(request):
        _log_request("/tutor", 401, t0, 0, [], None, info)
        return JSONResponse({"detail": "invalid token"}, status_code=401)
    messages, body_bytes = [], 0
    try:
        messages, _ = await read_payload(request)
        body_bytes = len(await request.body())
        out = TutorResponse(**await tutor_logic(messages, t0)).model_dump()
    except Exception as e:
        info.update(outcome="fallback_exception", error=type(e).__name__)
        out = {"answer": FALLBACK, "action": "answer", "citations": []}
    _log_request("/tutor", 200, t0, body_bytes, messages, None, info)
    return out


@app.post("/guard")
async def guard(request: Request):
    t0 = time.time()
    info = logs.start_request()
    if not authorized(request):
        _log_request("/guard", 401, t0, 0, [], None, info)
        return JSONResponse({"detail": "invalid token"}, status_code=401)
    messages, cand, body_bytes = [], None, 0
    try:
        messages, cand = await read_payload(request)
        body_bytes = len(await request.body())
        out = GuardResponse(**await guard_logic(messages, cand, t0)).wire()
    except Exception as e:
        info.update(outcome="fallback_exception", error=type(e).__name__)
        out = {"input": {"decision": "allow", "category": "normal"}}
        if cand is not None:
            out["output"] = {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}
    _log_request("/guard", 200, t0, body_bytes, messages, cand, info)
    return out
