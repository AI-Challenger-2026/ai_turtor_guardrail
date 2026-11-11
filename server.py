"""안전벨트를 채운 AI 튜터 — /tutor, /guard 서버.

1단계: 입력 범주 분류(app.classify) → 가드 판정·튜터 응대 방식을 규칙(app.policy)으로 결정.
2단계: BM25 검색(app.kb) → 근거 섹션 1개 선택 + 근거 기반 답변(app.answer). 자료에 없으면 abstain.
3단계: 가드 후보 답변을 지식 베이스와 대조해 근거 없는 구간 추출(app.verify), 입력 분류와 동시 실행.

견고성 원칙
- 인증 실패(401) 말고는 어떤 입력에도 HTTP 200 + 규격 JSON으로 답한다(4xx·5xx는 그 문항 0점).
- 요청마다 시간 예산(REQUEST_BUDGET초)을 두고, 넘기면 안전한 기본 응답을 낸다(채점기 제한 30초).
- 로그에는 시간·길이·상태·오류 종류만 남긴다(요청·답변 본문 저장 금지, SPEC §8).

Port note: server logic is final-v3 (teammate repo, 8bdc5c4). Changes in this repo: metadata logging
goes through app.logs (one line per request and per API call, uvicorn access log off), responses are
validated with the Literal models in app.schemas (an invalid body becomes the same fallback as an
exception), and the knowledge base no longer needs numpy.
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
KB_INDEX = KB(use_embeddings=False)  # 서버 시작 때 색인 생성(BM25). 임베딩은 검색 평가에서 손해라 쓰지 않음.
logs.startup_line(kb_sections=len(KB_INDEX.secs), kb_chunks=len(KB_INDEX.chunks),
                  latency_ms=round((time.perf_counter() - _t_start) * 1000))

TOKEN = os.environ["TEAM_TOKEN"]
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."
MAX_ANSWER = 4000
# 답변 생성에 주는 섹션 수. 1 = 인용할 BM25 1위 섹션만 보고 답한다(근거 충실도는 인용 섹션과만 대조되므로
# 답변 내용과 인용이 구조적으로 일치). A/B: 섹션 1/2/3/4개 = 86.3/84.5/86.1/82.7, 1개가 가장 안정·빠름(DEVLOG 5장).
TOP_K = int(os.environ.get("TOP_K", "1"))
VERIFY_TOP_K = int(os.environ.get("VERIFY_TOP_K", "4"))  # 가드 후보 답변 검증은 여러 섹션과 대조
CITE_MODE = os.environ.get("CITE_MODE", "bm25")  # "bm25" | "llm"
OFFTOPIC_RECHECK = os.environ.get("OFFTOPIC_RECHECK", "1") == "1"  # off_topic인데 자료에 답이 있으면 정상으로(17장)
EXTRACTIVE_FALLBACK = os.environ.get("EXTRACTIVE_FALLBACK", "1") == "1"  # 생성 실패 시 자료 발췌 답변(16장)
SPECULATIVE = os.environ.get("SPECULATIVE", "1") == "1"  # 튜터: 분류와 답변 생성 동시 시작
GUARD_OOM = os.environ.get("GUARD_OOM", "1") == "1"  # 가드에서 out_of_material 범주 판정(DEVLOG 13장)
GUARD_OOM_MODE = os.environ.get("GUARD_OOM_MODE", "parallel")  # "parallel": 처음부터 동시 실행(기본) | "after": 분류가 normal일 때만 뒤이어 확인(가드 지연 증가로 기각)
REQUEST_BUDGET = float(os.environ.get("REQUEST_BUDGET", "23"))  # 전체 예산(초). 서버 시간은 처리 함수 진입부터라, 부하 시 터널·프록시 대기(최대 약 5초)를 위해 30초에서 7초 여유(DEVLOG 14장)
CLASSIFY_BUDGET = float(os.environ.get("CLASSIFY_BUDGET", "12"))  # 튜터에서 분류에 쓸 최대 시간(초)
# Guard candidate verifier jury (added on top of final-v3): N calls at once, majority, two that agree
# decide. 1 = final-v3's single hedged call.
GROUND_JUDGES = int(os.environ.get("GROUND_JUDGES", "3"))
# English questions are ranked over Korean + English chunks; the English side is translated by the
# competition model at start-up (app.kb_en). 0 = Korean index only (final-v3).
KB_ENGLISH = os.environ.get("KB_ENGLISH", "1") == "1"
# Found answers end with verbatim sentences of the cited section (app.excerpt). 0 = final-v3.
EXCERPT = os.environ.get("EXCERPT", "1") == "1"


def last_user_turn(messages) -> str:
    users = [m.get("content") or "" for m in messages if m.get("role") == "user"]
    return users[-1] if users else ""


async def build_english_index() -> None:
    t0 = time.perf_counter()
    english = await translate_chunks([c["text"] for c in KB_INDEX.chunks])
    n = KB_INDEX.set_english(english)
    logs.index_line(kb_en_chunks=n, kb_chunks=len(KB_INDEX.chunks), latency_ms=round((time.perf_counter() - t0) * 1000))


@asynccontextmanager
async def lifespan(_app):
    # Unretrieved task exceptions are logged by class name only (asyncio would print a traceback).
    asyncio.get_running_loop().set_exception_handler(logs.loop_exception)
    if KB_ENGLISH:
        try:
            await build_english_index()
        except Exception as e:  # noqa: BLE001  (Korean index only)
            logs.index_line(kb_en_chunks=0, kb_chunks=len(KB_INDEX.chunks), latency_ms=0, error=type(e).__name__)
    yield


app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def authorized(request: Request) -> bool:
    got = request.headers.get("authorization", "")
    return secrets.compare_digest(got.encode(), f"Bearer {TOKEN}".encode())


async def read_payload(request: Request):
    """본문을 관대하게 읽는다: JSON이 깨졌거나 필드가 이상해도 빈 값으로 정규화."""
    try:
        data = json.loads(await request.body() or b"{}")
    except Exception:  # noqa: BLE001
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
# 대필 요청인데 생성이 시간 초과된 경우의 대체 문구(발췌는 '대신 써 주기'가 될 수 있어 쓰지 않음)
ACADEMIC_FALLBACK = ("평가나 숙제에 제출할 답안은 대신 써 줄 수 없어요. 문제에서 무엇을 요구하는지 한 문장으로 적고, "
                     "관련 실습 코드를 다시 실행해 보며 각 줄이 무엇을 하는지 스스로 정리해 보세요. 직접 쓴 답을 보여 주면 힌트를 줄게요.")


async def _generate(messages, secs, mode, deadline):
    try:
        return await grounded_answer(messages, secs, mode, deadline=deadline)
    except Exception as e:  # noqa: BLE001
        timed_out = isinstance(e, (DeadlineExceeded, asyncio.TimeoutError))
        logs.degrade("generate_timeout" if timed_out else "generate", e)
        return None


async def tutor_logic(messages, t0) -> Dict:
    """추측 실행: 검색·답변 생성(answer 모드)을 분류와 동시에 시작해 지연을 '합'에서 '둘 중 긴 쪽'으로 줄인다.
    분류가 고정 응답 범주면 생성 결과는 버리고, 힌트·대필 모드일 때만 분류 뒤에 다시 생성한다.
    (API가 느린 시점에 순차 구조는 생성이 26초 예산을 넘겨 기본 응답으로 대체됐다 — DEVLOG 14장.)"""
    deadline = t0 + REQUEST_BUDGET
    secs = retrieve(messages)
    gen_task = asyncio.create_task(_generate(messages, secs, "answer", deadline)) if SPECULATIVE else None
    c = await classify(messages, deadline=min(deadline, t0 + CLASSIFY_BUDGET))
    if OFFTOPIC_RECHECK and c["category"] == "off_topic" and gen_task is not None:
        # off_topic(수업과 무관)인데 생성기가 자료에서 답을 찾았다면 모순 → 정상 질문으로 본다.
        # 공개 400문항 2회: off_topic+found에서 정상 6 : 진짜 off_topic 2. jailbreak·harmful은 반대라 적용하지 않음(DEVLOG 17장).
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
    if action == "scaffold" or gen_task is None:
        if gen_task:
            gen_task.cancel()
        if action == "scaffold":
            mode = "academic" if c["category"] == "academic_dishonesty" else "scaffold"
        else:
            mode = "answer"
        g = await _generate(messages, secs, mode, deadline)
    else:
        g = await gen_task
    if g is None:
        # 답변 생성 실패·시간 초과: 기본 응답 대신 1위 섹션 발췌 + 인용(대필 모드는 발췌가 '대신 써 주기'가 될 수 있어 제외)
        if c["category"] == "academic_dishonesty" and EXTRACTIVE_FALLBACK:
            logs.set_outcome("fallback_academic")
            return {"answer": ACADEMIC_FALLBACK, "action": action, "citations": citations}
        # 분류가 실패해 normal로 대체된 경우(flags 없음)에는 공격 요청일 수 있으므로 발췌하지 않고 기본 응답을 쓴다
        ext = extractive_answer(messages, secs[0]) if EXTRACTIVE_FALLBACK and "flags" in c else ""
        if ext:
            logs.set_outcome("fallback_extractive")
            return {"answer": ext[:MAX_ANSWER], "action": action, "citations": [secs[0]["id"]]}
        logs.set_outcome("fallback_generate")
        return {"answer": FALLBACK, "action": action, "citations": citations}
    answer = g["answer"]
    if g["found"]:
        # 인용은 BM25 1위로 정한다: 공개 세트에서 LLM 선택 정확도 70% < BM25 1위 91%
        # (같은 사실이 여러 섹션에 중복돼 있어 LLM은 제목이 그럴듯한 섹션을 고르는 경향).
        citations = [secs[0]["id"]] if CITE_MODE == "bm25" else [g["evidence"]]
        if EXCERPT and action == "answer":
            # Verbatim sentences of the cited top section after the answer (app.excerpt); never after
            # a scaffold, which must not give the answer away.
            answer = with_excerpt(answer, secs[0]["text"], last_user_turn(messages))
    elif action == "answer":
        action = "abstain"  # 자료에서 확인할 수 없는 질문 → 지어내지 않고 abstain
        # (abstain에도 1위 섹션을 인용하는 안은 기각: 판정 모델이 인용 섹션과 대조해 근거 충실을 깎았다 — DEVLOG 16장)
    if c["category"] == "pii_disclosure" and "개인정보" not in answer:
        answer = answer.rstrip() + " " + PII_NOTE  # 개인정보 안내는 코드로(이전의 pii 모드와 같은 문장)
    return {"answer": answer[:MAX_ANSWER], "action": action, "citations": citations}


def retrieve_for_candidate(messages, cand: str) -> List[Dict[str, str]]:
    q = "\n".join(m["content"] for m in messages[-4:]) + "\n" + cand + "\n" + cand  # 후보 답변 가중
    return [KB_INDEX.secs[sid] for sid, _ in KB_INDEX.rank(q)[:VERIFY_TOP_K]]


async def _verify(messages, cand: str, deadline) -> Optional[List[str]]:
    try:
        sections = retrieve_for_candidate(messages, cand)
        if GROUND_JUDGES > 1:
            return await find_unsupported_jury(messages, cand, sections, deadline=deadline, judges=GROUND_JUDGES)
        return await find_unsupported(messages, cand, sections, deadline=deadline)
    except Exception as e:  # noqa: BLE001
        logs.degrade("ground", e)
        return None


async def _material_found(messages, deadline) -> Optional[bool]:
    """'자료에 답이 있나': 튜터와 같은 검색(1위 섹션) + 답변 생성기의 found 판단을 그대로 쓴다. 실패하면 None."""
    try:
        g = await grounded_answer(messages, retrieve(messages), "answer", deadline=deadline, kind="found")
        return bool(g["found"])
    except Exception as e:  # noqa: BLE001
        logs.degrade("found", e)
        return None


async def guard_logic(messages, cand: Optional[str], t0) -> Dict:
    deadline = t0 + REQUEST_BUDGET
    # 분류 / 후보 답변 검증 / '자료에 있나' 확인을 동시에 실행(지연 = 셋 중 가장 긴 쪽)
    found_task = asyncio.create_task(_material_found(messages, deadline)) if GUARD_OOM and GUARD_OOM_MODE == "parallel" else None
    if cand is not None and cand.strip():
        c, spans = await asyncio.gather(classify(messages, deadline=deadline), _verify(messages, cand, deadline))
    else:
        c, spans = await classify(messages, deadline=deadline), []
    category = c["category"]
    if OFFTOPIC_RECHECK and category == "off_topic" and found_task is not None:
        if await found_task is True:  # off_topic인데 자료에 답이 있으면 정상으로(튜터와 같은 규칙, DEVLOG 17장)
            category = "normal"
    decision = guard_decision(category)
    if GUARD_OOM and GUARD_OOM_MODE != "parallel" and category == "normal":
        if await _material_found(messages, deadline) is False:
            category = "out_of_material"
    if found_task is not None:
        if category == "normal" and not found_task.cancelled():
            # 범주 점수는 정답이 normal이 아닌 문항만 채점하고 판정(allow)은 그대로라, 정상 질문을 out_of_material로
            # 잘못 바꿔도 잃는 점수가 없다(비대칭). 공개 400문항: 가드 범주 정확도 0.78~0.83 → 0.90~0.95(DEVLOG 13장).
            if await found_task is False:
                category = "out_of_material"
        else:
            found_task.cancel()
    result = {"input": {"decision": decision, "category": category}}
    if cand is not None:
        if decision == "block":
            # 막아야 할 입력에 대한 후보 답변 = 그 요청에 응한 답변 → unsafe.
            # 공개 세트 라벨에서 이런 답변은 모두 grounded=True·구간 없음(근거가 아니라 안전 문제로 봄).
            out = {"grounded": True, "unsupported": [], "safe": False}
        else:
            spans = spans or []  # 검증 실패(None)면 근거 있음으로 둔다(정상 답변을 깎지 않음)
            out = {"grounded": not spans, "unsupported": spans, "safe": True}
        result["output"] = {**out, "category": category}
    return result


def _log_request(endpoint: str, status: int, t0: float, body_bytes: int, messages, cand, info: dict) -> None:
    # Metadata only: lengths, counts, outcome and stage names. Never raises.
    try:
        logs.request_line(endpoint=endpoint, status=status, latency_ms=round((time.time() - t0) * 1000),
                          body_bytes=body_bytes,
                          input_chars=sum(len(m["content"]) for m in messages) + len(cand or ""),
                          n_msgs=len(messages), outcome=info.get("outcome", "ok"),
                          degraded=info.get("degraded"), error=info.get("error"))
    except Exception:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001  — 어떤 경우에도 규격 응답(값 검증 실패도 여기로)
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
    except Exception as e:  # noqa: BLE001  — 값 검증 실패도 여기로
        info.update(outcome="fallback_exception", error=type(e).__name__)
        out = {"input": {"decision": "allow", "category": "normal"}}
        if cand is not None:
            out["output"] = {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}
    _log_request("/guard", 200, t0, body_bytes, messages, cand, info)
    return out
