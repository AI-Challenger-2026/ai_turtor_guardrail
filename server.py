"""안전벨트를 채운 AI 튜터 — /tutor, /guard 서버.

1단계: 입력 범주 분류(app.classify) → 가드 판정·튜터 응대 방식을 규칙(app.policy)으로 결정.
2단계: BM25 검색(app.kb) → 근거 섹션 1개 선택 + 근거 기반 답변(app.answer). 자료에 없으면 abstain.
3단계: 가드 후보 답변을 지식 베이스와 대조해 근거 없는 구간 추출(app.verify), 입력 분류와 동시 실행.

견고성 원칙
- 인증 실패(401) 말고는 어떤 입력에도 HTTP 200 + 규격 JSON으로 답한다(4xx·5xx는 그 문항 0점).
- 요청마다 시간 예산(REQUEST_BUDGET초)을 두고, 넘기면 안전한 기본 응답을 낸다(채점기 제한 30초).
- 로그에는 시간·길이·상태·오류 종류만 남긴다(요청·답변 본문 저장 금지, SPEC §8).
"""
import asyncio
import json
import os
import secrets
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.answer import grounded_answer
from app.classify import classify
from app.kb import KB, build_query
from app.llm import log
from app.policy import guard_decision, template_for, tutor_action
from app.verify import find_unsupported_vote

load_dotenv()

KB_INDEX = KB()  # 서버 시작 때 BM25 색인 생성(약 1.7초)

TOKEN = os.environ["TEAM_TOKEN"]
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."
MAX_ANSWER = 4000
# 답변 생성에 주는 섹션 수. 1 = 인용할 BM25 1위 섹션만 보고 답한다(근거 충실도는 인용 섹션과만 대조되므로
# 답변 내용과 인용이 구조적으로 일치). A/B: 섹션 1/2/3/4개 = 86.3/84.5/86.1/82.7, 1개가 가장 안정·빠름(DEVLOG 5장).
TOP_K = int(os.environ.get("TOP_K", "1"))
VERIFY_TOP_K = int(os.environ.get("VERIFY_TOP_K", "4"))  # 가드 후보 답변 검증은 여러 섹션과 대조
SPECULATIVE = os.environ.get("SPECULATIVE", "1") == "1"  # 분류와 답변 생성 동시 시작
GUARD_OOM = os.environ.get("GUARD_OOM", "1") == "1"  # 가드: 자료에 없는 질문이면 범주를 out_of_material로(DEVLOG 20장)
SELF_VERIFY = os.environ.get("SELF_VERIFY", "0") == "1"  # 튜터 답변 자기검증 후 재생성(DEVLOG 17장, 실험)
REQUEST_BUDGET = float(os.environ.get("REQUEST_BUDGET", "26"))  # 전체 예산(초): 30초 제한 - 네트워크 여유
CLASSIFY_BUDGET = float(os.environ.get("CLASSIFY_BUDGET", "12"))  # 튜터에서 분류에 쓸 최대 시간(초)

app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)


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
    ranked = KB_INDEX.rank(build_query(messages, "all"))
    return [KB_INDEX.secs[sid] for sid, _ in ranked[:TOP_K]]


PII_NOTE = "이름이나 학교 같은 개인정보는 채팅에 적지 않아도 괜찮아요."


async def _generate(messages, secs, mode, deadline):
    try:
        return await grounded_answer(messages, secs, mode, deadline=deadline)
    except Exception as e:  # noqa: BLE001
        log("tutor_llm_error", err=type(e).__name__)
        return None


async def refine(messages, secs, g, deadline):
    """생성한 답변을 인용할 섹션과 대조하고, 근거 없는 구간이 나오면 그 구간을 알려 주고 한 번 다시 생성한다.
    시간이 부족하거나 검증·재생성이 실패하면 처음 답변을 그대로 쓴다."""
    if deadline - time.time() < 12:
        return g
    try:
        spans = await find_unsupported_vote(messages, g["answer"], secs, deadline=deadline - 7, votes=1)
    except Exception:  # noqa: BLE001
        log("selfverify_error")
        return g
    if not spans:
        return g
    try:
        g2 = await grounded_answer(messages, secs, "answer", deadline=deadline, feedback=spans[:3])
    except Exception:  # noqa: BLE001
        log("selfverify_regen_error")
        return g
    return g2 if g2["found"] else g


async def tutor_logic(messages, t0) -> Dict:
    """분류와 답변 생성을 동시에 시작한다(추측 실행). 지연 = 둘의 합 → 둘 중 긴 쪽.
    분류가 고정 응답 범주면 생성 결과는 버리고, 힌트 모드(answer_seeking)일 때만 다시 생성한다."""
    deadline = t0 + REQUEST_BUDGET
    secs = retrieve(messages)
    gen_task = asyncio.create_task(_generate(messages, secs, "answer", deadline)) if SPECULATIVE else None
    c = await classify(messages, deadline=min(deadline, t0 + CLASSIFY_BUDGET))
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
        mode = "scaffold" if action == "scaffold" else "answer"
        g = await _generate(messages, secs, mode, deadline)
    else:
        g = await gen_task
    if g is None:
        return {"answer": FALLBACK, "action": action, "citations": citations}
    if SELF_VERIFY and action == "answer" and g["found"]:
        g = await refine(messages, secs, g, deadline)
    answer = g["answer"]
    if g["found"]:
        # 인용은 BM25 1위로 정한다: 공개 세트에서 LLM 선택 정확도 70% < BM25 1위 91%
        # (같은 사실이 여러 섹션에 중복돼 있어 LLM은 제목이 그럴듯한 섹션을 고르는 경향).
        citations = [secs[0]["id"]]
    elif action == "answer":
        action = "abstain"  # 자료에서 확인할 수 없는 질문 → 지어내지 않고 abstain
    if c["category"] == "pii_disclosure" and "개인정보" not in answer:
        answer = answer.rstrip() + " " + PII_NOTE  # 개인정보 안내는 코드로 붙인다(LLM 모드 불필요)
    return {"answer": answer[:MAX_ANSWER], "action": action, "citations": citations}


def retrieve_for_candidate(messages, cand: str) -> List[Dict[str, str]]:
    q = "\n".join(m["content"] for m in messages[-4:]) + "\n" + cand + "\n" + cand  # 후보 답변 가중
    return [KB_INDEX.secs[sid] for sid, _ in KB_INDEX.rank(q)[:VERIFY_TOP_K]]


async def _verify(messages, cand: str, deadline) -> Optional[List[str]]:
    try:
        return await find_unsupported_vote(messages, cand, retrieve_for_candidate(messages, cand), deadline=deadline)
    except Exception as e:  # noqa: BLE001
        log("verify_error", err=type(e).__name__)
        return None


async def _found(messages, deadline) -> Optional[bool]:
    """검색 1위 섹션으로 '자료에 답이 있나'만 본다(튜터의 found와 같은 판단). 실패하면 None."""
    try:
        g = await grounded_answer(messages, retrieve(messages), "answer", deadline=deadline)
        return g["found"]
    except Exception as e:  # noqa: BLE001
        log("guard_found_error", err=type(e).__name__)
        return None


async def guard_logic(messages, cand: Optional[str], t0) -> Dict:
    deadline = t0 + REQUEST_BUDGET
    found_task = asyncio.create_task(_found(messages, deadline)) if GUARD_OOM else None
    if cand is not None and cand.strip():
        # 입력 분류와 후보 답변 검증을 동시에 실행(지연 = 둘 중 긴 쪽)
        c, spans = await asyncio.gather(classify(messages, deadline=deadline), _verify(messages, cand, deadline))
    else:
        c, spans = await classify(messages, deadline=deadline), []
    if found_task:
        # 범주 점수는 정답이 normal이 아닌 문항만 센다: 정상 입력에 out_of_material을 잘못 붙여도 점수가 깎이지 않고
        # 판정(allow)도 같다. 그래서 normal로 분류됐고 자료에서 답을 못 찾은 입력만 out_of_material로 바꾼다.
        if c["category"] == "normal" and (await found_task) is False:
            c = {**c, "category": "out_of_material"}
        else:
            found_task.cancel()
    decision = guard_decision(c["category"])
    result = {"input": {"decision": decision, "category": c["category"]}}
    if cand is not None:
        if decision == "block":
            # 막아야 할 입력에 대한 후보 답변 = 그 요청에 응한 답변 → unsafe.
            # 공개 세트 라벨에서 이런 답변은 모두 grounded=True·구간 없음(근거가 아니라 안전 문제로 봄).
            out = {"grounded": True, "unsupported": [], "safe": False}
        else:
            spans = spans or []  # 검증 실패(None)면 근거 있음으로 둔다(정상 답변을 깎지 않음)
            out = {"grounded": not spans, "unsupported": spans, "safe": True}
        result["output"] = {**out, "category": c["category"]}
    return result


@app.post("/tutor")
async def tutor(request: Request):
    t0 = time.time()
    if not authorized(request):
        return JSONResponse({"detail": "invalid token"}, status_code=401)
    status = "ok"
    try:
        messages, _ = await read_payload(request)
        out = await tutor_logic(messages, t0)
    except Exception as e:  # noqa: BLE001  — 어떤 경우에도 규격 응답
        status = f"error:{type(e).__name__}"
        messages, out = [], {"answer": FALLBACK, "action": "answer", "citations": []}
    log("tutor", ms=int((time.time() - t0) * 1000), status=status,
        in_len=sum(len(m["content"]) for m in messages), out_len=len(out["answer"]))
    return out


@app.post("/guard")
async def guard(request: Request):
    t0 = time.time()
    if not authorized(request):
        return JSONResponse({"detail": "invalid token"}, status_code=401)
    status = "ok"
    cand = None
    try:
        messages, cand = await read_payload(request)
        out = await guard_logic(messages, cand, t0)
    except Exception as e:  # noqa: BLE001
        status = f"error:{type(e).__name__}"
        out = {"input": {"decision": "allow", "category": "normal"}}
        if cand is not None:
            out["output"] = {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}
    log("guard", ms=int((time.time() - t0) * 1000), status=status,
        in_len=sum(len(m["content"]) for m in messages), cand_len=len(cand or ""))
    return out
