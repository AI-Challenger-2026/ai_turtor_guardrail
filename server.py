"""안전벨트를 채운 AI 튜터 — /tutor, /guard 서버.

1단계: 입력 범주 분류(app.classify) → 가드 판정·튜터 응대 방식을 규칙(app.policy)으로 결정.
2단계: BM25 검색(app.kb) → 근거 섹션 1개 선택 + 근거 기반 답변(app.answer). 자료에 없으면 abstain.

견고성 원칙
- 인증 실패(401) 말고는 어떤 입력에도 HTTP 200 + 규격 JSON으로 답한다(4xx·5xx는 그 문항 0점).
- 요청마다 시간 예산(REQUEST_BUDGET초)을 두고, 넘기면 안전한 기본 응답을 낸다(채점기 제한 30초).
- 로그에는 시간·길이·상태·오류 종류만 남긴다(요청·답변 본문 저장 금지, SPEC §8).
"""
import json
import os
import secrets
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.answer import grounded_answer
from app.classify import classify
from app.kb import KB, build_query
from app.llm import log
from app.policy import guard_decision, template_for, tutor_action

KB_INDEX = KB(use_embeddings=False)  # 서버 시작 때 색인 생성(BM25). 임베딩은 검색 평가에서 손해라 쓰지 않음.

TOKEN = os.environ["TEAM_TOKEN"]
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."
MAX_ANSWER = 4000
TOP_K = int(os.environ.get("TOP_K", "4"))
CITE_MODE = os.environ.get("CITE_MODE", "bm25")  # "bm25" | "llm"
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


async def tutor_logic(messages, t0) -> Dict:
    deadline = t0 + REQUEST_BUDGET
    c = await classify(messages, deadline=min(deadline, t0 + CLASSIFY_BUDGET))
    action = tutor_action(c["category"], c["harmful_type"])
    citations: List[str] = []
    fixed = template_for(c["category"], c["harmful_type"])
    if fixed is not None:
        answer = fixed
    else:
        mode = "scaffold" if action == "scaffold" else ("pii" if c["category"] == "pii_disclosure" else "answer")
        try:
            secs = retrieve(messages)
            g = await grounded_answer(messages, secs, mode, deadline=deadline)
            answer = g["answer"]
            if g["found"]:
                # 인용은 BM25 1위로 정한다: 공개 세트에서 LLM 선택 정확도 70% < BM25 1위 91%
                # (같은 사실이 여러 섹션에 중복돼 있어 LLM은 제목이 그럴듯한 섹션을 고르는 경향).
                citations = [secs[0]["id"]] if CITE_MODE == "bm25" else [g["evidence"]]
            elif action == "answer":
                action = "abstain"  # 자료에서 확인할 수 없는 질문 → 지어내지 않고 abstain
        except Exception as e:  # noqa: BLE001
            log("tutor_llm_error", err=type(e).__name__)
            answer = FALLBACK
    return {"answer": answer[:MAX_ANSWER], "action": action, "citations": citations}


async def guard_logic(messages, cand: Optional[str], t0) -> Dict:
    c = await classify(messages, deadline=t0 + REQUEST_BUDGET)
    decision = guard_decision(c["category"])
    result = {"input": {"decision": decision, "category": c["category"]}}
    if cand is not None:
        # 1단계 임시 규칙: 막아야 할 입력에 대한 후보 답변은 그 요청에 응한 것으로 보고 unsafe.
        # (공개 세트 후보 답변 86개에서 unsafe 19개가 모두 block 범주였음.) 근거 판정은 3단계에서.
        result["output"] = {"grounded": True, "unsupported": [], "safe": decision == "allow", "category": c["category"]}
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
    log("tutor", ms=int((time.time() - t0) * 1000), status=status, action=out["action"],
        in_turns=len(messages), out_len=len(out["answer"]), n_cite=len(out["citations"]))
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
    log("guard", ms=int((time.time() - t0) * 1000), status=status, decision=out["input"]["decision"],
        has_cand=cand is not None)
    return out
