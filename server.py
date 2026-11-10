"""안전벨트를 채운 AI 튜터 — /tutor, /guard 서버.

1단계: 입력 범주 분류(app.classify) → 가드 판정·튜터 응대 방식을 규칙(app.policy)으로 결정.
2단계: BM25 검색(app.kb) → 근거 섹션 1개 선택 + 근거 기반 답변(app.answer). 자료에 없으면 abstain.
로그에는 시간·길이·상태·오류 종류만 남긴다(요청·답변 본문 저장 금지, SPEC §8).
"""
import os
import secrets
import time
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.answer import grounded_answer
from app.classify import classify
from app.kb import KB, build_query
from app.llm import log
from app.policy import guard_decision, template_for, tutor_action

KB_INDEX = KB(use_embeddings=False)  # 서버 시작 때 색인 생성(BM25). 임베딩은 검색 평가에서 손해라 쓰지 않음.

TOKEN = os.environ["TEAM_TOKEN"]
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."
MAX_ANSWER = 4000


def check_token(authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="invalid token")


app = FastAPI(redirect_slashes=False, dependencies=[Depends(check_token)])


class TutorRequest(BaseModel):
    messages: List[Dict[str, str]]


class GuardRequest(BaseModel):
    messages: List[Dict[str, str]]
    candidate_answer: Optional[str] = None


TOP_K = int(os.environ.get("TOP_K", "4"))
CITE_MODE = os.environ.get("CITE_MODE", "bm25")  # "bm25" | "llm"


def retrieve(messages) -> List[Dict[str, str]]:
    ranked = KB_INDEX.rank(build_query(messages, "all"))
    return [KB_INDEX.secs[sid] for sid, _ in ranked[:TOP_K]]


@app.post("/tutor")
async def tutor(req: TutorRequest):
    t0 = time.time()
    c = await classify(req.messages)
    action = tutor_action(c["category"], c["harmful_type"])
    citations: List[str] = []
    fixed = template_for(c["category"], c["harmful_type"])
    if fixed is not None:
        answer = fixed
    else:
        mode = "scaffold" if action == "scaffold" else ("pii" if c["category"] == "pii_disclosure" else "answer")
        try:
            secs = retrieve(req.messages)
            g = await grounded_answer(req.messages, secs, mode)
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
    answer = answer[:MAX_ANSWER]
    log("tutor", ms=int((time.time() - t0) * 1000), action=action, in_turns=len(req.messages),
        out_len=len(answer), n_cite=len(citations))
    return {"answer": answer, "action": action, "citations": citations}


@app.post("/guard")
async def guard(req: GuardRequest):
    t0 = time.time()
    c = await classify(req.messages)
    decision = guard_decision(c["category"])
    result = {"input": {"decision": decision, "category": c["category"]}}
    if req.candidate_answer is not None:
        # 1단계 임시 규칙: 막아야 할 입력에 대한 후보 답변은 그 요청에 응한 것으로 보고 unsafe.
        # (공개 세트 후보 답변 86개에서 unsafe 19개가 모두 block 범주였음.) 근거 판정은 3단계에서.
        result["output"] = {"grounded": True, "unsupported": [], "safe": decision == "allow", "category": c["category"]}
    log("guard", ms=int((time.time() - t0) * 1000), decision=decision, has_cand=req.candidate_answer is not None)
    return result
