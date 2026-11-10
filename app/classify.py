"""입력 범주 분류기(진입점).

LLM은 예/아니오 신호 9개만 뽑고(app/classify_feat.py), 범주는 코드가 우선순위 규칙으로 정한다.
같은 입력을 VOTES번 병렬 분류해 신호별 과반으로 결정한다(실행마다 흔들리는 판정 안정화).
범주 정의는 공개 세트 라벨을 관찰해 팀이 직접 세운 규칙이다(SPEC §5: 정의는 데이터로 세움).
예시 문장은 공개 세트 문항을 복사하지 않고 팀이 새로 쓴 것이다.
이전 방식(10개 범주를 직접 고르는 v1·v2 프롬프트)은 DEVLOG 1장, 커밋 ebb446e~bb4c9cb에 있다.
"""
import asyncio
import os
from typing import Dict, List

from . import classify_feat
from .llm import chat_json, log

EFFORT = os.environ.get("CLASSIFY_EFFORT", "low")  # minimal보다 block F1 약 +4%p(DEVLOG 1장)
VOTES = int(os.environ.get("CLASSIFY_VOTES", "3"))  # 3회 다수결: block F1 0.902→0.922, 흔들림 감소(DEVLOG 5장)
MAX_TOKENS = 2000  # 운영 API: 비스트리밍 응답은 2000 이하
TIMEOUT = 20
FEWSHOT_K = int(os.environ.get("FEWSHOT_K", "0"))  # 공개 세트 유사 예시 수(0이면 사용 안 함)
_FEWSHOT = None


def set_fewshot(fs):
    """평가 스크립트가 예시 창고를 바꿔 끼울 때 쓴다(교차 검증)."""
    global _FEWSHOT
    _FEWSHOT = fs


def _fewshot():
    global _FEWSHOT
    if _FEWSHOT is None and FEWSHOT_K > 0:
        from .fewshot import FewShot
        _FEWSHOT = FewShot()
    return _FEWSHOT


def render(messages: List[Dict[str, str]]) -> str:
    lines = []
    for i, m in enumerate(messages):
        role = "학습자" if m.get("role") == "user" else "튜터"
        tag = " [마지막 학습자 발화 — 분류 대상]" if i == len(messages) - 1 else ""
        lines.append(f"<{role}{tag}>\n{m.get('content', '')}\n</{role}>")
    return "\n".join(lines)


async def classify(messages: List[Dict[str, str]], deadline=None) -> Dict[str, str]:
    """{"category", "harmful_type", "flags"}를 돌려준다. 실패하면 normal로 둔다(정상 입력 차단을 피함)."""
    try:
        kw = dict(effort=EFFORT, max_tokens=MAX_TOKENS, timeout=TIMEOUT, deadline=deadline, hedge_after=7.0)
        user = render(messages)
        fs = _fewshot()
        if fs is not None and FEWSHOT_K > 0:
            user = fs.render(messages, FEWSHOT_K) + "\n\n[분류 대상 대화]\n" + user
        outs = await asyncio.gather(*[chat_json(classify_feat.SYSTEM, user, **kw) for _ in range(max(1, VOTES))],
                                    return_exceptions=True)
        outs = [o for o in outs if isinstance(o, dict)]
        if not outs:
            raise RuntimeError("all votes failed")
        flags = {k: sum(o.get(k) is True for o in outs) * 2 > len(outs) for k in classify_feat.FLAGS}
        return {**classify_feat.decide(flags), "flags": flags}
    except Exception as e:  # noqa: BLE001
        log("classify_error", err=type(e).__name__)
        return {"category": "normal", "harmful_type": "none"}
