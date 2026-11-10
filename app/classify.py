"""입력 범주 분류기(진입점).

LLM은 예/아니오 신호 9개만 뽑고(app/classify_feat.py), 범주는 코드가 우선순위 규칙으로 정한다.
같은 입력을 VOTES번 병렬 분류해 신호별 과반으로 결정한다(실행마다 흔들리는 판정 안정화).
범주 정의는 공개 세트 라벨을 관찰해 팀이 직접 세운 규칙이다(SPEC §5: 정의는 데이터로 세움).
예시 문장은 공개 세트 문항을 복사하지 않고 팀이 새로 쓴 것이다.
이전 방식(10개 범주를 직접 고르는 v1·v2 프롬프트)은 DEVLOG 1장, 커밋 ebb446e~bb4c9cb에 있다.
"""
import asyncio
import os
import time
from typing import Dict, List

from . import classify_feat
from .llm import chat_json, log

EFFORT = os.environ.get("CLASSIFY_EFFORT", "low")  # minimal보다 block F1 약 +4%p(DEVLOG 1장)
VOTES = int(os.environ.get("CLASSIFY_VOTES", "3"))  # 3회 다수결: block F1 0.902→0.922, 흔들림 감소(DEVLOG 5장)
EXTRA_VOTES = int(os.environ.get("CLASSIFY_EXTRA_VOTES", "0"))  # 적응형 투표: 첫 VOTES표가 갈리면 추가로 받는 표 수(0=끔)
MAX_TOKENS = 2000  # 운영 API: 비스트리밍 응답은 2000 이하
TIMEOUT = 20


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
        calls = lambda n: asyncio.gather(*[chat_json(classify_feat.SYSTEM, user, **kw) for _ in range(n)],
                                         return_exceptions=True)
        outs = [o for o in await calls(max(1, VOTES)) if isinstance(o, dict)]
        if not outs:
            raise RuntimeError("all votes failed")
        # 적응형 투표: 어떤 신호든 표가 갈렸고 시간이 남으면 표를 더 받는다(만장일치면 그대로 끝)
        split = any(0 < sum(o.get(k) is True for o in outs) < len(outs) for k in classify_feat.FLAGS)
        if EXTRA_VOTES and split and (deadline is None or deadline - time.time() > 8):
            outs += [o for o in await calls(EXTRA_VOTES) if isinstance(o, dict)]
        flags = {k: sum(o.get(k) is True for o in outs) * 2 > len(outs) for k in classify_feat.FLAGS}
        return {**classify_feat.decide(flags), "flags": flags}
    except Exception as e:  # noqa: BLE001
        log("classify_error", err=type(e).__name__)
        return {"category": "normal", "harmful_type": "none"}
