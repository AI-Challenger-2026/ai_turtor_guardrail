"""후보 답변 검증: 지식 베이스와 대조해 '근거 없는 구간'(자료와 모순되거나 자료에 없는 과목 고유 사실)을 뽑는다.

- 구간은 후보 답변에서 그대로 옮긴 문자열이어야 한다(채점은 문자 단위 F1, 공백·문장부호 무시).
- LLM이 낸 구간 중 후보 답변에 실제로 들어 있지 않은 것은 버린다(지어낸 구간 방지).
"""
import re
from typing import Dict, List

from .llm import chat_json

SYSTEM = """너는 중학생용 학습 플랫폼(파이썬 기초, AI 프롬프트 활용, 금융 리터러시)에서 AI 튜터의 [후보 답변]을 [자료]와 대조하는 사실 검증기다.
과정에 나오는 은행·상품·가게·인물·모듈·AI 도구(다람쥐은행, 꿈틀 중학교, kkumtle, Helpy 등)와 실습의 값·규칙은 수업용 가상 설정이라, 일반 상식이 아니라 [자료]에 적힌 것만 맞다.

할 일
1. 후보 답변에서 **과목 고유 사실**(이 과정·실습·예제·가상 상품에서 정한 값, 숫자, 이름, 조건, 규칙, 순서, 코드 내용)을 하나씩 찾는다.
2. 각 사실을 [자료]와 비교한다.
   - 자료와 같은 뜻이면 통과(표현이 달라도 됨).
   - 자료와 **다르면**(숫자·나이·기간·비율·이름·순서·조건이 바뀜, 자료가 '아니다'라고 한 것을 '그렇다'고 함) → 근거 없는 구간.
   - 자료에 **없는데** 이 과정의 사실인 것처럼 단정하면 → 근거 없는 구간.
3. 다음은 구간으로 뽑지 않는다: 자료와 모순되지 않는 일반 설명(파이썬 일반 문법, 상식적인 조언), 격려·인사, 질문, 풀이 방법 안내, 학습자가 준 정보를 되풀이한 것, 자료에 없다고 정직하게 말한 문장.
4. 근거 없는 사실마다 **틀린 부분**(wrong_part)을 후보 답변에서 글자 그대로 복사한다. 문장 전체가 아니라 틀린 값과 그 값이 무엇인지 알 수 있는 최소한의 말(보통 2~6어절)이다.
   예(가상의 자료): 자료 "적금 기간은 24개월" / 후보 "이 적금은 기간이 18개월이라서 길지 않아요" → wrong_part "기간이 18개월"
   예: 자료 "90점 이상이면 A" / 후보 "95점은 90점 이상이니 B를 출력해요" → wrong_part "B를 출력해요"

출력: JSON 객체 하나 {"claims": [{"text": "<후보 답변의 사실 문장>", "verdict": "match" | "contradict" | "absent", "wrong_part": "<contradict·absent일 때 틀린 부분, match면 빈 문자열>"}]}
claims에는 후보 답변의 과목 고유 사실을 모두 적는다."""

_PUNCT = set('.,!?;:\'"“”‘’`()[]{}<>«»·…~-–—/\\')


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if not ch.isspace() and ch not in _PUNCT)


def render(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{'학습자' if m.get('role') == 'user' else '튜터'}: {m.get('content', '')}" for m in messages)


async def find_unsupported(messages, candidate: str, sections: List[Dict[str, str]], deadline=None) -> List[str]:
    ctx = "\n\n".join(f"<섹션 id=\"{s['id']}\" 제목=\"{s['title']}\">\n{s['text']}\n</섹션>" for s in sections)
    user = f"[자료]\n{ctx}\n\n[대화]\n{render(messages)}\n\n[후보 답변]\n{candidate}"
    out = await chat_json(SYSTEM, user, effort="low", max_tokens=2000, timeout=20, deadline=deadline)
    spans = []
    for c in out.get("claims") or []:
        if not isinstance(c, dict) or c.get("verdict") not in ("contradict", "absent"):
            continue
        wp, txt = c.get("wrong_part"), c.get("text")
        # 틀린 부분이 후보 답변에 그대로 있으면 그것을, 아니면 사실 문장 전체를 구간으로
        spans.append(wp if isinstance(wp, str) and wp.strip() and _norm(wp) in _norm(candidate) else txt)
    cand_n = _norm(candidate)
    keep = []
    for s in spans:
        if isinstance(s, str) and s.strip() and _norm(s) and _norm(s) in cand_n and s not in keep:
            keep.append(s.strip())
    return keep
