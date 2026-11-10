"""'자료에 답이 있나' 전용 확인: 답이 적힌 문장을 섹션에서 그대로 인용하게 하고, 인용이 실제로 섹션에 있는지 코드로 검증한다.
LLM의 '있다'는 말이 아니라 검증 가능한 증거(인용)로 판단한다. 답변 생성과 동시에 실행한다.
"""
import re
from typing import Dict, List

from .llm import chat_json

SYSTEM = """너는 중학생용 학습 플랫폼의 자료 확인 도우미다. [자료] 섹션 하나와 학습자의 질문이 주어진다.
질문에 대한 답이 [자료]에 **직접 적혀 있는지**만 판단한다. 답을 만들거나 추론해서 채우지 않는다.

- 질문이 이름을 붙여 묻는 개념·규칙·기준·점수·등급·지수·수수료·기능·횟수(예: "○○ 규칙", "○○법", "○○ 기준")가 있으면, 그 이름 또는 같은 뜻이 [자료]에 실제로 있어야 한다. 비슷한 주제만 있으면 없는 것이다.
- 실습·예제의 값·이름·순서·조건·출력을 묻는 질문은, 그 값이 [자료]에 적혀 있거나 [자료]의 값으로 바로 계산할 수 있으면 있는 것이다.
- 학습자 코드가 자료의 예제와 왜 다른지 묻는 질문은, 비교할 예제 값이 [자료]에 있으면 있는 것이다.

있으면 답의 근거가 되는 문장을 [자료]에서 **글자 그대로** 1~2개 복사한다(표의 한 줄도 가능).
출력: JSON 객체 하나 {"answer_in_material": true 또는 false, "quotes": ["<자료에서 그대로 복사한 문장>", ...]}"""


# 자료에 없는 질문은 대개 이름 붙은 대상("○○ 규칙", "○○ 기준", "○○ 점수")을 묻는다. 이런 질문에만 확인을 적용해
# 확인 단계의 과잉 엄격함(정상 질문 약 14%를 '없음'으로 판정)이 정상 질문 전체에 퍼지지 않게 한다.
NAMED_TARGET = re.compile(r"(규칙|기준|등급|지수|수수료|점수|분류법|배점|만점|횟수|기능|rule|score|fee|limit)", re.I)


def needs_check(messages) -> bool:
    users = [m.get("content", "") for m in messages if m.get("role") == "user"]
    return bool(users) and bool(NAMED_TARGET.search(users[-1]))


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s.lower())


def _grams(s: str, n: int = 4):
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def quote_in_section(quote: str, section_text: str, thresh: float = 0.8) -> bool:
    """인용문의 글자 4-gram 중 thresh 이상이 섹션에 있으면 실제 인용으로 본다(띄어쓰기·문장부호 차이 허용)."""
    q = _norm(quote)
    if len(q) < 6:
        return False
    sec = _norm(section_text)
    if q in sec:
        return True
    g = _grams(q)
    return bool(g) and len(g & _grams(sec)) / len(g) >= thresh


def render(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{'학습자' if m.get('role') == 'user' else '튜터'}: {m.get('content', '')}" for m in messages[-5:])


async def check_support(messages, section: Dict[str, str], deadline=None) -> bool:
    user = f"[자료]\n<섹션 id=\"{section['id']}\">\n{section['text']}\n</섹션>\n\n[대화]\n{render(messages)}\n\n마지막 학습자 발화의 답이 자료에 직접 적혀 있나?"
    out = await chat_json(SYSTEM, user, effort="low", max_tokens=1500, timeout=20, deadline=deadline)
    if out.get("answer_in_material") is not True:
        return False
    quotes = [q for q in (out.get("quotes") or []) if isinstance(q, str)]
    return any(quote_in_section(q, section["text"]) for q in quotes)
