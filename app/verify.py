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
1. 후보 답변에서 **과목 고유 사실 주장**만 찾는다. 과목 고유 사실 주장이란 이 과정·실습·예제·가상 상품이 **무엇인지·어떻게 정해져 있는지 단정하는 내용**이다: 값·숫자·금액·비율·나이·기간, 이름·명령어·파일명·코드 내용, 규칙·조건·단계·순서, 실습 결과·출력.
2. 다음은 사실 주장이 아니므로 **claims에 넣지 않는다**(검사 대상 아님):
   - 튜터의 응대 문장: 거절("~는 보여 드릴 수 없어요", "~는 도와줄 수 없어요"), 수업으로 돌리기("~는 수업과 관련이 없어요"), 도움 안내, 격려·공감·인사.
   - 개인정보 안내("이름이나 학교는 적지 않아도 돼요" 등).
   - 자료에 없다고 말하는 문장("자료에 ~가 나와 있지 않아요", "~는 확인할 수 없어요").
   - 질문·되묻기("무엇이 궁금한가요?", "~는 얼마가 될까요?"), 학습자에게 해 보라고 하는 안내·힌트("~를 떠올려 보세요", "~를 비교해 보세요").
   - 자료와 모순되지 않는 일반 설명(파이썬 일반 문법, 상식적인 조언), 학습자가 말한 내용을 되풀이한 것, 섹션 ID 표기.
3. 찾은 사실 주장을 [자료]와 비교한다.
   - 자료와 같은 뜻이면 match(표현이 달라도 됨). 자료의 값으로 바르게 계산한 결과도 match.
   - 자료와 **다르면**(숫자·나이·기간·비율·이름·순서·조건이 바뀜, 자료가 '아니다'라고 한 것을 '그렇다'고 함) → contradict.
   - 자료에 전혀 없는 과목 고유 값·규칙·이름을 이 과정의 것처럼 **지어내 단정**하면 → absent. 자료에서 확인되는 내용을 다르게 표현했을 뿐이면 absent가 아니다.
4. contradict·absent인 사실마다 **틀린 부분**(wrong_part)을 후보 답변에서 글자 그대로 복사한다. 문장 전체가 아니라 틀린 값과 그 값이 무엇인지 알 수 있는 최소한의 말(보통 2~6어절)이다.
   예(가상의 자료): 자료 "적금 기간은 24개월" / 후보 "이 적금은 기간이 18개월이라서 길지 않아요" → wrong_part "기간이 18개월"
   예: 자료 "90점 이상이면 A" / 후보 "95점은 90점 이상이니 B를 출력해요" → wrong_part "B를 출력해요"

출력: JSON 객체 하나 {"claims": [{"text": "<후보 답변의 사실 문장>", "verdict": "match" | "contradict" | "absent", "wrong_part": "<contradict·absent일 때 틀린 부분, match면 빈 문자열>"}]}
claims에는 후보 답변의 과목 고유 사실을 모두 적는다."""

_PUNCT = set('.,!?;:\'"“”‘’`()[]{}<>«»·…~-–—/\\')


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if not ch.isspace() and ch not in _PUNCT)

# 사실 주장이 아닌 문장(응대·안내·질문)에 걸린 구간은 버린다. LLM이 프롬프트의 제외 지시를 자주 어겨서
# 코드로 한 번 더 거른다. 공개 정답 답변(튜닝용 절반)에서 관찰한 오탐 유형을 일반 패턴으로 적은 것.
_NON_CLAIM = re.compile(
    r"(나와 있지 않|나와있지 않|설명되어 있지 않|적혀 있지 않|자료에(는)? 없|자료에서 확인할 수 없|확인할 수 없"  # 자료에 없음
    r"|수 없어요|수 없습니다|드릴 수 없|줄 수 없"                                              # 거절
    r"|수업과 (관련이 )?(없|무관)|수업과 상관"                                                 # 수업 전환
    r"|개인정보|(이름|학교|학년)[^.?!]{0,15}적지 않아도"                                                                  # 개인정보 안내
    r"|보세요|볼래요|볼까요|봐요\.?$|해 봐요|생각해 봐|떠올려|살펴보고|찾아보"                 # 힌트·권유
    r"|도와(드릴|줄)게요|도울 수 있어요|함께 (배울|공부할) 수)"                                 # 도움 안내
)


def _sentence_of(candidate: str, span: str) -> str:
    """후보 답변에서 span이 들어 있는 문장(마침표·물음표·느낌표·줄바꿈 기준)."""
    for sent in re.split(r"(?<=[.?!。])\s+|\n+", candidate):
        if _norm(span) and _norm(span) in _norm(sent):
            return sent
    return ""


def _is_non_claim(candidate: str, span: str) -> bool:
    sent = _sentence_of(candidate, span).strip()
    return bool(sent) and (sent.endswith("?") or bool(_NON_CLAIM.search(sent)))


def render(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{'학습자' if m.get('role') == 'user' else '튜터'}: {m.get('content', '')}" for m in messages)


async def find_unsupported(messages, candidate: str, sections: List[Dict[str, str]], deadline=None) -> List[str]:
    ctx = "\n\n".join(f"<섹션 id=\"{s['id']}\" 제목=\"{s['title']}\">\n{s['text']}\n</섹션>" for s in sections)
    user = f"[자료]\n{ctx}\n\n[대화]\n{render(messages)}\n\n[후보 답변]\n{candidate}"
    out = await chat_json(SYSTEM, user, effort="low", max_tokens=2000, timeout=20, deadline=deadline, hedge_after=9.0,
                          kind="ground")
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
        if isinstance(s, str) and s.strip() and _norm(s) and _norm(s) in cand_n and s not in keep \
                and not _is_non_claim(candidate, s):
            keep.append(s.strip())
    return keep
