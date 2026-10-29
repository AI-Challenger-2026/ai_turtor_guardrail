"""응답 모드, 질문 유형 분류, 학습자 수준 판단 (answer modes / question type / learner level).

- 질문 유형 분류(question type classification)와 학습자 수준 판단은 규칙 기반(rule-based)이다.
  LLM 호출 없이 결정적으로 동작하므로 비용이 들지 않고 테스트하기 쉽다.
  평가(run_eval)에서 qa_set 의 q_type 과 비교해 분류 정확도를 함께 보고한다.
- 모드별 지시문은 prompts/answer_modes.yaml 에서 읽는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Sequence

from app.prompts import PromptStore, render_template

QuestionType = Literal["concept", "exercise", "confusion", "followup"]
LearnerLevel = Literal["beginner", "intermediate"]

_CONFUSION_KW = (
    "헷갈", "햇갈", "이해가 안", "이해 안", "이해못", "이해 못", "모르겠", "잘 모르", "무슨 말", "무슨말",
    "다시 설명", "다시 알려", "어려워", "어렵다", "어렵네", "뭔 소리", "뭔소리", "??",
)
_EXERCISE_KW = (
    "코드 짜", "코드짜", "코드 작성", "작성해", "짜줘", "짜 줘", "풀어", "풀이", "문제", "정답", "답 알려", "답알려",
    "답이 뭐", "출력 결과", "결과가 뭐", "만들어줘", "만들어 줘", "구현", "프로그램", "과제", "숙제", "연습",
    "에러", "오류", "안 돌아", "안돌아", "틀렸", "고쳐",
)
_FOLLOWUP_PREFIX = (
    "그럼", "그러면", "그건", "그거", "그게", "이건", "이거", "저건", "저거", "그래서", "왜요", "왜?", "왜 ", "또",
    "다른 건", "다른건", "아까", "방금", "위에", "그 다음", "그다음", "그리고", "근데", "그런데",
)
_FOLLOWUP_PATTERN = re.compile(r"^\s*(\d+)\s*번")
_INTERMEDIATE_KW = (
    "컴프리헨션", "comprehension", "제너레이터", "generator", "이터레이터", "iterator", "가변", "불변", "mutable",
    "immutable", "시간복잡도", "시간 복잡도", "참조", "얕은 복사", "깊은 복사", "deepcopy", "람다", "lambda",
    "데코레이터", "decorator", "클로저", "closure", "*args", "**kwargs", "언패킹", "unpacking", "해시", "hash",
    "스코프", "scope", "재귀", "recursion", "타입 힌트", "type hint",
)
_BEGINNER_KW = (
    "처음", "초보", "입문", "왕초보", "기초", "쉽게", "잘 몰라", "하나도 모르", "코딩 처음", "배운 지 얼마",
)


def is_followup_like(question: str, history_len: int, max_short_chars: int = 20) -> bool:
    """맥락 의존적인 짧은 후속 질문처럼 보이는지 (context-dependent follow-up heuristic)."""
    if history_len == 0:
        return False
    q = question.strip()
    if q.startswith(_FOLLOWUP_PREFIX) or _FOLLOWUP_PATTERN.match(q):
        return True
    return len(q) <= max_short_chars and not _looks_self_contained(q)


_TOPIC_TERMS = ("리스트", "튜플", "딕셔너리", "변수", "자료형", "함수", "반복문", "조건문", "for", "while", "if",
                "문자열", "정수", "실수", "print", "return", "range", "인덱스", "슬라이싱", "def", "매개변수", "원소")


def _looks_self_contained(q: str) -> bool:
    # 주제어(파이썬 용어)가 들어 있고 너무 짧지 않으면 독립 질문으로 본다.
    return len(q) >= 10 and any(t in q for t in _TOPIC_TERMS)


def classify_question(question: str, history: Sequence[dict[str, str]] | None = None, max_short_chars: int = 20) -> QuestionType:
    """질문 유형 분류: confusion > exercise > followup > concept 순서로 판단한다."""
    history = history or []
    q = question.strip()
    low = q.lower()
    if any(k in low for k in _CONFUSION_KW):
        return "confusion"
    if any(k in low for k in _EXERCISE_KW):
        return "exercise"
    if is_followup_like(q, len(history), max_short_chars):
        return "followup"
    return "concept"


def estimate_learner_level(messages: Sequence[dict[str, str]]) -> LearnerLevel:
    """대화 이력(학습자 발화)에서 초급/중급을 추정한다. 근거가 부족하면 초급(beginner)."""
    user_text = " ".join(m["content"] for m in messages if m.get("role") == "user").lower()
    beginner = sum(1 for k in _BEGINNER_KW if k in user_text)
    intermediate = sum(1 for k in _INTERMEDIATE_KW if k.lower() in user_text)
    if intermediate >= 2 and intermediate > beginner:
        return "intermediate"
    if intermediate >= 1 and beginner == 0 and len(user_text) > 40:
        return "intermediate"
    return "beginner"


@dataclass
class ModeContext:
    answer_mode: str
    question_type: QuestionType
    learner_level: LearnerLevel
    mode_instruction: str
    level_instruction: str
    question_type_label: str
    learner_level_label: str


class ModeInstructions:
    """answer_modes.yaml 을 읽어 프롬프트 변수로 만든다."""

    def __init__(self, prompts: PromptStore) -> None:
        self.prompts = prompts

    def build(self, answer_mode: str, question_type: QuestionType, learner_level: LearnerLevel) -> ModeContext:
        data = self.prompts.yaml("answer_modes.yaml")
        modes = data.get("modes", {})
        if answer_mode not in modes:
            raise ValueError(f"알 수 없는 응답 모드 (unknown answer mode): {answer_mode}")
        qtype_instruction = data.get("question_types", {}).get(question_type, "")
        mode_instruction = render_template(modes[answer_mode], qtype_instruction=qtype_instruction.strip()).strip()
        level_instruction = data.get("levels", {}).get(learner_level, "").strip()
        labels = data.get("labels", {})
        return ModeContext(
            answer_mode=answer_mode,
            question_type=question_type,
            learner_level=learner_level,
            mode_instruction=mode_instruction,
            level_instruction=level_instruction,
            question_type_label=labels.get("question_types", {}).get(question_type, question_type),
            learner_level_label=labels.get("levels", {}).get(learner_level, learner_level),
        )
