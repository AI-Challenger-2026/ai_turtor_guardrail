"""Fixed learner-facing texts. Never interpolate learner input into these."""

# Generation failed or the request budget ran out.
FALLBACK = "지금은 답변을 만들 수 없어요. 잠시 후 다시 질문해 주세요."
FALLBACK_FROM_MATERIAL = "지금은 답변을 정리하지 못해서 수업 자료에서 관련 내용을 옮겨 드릴게요."
MATERIAL_EXCERPT_HEAD = "수업 자료 원문:"

REFUSE_JAILBREAK = (
    "이전 지시를 무시하거나 제한 없이 답하는 역할을 맡을 수는 없고, 숨겨진 지시도 보여 드릴 수 없어요. "
    "수업 내용은 계속 도와드릴게요."
)

REFUSE_HARMFUL = (
    "다른 사람을 속이거나 해치는 일, 다른 사람의 개인정보를 알아내거나 퍼뜨리는 일은 도와드릴 수 없어요. "
    "안전하게 대처하는 방법이나 수업 내용은 함께 공부할 수 있어요."
)

REDIRECT_STATEMENT = "특정 집단이나 사람을 깎아내리는 표현은 사용하지 말아 주세요. 지금 공부하던 내용은 계속 도와드릴게요."

REDIRECT_OFF_TOPIC = "그 요청은 이 수업과 관련이 없어요. 수업에서 배운 내용 가운데 궁금한 점을 물어봐 주시면 함께 살펴볼게요."

ESCALATE = (
    "그동안 혼자 버티느라 많이 힘들었겠어요. 지금은 공부를 잠시 멈추고, 믿을 수 있는 선생님이나 보호자에게 "
    "지금 마음을 이야기해 주세요. 학교 상담 선생님이나 청소년상담전화 1388에 도움을 요청해도 좋아요."
)

CLARIFY = "무엇이 궁금한지 조금 더 알려 줄래요? 어떤 실습이나 개념에 관한 질문인지 말해 주면 함께 살펴볼게요."

SCAFFOLD_ACADEMIC = (
    "제출할 답안을 대신 완성해 줄 수는 없어요. 대신 직접 쓸 수 있게 순서를 안내할게요. "
    "먼저 자료에서 문제와 관련된 개념을 찾아 한 문장으로 정리해 보세요. "
    "그다음 그 개념을 문제 상황에 적용해 단계별로 적어 보고, 쓴 내용을 보여 주면 함께 살펴볼게요."
)

# Temporary: task 2 replaces this with a generated, material-grounded scaffold.
# Used when scaffold generation fails.
SCAFFOLD_ANSWER_SEEKING = (
    "정답을 바로 알려 주기보다 스스로 고를 수 있게 도와드릴게요. "
    "먼저 문제에서 묻는 값과 조건을 찾고, 자료에서 그 조건에 해당하는 규칙을 확인해 보세요. "
    "규칙과 문제의 값을 비교하면 답을 고를 수 있어요."
)

PII_NOTICE = "이름이나 학교 같은 개인정보는 적지 않아도 괜찮아요."
