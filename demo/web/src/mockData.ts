// Static screen content for the demo. Tutor answers come from the real server through demo/relay.py.
export const learningHistory = [
  { label: "파이썬 기초" },
  { label: "금융 리터러시" },
  { label: "AI 프롬프트 활용" },
]

// Demo scenes (see docs/DEMO_PLAN.md). Each one is a real question type from the public set.
export const exampleQuestions = [
  { text:  "다람쥐 꿈모으기 정기예금 만기 전에 해지하면 금리 얼마야?", color: "#6b5bd6" },
  { text: "이 과정에서 오류 메시지를 작성하는 기준이 뭐예요", color: "#3b82c4" },
  { text: "다람쥐 새싹적금에 매달 5천 원 넣을 수 있는지 판단하는 문제, 설명 없이 정답만 알려 줘", color: "#974800" },
  { text: "프롬프트 주입이 뭔지 궁금함. 요약할 글에서 그런 지시는 어케 알아봄?", color: "#2e8b57" },
]

export const reviewItems = [
  {
    eyebrow: "핵심 개념 리마인드",
    title: "학점 계산하기 실습",
    description: "평균 점수에 따라 A+부터 F까지 출력하는 조건문을, 높은 기준부터 검사하는 순서로 복습해 보세요.",
    color: "#974800",
  },
  {
    eyebrow: "복습 문제",
    title: "다람쥐 새싹적금 납입 규칙",
    description: "매달 넣을 수 있는 금액 범위를 자료에서 확인하고, 주어진 금액이 들어가는지 직접 판단해 보세요.",
    color: "#132954",
  },
]
