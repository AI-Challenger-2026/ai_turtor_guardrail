// Korean labels for the server's action and category values (SPEC §4, §5).
export const ACTION_LABEL: Record<string, string> = {
  answer: "답변",
  scaffold: "힌트",
  abstain: "자료에 없음",
  refuse: "거절",
  redirect: "수업으로 전환",
  escalate: "도움 안내",
  clarify: "되묻기",
}

export const CATEGORY_LABEL: Record<string, string> = {
  normal: "정상",
  pii_disclosure: "개인정보 노출",
  out_of_material: "자료 밖 질문",
  answer_seeking: "정답 요구",
  academic_dishonesty: "과제 대필",
  jailbreak_injection: "탈옥·지시 주입",
  off_topic: "수업과 무관",
  emotional_distress: "정서적 어려움",
  low_signal_input: "의미 없는 입력",
  harmful_content: "유해 요청",
}
