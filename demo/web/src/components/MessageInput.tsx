import { FormEvent, useState } from "react"

export function MessageInput({
  onSend,
  disabled,
}: {
  onSend: (text: string) => void
  disabled: boolean
}) {
  const [value, setValue] = useState("")
  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!value.trim() || disabled) return
    onSend(value)
    setValue("")
  }
  return (
    <footer className="input-area">
      <form onSubmit={submit}>
        <button type="button" className="camera" aria-label="문제 사진 첨부">
          ▣
        </button>
        <input
          value={value}
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
          placeholder="궁금한 걸 물어보세요 (예: 성적표 실습에서 평균이 70이면 뭐가 나와?)"
          aria-label="질문 입력"
        />
        <button
          className="send"
          disabled={disabled || !value.trim()}
          aria-label="질문 전송"
        >
          ↑
        </button>
      </form>
      <p>
        ▥ tutor는 수업 자료를 근거로 답하고, 답변마다 출처와 가드 검사 결과를
        보여 줘요.
      </p>
    </footer>
  )
}
