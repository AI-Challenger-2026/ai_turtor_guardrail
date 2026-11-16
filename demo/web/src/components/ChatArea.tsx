import { useEffect, useRef } from "react"
import type { Message } from "../App"
import { ACTION_LABEL, CATEGORY_LABEL } from "../labels"
import { AnswerText, ExcerptText } from "./AnswerText"
import { exampleQuestions } from "../mockData"
import type { TutorResponse } from "../services/tutorService"

function GuardLine({ response, onClick }: { response: TutorResponse; onClick: () => void }) {
  if (response.action === "error") return null
  let cls = "pending"
  let text = "가드 검사 중…"
  if (response.guardState === "error") {
    cls = "warn"
    text = "가드 검사에 실패했어요"
  } else if (response.guardState === "done" && response.guard) {
    const g = response.guard
    const cat = CATEGORY_LABEL[g.input.category] ?? g.input.category
    if (g.input.decision === "block") {
      cls = "block"
      text = `가드: 입력 차단 · ${cat}`
    } else {
      const o = g.output
      const verdict = !o ? "" : o.grounded ? " · 답변 근거 확인" : ` · 근거 없는 구간 ${o.unsupported.length}곳`
      cls = o && !o.grounded ? "warn" : "ok"
      text = `가드: 입력 허용 · ${cat}${verdict}`
    }
  }
  return (
    <button className={`guard-line ${cls}`} onClick={onClick}>
      ◈ {text}
    </button>
  )
}

function TutorAnswer({
  response,
  onSource,
  onGuard,
}: {
  response: TutorResponse
  onSource: () => void
  onGuard: () => void
}) {
  return (
    <div className="answer-wrap">
      <div className="tutor-name">
        <span>●</span> tutor <small>방금</small>
        {response.action !== "error" && (
          <b className={`action-badge action-${response.action}`}>{ACTION_LABEL[response.action] ?? response.action}</b>
        )}
      </div>
      <article className="answer-card">
        <AnswerText text={response.main} />
        {response.excerpt && (
          <div className="excerpt-block">
            <b>수업 자료 원문</b>
            <ExcerptText text={response.excerpt} />
          </div>
        )}
        {response.note && <p className="pii-note">ⓘ {response.note}</p>}
        {response.sources.length > 0 && (
          <div className="answer-resources">
            {response.sources.map((source) => (
              <button className="source-chip" key={source.id} onClick={onSource}>
                ▣ {source.id} · {source.title} ↗
              </button>
            ))}
          </div>
        )}
        <div className="feedback-row">
          <GuardLine response={response} onClick={onGuard} />
        </div>
      </article>
    </div>
  )
}

type Props = {
  messages: Message[]
  typing: boolean
  onPrompt: (text: string) => void
  onSource: (id: number) => void
  onGuard: (id: number) => void
}
export function ChatArea({ messages, typing, onPrompt, onSource, onGuard }: Props) {
  const started = messages.length > 0 || typing
  const conversationEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [messages, typing])

  return (
    <section className={`chat-area ${started ? "started" : ""}`}>
      {!started ? (
        <div className="empty-state">
          <div className="tutor-mark">
            <span>
              <img src="assets/928fb.svg" alt="" />
            </span>
          </div>
          <h1>오늘 뭐가 궁금해요?</h1>
          
          <div className="prompt-list">
            <p>이런 질문은 어때요?</p>
            <div>
              {exampleQuestions.map((question) => (
                <button key={question.text} onClick={() => onPrompt(question.text)}>
                  <i style={{ background: question.color }} />
                  {question.text}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="conversation">
          <div className="messages" aria-live="polite">
            {messages.map((message) =>
              message.role === "student" ? (
                <div className="message-row student" key={message.id}>
                  <p>{message.text}</p>
                </div>
              ) : (
                <TutorAnswer
                  key={message.id}
                  response={message.response}
                  onSource={() => onSource(message.id)}
                  onGuard={() => onGuard(message.id)}
                />
              ),
            )}
            {typing && (
              <div className="message-row tutor">
                <span className="message-avatar">✦</span>
                <p className="typing">
                  <i />
                  <i />
                  <i />
                </p>
              </div>
            )}
            <div className="conversation-end" ref={conversationEndRef} aria-hidden="true" />
          </div>
        </div>
      )}
    </section>
  )
}
