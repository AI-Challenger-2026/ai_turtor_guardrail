import { useState, type FormEvent } from "react"
import type { PanelView } from "../App"
import { CATEGORY_LABEL } from "../labels"
import { checkGuard, type ChatTurn, type GuardResult, type TutorResponse } from "../services/tutorService"

// Mark every occurrence of the unsupported spans inside the answer text.
function Highlight({ text, spans }: { text: string; spans: string[] }) {
  const parts: Array<{ t: string; hit: boolean }> = []
  let rest = text
  while (rest) {
    let best = -1
    let span = ""
    for (const s of spans) {
      const i = s ? rest.indexOf(s) : -1
      if (i >= 0 && (best < 0 || i < best)) {
        best = i
        span = s
      }
    }
    if (best < 0) {
      parts.push({ t: rest, hit: false })
      break
    }
    if (best > 0) parts.push({ t: rest.slice(0, best), hit: false })
    parts.push({ t: span, hit: true })
    rest = rest.slice(best + span.length)
  }
  return (
    <blockquote className="guard-text">
      {parts.map((p, i) => (p.hit ? <mark key={i}>{p.t}</mark> : <span key={i}>{p.t}</span>))}
    </blockquote>
  )
}

function GuardSummary({ guard }: { guard: GuardResult }) {
  const blocked = guard.input.decision === "block"
  const o = guard.output
  return (
    <dl className="guard-summary">
      <div>
        <dt>입력 판정</dt>
        <dd className={blocked ? "bad" : "good"}>{blocked ? "차단" : "허용"}</dd>
      </div>
      <div>
        <dt>범주</dt>
        <dd>{CATEGORY_LABEL[guard.input.category] ?? guard.input.category}</dd>
      </div>
      {blocked ? (
        <p className="guard-note">막아야 할 입력이라 답변을 내보내지 않도록 표시해요.</p>
      ) : (
        o && (
          <>
            <div>
              <dt>답변 근거</dt>
              <dd className={o.grounded ? "good" : "bad"}>{o.grounded ? "자료로 확인됨" : "근거 없음"}</dd>
            </div>
            {o.unsupported.length > 0 && (
              <div>
                <dt>틀린 구간</dt>
                <dd className="bad">{o.unsupported.map((s) => `“${s}”`).join(", ")}</dd>
              </div>
            )}
            <div>
              <dt>안전</dt>
              <dd className={o.safe ? "good" : "bad"}>{o.safe ? "안전" : "안전하지 않음"}</dd>
            </div>
          </>
        )
      )}
    </dl>
  )
}

function GuardView({ response, guardTurns }: { response: TutorResponse | null; guardTurns: ChatTurn[] }) {
  const [candidate, setCandidate] = useState("")
  const [result, setResult] = useState<{ text: string; guard: GuardResult } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    const text = candidate.trim()
    if (!text || busy || !guardTurns.length) return
    setBusy(true)
    setError("")
    try {
      setResult({ text, guard: await checkGuard(guardTurns, text) })
    } catch {
      setError("가드 검사에 실패했어요. 다시 시도해 주세요.")
    }
    setBusy(false)
  }

  return (
    <div className="source-view guard-view">
      <article>
        <b>◈ 이 답변의 가드 결과</b>
        {!response && <p className="guard-note">질문을 보내면 가드가 튜터의 답변을 다시 검사해요.</p>}
        {response?.guardState === "pending" && <p className="guard-note">검사 중…</p>}
        {response?.guardState === "error" && <p className="guard-note">검사에 실패했어요.</p>}
        {response?.guardState === "done" && response.guard && (
          <>
            {!!response.guard.output?.unsupported.length && (
              <Highlight text={response.answer} spans={response.guard.output.unsupported} />
            )}
            <GuardSummary guard={response.guard} />
          </>
        )}
      </article>
      <article>
        <b>◈ 후보 답변 직접 검사</b>
        <p className="guard-note">
          {guardTurns.length
            ? "지금 대화의 마지막 질문을 기준으로, 아래 답변이 자료와 맞는지 가드가 검사해요."
            : "먼저 질문을 하나 보내 주세요. 가드는 그 대화를 기준으로 검사해요."}
        </p>
        <form className="guard-form" onSubmit={submit}>
          <textarea
            value={candidate}
            onChange={(e) => setCandidate(e.target.value)}
            placeholder="예: 평균 70점은 70점대에 해당해서 C를 출력해요."
            rows={3}
            disabled={!guardTurns.length}
          />
          <button disabled={busy || !candidate.trim() || !guardTurns.length}>{busy ? "검사 중…" : "검사하기"}</button>
        </form>
        {error && <p className="guard-note bad">{error}</p>}
        {result && (
          <>
            <Highlight text={result.text} spans={result.guard.output?.unsupported ?? []} />
            <GuardSummary guard={result.guard} />
          </>
        )}
      </article>
    </div>
  )
}

export function RightPanel({
  open,
  view,
  response,
  guardTurns,
  onViewChange,
  onClose,
}: {
  open: boolean
  view: PanelView
  response: TutorResponse | null
  guardTurns: ChatTurn[]
  onViewChange: (view: PanelView) => void
  onClose: () => void
}) {
  const source = response?.sources[0]
  return (
    <aside className={`right-panel ${open ? "is-open" : ""}`}>
      <div className="right-heading">
        <div className="panel-tabs" aria-label="패널 보기 선택">
          <button className={view === "source" ? "active" : ""} onClick={() => onViewChange("source")} disabled={!source}>
            ▣ 출처
          </button>
          <button className={view !== "source" || !source ? "active" : ""} onClick={() => onViewChange("guard")}>
            ◈ 가드
          </button>
        </div>
        <button onClick={onClose} aria-label="패널 닫기">
          ×
        </button>
      </div>
      {view === "source" && source && (
        <div className="source-unit">
          {source.subject} · {source.id}
        </div>
      )}
      {view === "source" && source ? (
        <div className="source-view">
          <article>
            <b>▣ 수업 자료 원문</b>
            <blockquote>{response?.excerpt || "이 답변에는 원문 문장이 붙지 않았어요."}</blockquote>
            <small>
              {source.id} {source.title}
            </small>
          </article>
        </div>
      ) : (
        // Guard view is the default; it also covers "source" when the answer has no source.
        <GuardView response={response} guardTurns={guardTurns} />
      )}
      <div className="sheet-handle" />
    </aside>
  )
}
