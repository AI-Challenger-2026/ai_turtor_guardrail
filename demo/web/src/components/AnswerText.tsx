// Display-only formatting of the tutor's answer text. The text itself is shown exactly as the server returned it;
// headings, Q&A, quotes, lists and long paragraphs are only laid out for reading.
import { useState, type ReactNode } from "react"

const LIST_ITEM = /^(?:[-•·]\s+|\d+[.)]\s+)/
const MD_HEADING = /^#{1,6}\s+(.*)$/
const QUESTION = /^Q[.:]\s*(.*)$/
const ANSWER = /^A[.:]\s*(.*)$/
const FOLD_AFTER = 8 // blocks shown before "전체 보기"

function inline(text: string) {
  // `code` spans
  return text.split(/(`[^`]+`)/g).map((part, i) =>
    part.startsWith("`") && part.endsWith("`") && part.length > 2 ? <code key={i}>{part.slice(1, -1)}</code> : part,
  )
}

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+(?=\S)/).filter(Boolean)
}

// A short line with no sentence ending reads as a section heading ("울타리로 자료와 지시 구분하기").
function looksLikeHeading(line: string) {
  return line.length <= 30 && !/[.!?…:)\]"”'’>]$/.test(line) && !/[,，]/.test(line)
}

// Code detection: a line with no Hangul outside string literals that looks like Python.
const PY_START = /^(if|elif|else|for|while|def|return|print|import|from|class|try|except|with|break|continue|pass)\b/
function isCodeLine(raw: string) {
  const t = raw.trim()
  if (!t) return false
  const bare = t.replace(/(['"])(?:\\.|(?!\1).)*\1/g, '""')
  if (/[가-힣]/.test(bare)) return false
  if (PY_START.test(t)) return true
  if (/^[A-Za-z_][\w.[\]]*\s*(=|\+=|-=|\*=|\/=)\s*\S/.test(t)) return true
  if (/^[A-Za-z_][\w.]*\(.*\)\s*$/.test(t)) return true
  return /^\s+\S/.test(raw) && /[=():[\]]/.test(t)
}

function dedent(lines: string[]) {
  const pad = Math.min(...lines.filter((l) => l.trim()).map((l) => l.match(/^\s*/)![0].length))
  return lines.map((l) => l.slice(pad)).join("\n").replace(/\s+$/, "")
}

const PY_TOKEN =
  /(#.*$)|('(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")|\b(if|elif|else|for|while|def|return|import|from|in|not|and|or|True|False|None|break|continue|pass|class|try|except|with|as|lambda)\b|\b(print|input|int|str|float|len|range|map|list|dict|set|sum|open|split)\b|\b(\d+(?:\.\d+)?)\b/gm

function highlight(code: string) {
  const out: ReactNode[] = []
  let last = 0
  for (const m of code.matchAll(PY_TOKEN)) {
    const i = m.index ?? 0
    if (i > last) out.push(code.slice(last, i))
    const cls = m[1] ? "tk-com" : m[2] ? "tk-str" : m[3] ? "tk-kw" : m[4] ? "tk-fn" : "tk-num"
    out.push(<span key={i} className={cls}>{m[0]}</span>)
    last = i + m[0].length
  }
  if (last < code.length) out.push(code.slice(last))
  return out
}

type Block =
  | { kind: "code"; text: string }
  | { kind: "p"; text: string; inAnswer: boolean }
  | { kind: "ul" | "ol"; items: string[] }
  | { kind: "h"; text: string }
  | { kind: "q"; text: string }
  | { kind: "a"; text: string }
  | { kind: "quote"; text: string }

function toBlocks(raw: string): Block[] {
  // "… 순서: 1) A 2) B 3) C" written inline: break it into one item per line.
  const text = raw.replace(/\s+(?=\d+\)\s)/g, (m, _offset, whole: string) =>
    (whole.match(/\d+\)\s/g) || []).length >= 2 ? "\n" : m,
  )
  // Pull code out first (it needs the untrimmed lines for indentation); mark each code run as one placeholder line.
  const rawLines = text.split("\n")
  const codes: string[] = []
  const prepared: string[] = []
  for (let r = 0; r < rawLines.length; r++) {
    if (/^\s*```/.test(rawLines[r])) {
      const body: string[] = []
      while (++r < rawLines.length && !/^\s*```/.test(rawLines[r])) body.push(rawLines[r])
      prepared.push(`\u0000CODE${codes.length}`)
      codes.push(dedent(body))
      continue
    }
    if (isCodeLine(rawLines[r])) {
      const body = [rawLines[r]]
      // keep going over code lines; a blank line stays inside the block only if code follows it
      while (r + 1 < rawLines.length) {
        const next = rawLines[r + 1]
        const after = rawLines[r + 2] ?? ""
        if (isCodeLine(next)) body.push(rawLines[++r])
        else if (!next.trim() && isCodeLine(after)) body.push(rawLines[++r])
        else break
      }
      prepared.push(`\u0000CODE${codes.length}`)
      codes.push(dedent(body))
      continue
    }
    prepared.push(rawLines[r])
  }

  const lines = prepared.map((l) => l.trim()).filter(Boolean)
  const blocks: Block[] = []
  let inAnswer = false
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i]

    const code = line.match(/^\u0000CODE(\d+)$/)
    if (code) {
      blocks.push({ kind: "code", text: codes[Number(code[1])] })
      continue
    }

    // Quoted example, possibly spread over several lines: “ … ”
    if (/^[“"]/.test(line) && !/[”"]$/.test(line.slice(1))) {
      const parts = [line]
      while (i + 1 < lines.length && !/[”"]/.test(parts[parts.length - 1].slice(parts.length === 1 ? 1 : 0))) {
        parts.push(lines[++i])
      }
      blocks.push({ kind: "quote", text: parts.join("\n") })
      continue
    }
    if (/^[“"].*[”"]$/.test(line) && line.length > 2) {
      blocks.push({ kind: "quote", text: line })
      continue
    }

    const md = line.match(MD_HEADING)
    if (md) line = md[1].trim()
    const q = line.match(QUESTION)
    if (q) {
      blocks.push({ kind: "q", text: q[1] })
      inAnswer = false
      continue
    }
    const a = line.match(ANSWER)
    if (a) {
      blocks.push({ kind: "a", text: a[1] })
      inAnswer = true
      continue
    }
    if (md || (i > 0 && looksLikeHeading(line) && !LIST_ITEM.test(line))) {
      blocks.push({ kind: "h", text: line })
      inAnswer = false
      continue
    }

    const m = line.match(LIST_ITEM)
    if (m) {
      const kind = /\d/.test(m[0]) ? "ol" : "ul"
      const item = line.slice(m[0].length)
      const last = blocks[blocks.length - 1]
      if (last && last.kind === kind) last.items.push(item)
      else blocks.push({ kind, items: [item] })
      continue
    }
    // Long paragraphs: two sentences per paragraph.
    const sentences = splitSentences(line)
    for (let k = 0; k < sentences.length; k += 2) {
      blocks.push({ kind: "p", text: sentences.slice(k, k + 2).join(" "), inAnswer })
    }
  }
  return blocks
}

function renderBlock(b: Block, i: number) {
  switch (b.kind) {
    case "code":
      return (
        <pre key={i} className="answer-code">
          <code>{highlight(b.text)}</code>
        </pre>
      )
    case "h":
      return <h4 key={i}>{inline(b.text)}</h4>
    case "q":
      return (
        <p key={i} className="qa-q">
          <span className="qa-mark">Q</span>
          {inline(b.text)}
        </p>
      )
    case "a":
      return (
        <p key={i} className="qa-a">
          <span className="qa-mark">A</span>
          {inline(b.text)}
        </p>
      )
    case "quote":
      return (
        <blockquote key={i} className="answer-quote">
          {inline(b.text)}
        </blockquote>
      )
    case "ul":
    case "ol": {
      const List = b.kind
      return (
        <List key={i}>
          {b.items.map((item, j) => (
            <li key={j}>{inline(item)}</li>
          ))}
        </List>
      )
    }
    default:
      if (i === 0) {
        // The first sentence is usually the direct answer: emphasise it.
        const [lead, ...rest] = splitSentences(b.text)
        return (
          <p key={i}>
            <strong className="answer-lead">{inline(lead)}</strong>
            {rest.length > 0 && <> {inline(rest.join(" "))}</>}
          </p>
        )
      }
      return (
        <p key={i} className={b.inAnswer ? "qa-body" : undefined}>
          {inline(b.text)}
        </p>
      )
  }
}

export function AnswerText({ text }: { text: string }) {
  const blocks = toBlocks(text)
  const [expanded, setExpanded] = useState(false)
  const foldable = blocks.length > FOLD_AFTER + 2
  const shown = foldable && !expanded ? blocks.slice(0, FOLD_AFTER) : blocks
  return (
    <div className="answer-text">
      {shown.map(renderBlock)}
      {foldable && (
        <button className="answer-more" onClick={() => setExpanded(!expanded)}>
          {expanded ? "접기 ▴" : `전체 보기 ▾ (${blocks.length - FOLD_AFTER}개 문단 더)`}
        </button>
      )}
    </div>
  )
}

export function ExcerptText({ text }: { text: string }) {
  return <p>{inline(text)}</p>
}
