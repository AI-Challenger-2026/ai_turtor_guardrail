// Display-only formatting of the tutor's answer text. The text itself is shown exactly as the server returned it;
// only line breaks, list markers and long paragraphs are laid out for reading.

const LIST_ITEM = /^(?:[-•·]\s+|\d+[.)]\s+)/

function inline(text: string) {
  // `code` spans
  return text.split(/(`[^`]+`)/g).map((part, i) =>
    part.startsWith("`") && part.endsWith("`") && part.length > 2 ? <code key={i}>{part.slice(1, -1)}</code> : part,
  )
}

function splitSentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+(?=\S)/).filter(Boolean)
}

type Block = { kind: "p"; text: string } | { kind: "ul" | "ol"; items: string[] }

function toBlocks(raw: string): Block[] {
  // "… 순서: 1) A 2) B 3) C" written inline: break it into one item per line.
  const text = raw.replace(/\s+(?=\d+\)\s)/g, (m, offset, whole) =>
    (whole.match(/\d+\)\s/g) || []).length >= 2 ? "\n" : m,
  )
  const blocks: Block[] = []
  for (const line of text.split("\n").map((l) => l.trim()).filter(Boolean)) {
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
    for (let i = 0; i < sentences.length; i += 2) blocks.push({ kind: "p", text: sentences.slice(i, i + 2).join(" ") })
  }
  return blocks
}

export function AnswerText({ text }: { text: string }) {
  const blocks = toBlocks(text)
  return (
    <div className="answer-text">
      {blocks.map((b, i) => {
        if (b.kind === "p") {
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
          return <p key={i}>{inline(b.text)}</p>
        }
        const List = b.kind
        return (
          <List key={i}>
            {b.items.map((item, j) => (
              <li key={j}>{inline(item)}</li>
            ))}
          </List>
        )
      })}
    </div>
  )
}

export function ExcerptText({ text }: { text: string }) {
  return <p>{inline(text)}</p>
}
