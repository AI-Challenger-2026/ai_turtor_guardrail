// Calls the demo relay (demo/relay.py). Paths are relative because the page may be served under a tunnel prefix.

export type ChatTurn = { role: "user" | "assistant"; content: string }

export type Source = { id: string; title: string; subject: string }

export type GuardResult = {
  input: { decision: "allow" | "block"; category: string }
  output?: { grounded: boolean; unsupported: string[]; safe: boolean }
}

export type TutorResponse = {
  action: string
  answer: string // full answer text, sent back as conversation history
  main: string
  excerpt: string
  note: string
  sources: Source[]
  citations: string[]
  guard: GuardResult | null
  guardState: "pending" | "done" | "error"
  error?: string
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  const data = await res.json()
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`)
  return data as T
}

export async function getTutorResponse(messages: ChatTurn[]): Promise<TutorResponse> {
  try {
    const data = await post<{
      raw: { answer: string; action: string; citations: string[] }
      main: string
      excerpt: string
      note: string
      sources: Source[]
    }>("api/tutor", { messages })
    return {
      action: data.raw.action,
      answer: data.raw.answer,
      main: data.main,
      excerpt: data.excerpt,
      note: data.note,
      sources: data.sources,
      citations: data.raw.citations,
      guard: null,
      guardState: "pending",
    }
  } catch (e) {
    return {
      action: "error",
      answer: "",
      main: "서버에 연결하지 못했어요. 잠시 후 다시 시도해 주세요.",
      excerpt: "",
      note: "",
      sources: [],
      citations: [],
      guard: null,
      guardState: "error",
      error: e instanceof Error ? e.message : String(e),
    }
  }
}

export function checkGuard(messages: ChatTurn[], candidate?: string): Promise<GuardResult> {
  return post<GuardResult>("api/guard", candidate ? { messages, candidate_answer: candidate } : { messages })
}
