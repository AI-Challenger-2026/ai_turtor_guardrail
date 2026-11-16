import { useEffect, useMemo, useState } from "react"
import { ChatArea } from "./components/ChatArea"
import { MessageInput } from "./components/MessageInput"
import { RightPanel } from "./components/RightPanel"
import { Sidebar } from "./components/Sidebar"
import { TopBar } from "./components/TopBar"
import {
  checkGuard,
  getTutorResponse,
  type ChatTurn,
  type TutorResponse,
} from "./services/tutorService"

export type Message =
  | { id: number; role: "student"; text: string }
  | { id: number; role: "tutor"; response: TutorResponse }

export type PanelView = "review" | "source" | "guard"

function toTurns(messages: Message[]): ChatTurn[] {
  return messages.flatMap((m): ChatTurn[] =>
    m.role === "student"
      ? [{ role: "user", content: m.text }]
      : m.response.answer
        ? [{ role: "assistant", content: m.response.answer }]
        : [],
  )
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [typing, setTyping] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [panelOpen, setPanelOpen] = useState(
    () => !window.matchMedia("(max-width: 1100px)").matches,
  )
  const [panelView, setPanelView] = useState<PanelView>("review")
  const [activeId, setActiveId] = useState<number | null>(null)

  useEffect(() => {
    const mobileLayout = window.matchMedia("(max-width: 1100px)")
    const handleLayoutChange = (event: MediaQueryListEvent) => setPanelOpen(!event.matches)
    mobileLayout.addEventListener("change", handleLayoutChange)
    return () => mobileLayout.removeEventListener("change", handleLayoutChange)
  }, [])

  const activeResponse = useMemo(() => {
    const m = messages.find((item) => item.id === activeId)
    return m && m.role === "tutor" ? m.response : null
  }, [messages, activeId])

  // The conversation up to the latest student message: the context the guard checks a candidate answer against.
  const guardTurns = useMemo(() => {
    let last = -1
    messages.forEach((m, i) => {
      if (m.role === "student") last = i
    })
    return last < 0 ? [] : toTurns(messages.slice(0, last + 1))
  }, [messages])

  const updateResponse = (id: number, patch: Partial<TutorResponse>) =>
    setMessages((current) =>
      current.map((m) =>
        m.id === id && m.role === "tutor" ? { ...m, response: { ...m.response, ...patch } } : m,
      ),
    )

  const sendMessage = async (text: string) => {
    const clean = text.trim()
    if (!clean || typing) return
    const history: Message[] = [...messages, { id: Date.now(), role: "student", text: clean }]
    setMessages(history)
    setTyping(true)
    const turns = toTurns(history)
    const response = await getTutorResponse(turns)
    const id = Date.now() + 1
    setMessages((current) => [...current, { id, role: "tutor", response }])
    setActiveId(id)
    if (response.sources.length && window.matchMedia("(min-width: 1101px)").matches) {
      setPanelView("source")
      setPanelOpen(true)
    }
    setTyping(false)
    if (response.guardState === "pending") {
      // The guard re-checks the tutor's own answer against the same conversation.
      checkGuard(turns, response.answer)
        .then((guard) => updateResponse(id, { guard, guardState: "done" }))
        .catch(() => updateResponse(id, { guardState: "error" }))
    }
  }

  // ?ask=<question>&panel=<review|source|guard>: open a demo scene directly (used for recording and screenshots).
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const ask = params.get("ask")
    const panel = params.get("panel")
    if (ask) {
      sendMessage(ask).then(() => {
        if (panel === "review" || panel === "source" || panel === "guard") setPanelView(panel)
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const reset = () => {
    setMessages([])
    setTyping(false)
    setSidebarOpen(false)
    setActiveId(null)
    setPanelView("review")
  }

  const openPanel = (id: number, view: PanelView) => {
    setActiveId(id)
    setPanelView(view)
    setPanelOpen(true)
  }

  return (
    <div className="app-shell">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onReset={reset}
        hasAnswer={Boolean(activeResponse)}
      />
      {sidebarOpen && (
        <button className="backdrop sidebar-backdrop" aria-label="메뉴 닫기" onClick={() => setSidebarOpen(false)} />
      )}
      <main className="main-column">
        <TopBar onMenu={() => setSidebarOpen(true)} onPanel={() => setPanelOpen(true)} />
        <ChatArea
          messages={messages}
          typing={typing}
          onPrompt={sendMessage}
          onSource={(id) => openPanel(id, "source")}
          onGuard={(id) => openPanel(id, "guard")}
        />
        <MessageInput onSend={sendMessage} disabled={typing} />
      </main>
      {panelOpen && (
        <button className="backdrop panel-backdrop" aria-label="패널 닫기" onClick={() => setPanelOpen(false)} />
      )}
      <RightPanel
        open={panelOpen}
        view={panelView}
        response={activeResponse}
        guardTurns={guardTurns}
        onViewChange={setPanelView}
        onClose={() => setPanelOpen(false)}
      />
      {!panelOpen && (
        <button className="panel-reopen" onClick={() => setPanelOpen(true)} aria-label="패널 열기">
          ‹
        </button>
      )}
    </div>
  )
}
