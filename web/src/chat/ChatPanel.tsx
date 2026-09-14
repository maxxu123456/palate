import { useRef, useState } from "react"

import { api, asBrief, streamChat, type FilmBrief } from "../api/client"
import type { AgentEvent } from "../api/events"
import { Transcript, type Turn } from "./Transcript"

function Composer({
  onAsk,
  onStop,
  asking,
}: {
  onAsk: (message: string) => void
  onStop: () => void
  asking: boolean
}) {
  const [draft, setDraft] = useState("")
  const submit = () => {
    if (draft.trim() === "" || asking) return
    onAsk(draft.trim())
    setDraft("")
  }
  return (
    <div className="composer">
      <textarea
        value={draft}
        placeholder="what are you in the mood for"
        aria-label="what are you in the mood for"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault()
            submit()
          }
        }}
      />
      {asking ? (
        <button onClick={onStop}>Stop</button>
      ) : (
        <button onClick={submit} disabled={draft.trim() === ""}>
          Ask
        </button>
      )}
    </div>
  )
}

/** One agent run at a time, written into the transcript. The films go to the listing. */
export function ChatPanel({
  onFilms,
  onProblem,
}: {
  onFilms: (films: FilmBrief[]) => void
  onProblem: (error: Error) => void
}) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [asking, setAsking] = useState(false)
  const session = useRef<string | null>(null)
  const runId = useRef<string | null>(null)
  const abort = useRef<AbortController | null>(null)
  const counter = useRef(0)

  function say(who: Turn["who"], text: string) {
    counter.current += 1
    const id = counter.current
    setTurns((old) => [...old, { id, who, text, tools: [] }])
  }

  function open(edit: (turn: Turn) => Turn) {
    setTurns((old) => old.map((turn, index) => (index === old.length - 1 ? edit(turn) : turn)))
  }

  function absorb(event: AgentEvent) {
    switch (event.type) {
      case "run.started":
        session.current = event.session_id
        runId.current = event.run_id
        say("palate", "")
        return
      case "text.delta":
        // TODO: the thinking channel belongs in the trace line, not in the answer.
        if (event.channel !== "answer") return
        open((turn) => ({ ...turn, text: turn.text + event.text }))
        return
      case "tool.finished": {
        const line = { callId: event.call_id, text: `> ${event.summary}` }
        open((turn) => ({ ...turn, tools: [...turn.tools, line] }))
        return
      }
      case "recommendations":
        // The preamble already streamed. The per film lines are the listing, not the transcript.
        onFilms(event.films.map(asBrief))
        return
      case "run.failed":
        onProblem(new Error(event.message))
        return
      default:
        return
    }
  }

  async function ask(message: string) {
    say("you", message)
    setAsking(true)
    abort.current = new AbortController()
    try {
      await streamChat({ message, session_id: session.current }, absorb, abort.current.signal)
    } catch (error) {
      if ((error as Error).name !== "AbortError") onProblem(error as Error)
    } finally {
      setAsking(false)
      abort.current = null
    }
  }

  // Cancel the run as well as the read, or the server keeps spending on a stream nobody holds.
  function stop() {
    if (runId.current !== null) void api.cancel(runId.current)
    abort.current?.abort()
  }

  return (
    <section className="pane">
      <Transcript turns={turns} streaming={asking} />
      <Composer onAsk={(message) => void ask(message)} onStop={stop} asking={asking} />
    </section>
  )
}
