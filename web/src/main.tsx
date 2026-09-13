import { StrictMode, useEffect, useRef, useState } from "react"
import { createRoot } from "react-dom/client"

import { api, asBrief, streamChat, type FilmBrief, type Health, type Recommended } from "./api/client"
import type { AgentEvent } from "./api/events"
import { FilmDetail } from "./film/FilmDetail"
import { FilmRow } from "./film/FilmRow"
import { ProfilePanel } from "./profile/ProfilePanel"
import { Waterfall } from "./traces/Waterfall"
import "./styles.css"

type View = "programme" | "taste" | "traces"

type Listed = FilmBrief | Recommended

interface Trace {
  id: string
  line: string
  detail: string
}

interface Turn {
  who: "you" | "palate"
  text: string
  traces: Trace[]
}

function ranked(film: Listed): Recommended | undefined {
  return "score" in film ? film : undefined
}

function lastTurn(turns: Turn[], change: (turn: Turn) => Turn): Turn[] {
  if (turns.length === 0) return turns
  return turns.map((turn, index) => (index === turns.length - 1 ? change(turn) : turn))
}

function Traces({ traces }: { traces: Trace[] }) {
  const [open, setOpen] = useState<string | null>(null)
  return (
    <>
      {traces.map((trace) => (
        <div key={trace.id}>
          <button
            className="trace"
            aria-expanded={open === trace.id}
            onClick={() => setOpen(open === trace.id ? null : trace.id)}
          >
            {trace.line}
          </button>
          {open === trace.id ? <div className="trace__detail">{trace.detail}</div> : null}
        </div>
      ))}
    </>
  )
}

function Listing({ films, reflow }: { films: Listed[]; reflow: number }) {
  const [open, setOpen] = useState<number | null>(null)
  if (films.length === 0) {
    return <p className="empty">Nothing listed yet. Say what you want above, or ask on the right.</p>
  }
  return (
    <ol className="listing listing--reflow" key={reflow}>
      {films.map((film) => (
        <li key={film.film_id}>
          <FilmRow
            film={film}
            open={open === film.film_id}
            onToggle={() => setOpen(open === film.film_id ? null : film.film_id)}
          >
            <FilmDetail filmId={film.film_id} ranked={ranked(film)} />
          </FilmRow>
        </li>
      ))}
    </ol>
  )
}

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

function App() {
  const [view, setView] = useState<View>("programme")
  const [films, setFilms] = useState<Listed[]>([])
  const [reflow, setReflow] = useState(0)
  const [turns, setTurns] = useState<Turn[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [asking, setAsking] = useState(false)
  const [wanted, setWanted] = useState("")
  const session = useRef<string | null>(null)
  const runId = useRef<string | null>(null)
  const abort = useRef<AbortController | null>(null)

  function showProblem(error: Error) {
    setNotice(error.message)
  }

  function list(found: Listed[]) {
    setFilms(found)
    setReflow((n) => n + 1)
  }

  useEffect(() => {
    api.health().then(setHealth).catch(showProblem)
  }, [])

  function programme(query: string) {
    setNotice(null)
    setView("programme")
    api
      .recommend(query)
      .then((answer) => list(answer.films))
      .catch(showProblem)
  }

  function absorb(event: AgentEvent) {
    switch (event.type) {
      case "run.started":
        session.current = event.session_id
        runId.current = event.run_id
        setTurns((old) => [...old, { who: "palate", text: "", traces: [] }])
        return
      case "text.delta":
        if (event.channel !== "answer") return
        setTurns((old) => lastTurn(old, (turn) => ({ ...turn, text: turn.text + event.text })))
        return
      case "tool.finished": {
        const trace = {
          id: event.call_id,
          line: `> ${event.summary}`,
          detail: JSON.stringify(event.meta, null, 2),
        }
        setTurns((old) => lastTurn(old, (turn) => ({ ...turn, traces: [...turn.traces, trace] })))
        return
      }
      case "recommendations":
        list(event.films.map(asBrief))
        setView("programme")
        return
      case "run.failed":
        setNotice(event.message)
        return
      default:
        return
    }
  }

  async function ask(message: string) {
    setNotice(null)
    setTurns((old) => [...old, { who: "you", text: message, traces: [] }])
    setAsking(true)
    abort.current = new AbortController()
    try {
      await streamChat({ message, session_id: session.current }, absorb, abort.current.signal)
    } catch (error) {
      if ((error as Error).name !== "AbortError") showProblem(error as Error)
    } finally {
      setAsking(false)
      abort.current = null
    }
  }

  function stop() {
    if (runId.current !== null) void api.cancel(runId.current)
    abort.current?.abort()
  }

  return (
    <div className="shell">
      <header className="masthead">
        <span className="masthead__name">palate</span>
        <span className="masthead__state num">
          {health === null ? null : (
            <>
              <span>{health.chat.model}</span>
              <span>{health.corpus.eligible} films</span>
              <span>{health.corpus.rated} rated</span>
            </>
          )}
        </span>
      </header>
      <div className="panes">
        <section className="pane pane--listing">
          <div className="pane__head">
            <nav className="views">
              <button aria-current={view === "programme"} onClick={() => setView("programme")}>
                Programme
              </button>
              <button aria-current={view === "taste"} onClick={() => setView("taste")}>
                Taste
              </button>
              <button aria-current={view === "traces"} onClick={() => setView("traces")}>
                Traces
              </button>
            </nav>
            <form
              onSubmit={(event) => {
                event.preventDefault()
                if (wanted.trim().length >= 3) programme(wanted.trim())
              }}
            >
              <input
                className="find"
                value={wanted}
                placeholder="programme for"
                aria-label="programme for"
                onChange={(event) => setWanted(event.target.value)}
              />
            </form>
          </div>
          {notice === null ? null : <p className="notice">{notice}</p>}
          {view === "taste" ? <ProfilePanel /> : null}
          {view === "traces" ? <Waterfall /> : null}
          {view === "programme" ? <Listing films={films} reflow={reflow} /> : null}
        </section>
        <section className="pane">
          <div className="transcript">
            {turns.length === 0 ? (
              <p className="empty">Ask for what you are in the mood for.</p>
            ) : (
              turns.map((turn, index) => (
                <div className={`turn turn--${turn.who}`} key={index}>
                  <div className="turn__who">{turn.who}</div>
                  <div className="turn__body">{turn.text}</div>
                  <Traces traces={turn.traces} />
                </div>
              ))
            )}
          </div>
          <Composer onAsk={(message) => void ask(message)} onStop={stop} asking={asking} />
        </section>
      </div>
    </div>
  )
}

const root = document.getElementById("root")
if (root !== null) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}
