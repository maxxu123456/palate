import { StrictMode, useEffect, useState } from "react"
import { createRoot } from "react-dom/client"

import { api, type FilmBrief, type Health, type Recommended } from "./api/client"
import { ChatPanel } from "./chat/ChatPanel"
import { CHAT_ENABLED } from "./config"
import { FilmDetail } from "./film/FilmDetail"
import { FilmRow } from "./film/FilmRow"
import { ProfilePanel } from "./profile/ProfilePanel"
import { Waterfall } from "./traces/Waterfall"
import "./styles.css"

type View = "programme" | "taste" | "traces"

type Listed = FilmBrief | Recommended

function ranked(film: Listed): Recommended | undefined {
  return "score" in film ? film : undefined
}

function Listing({ films, reflow }: { films: Listed[]; reflow: number }) {
  const [open, setOpen] = useState<number | null>(null)
  if (films.length === 0) {
    return <p className="empty">Nothing listed yet. Say what you want above.</p>
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

function App() {
  const [view, setView] = useState<View>("programme")
  const [films, setFilms] = useState<Listed[]>([])
  const [reflow, setReflow] = useState(0)
  const [health, setHealth] = useState<Health | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [wanted, setWanted] = useState("")

  function showProblem(error: Error) {
    setNotice(error.message)
  }

  function list(found: Listed[]) {
    setFilms(found)
    setReflow((n) => n + 1)
    setView("programme")
  }

  useEffect(() => {
    api.health().then(setHealth).catch(showProblem)
  }, [])

  function programme(query: string) {
    setNotice(null)
    api
      .recommend(query)
      .then((answer) => list(answer.films))
      .catch(showProblem)
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
      <div className={CHAT_ENABLED ? "panes" : "panes panes--solo"}>
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
        {CHAT_ENABLED ? <ChatPanel onFilms={list} onProblem={showProblem} /> : null}
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
